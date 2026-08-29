"""
yfinance data client — zero-cost replacement for FMP.

yfinance wraps Yahoo Finance and provides:
  - Full daily OHLCV history (adjusted)
  - Company profile (sector, industry, market cap, description)
  - Income statement, balance sheet, cash flow (annual + quarterly)
  - Key metrics (P/E, EV/EBITDA, ROE, etc.) via .info dict
  - Institutional holders and insider transactions
  - Recent news headlines

S&P 500 universe is fetched from Wikipedia (no API key needed).
NASDAQ-100 universe is fetched from Wikipedia as well.
"""
import time
import requests
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional
from loguru import logger


# ── Universe helpers ──────────────────────────────────────────────────────────

# Wikipedia blocks the default pandas/requests User-Agent — use a browser UA.
_WIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}

# Hardcoded fallback — top 100 liquid US equities used when Wikipedia is down.
_SP500_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO",
    "JPM","TSLA","UNH","V","XOM","MA","PG","HD","COST","JNJ","MRK","ABBV",
    "BAC","CRM","ORCL","CVX","WMT","NFLX","AMD","TMO","KO","LIN","PEP","ACN",
    "MCD","CSCO","ADBE","ABT","WFC","CAT","GE","TXN","NOW","QCOM","AMGN",
    "IBM","INTU","ISRG","MS","GS","SPGI","BLK","SYK","SCHW","RTX","BKNG",
    "VRTX","PLD","AXP","ADI","DE","MDT","GILD","ETN","MMC","CB","LRCX","REGN",
    "MU","ELV","BSX","ADP","PH","ZTS","PANW","KLAC","SNPS","CME","CI","CDNS",
    "SO","PYPL","DUK","SHW","TJX","APH","MCO","FI","AON","USB","ITW","PNC",
    "ECL","CARR","TGT","CL","GD","NOC","MMM","HCA","COF","WELL","CTAS","NSC",
]


def get_sp500_symbols() -> List[str]:
    """Fetch current S&P 500 tickers from Wikipedia."""
    try:
        table = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            header=0,
            storage_options={"User-Agent": _WIKI_HEADERS["User-Agent"]},
        )[0]
        symbols = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"S&P 500: {len(symbols)} symbols fetched from Wikipedia")
        return symbols
    except Exception as exc:
        logger.warning(f"Could not fetch S&P 500 from Wikipedia: {exc} — using fallback list")
        return _SP500_FALLBACK.copy()


# Top NASDAQ-100 names not typically in S&P 500 (growth/tech supplement)
_NDX_EXTRA_FALLBACK = [
    "MELI","ASML","TEAM","DDOG","CRWD","SNOW","ZS","OKTA","FTNT","MDB",
    "HUBS","WBD","GEHC","CEG","FANG","IDXX","ROST","FAST","CSGP","ANSS",
]


def get_nasdaq100_symbols() -> List[str]:
    """
    Return NASDAQ-100 tickers.
    The Wikipedia Nasdaq-100 page no longer exposes a parseable constituent
    table, so we return a curated supplement of NDX names not in the S&P 500.
    """
    logger.info(f"NASDAQ-100: using curated supplement ({len(_NDX_EXTRA_FALLBACK)} tickers)")
    return _NDX_EXTRA_FALLBACK.copy()


# ── yfinance client ───────────────────────────────────────────────────────────

class YFClient:
    """
    Thin wrapper around yfinance with caching, error handling,
    and a consistent interface matching what the rest of the engine expects.
    """

    def __init__(self, throttle: float = 0.1):
        self.throttle = throttle
        self._cache: Dict[str, yf.Ticker] = {}

    def _ticker(self, symbol: str) -> yf.Ticker:
        if symbol not in self._cache:
            self._cache[symbol] = yf.Ticker(symbol)
        return self._cache[symbol]

    # ── Price History ─────────────────────────────────────────────────────────

    def get_daily_ohlcv(self, symbol: str, days: int = 400) -> pd.DataFrame:
        """
        Return daily adjusted OHLCV sorted oldest → newest.
        Columns: date, Open, High, Low, Close, Volume, Adj_Close
        """
        try:
            period = f"{min(days // 30 + 2, 60)}mo"
            tk = self._ticker(symbol)
            df = tk.history(period=period, auto_adjust=True, actions=False)
            if df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            # yfinance may use "Date" or "Datetime" as index column name
            date_col = "Datetime" if "Datetime" in df.columns else "Date"
            df = df.rename(columns={
                date_col: "date",
                "Open":   "Open",
                "High":   "High",
                "Low":    "Low",
                "Close":  "Close",
                "Volume": "Volume",
            })
            # Strip timezone info so pandas doesn't complain
            df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(None) if pd.api.types.is_datetime64tz_dtype(df["date"]) else pd.to_datetime(df["date"])
            df["Adj_Close"] = df["Close"]
            cols = ["date", "Open", "High", "Low", "Close", "Volume", "Adj_Close"]
            df = df[[c for c in cols if c in df.columns]]
            df = df.sort_values("date").reset_index(drop=True)
            # Drop incomplete intraday bar (today's row often has NaN OHLC but a volume)
            df = df.dropna(subset=["Close"]).reset_index(drop=True)
            time.sleep(self.throttle)
            return df
        except Exception as exc:
            logger.debug(f"OHLCV fetch failed for {symbol}: {exc}")
            return pd.DataFrame()

    # ── Company Info ──────────────────────────────────────────────────────────

    def get_company_profile(self, symbol: str) -> Dict:
        """Return company info dict with sector, market cap, description."""
        try:
            info = self._ticker(symbol).info
            return {
                "symbol":        symbol,
                "companyName":   info.get("longName", symbol),
                "sector":        info.get("sector", ""),
                "industry":      info.get("industry", ""),
                "description":   info.get("longBusinessSummary", ""),
                "exchangeShortName": info.get("exchange", ""),
                "mktCap":        info.get("marketCap"),
                "country":       info.get("country", "US"),
                "website":       info.get("website", ""),
                "fullTimeEmployees": info.get("fullTimeEmployees"),
            }
        except Exception as exc:
            logger.debug(f"Profile fetch failed for {symbol}: {exc}")
            return {"symbol": symbol}

    # ── Fundamental Metrics ───────────────────────────────────────────────────

    def get_key_metrics_ttm(self, symbol: str) -> Dict:
        """
        Extract key financial metrics from yfinance .info dict.
        Maps to the same field names the fundamental analyser expects.
        """
        try:
            info = self._ticker(symbol).info
            # yfinance returns ratios as decimals for some, percentages for others
            def _pct(key):
                v = info.get(key)
                return float(v) * 100 if v is not None else None

            def _f(key):
                v = info.get(key)
                return float(v) if v is not None else None

            return {
                # Quality
                "roicTTM":                  _pct("returnOnEquity"),   # proxy
                "roeTTM":                   _pct("returnOnEquity"),
                "freeCashFlowYieldTTM":     None,  # computed below
                "debtToEquityTTM":          _f("debtToEquity"),
                "currentRatioTTM":          _f("currentRatio"),
                # Valuation
                "enterpriseValueOverEBITDATTM": _f("enterpriseToEbitda"),
                "priceToFreeCashFlowsRatioTTM": _f("priceToFreeCashflows"),
                # Extra
                "_marketCap":   info.get("marketCap"),
                "_freeCashflow":info.get("freeCashflow"),
                "_price":       info.get("currentPrice") or info.get("regularMarketPrice"),
                "_forwardPE":   _f("forwardPE"),
                "_trailingPE":  _f("trailingPE"),
                "_beta":        _f("beta"),
            }
        except Exception as exc:
            logger.debug(f"Key metrics failed for {symbol}: {exc}")
            return {}

    def get_ratios_ttm(self, symbol: str) -> Dict:
        try:
            info = self._ticker(symbol).info
            def _f(key):
                v = info.get(key)
                return float(v) if v is not None else None
            def _pct(key):
                v = info.get(key)
                return float(v) * 100 if v is not None else None

            peg = _f("pegRatio")
            # If PEG is missing, approximate: forward PE / (EPS growth * 100)
            return {
                "pegRatioTTM":              peg,
                "netProfitMarginTTM":       _pct("profitMargins"),
                "grossProfitMarginTTM":     _pct("grossMargins"),
                "priceToBookRatioTTM":      _f("priceToBook"),
                "priceEarningsRatioTTM":    _f("trailingPE"),
                "operatingMarginTTM":       _pct("operatingMargins"),
                "ebitdaMarginsTTM":         _pct("ebitdaMargins"),
            }
        except Exception as exc:
            logger.debug(f"Ratios failed for {symbol}: {exc}")
            return {}

    def get_income_statement(
        self, symbol: str, period: str = "quarter", limit: int = 8
    ) -> List[Dict]:
        """
        Return list of income statement dicts (newest first).
        period: "quarter" | "annual"
        """
        try:
            tk = self._ticker(symbol)
            if period == "quarter":
                fin = tk.quarterly_financials
            else:
                fin = tk.financials

            if fin is None or fin.empty:
                return []

            results = []
            for col in fin.columns[:limit]:
                row = {"date": str(col)}
                eps_shares = None
                try:
                    shares = tk.info.get("sharesOutstanding") or 1
                    net_income = fin.loc["Net Income", col] if "Net Income" in fin.index else None
                    revenue    = fin.loc["Total Revenue", col] if "Total Revenue" in fin.index else None
                    if net_income and shares:
                        eps_shares = float(net_income) / float(shares)
                    row["eps"]     = eps_shares
                    row["revenue"] = float(revenue) if revenue is not None else None
                    row["netIncome"] = float(net_income) if net_income is not None else None
                except Exception:
                    pass
                results.append(row)
            return results
        except Exception as exc:
            logger.debug(f"Income statement failed for {symbol}: {exc}")
            return []

    def get_earnings_surprises(self, symbol: str, limit: int = 4) -> List[Dict]:
        """Return recent EPS surprise data."""
        try:
            tk = self._ticker(symbol)
            hist = tk.earnings_history
            if hist is None or hist.empty:
                return []
            results = []
            for _, row in hist.head(limit).iterrows():
                results.append({
                    "actualEarningResult": row.get("epsActual"),
                    "estimatedEarning":    row.get("epsEstimate"),
                    "date": str(row.name) if hasattr(row, "name") else "",
                })
            return results
        except Exception as exc:
            logger.debug(f"Earnings surprises failed for {symbol}: {exc}")
            return []

    def get_institutional_holders(self, symbol: str) -> List[Dict]:
        try:
            tk = self._ticker(symbol)
            ih = tk.institutional_holders
            if ih is None or ih.empty:
                return []
            return ih.to_dict("records")
        except Exception as exc:
            logger.debug(f"Institutional holders failed for {symbol}: {exc}")
            return []

    def get_insider_transactions(self, symbol: str, limit: int = 20) -> List[Dict]:
        try:
            tk = self._ticker(symbol)
            it = tk.insider_transactions
            if it is None or it.empty:
                return []
            results = []
            for _, row in it.head(limit).iterrows():
                text = str(row.get("Text", "")).upper()
                t_type = "P-PURCHASE" if "PURCHASE" in text or "BUY" in text else "S-SALE"
                results.append({"transactionType": t_type, "shares": row.get("Shares")})
            return results
        except Exception as exc:
            logger.debug(f"Insider transactions failed for {symbol}: {exc}")
            return []

    def get_short_interest(self, symbol: str) -> Dict:
        """Return short interest metrics from yfinance."""
        try:
            info = self._ticker(symbol).info
            pct_raw = info.get("shortPercentOfFloat")
            ratio_raw = info.get("shortRatio")
            shares_short = info.get("sharesShort")
            return {
                "short_pct_float": round(float(pct_raw) * 100, 2) if pct_raw is not None else None,
                "short_ratio":     round(float(ratio_raw), 1)      if ratio_raw is not None else None,
                "shares_short":    int(shares_short)                if shares_short else None,
            }
        except Exception as exc:
            logger.debug(f"Short interest failed for {symbol}: {exc}")
            return {}

    def get_next_earnings_date(self, symbol: str) -> Dict:
        """Return next scheduled earnings date and days until it."""
        from datetime import date as _date, datetime as _dt
        try:
            tk = self._ticker(symbol)
            # Try .calendar (DataFrame or dict depending on yfinance version)
            cal = tk.calendar
            dates = []
            if cal is not None:
                if hasattr(cal, "loc") and "Earnings Date" in (cal.index if hasattr(cal, "index") else []):
                    raw = cal.loc["Earnings Date"]
                    dates = list(raw) if hasattr(raw, "__iter__") else [raw]
                elif isinstance(cal, dict):
                    raw = cal.get("Earnings Date", [])
                    dates = list(raw) if hasattr(raw, "__iter__") else [raw]

            # Fallback: earningsDate from info (Unix timestamp or ISO string)
            if not dates:
                ed = tk.info.get("earningsDate")
                if ed:
                    dates = [ed] if not isinstance(ed, list) else ed

            today = _date.today()
            future = []
            for d in dates:
                try:
                    if isinstance(d, (int, float)):
                        dt = _dt.utcfromtimestamp(d).date()
                    elif hasattr(d, "date"):
                        dt = d.date()
                    else:
                        dt = _dt.strptime(str(d)[:10], "%Y-%m-%d").date()
                    if dt >= today:
                        future.append(dt)
                except Exception:
                    pass

            if future:
                next_d = min(future)
                return {
                    "next_earnings_date": str(next_d),
                    "days_to_earnings":   (next_d - today).days,
                }
        except Exception as exc:
            logger.debug(f"Earnings date failed for {symbol}: {exc}")
        return {}

    def get_analyst_estimates(self, symbol: str, **kwargs) -> List[Dict]:
        try:
            info = self._ticker(symbol).info
            return [{
                "estimatedEpsAvg":     info.get("forwardEps"),
                "estimatedRevenueAvg": info.get("revenueEstimatesAvg"),
            }]
        except Exception:
            return []

    def get_price_target_summary(self, symbol: str) -> Dict:
        try:
            info = self._ticker(symbol).info
            return {
                "targetHigh":     info.get("targetHighPrice"),
                "targetLow":      info.get("targetLowPrice"),
                "targetMean":     info.get("targetMeanPrice"),
                "totalAnalysts":  info.get("numberOfAnalystOpinions"),
            }
        except Exception:
            return {}

    def get_stock_news(self, symbol: str, limit: int = 20) -> List[Dict]:
        """Return recent news articles via yfinance."""
        try:
            tk = self._ticker(symbol)
            news = tk.news or []
            results = []
            for n in news[:limit]:
                results.append({
                    "title":         n.get("title", ""),
                    "text":          n.get("summary", ""),
                    "url":           n.get("link", ""),
                    "publishedDate": str(n.get("providerPublishTime", "")),
                })
            return results
        except Exception as exc:
            logger.debug(f"News fetch failed for {symbol}: {exc}")
            return []

    # ── Screener (no API — derived from universe lists) ───────────────────────

    def get_stock_screener(
        self,
        market_cap_min: int = 1_000_000_000,
        volume_min: int = 500_000,
        price_min: float = 10.0,
        limit: int = 500,
    ) -> List[Dict]:
        """
        Build a screened universe by fetching batch info for S&P500+NDX.
        Returns list of dicts with symbol, marketCap, volume, price.
        """
        symbols = list(set(get_sp500_symbols() + get_nasdaq100_symbols()))
        logger.info(f"Bulk-downloading screener info for {len(symbols)} symbols …")

        # yfinance batch download for speed
        chunk_size = 50
        results = []
        for i in range(0, min(len(symbols), limit * 2), chunk_size):
            batch = symbols[i: i + chunk_size]
            try:
                tickers = yf.Tickers(" ".join(batch))
                for sym in batch:
                    try:
                        info = tickers.tickers[sym].fast_info
                        mkt_cap  = getattr(info, "market_cap",    None)
                        volume   = getattr(info, "three_month_average_volume", None) or getattr(info, "last_volume", None)
                        price    = getattr(info, "last_price",     None)
                        exchange = getattr(info, "exchange",       "")
                        if (
                            mkt_cap  and mkt_cap  >= market_cap_min
                            and price    and price    >= price_min
                            and exchange in ("NMS", "NYQ", "NGM", "NCM", "PCX", "ASE", "NYSE", "NASDAQ")
                        ):
                            results.append({
                                "symbol":    sym,
                                "marketCap": mkt_cap,
                                "volume":    volume or 0,
                                "price":     price,
                            })
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug(f"Batch screener chunk failed: {exc}")
            time.sleep(0.2)

        logger.info(f"Screener: {len(results)} stocks passed basic filters")
        return results[:limit]

    def get_sp500_constituents(self) -> List[str]:
        return get_sp500_symbols()

    def get_nasdaq100_constituents(self) -> List[str]:
        return get_nasdaq100_symbols()

"""
Financial Modeling Prep (FMP) API client.

Handles all market data ingestion:
  - US equity universe screening
  - Daily OHLCV + adjusted price history
  - Fundamental metrics (ROIC, FCF, EPS, ratios)
  - Institutional ownership, insider transactions
  - SEC filings metadata
  - Company news headlines

Rate-limiting, retry logic, and response validation are built-in.
"""
import time
import requests
import pandas as pd
from typing import Any, Dict, List, Optional
from loguru import logger

from config.settings import FMP_API_KEY

_BASE = "https://financialmodelingprep.com/api"


class FMPClient:
    def __init__(self, api_key: str = FMP_API_KEY, throttle: float = 0.35):
        if not api_key:
            raise ValueError("FMP_API_KEY is not set. Add it to your .env file.")
        self.api_key = api_key
        self.throttle = throttle
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    # ── Internal ─────────────────────────────────────────────────────────────

    def _get(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        version: int = 3,
        retries: int = 3,
    ) -> Any:
        url = f"{_BASE}/v{version}/{endpoint}"
        payload = {"apikey": self.api_key, **(params or {})}

        for attempt in range(retries):
            try:
                r = self.session.get(url, params=payload, timeout=30)
                r.raise_for_status()
                time.sleep(self.throttle)
                return r.json()
            except requests.HTTPError as exc:
                if r.status_code == 429:
                    wait = 2 ** (attempt + 2)
                    logger.warning(f"Rate-limited by FMP. Waiting {wait}s …")
                    time.sleep(wait)
                else:
                    logger.error(f"HTTP {r.status_code} for {url}: {exc}")
                    return {}
            except requests.RequestException as exc:
                logger.warning(f"Request failed (attempt {attempt+1}/{retries}): {exc}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return {}

    # ── Universe ──────────────────────────────────────────────────────────────

    def get_stock_screener(
        self,
        market_cap_min: int = 1_000_000_000,
        volume_min: int = 500_000,
        price_min: float = 10.0,
        exchange: str = "NYSE,NASDAQ",
        limit: int = 500,
    ) -> List[Dict]:
        """Return a list of US equity tickers passing basic liquidity gates."""
        data = self._get(
            "stock-screener",
            {
                "marketCapMoreThan": market_cap_min,
                "volumeMoreThan": volume_min,
                "priceMoreThan": price_min,
                "exchange": exchange,
                "isActivelyTrading": "true",
                "country": "US",
                "limit": limit,
            },
        )
        return data if isinstance(data, list) else []

    def get_sp500_constituents(self) -> List[str]:
        """Return current S&P 500 tickers."""
        data = self._get("sp500_constituent")
        return [d["symbol"] for d in data] if isinstance(data, list) else []

    def get_nasdaq100_constituents(self) -> List[str]:
        """Return current NASDAQ-100 tickers."""
        data = self._get("nasdaq_constituent")
        return [d["symbol"] for d in data] if isinstance(data, list) else []

    # ── Price History ─────────────────────────────────────────────────────────

    def get_daily_ohlcv(self, symbol: str, days: int = 400) -> pd.DataFrame:
        """
        Fetch daily OHLCV + adjusted close.
        Returns a DataFrame sorted oldest → newest.
        """
        data = self._get(
            f"historical-price-full/{symbol}", {"timeseries": days}
        )
        if not data or "historical" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data["historical"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df.rename(
            columns={
                "open":     "Open",
                "high":     "High",
                "low":      "Low",
                "close":    "Close",
                "volume":   "Volume",
                "adjClose": "Adj_Close",
            }
        )
        # keep only the columns we actually use
        keep = [c for c in ["date", "Open", "High", "Low", "Close", "Volume", "Adj_Close"] if c in df.columns]
        return df[keep]

    # ── Fundamentals ──────────────────────────────────────────────────────────

    def get_key_metrics(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Annual key metrics: ROIC, FCF yield, P/E, EV/EBITDA, etc."""
        data = self._get(f"key-metrics/{symbol}", {"limit": limit, "period": "annual"})
        return data if isinstance(data, list) else []

    def get_key_metrics_ttm(self, symbol: str) -> Dict:
        """Trailing-twelve-month snapshot for quick screening."""
        data = self._get(f"key-metrics-ttm/{symbol}")
        if isinstance(data, list) and data:
            return data[0]
        return {}

    def get_ratios(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Valuation ratios: PEG, EV/EBITDA, P/FCF, P/B, etc."""
        data = self._get(f"ratios/{symbol}", {"limit": limit, "period": "annual"})
        return data if isinstance(data, list) else []

    def get_ratios_ttm(self, symbol: str) -> Dict:
        data = self._get(f"ratios-ttm/{symbol}")
        if isinstance(data, list) and data:
            return data[0]
        return {}

    def get_income_statement(
        self, symbol: str, period: str = "quarter", limit: int = 8
    ) -> List[Dict]:
        """Income statements for EPS growth and revenue acceleration analysis."""
        data = self._get(
            f"income-statement/{symbol}", {"limit": limit, "period": period}
        )
        return data if isinstance(data, list) else []

    def get_balance_sheet(self, symbol: str, limit: int = 4) -> List[Dict]:
        data = self._get(
            f"balance-sheet-statement/{symbol}", {"limit": limit, "period": "annual"}
        )
        return data if isinstance(data, list) else []

    def get_cash_flow(self, symbol: str, limit: int = 4) -> List[Dict]:
        data = self._get(
            f"cash-flow-statement/{symbol}", {"limit": limit, "period": "annual"}
        )
        return data if isinstance(data, list) else []

    def get_financial_growth(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Pre-calculated YoY growth rates from FMP."""
        data = self._get(
            f"financial-growth/{symbol}", {"limit": limit, "period": "annual"}
        )
        return data if isinstance(data, list) else []

    def get_company_profile(self, symbol: str) -> Dict:
        """Sector, industry, description, country, market cap."""
        data = self._get(f"profile/{symbol}")
        if isinstance(data, list) and data:
            return data[0]
        return {}

    # ── Ownership & Insider Data ──────────────────────────────────────────────

    def get_institutional_holders(self, symbol: str) -> List[Dict]:
        """13-F based institutional holder list (top holders)."""
        data = self._get(f"institutional-holder/{symbol}")
        return data if isinstance(data, list) else []

    def get_insider_transactions(self, symbol: str, limit: int = 20) -> List[Dict]:
        """Form 4 insider buy/sell transactions."""
        data = self._get("insider-trading", {"symbol": symbol, "limit": limit})
        return data if isinstance(data, list) else []

    # ── Earnings & Analyst Estimates ─────────────────────────────────────────

    def get_earnings_surprises(self, symbol: str, limit: int = 8) -> List[Dict]:
        """Historical EPS surprises vs. analyst consensus."""
        data = self._get(f"earnings-surprises/{symbol}", {"limit": limit})
        return data if isinstance(data, list) else []

    def get_analyst_estimates(
        self, symbol: str, period: str = "quarter", limit: int = 4
    ) -> List[Dict]:
        """Forward EPS and revenue consensus estimates."""
        data = self._get(
            f"analyst-estimates/{symbol}", {"limit": limit, "period": period}
        )
        return data if isinstance(data, list) else []

    def get_earnings_calendar(self, symbol: str) -> List[Dict]:
        data = self._get(f"historical/earning_calendar/{symbol}")
        return data if isinstance(data, list) else []

    # ── News & Filings ────────────────────────────────────────────────────────

    def get_stock_news(self, symbol: str, limit: int = 30) -> List[Dict]:
        """Recent news articles for a ticker."""
        data = self._get("stock_news", {"tickers": symbol, "limit": limit})
        return data if isinstance(data, list) else []

    def get_sec_filings(
        self, symbol: str, filing_type: str = "10-Q", limit: int = 5
    ) -> List[Dict]:
        """Links to recent SEC filings."""
        data = self._get(f"sec_filings/{symbol}", {"type": filing_type, "limit": limit})
        return data if isinstance(data, list) else []

    # ── Price Target & Analyst Ratings ────────────────────────────────────────

    def get_price_target_summary(self, symbol: str) -> Dict:
        data = self._get(f"price-target-summary/{symbol}")
        if isinstance(data, list) and data:
            return data[0]
        return {}

    def get_analyst_stock_recommendations(self, symbol: str, limit: int = 5) -> List[Dict]:
        data = self._get(
            f"analyst-stock-recommendations/{symbol}", {"limit": limit}
        )
        return data if isinstance(data, list) else []

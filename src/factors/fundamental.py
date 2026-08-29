"""
Fundamental factor calculator — powered by yfinance (free, no API key).

Computes per-ticker metrics:
  Quality:   ROIC proxy, ROE, FCF yield, D/E, net/gross margin, current ratio
  Growth:    Quarterly EPS YoY growth, EPS acceleration, annual EPS growth,
             revenue growth, EPS surprise vs consensus
  Valuation: PEG, EV/EBITDA, P/FCF, P/B, trailing P/E
  Ownership: institutional holders count, insider net buy/sell
  Targets:   analyst price target mean/high, analyst count
"""
from typing import Dict, List, Optional
from loguru import logger

from src.data.yf_client import YFClient


def _safe_pct(a, b) -> Optional[float]:
    try:
        if b and b != 0 and a is not None:
            return ((float(a) - float(b)) / abs(float(b))) * 100.0
        return None
    except (TypeError, ZeroDivisionError):
        return None


def _safe(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


class FundamentalAnalyser:
    def __init__(self, client: YFClient):
        self.client = client

    def analyse(self, symbol: str) -> Dict:
        result: Dict = {"symbol": symbol}

        try:
            ttm = self.client.get_key_metrics_ttm(symbol)

            roe_raw = ttm.get("roeTTM")
            result["roic"]         = _safe(roe_raw)
            result["roe"]          = _safe(roe_raw)
            result["debt_equity"]  = _safe(ttm.get("debtToEquityTTM"))
            result["current_ratio"]= _safe(ttm.get("currentRatioTTM"))
            result["ev_ebitda"]    = _safe(ttm.get("enterpriseValueOverEBITDATTM"))
            result["p_fcf"]        = _safe(ttm.get("priceToFreeCashFlowsRatioTTM"))
            result["pe_ratio"]     = _safe(ttm.get("_trailingPE"))

            fcf  = ttm.get("_freeCashflow")
            mcap = ttm.get("_marketCap")
            if fcf and mcap and mcap > 0:
                result["fcf_yield"] = (_safe(fcf) / _safe(mcap)) * 100
            else:
                result["fcf_yield"] = 0.0

            ratios = self.client.get_ratios_ttm(symbol)
            result["peg_ratio"]   = _safe(ratios.get("pegRatioTTM"))
            result["net_margin"]  = _safe(ratios.get("netProfitMarginTTM"))
            result["gross_margin"]= _safe(ratios.get("grossProfitMarginTTM"))
            result["p_book"]      = _safe(ratios.get("priceToBookRatioTTM"))

            quarters = self.client.get_income_statement(symbol, period="quarter", limit=8)
            result["eps_growth_yoy"]   = None
            result["eps_growth_qoq"]   = None
            result["eps_acceleration"] = None
            result["rev_growth_yoy"]   = None

            if len(quarters) >= 5:
                eps_now = quarters[0].get("eps")
                eps_1q  = quarters[1].get("eps") if len(quarters) > 1 else None
                eps_4q  = quarters[4].get("eps") if len(quarters) > 4 else None
                rev_now = quarters[0].get("revenue")
                rev_4q  = quarters[4].get("revenue") if len(quarters) > 4 else None

                result["eps_growth_yoy"] = _safe_pct(eps_now, eps_4q)
                result["rev_growth_yoy"] = _safe_pct(rev_now, rev_4q)
                result["eps_growth_qoq"] = _safe_pct(eps_now, eps_1q)

                if len(quarters) >= 6:
                    eps_prev    = quarters[1].get("eps")
                    eps_prev_4q = quarters[5].get("eps") if len(quarters) > 5 else None
                    prev_yoy    = _safe_pct(eps_prev, eps_prev_4q)
                    curr_yoy    = result["eps_growth_yoy"]
                    if curr_yoy is not None and prev_yoy is not None:
                        result["eps_acceleration"] = curr_yoy - prev_yoy
                elif len(quarters) >= 3:
                    # Fallback: QoQ acceleration (change in QoQ growth rate)
                    eps_q0 = quarters[0].get("eps")
                    eps_q1 = quarters[1].get("eps")
                    eps_q2 = quarters[2].get("eps")
                    qoq_curr = _safe_pct(eps_q0, eps_q1)
                    qoq_prev = _safe_pct(eps_q1, eps_q2)
                    if qoq_curr is not None and qoq_prev is not None:
                        result["eps_acceleration"] = qoq_curr - qoq_prev

            annual = self.client.get_income_statement(symbol, period="annual", limit=4)
            result["annual_eps_growth"] = None
            if len(annual) >= 2:
                result["annual_eps_growth"] = _safe_pct(
                    annual[0].get("eps"), annual[1].get("eps")
                )

            surprises = self.client.get_earnings_surprises(symbol, limit=4)
            if surprises:
                s = surprises[0]
                result["eps_surprise_pct"] = _safe_pct(
                    s.get("actualEarningResult"), s.get("estimatedEarning")
                )
            else:
                result["eps_surprise_pct"] = None

            holders = self.client.get_institutional_holders(symbol)
            result["institutional_holders_count"] = len(holders) if holders else 0

            insiders = self.client.get_insider_transactions(symbol, limit=20)
            buys  = sum(1 for t in insiders if "PURCHASE" in str(t.get("transactionType","")).upper())
            sells = sum(1 for t in insiders if "SALE" in str(t.get("transactionType","")).upper())
            result["insider_buy_count"]  = buys
            result["insider_sell_count"] = sells
            result["insider_net"]        = buys - sells

            pt = self.client.get_price_target_summary(symbol)
            result["analyst_target_mean"] = _safe(pt.get("targetMean"))
            result["analyst_target_high"] = _safe(pt.get("targetHigh"))
            result["analyst_count"]       = _safe(pt.get("totalAnalysts"))

            # ── Beta & forward P/E (market sensitivity + earnings expectation) ─
            result["beta"]       = ttm.get("_beta")
            result["forward_pe"] = ttm.get("_forwardPE")

            # ── Sector / industry / description (for agent context) ────────────
            try:
                profile = self.client.get_company_profile(symbol)
                result["sector"]      = profile.get("sector", "")
                result["industry"]    = profile.get("industry", "")
                desc = profile.get("description", "") or ""
                result["company_description"] = desc[:400].strip()
            except Exception:
                pass

            # ── Short interest (squeeze potential / distribution signal) ───────
            try:
                si = self.client.get_short_interest(symbol)
                result["short_pct_float"] = si.get("short_pct_float")
                result["short_ratio"]     = si.get("short_ratio")
            except Exception:
                pass

            # ── Next earnings date (binary event / catalyst proximity) ─────────
            try:
                ed = self.client.get_next_earnings_date(symbol)
                result["next_earnings_date"] = ed.get("next_earnings_date")
                result["days_to_earnings"]   = ed.get("days_to_earnings")
            except Exception:
                pass

        except Exception as exc:
            logger.error(f"Fundamental analysis failed for {symbol}: {exc}")

        return result

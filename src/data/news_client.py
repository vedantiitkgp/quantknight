"""
News aggregation client.

Pulls headlines from:
  - Financial Modeling Prep (company-specific news)
  - Finnhub (real-time financial news feed)
  - SEC EDGAR RSS (8-K / 10-Q filing notifications)

All headlines are returned in a unified format for FinBERT sentiment scoring.
"""
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List
from loguru import logger

from config.settings import FMP_API_KEY, FINNHUB_API_KEY


class NewsClient:
    """Aggregates financial news headlines from multiple sources."""

    def __init__(self):
        self._fmp_key = FMP_API_KEY
        self._fh_key  = FINNHUB_API_KEY
        self._session = requests.Session()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get(self, url: str, params: dict, retries: int = 2) -> List | Dict:
        for attempt in range(retries):
            try:
                r = self._session.get(url, params=params, timeout=20)
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                logger.warning(f"News request failed ({attempt+1}/{retries}): {exc}")
                time.sleep(1.5)
        return []

    def _normalise(self, raw: dict, source: str) -> dict:
        """Map raw API fields to a unified schema."""
        return {
            "headline":   raw.get("title") or raw.get("headline", ""),
            "summary":    raw.get("text") or raw.get("summary", ""),
            "source":     source,
            "published":  raw.get("publishedDate") or raw.get("datetime", ""),
            "url":        raw.get("url") or raw.get("url", ""),
            "sentiment":  None,   # filled later by sentiment engine
        }

    # ── FMP Company News ──────────────────────────────────────────────────────

    def get_fmp_news(self, symbol: str, limit: int = 30) -> List[Dict]:
        data = self._get(
            "https://financialmodelingprep.com/api/v3/stock_news",
            {"tickers": symbol, "limit": limit, "apikey": self._fmp_key},
        )
        if not isinstance(data, list):
            return []
        return [self._normalise(a, "FMP") for a in data]

    # ── Finnhub Company News ──────────────────────────────────────────────────

    def get_finnhub_news(self, symbol: str, days_back: int = 7) -> List[Dict]:
        if not self._fh_key:
            return []
        today  = datetime.utcnow().strftime("%Y-%m-%d")
        from_d = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        data = self._get(
            "https://finnhub.io/api/v1/company-news",
            {"symbol": symbol, "from": from_d, "to": today, "token": self._fh_key},
        )
        if not isinstance(data, list):
            return []
        return [self._normalise(a, "Finnhub") for a in data]

    # ── Finnhub General Market News ──────────────────────────────────────────

    def get_market_news(self, category: str = "general") -> List[Dict]:
        if not self._fh_key:
            return []
        data = self._get(
            "https://finnhub.io/api/v1/news",
            {"category": category, "token": self._fh_key},
        )
        if not isinstance(data, list):
            return []
        return [self._normalise(a, "Finnhub-Market") for a in data[:20]]

    # ── SEC EDGAR RSS (8-K real-time filing alerts) ───────────────────────────

    def get_sec_8k_alerts(self, symbol: str) -> List[Dict]:
        """
        Pull the most recent Form 8-K (material event disclosures) from FMP
        SEC filing endpoint.  Returns a list of normalised headline dicts.
        """
        data = self._get(
            f"https://financialmodelingprep.com/api/v3/sec_filings/{symbol}",
            {"type": "8-K", "limit": 5, "apikey": self._fmp_key},
        )
        if not isinstance(data, list):
            return []
        results = []
        for f in data:
            results.append({
                "headline":  f"SEC 8-K Filing: {f.get('type','')} — {symbol}",
                "summary":   f.get("description", ""),
                "source":    "SEC-EDGAR",
                "published": f.get("fillingDate", ""),
                "url":       f.get("link", ""),
                "sentiment": None,
            })
        return results

    # ── Combined Fetch ────────────────────────────────────────────────────────

    def get_all_news(self, symbol: str, max_articles: int = 40) -> List[Dict]:
        """
        Aggregate from all sources, deduplicate by URL, and return up to
        max_articles sorted by publication date (newest first).
        """
        articles: List[Dict] = []
        # FMP legacy endpoints (/v3/stock_news, /v3/sec_filings) are permanently
        # blocked as of August 2025 — use Finnhub only.
        articles += self.get_finnhub_news(symbol, days_back=7)

        # Deduplicate
        seen = set()
        unique = []
        for a in articles:
            key = a.get("url") or a.get("headline")
            if key and key not in seen:
                seen.add(key)
                unique.append(a)

        # Sort newest first (best-effort; some APIs return mixed formats)
        def _parse_date(d: str) -> datetime:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(str(d)[:19], fmt)
                except Exception:
                    pass
            return datetime.min

        unique.sort(key=lambda a: _parse_date(a["published"]), reverse=True)
        return unique[:max_articles]

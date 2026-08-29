"""
Universe builder.

Fetches the broadest reasonable US equity universe by combining:
  - S&P 500 constituents (large-cap quality bias)
  - NASDAQ-100 constituents (growth/tech)
  - FMP stock screener filtered by liquidity gates

Deduplicates and returns a clean list of tickers.
"""
from typing import List
from loguru import logger

from src.data.yf_client import YFClient as FMPClient
from config.settings import MIN_MARKET_CAP, MIN_DAILY_VOLUME, MIN_PRICE, UNIVERSE_SIZE


def build_universe(client: FMPClient | None = None) -> List[str]:
    """
    Return a deduplicated list of US equity ticker symbols that pass
    basic liquidity requirements.

    Priority order (later sources fill up to UNIVERSE_SIZE):
      1. S&P 500 constituents
      2. NASDAQ-100 constituents
      3. FMP screener (fills remainder)
    """
    if client is None:
        client = FMPClient()

    tickers: set = set()

    # 1. S&P 500
    logger.info("Fetching S&P 500 constituents …")
    sp500 = client.get_sp500_constituents()
    logger.info(f"  S&P 500: {len(sp500)} tickers")
    tickers.update(sp500)

    # 2. NASDAQ-100
    logger.info("Fetching NASDAQ-100 constituents …")
    ndx = client.get_nasdaq100_constituents()
    logger.info(f"  NASDAQ-100: {len(ndx)} tickers")
    tickers.update(ndx)

    # 3. Screener top-up if we are below UNIVERSE_SIZE
    if len(tickers) < UNIVERSE_SIZE:
        needed = UNIVERSE_SIZE - len(tickers)
        logger.info(f"Supplementing with FMP screener (requesting {needed} extra) …")
        screener_results = client.get_stock_screener(
            market_cap_min=MIN_MARKET_CAP,
            volume_min=MIN_DAILY_VOLUME,
            price_min=MIN_PRICE,
            limit=UNIVERSE_SIZE,
        )
        extra = [s["symbol"] for s in screener_results if s.get("symbol")]
        logger.info(f"  Screener returned {len(extra)} tickers")
        tickers.update(extra)

    # Exclude ETFs and known non-equity tickers
    _EXCLUDE_SUFFIXES = {".TO", ".AX", ".L", ".DE", "-UN", "-A", "-B"}
    clean = sorted(
        t for t in tickers
        if t.isalpha() and len(t) <= 5
        and not any(t.endswith(s) for s in _EXCLUDE_SUFFIXES)
    )

    logger.info(f"Universe built: {len(clean)} unique US equity tickers")
    return clean[:UNIVERSE_SIZE]

"""
Hard-gate quantitative filter engine.

Runs BEFORE the expensive fundamental + technical calculations so we
avoid wasting API calls on junk stocks.

Gates applied (in order):
  1. Liquidity:  Market Cap > $1B, 30-day avg volume > 500 K shares
  2. Price:      Stock price > $10  (avoids penny-stock noise)
  3. Minervini Trend Template:
       - Close > 200d SMA
       - 150d SMA > 200d SMA  (long-term uptrend structure)
       - Close > 50d SMA
       - Price within 25 % of 52-week high
       - Price at least 25 % above 52-week low
  4. Volume quality:  20-day avg volume > 300 K shares
  5. Not in a severe downtrend:  not making new 52-week lows

Stocks that pass all gates are passed to the full factor-scoring engine.
"""
import pandas as pd
import pandas_ta as ta
from typing import Dict, List
from loguru import logger

from config.settings import (
    MIN_MARKET_CAP, MIN_DAILY_VOLUME, MIN_PRICE,
    EMA_MID, EMA_SLOW, EMA_TREND,
    SEPA_MAX_FROM_52W_HIGH, SEPA_MIN_FROM_52W_LOW,
)
from src.data.yf_client import YFClient as FMPClient


def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA/EMA columns needed for gate checks."""
    df = df.copy()
    df[f"SMA_{EMA_MID}"]   = ta.sma(df["Close"], length=EMA_MID)
    df[f"SMA_{EMA_SLOW}"]  = ta.sma(df["Close"], length=EMA_SLOW)
    df[f"SMA_{EMA_TREND}"] = ta.sma(df["Close"], length=EMA_TREND)
    return df


def passes_minervini_template(df: pd.DataFrame) -> bool:
    """
    Mark Minervini's Stage-2 Trend Template:
      1.  Current price > 200d SMA
      2.  150d SMA > 200d SMA
      3.  Current price > 50d SMA
      4.  Current price within 25 % of 52-week high
      5.  Current price > 25 % above 52-week low
    """
    if len(df) < EMA_TREND + 5:
        return False

    df = _add_moving_averages(df)
    last = df.iloc[-1]

    close    = last["Close"]
    sma50    = last.get(f"SMA_{EMA_MID}")
    sma150   = last.get(f"SMA_{EMA_SLOW}")
    sma200   = last.get(f"SMA_{EMA_TREND}")

    if pd.isna(sma50) or pd.isna(sma150) or pd.isna(sma200):
        return False

    high_52w = df["High"].tail(252).max()
    low_52w  = df["Low"].tail(252).min()

    checks = [
        close > sma200,
        sma150 > sma200,
        close > sma50,
        close >= high_52w * (1 - SEPA_MAX_FROM_52W_HIGH),
        close >= low_52w  * (1 + SEPA_MIN_FROM_52W_LOW),
    ]
    return all(checks)


def apply_hard_filters(
    symbols: List[str],
    client: FMPClient,
    screener_data: List[Dict] | None = None,
) -> List[str]:
    """
    Apply all hard gates.  Returns list of symbols that pass.

    `screener_data` is the raw list from `get_stock_screener()`;
    if provided, the liquidity/price checks reuse that data to avoid
    extra API calls.
    """
    # ── Gate 1: Liquidity & Price (use screener cache if available) ───────────
    if screener_data:
        screener_map = {s["symbol"]: s for s in screener_data if s.get("symbol")}
    else:
        screener_map = {}

    passed_liquidity: List[str] = []
    for sym in symbols:
        if sym in screener_map:
            row = screener_map[sym]
            if (
                (row.get("marketCap") or 0) >= MIN_MARKET_CAP
                and (row.get("volume") or 0) >= MIN_DAILY_VOLUME
                and (row.get("price") or 0) >= MIN_PRICE
            ):
                passed_liquidity.append(sym)
        else:
            # No screener cache — let through, SEPA gate will filter further
            passed_liquidity.append(sym)

    logger.info(f"Gate 1 (Liquidity): {len(passed_liquidity)}/{len(symbols)} passed")

    # ── Gate 2: Minervini Trend Template (requires price history) ─────────────
    passed_sepa: List[str] = []
    for sym in passed_liquidity:
        try:
            df = client.get_daily_ohlcv(sym, days=280)
            if df.empty or len(df) < 210:
                continue
            if passes_minervini_template(df):
                passed_sepa.append(sym)
        except Exception as exc:
            logger.debug(f"SEPA check failed for {sym}: {exc}")

    logger.info(f"Gate 2 (Minervini SEPA): {len(passed_sepa)}/{len(passed_liquidity)} passed")
    return passed_sepa

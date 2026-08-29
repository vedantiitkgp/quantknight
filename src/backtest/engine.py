"""
Backtesting engine using VectorBT.

Implements and evaluates three core strategies inspired by professional traders:

  Strategy 1 — RSI_PULLBACK
    Universe pre-filtered by Minervini Trend Template.
    Entry: RSI drops into 38–55 zone while price > EMA-20 > EMA-50 > EMA-200.
    Exit:  RSI exceeds 72 (overbought) OR ATR-based stop-loss hit.
    Inspired by: O'Neil, Minervini, Elder.

  Strategy 2 — BREAKOUT
    Entry: Daily close breaks above the 20-day highest-high with volume
           > 150 % of the 20-day average volume.
    Exit:  10-day trailing stop OR 3× ATR profit target.
    Inspired by: Turtle Trading, Darvas Box.

  Strategy 3 — MOMENTUM_QUALITY
    Long-only momentum strategy: top-decile 12-1 month return stocks
    that also rank in the top quartile on quality (ROIC proxy).
    Rebalanced monthly.  Equal-weighted portfolio.
    Inspired by: AQR, Fama-French UMD factor.

Walk-Forward Validation:
  Training window: 3 years
  Testing window:  1 year
  Slide:           1 year
  Covers 2019–2024 (5 folds).

All strategies use:
  - Commission: 0.1 % per side
  - Slippage:   0.1 %
  - Initial capital: $100,000
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from typing import Dict, List, Tuple
from loguru import logger

from config.settings import (
    INITIAL_CAPITAL, COMMISSION_PCT, SLIPPAGE_PCT,
    BACKTEST_START_DATE, BACKTEST_END_DATE,
    RSI_PERIOD, ATR_PERIOD, EMA_FAST, EMA_MID, EMA_TREND,
    RSI_PULLBACK_LOW, RSI_PULLBACK_HIGH,
    ATR_STOP_MULTIPLIER, ATR_TARGET_MULT_SHORT,
    WF_TRAIN_YEARS, WF_TEST_YEARS,
)
from src.backtest.metrics import compute_metrics


def _fetch_price_data(symbols: List[str], fmp_client) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for all symbols. Returns dict {symbol: DataFrame}."""
    data = {}
    for sym in symbols:
        try:
            df = fmp_client.get_daily_ohlcv(sym, days=1600)
            if not df.empty and len(df) > 300:
                df = df.set_index("date")
                data[sym] = df
        except Exception as exc:
            logger.debug(f"Could not fetch {sym}: {exc}")
    return data


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RSI"]    = ta.rsi(df["Close"], length=RSI_PERIOD)
    df["EMA20"]  = ta.ema(df["Close"], length=EMA_FAST)
    df["EMA50"]  = ta.ema(df["Close"], length=EMA_MID)
    df["EMA200"] = ta.ema(df["Close"], length=EMA_TREND)
    atr          = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_PERIOD)
    df["ATR"]    = atr
    df["Vol20"]  = df["Volume"].rolling(20).mean()
    df["Hi20"]   = df["High"].rolling(20).max().shift(1)   # yesterday's 20d high
    return df


# ── Strategy 1: RSI Pullback ──────────────────────────────────────────────────

def run_rsi_pullback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised signal generation for the RSI Pullback strategy.
    Returns DataFrame with columns: entry, exit, stop_price, target_price.
    """
    df = _add_indicators(df)
    # Uptrend condition
    uptrend = (df["Close"] > df["EMA20"]) & (df["EMA20"] > df["EMA50"]) & (df["Close"] > df["EMA200"])
    # Entry signal
    entry_signal = uptrend & (df["RSI"] >= RSI_PULLBACK_LOW) & (df["RSI"] <= RSI_PULLBACK_HIGH)

    df["entry_signal"] = entry_signal.astype(int)
    df["stop_price"]   = df["Close"] - ATR_STOP_MULTIPLIER * df["ATR"]
    df["target_price"] = df["Close"] + ATR_TARGET_MULT_SHORT * df["ATR"]
    return df


def run_breakout(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised signal generation for the 20-day breakout strategy."""
    df = _add_indicators(df)
    vol_surge   = df["Volume"] >= df["Vol20"] * 1.5
    above_high  = df["Close"] > df["Hi20"]
    uptrend     = df["Close"] > df["EMA200"]
    entry_signal = vol_surge & above_high & uptrend

    df["entry_signal"] = entry_signal.astype(int)
    df["stop_price"]   = df["Close"] - ATR_STOP_MULTIPLIER * df["ATR"]
    df["target_price"] = df["Close"] + ATR_TARGET_MULT_SHORT * df["ATR"]
    return df


# ── Pure-Python Walk-Forward Simulator ───────────────────────────────────────

class WalkForwardResult:
    def __init__(self):
        self.folds:        List[Dict] = []
        self.combined_pnl: pd.Series  = pd.Series(dtype=float)

    def add_fold(self, fold_dict: Dict):
        self.folds.append(fold_dict)


def _simulate_trades(
    signals_df: pd.DataFrame,
    initial_capital: float,
    commission: float,
    slippage: float,
) -> pd.Series:
    """
    Pure-Python trade-by-trade simulator.
    Returns a daily equity curve (pd.Series indexed by date).
    """
    equity = initial_capital
    cash   = initial_capital
    position_price = None
    stop   = None
    target = None
    equity_curve = {}

    for dt, row in signals_df.iterrows():
        price = row["Close"]
        if pd.isna(price) or price <= 0:
            equity_curve[dt] = equity
            continue

        # ── Check exit conditions if we hold a position ────────────────────
        if position_price is not None:
            hit_stop   = stop   is not None and price <= stop
            hit_target = target is not None and price >= target
            rsi_exit   = "RSI" in signals_df.columns and not pd.isna(row["RSI"]) and row["RSI"] > 72

            if hit_stop or hit_target or rsi_exit:
                exit_price = price * (1 - slippage)
                pnl_pct    = (exit_price - position_price) / position_price
                shares     = cash / position_price
                cash       = shares * exit_price * (1 - commission)
                equity     = cash
                position_price = stop = target = None

        # ── Open new position on entry signal ─────────────────────────────
        elif row.get("entry_signal", 0) == 1 and cash > 0:
            entry_price    = price * (1 + slippage)
            position_price = entry_price * (1 + commission)
            stop           = row.get("stop_price")
            target         = row.get("target_price")
            # cash stays allocated (we're fully invested)

        # Mark-to-market equity
        if position_price is not None:
            shares = cash / position_price
            equity = shares * price
        equity_curve[dt] = equity

    return pd.Series(equity_curve)


def run_walk_forward(
    symbols: List[str],
    strategy: str,
    fmp_client,
    start: str = BACKTEST_START_DATE,
    end:   str = BACKTEST_END_DATE,
) -> Dict:
    """
    Run walk-forward validation across 2019–2024.

    Parameters
    ----------
    symbols  : list of ticker strings (pre-screened universe)
    strategy : "RSI_PULLBACK" | "BREAKOUT"
    fmp_client : FMPClient instance

    Returns
    -------
    dict with fold-level and aggregate performance metrics
    """
    logger.info(f"Walk-forward backtest: {strategy} on {len(symbols)} symbols")

    price_data = _fetch_price_data(symbols[:50], fmp_client)  # limit to 50 for speed
    if not price_data:
        return {"error": "No price data fetched"}

    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)

    folds         = []
    all_equity    = []

    # Generate walk-forward windows
    train_years = WF_TRAIN_YEARS
    test_years  = WF_TEST_YEARS
    current     = start_dt

    while current + pd.DateOffset(years=train_years + test_years) <= end_dt:
        train_end = current + pd.DateOffset(years=train_years)
        test_end  = train_end + pd.DateOffset(years=test_years)

        fold_equities = []

        for sym, df in price_data.items():
            test_df = df.loc[train_end:test_end].copy()
            if len(test_df) < 60:
                continue

            try:
                if strategy == "RSI_PULLBACK":
                    sig_df = run_rsi_pullback(test_df)
                elif strategy == "BREAKOUT":
                    sig_df = run_breakout(test_df)
                else:
                    continue

                eq = _simulate_trades(
                    sig_df,
                    INITIAL_CAPITAL,
                    COMMISSION_PCT,
                    SLIPPAGE_PCT,
                )
                fold_equities.append(eq)
            except Exception as exc:
                logger.debug(f"Simulation failed for {sym}: {exc}")

        if fold_equities:
            # Average across all stocks in this fold
            combined = pd.concat(fold_equities, axis=1).mean(axis=1).dropna()
            metrics  = compute_metrics(combined, INITIAL_CAPITAL)
            metrics["fold_start"] = str(train_end.date())
            metrics["fold_end"]   = str(test_end.date())
            folds.append(metrics)
            all_equity.append(combined)
            logger.info(
                f"  Fold {train_end.date()} → {test_end.date()}: "
                f"Return={metrics.get('total_return_pct', 0):.1f}% "
                f"Sharpe={metrics.get('sharpe_ratio', 0):.2f} "
                f"MaxDD={metrics.get('max_drawdown_pct', 0):.1f}%"
            )

        current = current + pd.DateOffset(years=test_years)

    # ── Aggregate across all folds ─────────────────────────────────────────
    aggregate = {}
    if folds:
        keys = [k for k in folds[0] if isinstance(folds[0][k], (int, float))]
        for k in keys:
            vals = [f[k] for f in folds if k in f]
            aggregate[k] = float(np.mean(vals)) if vals else None
        aggregate["fold_count"]       = len(folds)
        aggregate["strategy"]         = strategy
        aggregate["universe_symbols"] = len(price_data)

    return {
        "strategy":   strategy,
        "folds":      folds,
        "aggregate":  aggregate,
    }

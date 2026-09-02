"""
Technical factor calculator.

Strategies implemented (inspired by professionals):
  - Minervini SEPA entry refinement (10-week pullback to MA)
  - Elder Triple Screen (trend + momentum + entry)
  - Turtle Breakout (20-day high with volume confirmation)
  - RSI Pullback in Uptrend (Weinstein Stage 2 + O'Neil)
  - MACD Bullish Crossover with trend alignment
  - Relative Strength vs SPY (CAN SLIM L factor)
  - On-Balance Volume (OBV) trend for accumulation detection

Returns per-ticker technical metrics AND detects which specific
entry setup is present (if any).
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger

from config.settings import (
    EMA_FAST, EMA_MID, EMA_SLOW, EMA_TREND,
    RSI_PERIOD, ATR_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD,
    RSI_PULLBACK_LOW, RSI_PULLBACK_HIGH,
    RSI_BREAKOUT_MIN, VOLUME_BREAKOUT_MULT,
    ATR_STOP_MULTIPLIER, ATR_TARGET_MULT_SHORT, ATR_TARGET_MULT_LONG,
)


def _safe(val) -> Optional[float]:
    try:
        v = float(val)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


class TechnicalAnalyser:
    def __init__(self):
        pass

    def compute(self, df: pd.DataFrame, symbol: str = "",
                benchmark_return_12m: Optional[float] = None) -> Dict:
        """
        Compute all technical indicators and detect entry setup.

        Parameters
        ----------
        df : pd.DataFrame
            Daily OHLCV with columns: date, Open, High, Low, Close, Volume
        symbol : str
            Ticker symbol (for logging only)

        Returns
        -------
        dict of raw metrics + detected entry setup + risk levels
        """
        result: Dict = {"symbol": symbol}
        if df.empty or len(df) < EMA_TREND + 10:
            result["error"] = "insufficient data"
            return result

        df = df.copy().reset_index(drop=True)

        # ── Moving Averages ────────────────────────────────────────────────────
        df["EMA_20"]  = ta.ema(df["Close"], length=EMA_FAST)
        df["EMA_50"]  = ta.ema(df["Close"], length=EMA_MID)
        df["EMA_150"] = ta.ema(df["Close"], length=EMA_SLOW)
        df["EMA_200"] = ta.ema(df["Close"], length=EMA_TREND)
        df["SMA_200"] = ta.sma(df["Close"], length=EMA_TREND)
        df["SMA_50"]  = ta.sma(df["Close"], length=EMA_MID)
        df["SMA_150"] = ta.sma(df["Close"], length=EMA_SLOW)

        # ── RSI ────────────────────────────────────────────────────────────────
        df["RSI"] = ta.rsi(df["Close"], length=RSI_PERIOD)

        # ── ATR (Average True Range) ───────────────────────────────────────────
        atr_df = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_PERIOD)
        df["ATR"] = atr_df

        # ── MACD ───────────────────────────────────────────────────────────────
        macd = ta.macd(df["Close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        if macd is not None and not macd.empty:
            df["MACD"]        = macd.iloc[:, 0]
            df["MACD_signal"] = macd.iloc[:, 1]
            df["MACD_hist"]   = macd.iloc[:, 2]
        else:
            df["MACD"] = df["MACD_signal"] = df["MACD_hist"] = None

        # ── Bollinger Bands ────────────────────────────────────────────────────
        bb = ta.bbands(df["Close"], length=BB_PERIOD, std=BB_STD)
        if bb is not None and not bb.empty:
            df["BB_upper"] = bb.iloc[:, 0]
            df["BB_mid"]   = bb.iloc[:, 1]
            df["BB_lower"] = bb.iloc[:, 2]
            df["BB_pct"]   = bb.iloc[:, 4]   # %B
        else:
            df["BB_upper"] = df["BB_mid"] = df["BB_lower"] = df["BB_pct"] = None

        # ── OBV (On-Balance Volume) ────────────────────────────────────────────
        df["OBV"]     = ta.obv(df["Close"], df["Volume"])
        df["OBV_EMA"] = ta.ema(df["OBV"], length=20)

        # ── Volume metrics ─────────────────────────────────────────────────────
        df["Vol_20MA"]    = df["Volume"].rolling(20).mean()
        df["Vol_ratio"]   = df["Volume"] / df["Vol_20MA"].replace(0, np.nan)

        # ── 52-week high / low ─────────────────────────────────────────────────
        lookback = min(252, len(df))
        high_52w  = df["High"].tail(lookback).max()
        low_52w   = df["Low"].tail(lookback).min()

        # ── Latest values ──────────────────────────────────────────────────────
        last  = df.iloc[-1]
        prev  = df.iloc[-2] if len(df) >= 2 else last
        prev2 = df.iloc[-3] if len(df) >= 3 else prev

        close    = last["Close"]
        ema20    = _safe(last["EMA_20"])
        ema50    = _safe(last["EMA_50"])
        ema150   = _safe(last["EMA_150"])
        ema200   = _safe(last["EMA_200"])
        sma200   = _safe(last["SMA_200"])
        rsi      = _safe(last["RSI"])
        atr      = _safe(last["ATR"])
        macd_v   = _safe(last["MACD"])
        macd_sig = _safe(last["MACD_signal"])
        macd_h   = _safe(last["MACD_hist"])
        prev_macd_h = _safe(prev["MACD_hist"])
        vol_ratio = _safe(last["Vol_ratio"])
        obv_trend = None
        if _safe(last["OBV"]) and _safe(last["OBV_EMA"]):
            obv_trend = 1.0 if last["OBV"] > last["OBV_EMA"] else -1.0

        # ── Store raw metrics ──────────────────────────────────────────────────
        result.update({
            "close":       close,
            "ema20":       ema20,
            "ema50":       ema50,
            "ema150":      ema150,
            "ema200":      ema200,
            "sma200":      sma200,
            "rsi":         rsi,
            "atr":         atr,
            "macd":        macd_v,
            "macd_signal": macd_sig,
            "macd_hist":   macd_h,
            "vol_ratio":   vol_ratio,
            "obv_trend":   obv_trend,
            "high_52w":    high_52w,
            "low_52w":     low_52w,
            "bb_pct":      _safe(last.get("BB_pct")),
        })

        # ── EMA Alignment Score (0–4: how many EMAs price is above) ───────────
        alignment = sum([
            1 if ema20  and close > ema20  else 0,
            1 if ema50  and close > ema50  else 0,
            1 if ema150 and close > ema150 else 0,
            1 if ema200 and close > ema200 else 0,
        ])
        result["ema_alignment"] = alignment   # 4 = perfect uptrend

        # ── 12-1 Month Momentum ───────────────────────────────────────────────
        # Standard academic momentum: skip recent 1 month
        if len(df) >= 252:
            price_252d_ago = df.iloc[-252]["Close"]
            price_21d_ago  = df.iloc[-21]["Close"]
            if price_252d_ago and price_252d_ago != 0:
                result["momentum_12_1"] = ((price_21d_ago / price_252d_ago) - 1) * 100
            else:
                result["momentum_12_1"] = None
        else:
            result["momentum_12_1"] = None

        # ── 3-Month Momentum ──────────────────────────────────────────────────
        if len(df) >= 63:
            price_63d_ago = df.iloc[-63]["Close"]
            if price_63d_ago and price_63d_ago != 0:
                result["momentum_3m"] = ((close / price_63d_ago) - 1) * 100
            else:
                result["momentum_3m"] = None
        else:
            result["momentum_3m"] = None

        # ── Relative Strength vs benchmark (SPY) ──────────────────────────────
        # RS ratio > 1.0 means stock outperforming the market (Qullamaggie filter)
        stock_12m = result.get("momentum_12_1")
        if benchmark_return_12m is not None and stock_12m is not None:
            try:
                bm_r  = 1 + benchmark_return_12m / 100
                stk_r = 1 + stock_12m / 100
                result["rs_vs_spy"] = round(stk_r / bm_r, 3) if bm_r != 0 else None
            except Exception:
                result["rs_vs_spy"] = None
        else:
            result["rs_vs_spy"] = None

        # ── Entry Setup Detection ─────────────────────────────────────────────
        setup, confidence = self._detect_entry_setup(
            close, ema20, ema50, ema150, ema200, rsi,
            macd_v, macd_sig, macd_h, prev_macd_h,
            vol_ratio, high_52w, alignment, obv_trend
        )
        result["entry_setup"]       = setup
        result["setup_confidence"]  = confidence

        # ── Risk Levels (ATR-based, fully deterministic) ───────────────────────
        if atr:
            result["stop_loss"]     = round(close - ATR_STOP_MULTIPLIER   * atr, 2)
            result["target_short"]  = round(close + ATR_TARGET_MULT_SHORT  * atr, 2)
            result["target_long"]   = round(close + ATR_TARGET_MULT_LONG   * atr, 2)
            result["risk_reward"]   = round(ATR_TARGET_MULT_SHORT / ATR_STOP_MULTIPLIER, 2)
        else:
            result["stop_loss"] = result["target_short"] = result["target_long"] = result["risk_reward"] = None

        return result

    # ── Entry Setup Logic ─────────────────────────────────────────────────────

    def _detect_entry_setup(
        self,
        close, ema20, ema50, ema150, ema200, rsi,
        macd, macd_sig, macd_hist, prev_macd_hist,
        vol_ratio, high_52w, alignment, obv_trend
    ) -> Tuple[str, float]:
        """
        Detect the highest-confidence entry setup present.

        Returns (setup_name, confidence_0_to_1).
        setup_name can be:
          "RSI_PULLBACK"   — pullback to oversold zone in uptrend
          "BREAKOUT"       — volume-confirmed 52-week high breakout
          "MACD_CROSSOVER" — MACD line crosses above signal in uptrend
          "SEPA_PULLBACK"  — Minervini pullback to 20 EMA in Stage 2
          "NONE"           — no clean entry setup detected
        """
        none = ("NONE", 0.0)

        # Require at least partial uptrend for ANY setup
        if alignment < 2 or not rsi:
            return none

        checks: list[Tuple[str, float]] = []

        # ── Setup 1: RSI Pullback in Uptrend ──────────────────────────────────
        if (
            alignment >= 3
            and RSI_PULLBACK_LOW <= rsi <= RSI_PULLBACK_HIGH
        ):
            confidence = 0.60
            # Bonus: OBV still trending up (institutional accumulation)
            if obv_trend == 1.0:
                confidence += 0.15
            # Bonus: EMA-20 acting as support (close near EMA-20)
            if ema20 and abs(close - ema20) / close < 0.02:
                confidence += 0.10
            checks.append(("RSI_PULLBACK", min(confidence, 0.95)))

        # ── Setup 2: Volume-Confirmed Breakout ────────────────────────────────
        if (
            high_52w
            and close >= high_52w * 0.995              # within 0.5 % of 52-week high
            and vol_ratio is not None
            and vol_ratio >= VOLUME_BREAKOUT_MULT
            and rsi >= RSI_BREAKOUT_MIN
        ):
            confidence = 0.65
            if alignment == 4:
                confidence += 0.15
            if obv_trend == 1.0:
                confidence += 0.10
            checks.append(("BREAKOUT", min(confidence, 0.95)))

        # ── Setup 3: MACD Bullish Crossover ───────────────────────────────────
        if (
            macd is not None and macd_sig is not None
            and macd_hist is not None and prev_macd_hist is not None
            and macd_hist > 0                        # histogram just turned positive
            and prev_macd_hist <= 0                  # was negative (crossover happened)
            and alignment >= 3
        ):
            confidence = 0.60
            if rsi and 45 <= rsi <= 65:
                confidence += 0.10
            checks.append(("MACD_CROSSOVER", min(confidence, 0.85)))

        # ── Setup 4: Minervini SEPA Pullback to 20 EMA ────────────────────────
        if (
            alignment == 4
            and ema20 is not None
            and abs(close - ema20) / close < 0.03    # within 3 % of 20 EMA
            and rsi and 40 <= rsi <= 58
        ):
            confidence = 0.70
            if obv_trend == 1.0:
                confidence += 0.10
            checks.append(("SEPA_PULLBACK", min(confidence, 0.90)))

        if not checks:
            return none

        # Return the highest-confidence setup
        return max(checks, key=lambda x: x[1])

"""
Composite multi-factor scorer.

Implements cross-sectional percentile ranking — the same approach used by
institutional factor investors (AQR, Dimensional, Two Sigma).

Rather than applying hard numeric thresholds (which are arbitrary and
regime-dependent), we rank each metric relative to the ENTIRE universe
on the same day.  A stock scoring in the 90th percentile of ROIC is
"better" than 90 % of its peers, regardless of the absolute ROIC value.

Factor groups and weights (configurable in settings.py):
  Quality   (30%): ROIC, ROE, FCF yield, D/E, net margin, current ratio
  Momentum  (25%): 12-1 month price return, 3-month return, EPS acceleration
  Technical (28%): EMA alignment, RSI setup score, volume, OBV, setup detected
  Value     (17%): PEG, EV/EBITDA, P/FCF (lower is better = inverted rank)

Composite score = weighted average of the four group percentile scores.
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from loguru import logger

from config.settings import FACTOR_WEIGHTS


# Metrics where LOWER value = better (we invert their rank)
_LOWER_IS_BETTER = {"debt_equity", "ev_ebitda", "p_fcf", "peg_ratio", "p_book"}

# Metrics in each factor group
_QUALITY_METRICS  = ["roic", "roe", "fcf_yield", "net_margin", "gross_margin", "current_ratio"]
_MOMENTUM_METRICS = ["momentum_12_1", "momentum_3m", "eps_growth_yoy", "eps_acceleration", "eps_surprise_pct"]
_TECHNICAL_METRICS= ["ema_alignment", "rsi_setup_score", "vol_ratio", "momentum_12_1_tech"]
_VALUE_METRICS    = ["peg_ratio", "ev_ebitda", "p_fcf", "p_book"]


def _percentile_rank(series: pd.Series, invert: bool = False) -> pd.Series:
    """Return per-element percentile rank (0–100) within the Series."""
    ranked = series.rank(pct=True, na_option="keep") * 100
    if invert:
        ranked = 100 - ranked
    return ranked


def _rsi_setup_score(rsi: float | None, ema_alignment: float | None) -> float:
    """
    Encode the RSI entry setup attractiveness as a 0–100 score.
    Best score: RSI in 40–55 zone (ideal pullback) + perfect EMA alignment.
    """
    if rsi is None:
        return 50.0
    alignment_bonus = (ema_alignment or 0) * 10  # 0–40 pts

    # RSI proximity to ideal pullback zone (40–52)
    ideal_rsi = 48.0
    distance  = abs(rsi - ideal_rsi)
    rsi_score = max(0.0, 60.0 - distance * 2)  # 60 at ideal, decays with distance

    return min(100.0, rsi_score + alignment_bonus)


def score_universe(records: List[Dict]) -> pd.DataFrame:
    """
    Accept a list of per-ticker raw metric dicts (output of fundamental +
    technical analysers) and return a DataFrame with composite scores
    and factor percentiles.

    Parameters
    ----------
    records : list of dicts, each containing 'symbol' + raw metric fields

    Returns
    -------
    pd.DataFrame sorted descending by composite_score
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.set_index("symbol") if "symbol" in df.columns else df

    # ── Derived helper columns ─────────────────────────────────────────────────
    df["rsi_setup_score"] = df.apply(
        lambda r: _rsi_setup_score(r.get("rsi"), r.get("ema_alignment")), axis=1
    )
    # Mirror momentum into technical space (price trend)
    df["momentum_12_1_tech"] = df.get("momentum_12_1", pd.Series(dtype=float, index=df.index))

    # ── Per-Metric Percentile Ranks ───────────────────────────────────────────
    pct_df = pd.DataFrame(index=df.index)

    for col in _QUALITY_METRICS + _MOMENTUM_METRICS + _TECHNICAL_METRICS + _VALUE_METRICS:
        if col not in df.columns:
            df[col] = np.nan
        invert = col in _LOWER_IS_BETTER
        pct_df[f"pct_{col}"] = _percentile_rank(df[col], invert=invert)

    # ── Group Scores ──────────────────────────────────────────────────────────

    def _group_score(metric_list: list, prefix: str = "pct_") -> pd.Series:
        cols = [f"{prefix}{m}" for m in metric_list if f"{prefix}{m}" in pct_df.columns]
        if not cols:
            return pd.Series(50.0, index=pct_df.index)
        sub = pct_df[cols].copy()
        # Fill NaN with 50 (neutral) so missing data doesn't penalise too much
        sub = sub.fillna(50.0)
        return sub.mean(axis=1)

    pct_df["quality_score"]   = _group_score(_QUALITY_METRICS)
    pct_df["momentum_score"]  = _group_score(_MOMENTUM_METRICS)
    pct_df["technical_score"] = _group_score(_TECHNICAL_METRICS)
    pct_df["value_score"]     = _group_score(_VALUE_METRICS)

    # ── Composite Weighted Score ───────────────────────────────────────────────
    w = FACTOR_WEIGHTS
    pct_df["composite_score"] = (
        pct_df["quality_score"]   * w["quality"]   +
        pct_df["momentum_score"]  * w["momentum"]  +
        pct_df["technical_score"] * w["technical"] +
        pct_df["value_score"]     * w["value"]
    )

    # ── Entry Setup Bonus (stocks with a detected setup get a nudge) ──────────
    if "entry_setup" in df.columns:
        setup_bonus = df["entry_setup"].map({
            "SEPA_PULLBACK":  5.0,
            "RSI_PULLBACK":   4.0,
            "MACD_CROSSOVER": 3.0,
            "BREAKOUT":       3.5,
            "NONE":           0.0,
        }).fillna(0.0)
        pct_df["composite_score"] += setup_bonus

    pct_df["composite_score"] = pct_df["composite_score"].clip(0, 105)

    # Merge back the raw metrics for downstream use
    out = pd.concat([df, pct_df[["quality_score", "momentum_score",
                                  "technical_score", "value_score",
                                  "composite_score"]]], axis=1)
    out = out.reset_index().rename(columns={"index": "symbol"}) \
        if out.index.name == "symbol" else out.reset_index(drop=True)

    out = out.sort_values("composite_score", ascending=False).reset_index(drop=True)

    logger.info(
        f"Composite scoring complete — top score: "
        f"{out['composite_score'].iloc[0]:.1f} "
        f"({out['symbol'].iloc[0] if 'symbol' in out.columns else 'N/A'})"
    )
    return out

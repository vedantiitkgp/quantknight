"""
Performance metrics for backtesting and paper trading evaluation.

Computes all standard institutional metrics:
  - Total return, CAGR
  - Sharpe Ratio (annualised, risk-free = 5 %)
  - Sortino Ratio (downside deviation only)
  - Calmar Ratio (CAGR / Max Drawdown)
  - Maximum Drawdown (%) and Max Drawdown Duration (trading days)
  - Win Rate, Profit Factor, Expectancy
  - Value at Risk (95 % confidence, 1-day)

These are computed from either an equity curve (pd.Series) or a
trade log (pd.DataFrame) depending on the calling context.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

RISK_FREE_RATE = 0.05    # annualised (5 %)
TRADING_DAYS   = 252


# ── Equity Curve Metrics ──────────────────────────────────────────────────────

def compute_metrics(equity_curve: pd.Series, initial_capital: float) -> Dict:
    """
    Full performance suite from a daily equity curve.

    Parameters
    ----------
    equity_curve   : pd.Series (index = dates, values = portfolio value)
    initial_capital: float

    Returns
    -------
    dict of metric names → float values
    """
    if equity_curve.empty or len(equity_curve) < 5:
        return {"error": "insufficient data"}

    eq = equity_curve.dropna()

    # ── Returns ───────────────────────────────────────────────────────────────
    daily_returns = eq.pct_change().dropna()
    total_return  = (eq.iloc[-1] / initial_capital) - 1.0
    n_years       = len(eq) / TRADING_DAYS
    cagr          = (eq.iloc[-1] / initial_capital) ** (1 / max(n_years, 0.001)) - 1

    # ── Volatility ────────────────────────────────────────────────────────────
    ann_vol       = daily_returns.std() * np.sqrt(TRADING_DAYS)
    downside      = daily_returns[daily_returns < 0]
    ann_downside_vol = downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 0 else 0

    # ── Drawdown ──────────────────────────────────────────────────────────────
    rolling_max   = eq.cummax()
    drawdown      = (eq - rolling_max) / rolling_max
    max_dd        = drawdown.min()
    # Drawdown duration
    in_dd         = drawdown < 0
    dd_duration   = 0
    current_dur   = 0
    for flag in in_dd:
        if flag:
            current_dur += 1
            dd_duration = max(dd_duration, current_dur)
        else:
            current_dur = 0

    # ── Risk-Adjusted Ratios ──────────────────────────────────────────────────
    excess_return = cagr - RISK_FREE_RATE
    sharpe        = excess_return / ann_vol       if ann_vol > 0       else 0.0
    sortino       = excess_return / ann_downside_vol if ann_downside_vol > 0 else 0.0
    calmar        = cagr / abs(max_dd)            if max_dd < 0        else 0.0

    # ── Value at Risk (Historical, 95 %) ──────────────────────────────────────
    var_95        = float(np.percentile(daily_returns, 5)) if len(daily_returns) > 0 else 0.0

    return {
        "total_return_pct":    round(total_return * 100, 2),
        "cagr_pct":            round(cagr * 100, 2),
        "ann_volatility_pct":  round(ann_vol * 100, 2),
        "sharpe_ratio":        round(sharpe, 3),
        "sortino_ratio":       round(sortino, 3),
        "calmar_ratio":        round(calmar, 3),
        "max_drawdown_pct":    round(max_dd * 100, 2),
        "max_dd_duration_days":dd_duration,
        "var_95_pct":          round(var_95 * 100, 2),
        "n_trading_days":      len(eq),
    }


# ── Trade Log Metrics (paper trading evaluation) ───────────────────────────────

def compute_trade_metrics(trades: List[Dict]) -> Dict:
    """
    Compute win rate, profit factor, and expectancy from a closed trade log.

    Each trade dict must have:
      - 'pnl_pct'   : float  (profit/loss as %)
      - 'hit_target': bool
      - 'hit_stop'  : bool
    """
    if not trades:
        return {"error": "no closed trades"}

    pnls    = [t["pnl_pct"] for t in trades if t.get("pnl_pct") is not None]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p <= 0]

    if not pnls:
        return {"error": "no pnl data"}

    win_rate      = len(wins) / len(pnls)
    gross_profit  = sum(wins)
    gross_loss    = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win  = np.mean(wins)  if wins   else 0.0
    avg_loss = np.mean(losses) if losses else 0.0

    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    return {
        "total_trades":    len(pnls),
        "win_count":       len(wins),
        "loss_count":      len(losses),
        "win_rate_pct":    round(win_rate * 100, 2),
        "avg_win_pct":     round(avg_win, 2),
        "avg_loss_pct":    round(avg_loss, 2),
        "gross_profit_pct":round(gross_profit, 2),
        "gross_loss_pct":  round(gross_loss, 2),
        "profit_factor":   round(profit_factor, 3),
        "expectancy_pct":  round(expectancy, 3),
    }


def format_metrics_report(metrics: Dict, title: str = "Performance Report") -> str:
    """Return a formatted plain-text performance summary."""
    lines = [
        f"\n{'=' * 55}",
        f"  {title}",
        f"{'=' * 55}",
    ]
    for k, v in metrics.items():
        if k == "error":
            lines.append(f"  ERROR: {v}")
            continue
        label = k.replace("_", " ").title()
        lines.append(f"  {label:<35} {v}")
    lines.append("=" * 55)
    return "\n".join(lines)

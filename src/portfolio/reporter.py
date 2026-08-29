"""
QuantKnight Report Generator.

Generates daily, weekly, and monthly markdown reports saved to data/reports/.
Each report includes:
  - Portfolio summary (equity, cash, P&L)
  - Positions opened/closed that period
  - Per-trade reasoning (why we entered, what happened)
  - Win rate, average gain/loss, risk metrics
  - Weekly/monthly cumulative view
"""
import json
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger

_TRADES_DIR  = "data/trades"
_REPORTS_DIR = "data/reports"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_trades_for_date(d: date) -> Optional[Dict]:
    path = f"{_TRADES_DIR}/{d}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _pct_bar(pct: float, width: int = 20) -> str:
    """ASCII progress bar for win-rate or fill percentage."""
    filled = int(max(0.0, min(1.0, pct / 100)) * width)
    return "█" * filled + "░" * (width - filled)


def _sign(n: float) -> str:
    return "+" if n >= 0 else ""


def _fmt_pnl(n: float) -> str:
    s = _sign(n)
    return f"{s}${n:,.0f}"


def _trade_summary_lines(exits: List[Dict]) -> List[str]:
    lines = []
    for e in exits:
        sym  = e.get("symbol", "?")
        dirn = e.get("direction", "LONG")
        shrs = e.get("shares", 0)
        ep   = e.get("entry_price", 0)
        xp   = e.get("exit_price", 0)
        pnl  = e.get("pnl_dollars", 0)
        pct  = e.get("pnl_pct", 0)
        rsn  = e.get("reason", "?")
        arrow = "▲" if pnl >= 0 else "▼"
        lines.append(
            f"  {arrow} **{sym}** ({dirn}) — "
            f"{shrs} shares @ ${ep:.2f} → ${xp:.2f} | "
            f"**{_fmt_pnl(pnl)} ({_sign(pct)}{pct:.1f}%)** — _{rsn}_"
        )
    return lines


# ── Daily Report ──────────────────────────────────────────────────────────────

def generate_daily(portfolio: Dict, trades: Dict, run_date: Optional[date] = None) -> str:
    """
    Generate a daily markdown report string and save to
    data/reports/daily_YYYY-MM-DD.md.
    """
    os.makedirs(_REPORTS_DIR, exist_ok=True)
    d = run_date or date.today()

    equity   = portfolio.get("total_equity", 0)
    cash     = portfolio.get("cash", 0)
    cum_pnl  = portfolio.get("cumulative_pnl", 0)
    positions = portfolio.get("positions", [])

    entries  = trades.get("entries", [])
    exits    = trades.get("exits", [])
    real_pnl = trades.get("realized_pnl", 0)
    unreal   = trades.get("unrealized_pnl", 0)
    daily_pnl = real_pnl + unreal

    # Stats
    wins   = [e for e in exits if e.get("pnl_dollars", 0) >= 0]
    losses = [e for e in exits if e.get("pnl_dollars", 0) < 0]
    win_rate = (len(wins) / len(exits) * 100) if exits else 0
    avg_win  = (sum(e["pnl_dollars"] for e in wins)   / len(wins))   if wins   else 0
    avg_loss = (sum(e["pnl_dollars"] for e in losses) / len(losses)) if losses else 0

    lines = [
        f"# QuantKnight Daily Report — {d}",
        f"> Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Portfolio Snapshot",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Equity | ${equity:,.0f} |",
        f"| Cash Available | ${cash:,.0f} |",
        f"| Daily P&L | {_fmt_pnl(daily_pnl)} |",
        f"| Realized Today | {_fmt_pnl(real_pnl)} |",
        f"| Unrealized (MTM) | {_fmt_pnl(unreal)} |",
        f"| Cumulative P&L | {_fmt_pnl(cum_pnl)} |",
        f"| Open Positions | {len(positions)} |",
        "",
    ]

    # Open positions
    if positions:
        lines += [
            "## Open Positions",
            "| Symbol | Dir | Shares | Entry | Stop | Target | Score | Type |",
            "|--------|-----|--------|-------|------|--------|-------|------|",
        ]
        for p in positions:
            lines.append(
                f"| {p['symbol']} | {p['direction']} | {p['shares']} "
                f"| ${p['entry_price']:.2f} | ${p.get('stop_loss',0) or 0:.2f} "
                f"| ${p.get('target',0) or 0:.2f} "
                f"| {p.get('composite_score',0) or 0:.1f} | {p.get('trade_type','swing')} |"
            )
        lines.append("")

    # Entries
    if entries:
        lines += ["## Trades Entered Today", ""]
        for e in entries:
            lines.append(
                f"- **{e['symbol']}** ({e['direction']}) — "
                f"{e['shares']} shares @ ${e['entry_price']:.2f} "
                f"| stop ${e.get('stop_loss',0) or 0:.2f} "
                f"| target ${e.get('target',0) or 0:.2f} "
                f"| cost ${e.get('cost',0):,.0f}"
            )
            if e.get("reason"):
                lines.append(f"  > _{e['reason']}_")
        lines.append("")

    # Exits
    if exits:
        lines += ["## Trades Closed Today", ""]
        win_lines  = _trade_summary_lines(wins)
        loss_lines = _trade_summary_lines(losses)
        if win_lines:
            lines += ["### Winners"] + win_lines + [""]
        if loss_lines:
            lines += ["### Losers"] + loss_lines + [""]

    # Day stats
    if exits:
        lines += [
            "## Today's Performance",
            f"| Trades Closed | Win Rate | Avg Win | Avg Loss |",
            f"|--------------|----------|---------|----------|",
            f"| {len(exits)} | {win_rate:.0f}% {_pct_bar(win_rate, 10)} "
            f"| {_fmt_pnl(avg_win)} | {_fmt_pnl(avg_loss)} |",
            "",
        ]

    if not entries and not exits:
        lines += ["_No trades executed today._", ""]

    report = "\n".join(lines)
    path = f"{_REPORTS_DIR}/daily_{d}.md"
    with open(path, "w") as f:
        f.write(report)
    logger.info(f"Daily report saved → {path}")
    return report


# ── Weekly Report ─────────────────────────────────────────────────────────────

def generate_weekly(portfolio: Dict, week_number: Optional[int] = None,
                    year: Optional[int] = None) -> str:
    """
    Aggregate trade data for the current ISO week and generate a weekly summary.
    Saved to data/reports/week_YYYY-WW.md.
    """
    os.makedirs(_REPORTS_DIR, exist_ok=True)
    today = date.today()
    yr    = year        or today.isocalendar()[0]
    wk    = week_number or today.isocalendar()[1]

    # Collect daily trade files for this week
    all_entries: List[Dict] = []
    all_exits:   List[Dict] = []
    daily_pnls:  List[float] = []

    for delta in range(7):
        d = date.fromisocalendar(yr, wk, 1) + timedelta(days=delta)
        t = _load_trades_for_date(d)
        if t:
            all_entries.extend(t.get("entries", []))
            all_exits.extend(t.get("exits", []))
            realized = t.get("realized_pnl", 0)
            unreal   = t.get("unrealized_pnl", 0)
            daily_pnls.append(realized + unreal)

    wins   = [e for e in all_exits if e.get("pnl_dollars", 0) >= 0]
    losses = [e for e in all_exits if e.get("pnl_dollars", 0) < 0]
    win_rate = (len(wins) / len(all_exits) * 100) if all_exits else 0
    total_pnl  = sum(e.get("pnl_dollars", 0) for e in all_exits)
    avg_win    = (sum(e["pnl_dollars"] for e in wins)   / len(wins))   if wins   else 0
    avg_loss   = (sum(e["pnl_dollars"] for e in losses) / len(losses)) if losses else 0
    profit_fac = (abs(sum(e["pnl_dollars"] for e in wins)) /
                  max(abs(sum(e["pnl_dollars"] for e in losses)), 1))

    equity = portfolio.get("total_equity", 0)
    cum_pnl = portfolio.get("cumulative_pnl", 0)

    lines = [
        f"# QuantKnight Weekly Report — {yr} Week {wk:02d}",
        f"> Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Week at a Glance",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Portfolio Equity | ${equity:,.0f} |",
        f"| Week P&L (realized) | {_fmt_pnl(total_pnl)} |",
        f"| Cumulative P&L | {_fmt_pnl(cum_pnl)} |",
        f"| Positions Opened | {len(all_entries)} |",
        f"| Positions Closed | {len(all_exits)} |",
        f"| Win Rate | {win_rate:.0f}% |",
        f"| Avg Win | {_fmt_pnl(avg_win)} |",
        f"| Avg Loss | {_fmt_pnl(avg_loss)} |",
        f"| Profit Factor | {profit_fac:.2f} |",
        "",
        f"Win rate bar: `{_pct_bar(win_rate)}`",
        "",
    ]

    if all_exits:
        lines += ["## All Closed Trades This Week", ""]
        lines += _trade_summary_lines(all_exits)
        lines.append("")

    if all_entries:
        lines += [
            "## Positions Opened This Week",
            "| Symbol | Dir | Shares | Entry | Score | Type |",
            "|--------|-----|--------|-------|-------|------|",
        ]
        for e in all_entries:
            lines.append(
                f"| {e['symbol']} | {e['direction']} | {e['shares']} "
                f"| ${e['entry_price']:.2f} "
                f"| {e.get('composite_score',0) or 0:.1f} | {e.get('trade_type','swing')} |"
            )
        lines.append("")

    report = "\n".join(lines)
    path = f"{_REPORTS_DIR}/week_{yr}-{wk:02d}.md"
    with open(path, "w") as f:
        f.write(report)
    logger.info(f"Weekly report saved → {path}")
    return report


# ── Monthly Report ────────────────────────────────────────────────────────────

def generate_monthly(portfolio: Dict, month: Optional[int] = None,
                     year: Optional[int] = None) -> str:
    """
    Aggregate trade data for the current calendar month and generate a monthly summary.
    Saved to data/reports/month_YYYY-MM.md.
    """
    os.makedirs(_REPORTS_DIR, exist_ok=True)
    today = date.today()
    yr    = year  or today.year
    mo    = month or today.month

    # Walk all trading days in month
    all_entries: List[Dict] = []
    all_exits:   List[Dict] = []

    # Determine days in month
    if mo == 12:
        next_month_first = date(yr + 1, 1, 1)
    else:
        next_month_first = date(yr, mo + 1, 1)

    d = date(yr, mo, 1)
    while d < next_month_first:
        t = _load_trades_for_date(d)
        if t:
            all_entries.extend(t.get("entries", []))
            all_exits.extend(t.get("exits", []))
        d += timedelta(days=1)

    wins   = [e for e in all_exits if e.get("pnl_dollars", 0) >= 0]
    losses = [e for e in all_exits if e.get("pnl_dollars", 0) < 0]
    win_rate   = (len(wins) / len(all_exits) * 100) if all_exits else 0
    total_pnl  = sum(e.get("pnl_dollars", 0) for e in all_exits)
    avg_win    = (sum(e["pnl_dollars"] for e in wins)   / len(wins))   if wins   else 0
    avg_loss   = (sum(e["pnl_dollars"] for e in losses) / len(losses)) if losses else 0
    profit_fac = (abs(sum(e["pnl_dollars"] for e in wins)) /
                  max(abs(sum(e["pnl_dollars"] for e in losses)), 1))

    # Expectancy = win_rate*avg_win - loss_rate*|avg_loss|
    loss_rate  = 100 - win_rate
    expectancy = (win_rate / 100 * avg_win) + (loss_rate / 100 * avg_loss)

    equity  = portfolio.get("total_equity", 0)
    cum_pnl = portfolio.get("cumulative_pnl", 0)

    import calendar
    month_name = calendar.month_name[mo]

    lines = [
        f"# QuantKnight Monthly Report — {month_name} {yr}",
        f"> Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Monthly Overview",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Portfolio Equity | ${equity:,.0f} |",
        f"| Month P&L (realized) | {_fmt_pnl(total_pnl)} |",
        f"| All-Time Cumulative P&L | {_fmt_pnl(cum_pnl)} |",
        f"| Positions Opened | {len(all_entries)} |",
        f"| Positions Closed | {len(all_exits)} |",
        f"| Winners | {len(wins)} |",
        f"| Losers | {len(losses)} |",
        f"| Win Rate | {win_rate:.0f}% |",
        f"| Avg Win | {_fmt_pnl(avg_win)} |",
        f"| Avg Loss | {_fmt_pnl(avg_loss)} |",
        f"| Profit Factor | {profit_fac:.2f} |",
        f"| Expectancy / Trade | {_fmt_pnl(expectancy)} |",
        "",
        f"Win rate bar: `{_pct_bar(win_rate)}`",
        "",
    ]

    # Top winners/losers
    sorted_exits = sorted(all_exits, key=lambda x: x.get("pnl_dollars", 0), reverse=True)
    if sorted_exits:
        lines += ["## Best & Worst Trades", ""]
        top3 = sorted_exits[:3]
        bot3 = sorted_exits[-3:][::-1]
        if top3:
            lines += ["### Top Winners"]
            lines += _trade_summary_lines(top3)
            lines.append("")
        if bot3 and bot3 != top3:
            lines += ["### Biggest Losers"]
            lines += _trade_summary_lines(bot3)
            lines.append("")

    # Symbol breakdown
    if all_exits:
        lines += ["## Trade-by-Trade Log", ""]
        lines += _trade_summary_lines(all_exits)
        lines.append("")

    report = "\n".join(lines)
    path = f"{_REPORTS_DIR}/month_{yr}-{mo:02d}.md"
    with open(path, "w") as f:
        f.write(report)
    logger.info(f"Monthly report saved → {path}")
    return report

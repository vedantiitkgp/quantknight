"""
Performance dashboard — run anytime to see paper trading accuracy.

Usage:
  python scripts/check_performance.py
  python scripts/check_performance.py --days 30
  python scripts/check_performance.py --resolve   # force-resolve open trades first
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
from datetime import date, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.models.database import get_session, PaperTrade, Recommendation, PerformanceStats
from src.paper_trading.tracker import PaperTradeTracker
from src.backtest.metrics import compute_trade_metrics, format_metrics_report

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Stock Engine Performance Dashboard")
    parser.add_argument("--days",    type=int, default=90,
                        help="Look-back window in calendar days (default: 90)")
    parser.add_argument("--resolve", action="store_true",
                        help="Resolve open paper trades before reporting")
    args = parser.parse_args()

    session = get_session()
    tracker = PaperTradeTracker()

    if args.resolve:
        console.print("Resolving open paper trades …")
        resolved = tracker.resolve_open_trades(session)
        console.print(f"  Resolved: [green]{resolved}[/green] trades\n")

    since = date.today() - timedelta(days=args.days)

    # ── Open Trades ────────────────────────────────────────────────────────────
    open_trades = (
        session.query(PaperTrade)
        .filter(PaperTrade.status == "OPEN", PaperTrade.open_date >= since)
        .order_by(PaperTrade.open_date.desc())
        .all()
    )

    # ── Closed Trades ──────────────────────────────────────────────────────────
    closed_trades = (
        session.query(PaperTrade)
        .filter(PaperTrade.status != "OPEN", PaperTrade.open_date >= since)
        .order_by(PaperTrade.close_date.desc())
        .all()
    )

    # ── Performance Metrics ────────────────────────────────────────────────────
    trade_dicts = [
        {"pnl_pct": t.pnl_pct or 0.0, "hit_target": t.hit_target, "hit_stop": t.hit_stop}
        for t in closed_trades
    ]
    metrics = compute_trade_metrics(trade_dicts) if trade_dicts else {}

    # ── Terminal Output ────────────────────────────────────────────────────────
    console.rule(f"[bold cyan]Stock Engine — Performance ({args.days}-day window)[/bold cyan]")

    # Summary panel
    if metrics and "error" not in metrics:
        summary = (
            f"  Closed Trades:   [bold]{metrics.get('total_trades', 0)}[/bold]  |  "
            f"Open: [bold]{len(open_trades)}[/bold]\n"
            f"  Win Rate:        [green]{metrics.get('win_rate_pct', 0):.1f}%[/green]\n"
            f"  Avg Win:         [green]+{metrics.get('avg_win_pct', 0):.1f}%[/green]  |  "
            f"Avg Loss: [red]{metrics.get('avg_loss_pct', 0):.1f}%[/red]\n"
            f"  Profit Factor:   [bold]{metrics.get('profit_factor', 0):.2f}[/bold]  "
            f"(>1.5 = strong)\n"
            f"  Expectancy:      [bold]{metrics.get('expectancy_pct', 0):+.2f}%[/bold] per trade"
        )
        console.print(Panel(summary, title="Paper Trading Accuracy", expand=False))
    else:
        console.print(Panel("[yellow]No closed trades yet in this window.[/yellow]",
                            title="Paper Trading Accuracy", expand=False))

    # Open trades table
    if open_trades:
        ot = Table(title="Open Paper Trades", show_lines=True)
        ot.add_column("Symbol",     style="bold cyan", width=8)
        ot.add_column("Opened",     width=12)
        ot.add_column("Entry $",    width=9)
        ot.add_column("Stop $",     width=9)
        ot.add_column("Target $",   width=9)
        ot.add_column("Days Open",  width=10)

        for t in open_trades:
            days = (date.today() - t.open_date).days
            ot.add_row(
                t.symbol,
                str(t.open_date),
                f"${t.entry_price:.2f}"  if t.entry_price  else "—",
                f"${t.stop_loss:.2f}"    if t.stop_loss    else "—",
                f"${t.target_short:.2f}" if t.target_short else "—",
                str(days),
            )
        console.print(ot)

    # Closed trades table
    if closed_trades:
        ct = Table(title="Recent Closed Trades", show_lines=True)
        ct.add_column("Symbol",   style="bold", width=8)
        ct.add_column("Opened",   width=12)
        ct.add_column("Closed",   width=12)
        ct.add_column("P&L %",    width=9)
        ct.add_column("Result",   width=10)

        for t in closed_trades[:30]:
            pnl_str = f"{t.pnl_pct:+.1f}%" if t.pnl_pct is not None else "—"
            pnl_fmt = f"[green]{pnl_str}[/green]" if (t.pnl_pct or 0) > 0 else f"[red]{pnl_str}[/red]"
            status_fmt = {
                "WIN":     "[green]WIN[/green]",
                "LOSS":    "[red]LOSS[/red]",
                "EXPIRED": "[yellow]EXPIRED[/yellow]",
            }.get(t.status, t.status)
            ct.add_row(
                t.symbol,
                str(t.open_date),
                str(t.close_date) if t.close_date else "—",
                pnl_fmt,
                status_fmt,
            )
        console.print(ct)

    session.close()


if __name__ == "__main__":
    main()

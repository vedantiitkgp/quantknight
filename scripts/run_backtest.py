"""
Standalone backtest runner.

Runs walk-forward validation for RSI_PULLBACK and BREAKOUT strategies
on a sample of high-quality S&P 500 stocks and prints a full report.

Usage:
  python scripts/run_backtest.py
  python scripts/run_backtest.py --strategy RSI_PULLBACK --symbols AAPL MSFT NVDA
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.data.fmp_client import FMPClient
from src.screener.universe import build_universe
from src.backtest.engine import run_walk_forward
from src.backtest.metrics import format_metrics_report

console = Console()

# Default sample — high-liquidity, well-known stocks
DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "JPM", "UNH", "V", "MA",
    "LLY", "AVGO", "HD", "XOM", "PG",
    "MRK", "COST", "ABBV", "TSLA", "JNJ",
]


def main():
    parser = argparse.ArgumentParser(description="Stock Engine Backtester")
    parser.add_argument("--strategy", choices=["RSI_PULLBACK", "BREAKOUT", "BOTH"],
                        default="BOTH", help="Which strategy to backtest")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Custom list of tickers (defaults to 20 S&P 500 names)")
    args = parser.parse_args()

    symbols = args.symbols or DEFAULT_SYMBOLS
    client  = FMPClient()

    strategies = (
        ["RSI_PULLBACK", "BREAKOUT"] if args.strategy == "BOTH" else [args.strategy]
    )

    all_results = {}

    for strat in strategies:
        console.rule(f"[bold cyan]Backtesting: {strat}[/bold cyan]")
        result = run_walk_forward(
            symbols  = symbols,
            strategy = strat,
            fmp_client = client,
        )
        all_results[strat] = result

        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
            continue

        # Print fold-level table
        table = Table(title=f"{strat} — Walk-Forward Folds", show_lines=True)
        table.add_column("Fold Period",   width=24)
        table.add_column("Return %",      width=10)
        table.add_column("CAGR %",        width=10)
        table.add_column("Sharpe",        width=8)
        table.add_column("Sortino",       width=8)
        table.add_column("Max DD %",      width=10)
        table.add_column("Calmar",        width=8)

        for fold in result.get("folds", []):
            table.add_row(
                f"{fold.get('fold_start','')} → {fold.get('fold_end','')}",
                f"{fold.get('total_return_pct', 0):.1f}%",
                f"{fold.get('cagr_pct', 0):.1f}%",
                f"{fold.get('sharpe_ratio', 0):.2f}",
                f"{fold.get('sortino_ratio', 0):.2f}",
                f"{fold.get('max_drawdown_pct', 0):.1f}%",
                f"{fold.get('calmar_ratio', 0):.2f}",
            )
        console.print(table)

        # Aggregate summary
        agg = result.get("aggregate", {})
        console.print(f"\n[bold]Aggregate across {agg.get('fold_count',0)} folds:[/bold]")
        console.print(
            f"  CAGR (avg):       {agg.get('cagr_pct', 0):.1f}%\n"
            f"  Sharpe (avg):     {agg.get('sharpe_ratio', 0):.2f}\n"
            f"  Max DD (avg):     {agg.get('max_drawdown_pct', 0):.1f}%\n"
            f"  Calmar (avg):     {agg.get('calmar_ratio', 0):.2f}\n"
            f"  Sortino (avg):    {agg.get('sortino_ratio', 0):.2f}\n"
        )

    console.rule("[bold green]Backtest Complete[/bold green]")


if __name__ == "__main__":
    main()

"""
Backfill bull/bear analysis for open positions that are missing it.

Fetches fresh fundamental + technical + sentiment data for every open
position in portfolio.json that has no bull_thesis, then runs the
multi-agent debate and writes the results back to portfolio.json.
Finally regenerates data/reports/index.html.

Usage:
  python scripts/backfill_analysis.py           # all positions missing analysis
  python scripts/backfill_analysis.py INCY CF   # specific symbols only
  python scripts/backfill_analysis.py --all     # force re-run even if already filled
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
from datetime import datetime
from rich.console import Console
from rich.table import Table

from src.data.yf_client import YFClient
from src.factors.fundamental import FundamentalAnalyser
from src.factors.technical import TechnicalAnalyser
from src.factors.sentiment import SentimentAnalyser
from src.agents.orchestrator import AgentOrchestrator
from src.portfolio.manager import PortfolioManager
from src.portfolio.dashboard import generate_dashboard

console = Console()
PORTFOLIO_PATH = "data/portfolio.json"


def backfill(symbols: list[str] = None, force: bool = False) -> None:
    console.rule("[bold cyan]QuantKnight — Backfill Analysis[/bold cyan]")

    # ── Load portfolio ─────────────────────────────────────────────────────────
    with open(PORTFOLIO_PATH) as f:
        portfolio = json.load(f)

    positions = portfolio.get("positions", [])
    if not positions:
        console.print("[yellow]No open positions.[/yellow]")
        return

    # Determine which positions need backfilling
    targets = []
    for pos in positions:
        sym = pos["symbol"]
        if symbols and sym not in symbols:
            continue
        already_filled = bool(pos.get("bull_thesis") or pos.get("full_memo"))
        if already_filled and not force:
            console.print(f"  [dim]{sym}: already has analysis — skip (use --all to force)[/dim]")
            continue
        targets.append(pos)

    if not targets:
        console.print("[green]All positions already have analysis.[/green]")
        return

    console.print(f"\nWill analyse: [bold]{', '.join(p['symbol'] for p in targets)}[/bold]\n")

    # ── Initialise engines ─────────────────────────────────────────────────────
    client      = YFClient()
    fund_engine = FundamentalAnalyser(client)
    tech_engine = TechnicalAnalyser()
    sent_engine = SentimentAnalyser()
    orchestrator = AgentOrchestrator()

    updated = []
    failed  = []

    for pos in targets:
        sym = pos["symbol"]
        console.rule(f"[bold]{sym}[/bold]")

        try:
            # Fundamentals
            console.print("  Fundamentals …")
            fund = fund_engine.analyse(sym)

            # Technicals
            console.print("  Technicals …")
            df = client.get_daily_ohlcv(sym, days=400)
            if df.empty:
                raise ValueError("No price data returned")
            tech = tech_engine.compute(df, symbol=sym)

            # Sentiment
            console.print("  Sentiment …")
            sent = sent_engine.analyse(sym)

            # Build candidate — merge everything, keep stored entry levels
            candidate = {
                **fund,
                **tech,
                "symbol":          sym,
                "sentiment_score": sent["sentiment_score"],
                "sentiment_label": sent["sentiment_label"],
                "top_headlines":   sent.get("top_headlines", []),
                "full_news":       sent.get("raw_articles", []),
                # keep original entry price for context in agent prompts
                "entry_price":     pos.get("entry_price"),
            }

            # Agent debate
            console.print("  Running agent debate …")
            result = orchestrator.run(candidate)

            verdict_color = {"APPROVED": "green", "WATCH": "yellow", "REJECTED": "red"}.get(
                result.get("verdict", "WATCH"), "white"
            )
            console.print(f"  Verdict: [{verdict_color}]{result.get('verdict')}[/{verdict_color}]")

            # Write back into the position dict in-memory
            pos["bull_thesis"] = result.get("bull_thesis", "")
            pos["bear_risks"]  = result.get("bear_risks", "")
            pos["full_memo"]   = result.get("full_memo", "")
            # Store top news for the news panel
            raw = sent.get("raw_articles", [])
            if raw:
                scored = [a for a in raw if a.get("sentiment") is not None]
                scored.sort(key=lambda a: abs(a["sentiment"]), reverse=True)
                pos["full_news"] = [
                    {
                        "headline":  a["headline"],
                        "summary":   (a.get("summary") or "")[:250].strip(),
                        "sentiment": round(a["sentiment"], 3),
                        "source":    a.get("source", ""),
                        "published": str(a.get("published", ""))[:10],
                        "url":       a.get("url", ""),
                    }
                    for a in scored[:10]
                ]

            updated.append(sym)

        except Exception as exc:
            console.print(f"  [red]Failed: {exc}[/red]")
            failed.append((sym, str(exc)))

    # ── Save portfolio.json ────────────────────────────────────────────────────
    if updated:
        portfolio["as_of"] = datetime.now().isoformat()
        with open(PORTFOLIO_PATH, "w") as f:
            json.dump(portfolio, f, indent=2, default=str)
        console.print(f"\n[green]Saved portfolio.json[/green]  (updated: {', '.join(updated)})")
    else:
        console.print("\n[yellow]No positions were updated.[/yellow]")

    if failed:
        console.print("\n[red]Failures:[/red]")
        for sym, err in failed:
            console.print(f"  {sym}: {err}")

    # ── Regenerate dashboard ───────────────────────────────────────────────────
    console.rule("[bold]Regenerating dashboard[/bold]")
    pm = PortfolioManager()
    today_trades = pm.today_trades
    out = generate_dashboard(pm.portfolio, today_trades)
    console.print(f"  Dashboard → [bold]{out}[/bold]")

    # Summary table
    tbl = Table(title="Backfill Results", show_header=True)
    tbl.add_column("Symbol");  tbl.add_column("Status")
    for sym in updated:
        tbl.add_row(sym, "[green]✓ Updated[/green]")
    for sym, _ in failed:
        tbl.add_row(sym, "[red]✗ Failed[/red]")
    console.print(tbl)
    console.rule("[bold green]Done[/bold green]")


def main():
    parser = argparse.ArgumentParser(description="Backfill agent analysis for open positions")
    parser.add_argument("symbols", nargs="*", help="Specific symbols to backfill (default: all missing)")
    parser.add_argument("--all", dest="force", action="store_true",
                        help="Re-run even for positions that already have analysis")
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols] if args.symbols else None
    backfill(symbols=symbols, force=args.force)


if __name__ == "__main__":
    main()

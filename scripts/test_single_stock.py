"""
Quick single-stock analysis — useful for testing the engine on one ticker
before running the full nightly pipeline.

Usage:
  python scripts/test_single_stock.py AAPL
  python scripts/test_single_stock.py NVDA --no-agent
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
from rich.console import Console
from rich.panel import Panel

from src.data.yf_client import YFClient as FMPClient
from src.factors.fundamental import FundamentalAnalyser
from src.factors.technical import TechnicalAnalyser
from src.factors.sentiment import SentimentAnalyser
from src.agents.orchestrator import AgentOrchestrator

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Test the engine on a single ticker")
    parser.add_argument("symbol",    type=str, help="Ticker symbol e.g. AAPL")
    parser.add_argument("--no-agent", action="store_true",
                        help="Skip the multi-agent debate (faster)")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    console.rule(f"[bold cyan]Stock Engine — Single Stock Analysis: {symbol}[/bold cyan]")

    def _fmt(v, fmt=".2f", fallback="N/A"):
        """Format a numeric value; return fallback when None or NaN."""
        return format(v, fmt) if v is not None and v == v else fallback

    client      = FMPClient()
    fund_engine = FundamentalAnalyser(client)
    tech_engine = TechnicalAnalyser()
    sent_engine = SentimentAnalyser()

    # ── Fundamentals ──────────────────────────────────────────────────────────
    console.print(f"\n[bold]Fundamental Analysis …[/bold]")
    fund = fund_engine.analyse(symbol)
    console.print(f"  ROIC:             {fund.get('roic', 'N/A'):.1f}%")
    console.print(f"  ROE:              {fund.get('roe', 'N/A'):.1f}%")
    console.print(f"  FCF Yield:        {fund.get('fcf_yield', 'N/A'):.1f}%")
    console.print(f"  Debt/Equity:      {fund.get('debt_equity', 'N/A'):.2f}")
    console.print(f"  EPS Growth YoY:   {_fmt(fund.get('eps_growth_yoy'), '.1f')}%")
    console.print(f"  EPS Acceleration: {_fmt(fund.get('eps_acceleration'), '.1f')}%")
    console.print(f"  PEG Ratio:        {_fmt(fund.get('peg_ratio'), '.2f')}")
    console.print(f"  EV/EBITDA:        {_fmt(fund.get('ev_ebitda'), '.2f')}")
    console.print(f"  Insider Net:      {fund.get('insider_net', 'N/A')} (+ = net buyers)")

    # ── Technicals ────────────────────────────────────────────────────────────
    console.print(f"\n[bold]Technical Analysis …[/bold]")
    df = client.get_daily_ohlcv(symbol, days=400)
    if df.empty:
        console.print("[red]  No price data available.[/red]")
        return

    tech = tech_engine.compute(df, symbol=symbol)
    console.print(f"  Current Price:    ${_fmt(tech.get('close'))}")
    console.print(f"  RSI (14d):        {_fmt(tech.get('rsi'), '.1f')}")
    console.print(f"  EMA Alignment:    {tech.get('ema_alignment', 0)}/4")
    console.print(f"  ATR (14d):        ${_fmt(tech.get('atr'))}")
    console.print(f"  Volume Ratio:     {_fmt(tech.get('vol_ratio'), '.2f')}x 20d avg")
    console.print(f"  Momentum 12-1m:   {_fmt(tech.get('momentum_12_1'), '.1f')}%")
    console.print(f"  Entry Setup:      [bold yellow]{tech.get('entry_setup', 'NONE')}[/bold yellow]"
                  f" (confidence: {_fmt(tech.get('setup_confidence', 0), '.0%')})")

    if tech.get("entry_setup", "NONE") != "NONE":
        console.print(f"\n  [green]Entry Levels (ATR-based, deterministic):[/green]")
        console.print(f"    Stop-Loss:      ${tech.get('stop_loss', 0):.2f}")
        console.print(f"    Short Target:   ${tech.get('target_short', 0):.2f}")
        console.print(f"    Long Target:    ${tech.get('target_long', 0):.2f}")
        console.print(f"    Risk/Reward:    1 : {tech.get('risk_reward', 0):.1f}")

    # ── Sentiment ─────────────────────────────────────────────────────────────
    console.print(f"\n[bold]News Sentiment (FinBERT) …[/bold]")
    sent = sent_engine.analyse(symbol)
    console.print(f"  Articles scanned: {sent.get('article_count', 0)}")
    console.print(f"  Sentiment Score:  {sent.get('sentiment_score', 0):+.3f}")
    console.print(f"  Sentiment Label:  [bold]{sent.get('sentiment_label', 'Neutral')}[/bold]")
    if sent.get("top_headlines"):
        console.print("  Top Headlines:")
        for h in sent["top_headlines"][:3]:
            score = h.get("sentiment", 0)
            colour = "green" if score > 0.1 else ("red" if score < -0.1 else "yellow")
            console.print(f"    [{colour}]({score:+.2f})[/{colour}] {h['headline'][:90]}")

    # ── Agent Debate ──────────────────────────────────────────────────────────
    if not args.no_agent:
        console.print(f"\n[bold]Multi-Agent Debate …[/bold]")
        candidate = {**fund, **tech,
                     "sentiment_score": sent["sentiment_score"],
                     "sentiment_label": sent["sentiment_label"]}
        orchestrator = AgentOrchestrator()
        result = orchestrator.run(candidate)

        verdict_colour = {"APPROVED": "green", "WATCH": "yellow", "REJECTED": "red"}.get(
            result.get("verdict", "WATCH"), "white"
        )
        console.print(Panel(
            result.get("full_memo", "No memo generated."),
            title=f"[{verdict_colour}]VERDICT: {result.get('verdict', 'WATCH')}[/{verdict_colour}]",
            expand=True,
        ))
    else:
        console.print("\n[dim](Agent debate skipped — pass without --no-agent to enable)[/dim]")

    console.rule("[bold green]Analysis Complete[/bold green]")


if __name__ == "__main__":
    main()

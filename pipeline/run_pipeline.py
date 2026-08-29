"""
Main nightly pipeline — runs every weekday at 7:00 PM.

Stage 1  →  Build Universe        (S&P 500 + NASDAQ-100 + screener)
Stage 2  →  Hard Filter Gate      (Minervini Trend Template + liquidity)
Stage 3  →  Fundamental Analysis  (ROIC, FCF, EPS growth, insider, valuation)
Stage 4  →  Technical Analysis    (EMA, RSI, ATR, setup detection)
Stage 5  →  Composite Scoring     (cross-sectional percentile ranking)
Stage 6  →  Sentiment Analysis    (FinBERT on news for top 20 candidates)
Stage 7  →  Agent Debate          (Bull → Bear → Risk Manager for top 15)
Stage 8  →  Persist & Notify      (PostgreSQL + Telegram)
Stage 9  →  Resolve Paper Trades  (close any trade that hit target/stop)

Total runtime on a Mac M-series with FMP free tier: ~25–40 minutes.
Run it as:  python -m pipeline.run_pipeline
"""
import sys
import time
from datetime import date, datetime
from typing import Dict, List
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import track

# ── Engine Imports ─────────────────────────────────────────────────────────────
from config.settings import (
    MAX_AGENT_CANDIDATES, MAX_FINAL_PICKS, LOG_LEVEL, LOG_FILE
)
from src.data.yf_client import YFClient as FMPClient
from src.screener.universe import build_universe
from src.screener.filter_engine import apply_hard_filters
from src.factors.fundamental import FundamentalAnalyser
from src.factors.technical import TechnicalAnalyser
from src.factors.sentiment import SentimentAnalyser
from src.factors.composite_scorer import score_universe
from src.agents.orchestrator import AgentOrchestrator
from src.paper_trading.tracker import PaperTradeTracker
from src.models.database import (
    init_db, get_session, PipelineRun, Recommendation, Ticker
)
from src.notifications.telegram_bot import (
    send_pipeline_header, send_recommendation,
    send_error_alert, send_performance_update
)

console = Console()


# ── Logging Setup ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr,      level=LOG_LEVEL, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(LOG_FILE,        level="DEBUG",   rotation="7 days", retention="30 days")


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run(dry_run: bool = False) -> None:
    """
    Execute the full nightly pipeline.

    Parameters
    ----------
    dry_run : bool
        If True, skip Telegram delivery and DB writes (for local testing).
    """
    t_start = time.time()
    today   = date.today()
    console.rule(f"[bold cyan]Stock Engine — {today}[/bold cyan]")

    # ── Initialise DB (idempotent) ─────────────────────────────────────────────
    init_db()
    session = get_session()

    # Record this pipeline run
    run_record = PipelineRun(run_at=datetime.utcnow(), status="running")
    session.add(run_record)
    session.commit()

    client      = FMPClient()
    fund_engine = FundamentalAnalyser(client)
    tech_engine = TechnicalAnalyser()
    sent_engine = SentimentAnalyser()
    agent_orch  = AgentOrchestrator()
    paper_track = PaperTradeTracker(client)

    errors: List[str] = []

    try:
        # ── Stage 1: Universe ──────────────────────────────────────────────────
        console.print("\n[bold]Stage 1 / 9[/bold]  Building US equity universe …")
        universe = build_universe(client)
        console.print(f"  Universe: [green]{len(universe)} tickers[/green]")

        # ── Stage 2: Hard Filters ─────────────────────────────────────────────
        console.print("\n[bold]Stage 2 / 9[/bold]  Applying hard filter gates …")

        # Fetch screener data once (reused inside filter_engine)
        screener_data = client.get_stock_screener(limit=1000)

        # Process in batches to show progress
        batch_size = 50
        passed_filters: List[str] = []

        for i in track(range(0, len(universe), batch_size), description="Filtering …"):
            batch = universe[i: i + batch_size]
            passed = apply_hard_filters(batch, client, screener_data)
            passed_filters.extend(passed)

        console.print(
            f"  Passed SEPA + liquidity: [green]{len(passed_filters)} tickers[/green]"
        )

        if not passed_filters:
            raise RuntimeError("No stocks passed the hard filter gates. Market may be in a downtrend.")

        # ── Stage 3: Fundamental Analysis ─────────────────────────────────────
        console.print("\n[bold]Stage 3 / 9[/bold]  Running fundamental analysis …")
        fund_records: List[Dict] = []

        for sym in track(passed_filters, description="Fundamentals …"):
            try:
                rec = fund_engine.analyse(sym)
                fund_records.append(rec)
            except Exception as exc:
                errors.append(f"Fund {sym}: {exc}")

        console.print(f"  Fundamentals: [green]{len(fund_records)} stocks analysed[/green]")

        # ── Stage 4: Technical Analysis ────────────────────────────────────────
        console.print("\n[bold]Stage 4 / 9[/bold]  Computing technical indicators …")
        combined_records: List[Dict] = []

        for rec in track(fund_records, description="Technicals …"):
            sym = rec["symbol"]
            try:
                df = client.get_daily_ohlcv(sym, days=400)
                if df.empty:
                    continue
                tech = tech_engine.compute(df, symbol=sym)
                # Only keep stocks where an entry setup was detected
                if tech.get("entry_setup", "NONE") == "NONE":
                    continue
                merged = {**rec, **tech}
                combined_records.append(merged)
            except Exception as exc:
                errors.append(f"Tech {sym}: {exc}")

        console.print(
            f"  Technical setups detected: [green]{len(combined_records)} stocks[/green]"
        )

        if not combined_records:
            raise RuntimeError("No stocks have a valid technical entry setup tonight.")

        # ── Stage 5: Composite Scoring ────────────────────────────────────────
        console.print("\n[bold]Stage 5 / 9[/bold]  Cross-sectional composite scoring …")
        ranked_df = score_universe(combined_records)
        console.print(
            f"  Top composite score: "
            f"[bold yellow]{ranked_df['composite_score'].iloc[0]:.1f}[/bold yellow]"
            f" ({ranked_df['symbol'].iloc[0]})"
        )

        # Take top candidates for deeper analysis
        top_candidates = ranked_df.head(MAX_AGENT_CANDIDATES).to_dict("records")

        # ── Stage 6: Sentiment Analysis ────────────────────────────────────────
        console.print("\n[bold]Stage 6 / 9[/bold]  FinBERT news sentiment analysis …")

        for rec in track(top_candidates, description="Sentiment …"):
            sym = rec["symbol"]
            try:
                sent = sent_engine.analyse(sym)
                rec["sentiment_score"] = sent["sentiment_score"]
                rec["sentiment_label"] = sent["sentiment_label"]
                rec["top_headlines"]   = sent["top_headlines"]
            except Exception as exc:
                rec["sentiment_score"] = 0.0
                rec["sentiment_label"] = "Neutral"
                errors.append(f"Sentiment {sym}: {exc}")

        # ── Stage 7: Multi-Agent Debate ────────────────────────────────────────
        console.print("\n[bold]Stage 7 / 9[/bold]  Multi-agent Bull/Bear/Risk debate …")
        approved: List[Dict] = []
        watch:    List[Dict] = []

        for rec in track(top_candidates, description="Agent debate …"):
            result = agent_orch.run(rec)
            rec["bull_thesis"] = result.get("bull_thesis", "")
            rec["bear_risks"]  = result.get("bear_risks", "")
            rec["verdict"]     = result.get("verdict", "WATCH")
            rec["full_memo"]   = result.get("full_memo", "")
            if rec["verdict"] == "APPROVED":
                approved.append(rec)
            elif rec["verdict"] == "WATCH":
                watch.append(rec)

        final_picks = (approved + watch)[:MAX_FINAL_PICKS]
        console.print(
            f"  APPROVED: [green]{len(approved)}[/green]  |  "
            f"WATCH: [yellow]{len(watch)}[/yellow]  |  "
            f"REJECTED: [red]{len(top_candidates) - len(approved) - len(watch)}[/red]"
        )

        # ── Stage 8: Persist & Notify ─────────────────────────────────────────
        console.print("\n[bold]Stage 8 / 9[/bold]  Saving to database and sending alerts …")

        if not dry_run:
            send_pipeline_header(len(final_picks))

        for rank, rec in enumerate(final_picks, 1):
            sym = rec["symbol"]
            # ── Upsert Ticker ──────────────────────────────────────────────────
            try:
                ticker_row = session.query(Ticker).filter_by(symbol=sym).first()
                if not ticker_row:
                    profile = client.get_company_profile(sym)
                    ticker_row = Ticker(
                        symbol    = sym,
                        name      = profile.get("companyName", sym),
                        sector    = profile.get("sector", ""),
                        industry  = profile.get("industry", ""),
                        exchange  = profile.get("exchangeShortName", ""),
                        market_cap= profile.get("mktCap"),
                    )
                    session.add(ticker_row)
                    session.flush()
            except Exception as exc:
                logger.warning(f"Ticker upsert failed for {sym}: {exc}")

            # ── Save Recommendation ────────────────────────────────────────────
            try:
                db_rec = Recommendation(
                    run_id          = run_record.id,
                    symbol          = sym,
                    rec_date        = today,
                    entry_price     = rec.get("close"),
                    stop_loss       = rec.get("stop_loss"),
                    target_short    = rec.get("target_short"),
                    target_long     = rec.get("target_long"),
                    risk_reward     = rec.get("risk_reward"),
                    atr             = rec.get("atr"),
                    composite_score = rec.get("composite_score"),
                    sentiment_score = rec.get("sentiment_score"),
                    bull_thesis     = rec.get("bull_thesis"),
                    bear_risks      = rec.get("bear_risks"),
                    risk_verdict    = rec.get("verdict"),
                    full_memo       = rec.get("full_memo"),
                    horizon_short   = "1–4 weeks",
                    horizon_long    = "3–12 months",
                )
                session.add(db_rec)
                session.flush()
                rec["rec_db_id"] = db_rec.id
            except Exception as exc:
                logger.error(f"DB save failed for {sym}: {exc}")

            # ── Open Paper Trade ───────────────────────────────────────────────
            if not dry_run:
                paper_track.open_trade(rec, session=session)
                send_recommendation(rec, rank)

        session.commit()
        _print_results_table(final_picks, console)

        # ── Stage 9: Resolve Open Paper Trades ────────────────────────────────
        console.print("\n[bold]Stage 9 / 9[/bold]  Resolving open paper trades …")
        resolved = paper_track.resolve_open_trades(session)
        console.print(f"  Resolved: [green]{resolved}[/green] trades")

        # Send weekly performance report (every Friday)
        if date.today().weekday() == 4 and not dry_run:
            stats = paper_track.get_performance_stats(session)
            send_performance_update(stats)

        # ── Finalise Run Record ────────────────────────────────────────────────
        elapsed = time.time() - t_start
        run_record.universe_size  = len(universe)
        run_record.candidates_out = len(passed_filters)
        run_record.final_picks    = len(final_picks)
        run_record.duration_sec   = elapsed
        run_record.errors         = "; ".join(errors[:20]) if errors else None
        run_record.status         = "ok" if not errors else "partial"
        session.commit()

        console.rule(
            f"[bold green]Pipeline complete in {elapsed:.0f}s — "
            f"{len(final_picks)} picks delivered[/bold green]"
        )

    except Exception as exc:
        logger.error(f"Pipeline fatal error: {exc}")
        if not dry_run:
            send_error_alert("pipeline", str(exc))
        run_record.status = "failed"
        run_record.errors = str(exc)
        session.commit()
        raise
    finally:
        session.close()


def _print_results_table(picks: List[Dict], console: Console) -> None:
    """Pretty-print the final picks to the terminal."""
    table = Table(title="Tonight's Picks", show_lines=True)
    table.add_column("#",        style="dim", width=3)
    table.add_column("Symbol",   style="bold cyan", width=7)
    table.add_column("Score",    width=7)
    table.add_column("Setup",    width=16)
    table.add_column("Entry",    width=8)
    table.add_column("Stop",     width=8)
    table.add_column("T1",       width=8)
    table.add_column("R/R",      width=5)
    table.add_column("Verdict",  width=10)
    table.add_column("Sentiment",width=10)

    for i, rec in enumerate(picks, 1):
        verdict_fmt = {
            "APPROVED": "[green]APPROVED[/green]",
            "WATCH":    "[yellow]WATCH[/yellow]",
            "REJECTED": "[red]REJECTED[/red]",
        }.get(rec.get("verdict", ""), rec.get("verdict", ""))

        table.add_row(
            str(i),
            rec.get("symbol", ""),
            f"{rec.get('composite_score', 0):.1f}",
            rec.get("entry_setup", "N/A"),
            f"${rec.get('close', 0):.2f}",
            f"${rec.get('stop_loss', 0):.2f}" if rec.get("stop_loss") else "N/A",
            f"${rec.get('target_short', 0):.2f}" if rec.get("target_short") else "N/A",
            f"1:{rec.get('risk_reward', 0):.1f}" if rec.get("risk_reward") else "N/A",
            verdict_fmt,
            rec.get("sentiment_label", "Neutral"),
        )

    console.print(table)


# ── CLI Entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stock Engine Nightly Pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run pipeline without DB writes or Telegram delivery")
    args = parser.parse_args()
    run(dry_run=args.dry_run)

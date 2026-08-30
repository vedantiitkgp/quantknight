"""
QuantKnight Intraday Pipeline — 5× daily paper-trading engine.

Runs on GitHub Actions 5 times per US trading day:

  Mode        Time (ET)   cron (UTC)       Action
  ─────────────────────────────────────────────────────────────────
  morning     9:30 AM     30 13 * * 1-5    Full scan → open new positions
  midmorning  11:00 AM     0 15 * * 1-5    Secondary entries + price check
  midday      1:00 PM      0 17 * * 1-5    Trim losers, evaluate holds
  preclose    3:00 PM      0 19 * * 1-5    Auto-close ALL intraday shorts
  eod         4:30 PM     30 20 * * 1-5    Mark-to-market + compile reports

Usage:
  python -m pipeline.intraday_pipeline --mode morning
  python -m pipeline.intraday_pipeline --mode eod
"""
import argparse
import sys
import time
from datetime import date, datetime
from typing import Dict, List
from loguru import logger
from rich.console import Console
from rich.table import Table

from config.settings import (
    MAX_AGENT_CANDIDATES, MAX_FINAL_PICKS, LOG_LEVEL, LOG_FILE,
    IS_CI,
)
from src.data.yf_client import YFClient
from src.screener.universe import build_universe
from src.screener.filter_engine import apply_hard_filters
from src.factors.fundamental import FundamentalAnalyser
from src.factors.technical import TechnicalAnalyser
from src.factors.sentiment import SentimentAnalyser
from src.factors.composite_scorer import score_universe
from src.agents.orchestrator import AgentOrchestrator
from src.portfolio.manager import PortfolioManager
from src.portfolio.reporter import generate_daily, generate_weekly, generate_monthly
from src.portfolio.dashboard import generate_dashboard
from src.models.database import (
    init_db, get_session, PipelineRun, Recommendation, Ticker, DailyPnL,
)
from src.notifications.telegram_bot import (
    send_pipeline_header, send_recommendation,
    send_error_alert, send_performance_update,
)

console = Console()

# ── Logging ────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr, level=LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
)
logger.add(LOG_FILE, level="DEBUG", rotation="7 days", retention="30 days")


# ── Shared scan (Stages 1–7) ───────────────────────────────────────────────────

def _run_full_scan(client: YFClient, mode: str) -> List[Dict]:
    """
    Execute the full 7-stage stock-selection pipeline.
    Returns a list of recommendation dicts (max MAX_FINAL_PICKS).
    Includes APPROVED, WATCH, and REJECTED verdicts.
    """
    errors: List[str] = []

    # Stage 1: Universe
    console.print(f"\n[bold]Stage 1[/bold]  Building universe …")
    universe = build_universe(client)
    console.print(f"  {len(universe)} tickers")

    # Stage 2: Hard Filters
    console.print("\n[bold]Stage 2[/bold]  Applying SEPA / liquidity filters …")
    screener_data = client.get_stock_screener(limit=1000)
    passed_filters: List[str] = []
    batch_size = 50
    for i in range(0, len(universe), batch_size):
        batch = universe[i: i + batch_size]
        passed_filters.extend(apply_hard_filters(batch, client, screener_data))
    console.print(f"  {len(passed_filters)} passed filters")

    if not passed_filters:
        logger.warning("No stocks passed hard filter gates — market may be weak")
        return []

    # Stage 3: Fundamentals
    console.print("\n[bold]Stage 3[/bold]  Fundamental analysis …")
    fund_engine = FundamentalAnalyser(client)
    fund_records: List[Dict] = []
    for sym in passed_filters:
        try:
            fund_records.append(fund_engine.analyse(sym))
        except Exception as exc:
            errors.append(f"Fund {sym}: {exc}")
    console.print(f"  {len(fund_records)} fundamentals computed")

    # Stage 4: Technicals
    console.print("\n[bold]Stage 4[/bold]  Technical analysis …")
    tech_engine = TechnicalAnalyser()
    combined: List[Dict] = []
    for rec in fund_records:
        sym = rec["symbol"]
        try:
            df = client.get_daily_ohlcv(sym, days=400)
            if df.empty:
                continue
            tech = tech_engine.compute(df, symbol=sym)
            if tech.get("entry_setup", "NONE") == "NONE":
                continue
            combined.append({**rec, **tech})
        except Exception as exc:
            errors.append(f"Tech {sym}: {exc}")
    console.print(f"  {len(combined)} setups detected")

    if not combined:
        logger.warning("No technical setups found")
        return []

    # Stage 5: Composite Scoring
    console.print("\n[bold]Stage 5[/bold]  Composite scoring …")
    ranked_df   = score_universe(combined)
    top_cands   = ranked_df.head(MAX_AGENT_CANDIDATES).to_dict("records")

    # Stage 6: Sentiment
    console.print("\n[bold]Stage 6[/bold]  FinBERT sentiment …")
    sent_engine = SentimentAnalyser()
    for rec in top_cands:
        sym = rec["symbol"]
        try:
            sent = sent_engine.analyse(sym)
            rec["sentiment_score"] = sent["sentiment_score"]
            rec["sentiment_label"] = sent["sentiment_label"]
            rec["top_headlines"]   = sent["top_headlines"]
            # Forward full articles (headline + body snippet) so agents can
            # correlate specific news events with the fundamental data
            raw = sent.get("raw_articles", [])
            if raw:
                scored_raw = [a for a in raw if a.get("sentiment") is not None]
                scored_raw.sort(key=lambda x: abs(x["sentiment"]), reverse=True)
                rec["full_news"] = [
                    {
                        "headline":  a["headline"],
                        "summary":   (a.get("summary") or "")[:250].strip(),
                        "sentiment": round(a["sentiment"], 3),
                        "source":    a.get("source", ""),
                        "published": str(a.get("published", ""))[:10],
                    }
                    for a in scored_raw[:12]
                ]
        except Exception as exc:
            rec["sentiment_score"] = 0.0
            rec["sentiment_label"] = "Neutral"
            errors.append(f"Sent {sym}: {exc}")

    # Stage 7: Agent Debate
    console.print("\n[bold]Stage 7[/bold]  Multi-agent debate …")
    agent_orch = AgentOrchestrator()
    final_picks: List[Dict] = []
    approved_n = watch_n = rejected_n = 0
    for rec in top_cands:
        result = agent_orch.run(rec)
        rec.update({
            "bull_thesis": result.get("bull_thesis", ""),
            "bear_risks":  result.get("bear_risks", ""),
            "verdict":     result.get("verdict", "WATCH"),
            "full_memo":   result.get("full_memo", ""),
        })
        v = rec["verdict"]
        if v == "APPROVED":
            approved_n += 1
        elif v == "WATCH":
            watch_n += 1
        else:
            rejected_n += 1
        final_picks.append(rec)

    final_picks = final_picks[:MAX_FINAL_PICKS]
    console.print(
        f"  APPROVED [green]{approved_n}[/green]  "
        f"WATCH [yellow]{watch_n}[/yellow]  "
        f"REJECTED [red]{rejected_n}[/red]"
    )

    if errors:
        logger.warning(f"{len(errors)} non-fatal errors during scan")

    return final_picks


# ── Trade execution ────────────────────────────────────────────────────────────

def _enter_positions(pm: PortfolioManager, picks: List[Dict], mode: str,
                     session, run_id: int) -> None:
    """Open paper positions for APPROVED / WATCH / REJECTED picks."""
    for rec in picks:
        verdict = rec.get("verdict", "WATCH")
        position = pm.open_position(rec, mode=mode)
        if position is None:
            # Position already open — backfill thesis data if agents produced it
            if rec.get("bull_thesis") or rec.get("full_memo"):
                pm.update_position_analysis(
                    rec["symbol"],
                    rec.get("bull_thesis", ""),
                    rec.get("bear_risks", ""),
                    rec.get("full_memo", ""),
                    rec.get("full_news"),
                )
            continue

        # Persist to DB
        sym = rec["symbol"]
        try:
            if not session.query(Ticker).filter_by(symbol=sym).first():
                session.add(Ticker(symbol=sym, name=sym))
                session.flush()

            db_rec = Recommendation(
                run_id          = run_id,
                symbol          = sym,
                rec_date        = date.today(),
                entry_price     = rec.get("close"),
                stop_loss       = rec.get("stop_loss"),
                target_short    = rec.get("target_short"),
                atr             = rec.get("atr"),
                composite_score = rec.get("composite_score"),
                sentiment_score = rec.get("sentiment_score", 0),
                bull_thesis     = rec.get("bull_thesis", ""),
                bear_risks      = rec.get("bear_risks", ""),
                risk_verdict    = verdict,
                full_memo       = rec.get("full_memo", ""),
                horizon_short   = "intraday" if verdict == "REJECTED" else "1–4 weeks",
                horizon_long    = "intraday" if verdict == "REJECTED" else "3–12 months",
            )
            session.add(db_rec)
            session.flush()

            # Send Telegram alert
            try:
                send_recommendation(rec, 0)
            except Exception:
                pass

        except Exception as exc:
            logger.warning(f"DB entry for {sym} failed: {exc}")

    session.commit()


def _print_portfolio_table(pm: PortfolioManager) -> None:
    s   = pm.summary()
    tbl = Table(title="QuantKnight Portfolio", show_header=True)
    tbl.add_column("Metric")
    tbl.add_column("Value", justify="right")

    sign = "+" if s["cumulative_pnl"] >= 0 else ""
    tbl.add_row("Total Equity",     f"${s['equity']:,.0f}")
    tbl.add_row("Cash",             f"${s['cash']:,.0f}")
    tbl.add_row("Cumulative P&L",   f"{sign}${s['cumulative_pnl']:,.0f}")
    tbl.add_row("Open Positions",   str(s["open_positions"]))
    tbl.add_row("Today Entries",    str(s["today_entries"]))
    tbl.add_row("Today Exits",      str(s["today_exits"]))
    tbl.add_row("Realized Today",   f"${s['today_realized']:,.0f}")
    console.print(tbl)


def _persist_daily_pnl(pm: PortfolioManager, session) -> None:
    """Write/upsert today's P&L row to the daily_pnl table."""
    from sqlalchemy import text
    today   = date.today()
    port    = pm.portfolio
    trades  = pm.today_trades
    real    = trades.get("realized_pnl", 0.0)
    unreal  = trades.get("unrealized_pnl", 0.0)

    # Delete existing row for today then re-insert (upsert)
    try:
        session.query(DailyPnL).filter_by(trade_date=today).delete()
        row = DailyPnL(
            trade_date      = today,
            realized_pnl    = round(real, 2),
            unrealized_pnl  = round(unreal, 2),
            total_pnl       = round(real + unreal, 2),
            cumulative_pnl  = port.get("cumulative_pnl", 0),
            cash            = port.get("cash", 0),
            equity          = port.get("total_equity", 0),
            positions_count = len(port.get("positions", [])),
            trades_entered  = len(trades.get("entries", [])),
            trades_exited   = len(trades.get("exits", [])),
        )
        session.add(row)
        session.commit()
    except Exception as exc:
        logger.warning(f"daily_pnl persist failed: {exc}")
        session.rollback()


# ── Mode handlers ─────────────────────────────────────────────────────────────

def _mode_scan(client: YFClient, pm: PortfolioManager, mode: str,
               session, run_id: int) -> None:
    """morning / midmorning / midday — scan + enter new positions."""
    picks = _run_full_scan(client, mode)
    if picks:
        _enter_positions(pm, picks, mode, session, run_id)
    # Resolve stops/targets for existing swing positions after new entries
    closed = pm.resolve_stops_and_targets(client)
    if closed:
        logger.info(f"  Resolved {len(closed)} stops/targets")


def _mode_preclose(client: YFClient, pm: PortfolioManager) -> None:
    """Close all intraday (SHORT) positions before market close."""
    console.print("\n[bold cyan]PRECLOSE[/bold cyan]  Closing all intraday shorts …")
    closed = pm.close_all_intraday(client)
    console.print(f"  Closed {len(closed)} intraday positions")
    # Also check swing stop/targets
    closed_sw = pm.resolve_stops_and_targets(client)
    if closed_sw:
        logger.info(f"  Resolved {len(closed_sw)} swing stops/targets")


def _mode_eod(client: YFClient, pm: PortfolioManager, session) -> None:
    """End-of-day: mark to market, generate reports, persist P&L."""
    console.print("\n[bold cyan]EOD[/bold cyan]  Marking to market …")
    unrealized = pm.mark_to_market(client)
    console.print(f"  Unrealized P&L: ${unrealized:+,.0f}")

    _persist_daily_pnl(pm, session)

    console.print("  Generating reports …")
    generate_daily(pm.portfolio, pm.today_trades)

    today = date.today()
    # Weekly report every Friday
    if today.weekday() == 4:
        generate_weekly(pm.portfolio)
    # Monthly report on last calendar day or last trading-day-of-month heuristic
    import calendar
    last_day = calendar.monthrange(today.year, today.month)[1]
    if today.day == last_day or (today.weekday() == 4 and today.day >= last_day - 6):
        generate_monthly(pm.portfolio)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(mode: str) -> None:
    """
    Execute the intraday pipeline for the given mode.

    Parameters
    ----------
    mode : str
        One of: morning | midmorning | midday | preclose | eod
    """
    t_start = time.time()
    today   = date.today()
    console.rule(f"[bold cyan]QuantKnight — {mode.upper()} — {today}[/bold cyan]")

    init_db()
    session = get_session()
    client  = YFClient()
    pm      = PortfolioManager()

    # Record run
    run_record = PipelineRun(
        run_at   = datetime.utcnow(),
        run_mode = mode,
        status   = "running",
    )
    session.add(run_record)
    session.commit()

    try:
        if mode in ("morning", "midmorning", "midday"):
            _mode_scan(client, pm, mode, session, run_record.id)

        elif mode == "preclose":
            _mode_preclose(client, pm)

        elif mode == "eod":
            _mode_eod(client, pm, session)

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        # Print portfolio state after every run
        _print_portfolio_table(pm)

        # Regenerate dashboard after every run so it reflects live state
        try:
            path = generate_dashboard(pm.portfolio, pm.today_trades)
            console.print(f"  Dashboard updated → {path}")
        except Exception as exc:
            logger.warning(f"Dashboard generation failed: {exc}")

        elapsed = time.time() - t_start
        run_record.status       = "ok"
        run_record.duration_sec = round(elapsed, 1)
        session.commit()

        console.print(
            f"\n[bold green]✓ {mode.upper()} complete[/bold green] "
            f"in {elapsed:.0f}s"
        )

    except Exception as exc:
        logger.exception(f"Intraday pipeline {mode} failed: {exc}")
        run_record.status = "error"
        run_record.errors = str(exc)
        session.commit()
        try:
            send_error_alert(f"[{mode}] {exc}")
        except Exception:
            pass
        sys.exit(1)

    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantKnight Intraday Pipeline")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["morning", "midmorning", "midday", "preclose", "eod"],
        help="Pipeline mode to run",
    )
    args = parser.parse_args()
    run(args.mode)

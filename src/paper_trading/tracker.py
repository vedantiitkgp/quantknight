"""
Paper Trading Tracker.

Every nightly recommendation is logged as an open paper trade.
A daily resolve job checks if price hit the stop-loss or short-term target.

This provides an ongoing, statistically rigorous accuracy audit of the engine
without risking any real capital.

Usage:
  tracker = PaperTradeTracker()
  tracker.open_trade(rec)          # call after each recommendation
  tracker.resolve_open_trades()    # call at end of each pipeline run
  stats = tracker.get_performance_stats()
"""
from datetime import date
from typing import Dict, List, Optional
from loguru import logger
from sqlalchemy.orm import Session

from src.models.database import PaperTrade, PerformanceStats, Recommendation, get_session
from src.backtest.metrics import compute_trade_metrics
from src.data.yf_client import YFClient as FMPClient


class PaperTradeTracker:
    def __init__(self, client: Optional[FMPClient] = None):
        self.client = client or FMPClient()

    def open_trade(self, rec: Dict, session: Optional[Session] = None) -> None:
        """
        Log a new recommendation as an open paper trade.

        rec must contain: symbol, entry_price, stop_loss, target_short
        """
        close_session = False
        if session is None:
            session = get_session()
            close_session = True

        try:
            pt = PaperTrade(
                symbol      = rec["symbol"],
                rec_id      = rec.get("rec_db_id"),
                open_date   = date.today(),
                entry_price = rec.get("entry_price") or rec.get("close"),
                stop_loss   = rec.get("stop_loss"),
                target_short= rec.get("target_short"),
                status      = "OPEN",
            )
            session.add(pt)
            session.commit()
            logger.debug(f"Paper trade opened: {rec['symbol']} @ {pt.entry_price:.2f}")
        except Exception as exc:
            logger.error(f"Failed to open paper trade for {rec.get('symbol')}: {exc}")
            session.rollback()
        finally:
            if close_session:
                session.close()

    def resolve_open_trades(self, session: Optional[Session] = None) -> int:
        """
        Fetch latest prices and close any trade that has hit its target or stop.
        Returns the number of trades resolved.
        """
        close_session = False
        if session is None:
            session = get_session()
            close_session = True

        resolved = 0
        try:
            open_trades: List[PaperTrade] = (
                session.query(PaperTrade)
                .filter(PaperTrade.status == "OPEN")
                .all()
            )

            if not open_trades:
                logger.info("No open paper trades to resolve.")
                return 0

            # Group by symbol to minimise API calls
            symbols = list({t.symbol for t in open_trades})
            latest_prices: Dict[str, float] = {}

            for sym in symbols:
                try:
                    df = self.client.get_daily_ohlcv(sym, days=5)
                    if not df.empty:
                        latest_prices[sym] = float(df.iloc[-1]["Close"])
                except Exception as exc:
                    logger.debug(f"Price fetch failed for {sym}: {exc}")

            # Resolve
            for trade in open_trades:
                current = latest_prices.get(trade.symbol)
                if current is None:
                    continue

                hit_stop   = trade.stop_loss   is not None and current <= trade.stop_loss
                hit_target = trade.target_short is not None and current >= trade.target_short

                # Auto-expire after 90 calendar days if neither is hit
                days_open  = (date.today() - trade.open_date).days
                expired    = days_open > 90

                if hit_stop or hit_target or expired:
                    trade.close_date  = date.today()
                    trade.exit_price  = current
                    entry             = trade.entry_price or 1.0
                    trade.pnl_pct     = ((current - entry) / entry) * 100
                    trade.hit_target  = hit_target
                    trade.hit_stop    = hit_stop
                    trade.status      = "WIN" if hit_target else ("LOSS" if hit_stop else "EXPIRED")
                    resolved += 1

            session.commit()
            logger.info(f"Resolved {resolved} paper trades.")
        except Exception as exc:
            logger.error(f"Error resolving paper trades: {exc}")
            session.rollback()
        finally:
            if close_session:
                session.close()

        return resolved

    def get_performance_stats(self, session: Optional[Session] = None) -> Dict:
        """
        Calculate and return live accuracy/performance metrics from all
        closed paper trades.
        """
        close_session = False
        if session is None:
            session = get_session()
            close_session = True

        try:
            closed = (
                session.query(PaperTrade)
                .filter(PaperTrade.status.in_(["WIN", "LOSS", "EXPIRED"]))
                .all()
            )
            open_count = (
                session.query(PaperTrade)
                .filter(PaperTrade.status == "OPEN")
                .count()
            )

            trades = [
                {
                    "pnl_pct":    t.pnl_pct or 0.0,
                    "hit_target": t.hit_target,
                    "hit_stop":   t.hit_stop,
                    "status":     t.status,
                }
                for t in closed
            ]

            stats = compute_trade_metrics(trades) if trades else {}
            stats["open_trades"] = open_count

            # Save snapshot to DB
            if trades:
                snap = PerformanceStats(
                    total_trades  = stats.get("total_trades", 0),
                    open_trades   = open_count,
                    win_rate      = stats.get("win_rate_pct"),
                    avg_win_pct   = stats.get("avg_win_pct"),
                    avg_loss_pct  = stats.get("avg_loss_pct"),
                    profit_factor = stats.get("profit_factor"),
                    expectancy    = stats.get("expectancy_pct"),
                )
                session.add(snap)
                session.commit()

            return stats
        except Exception as exc:
            logger.error(f"Performance stats error: {exc}")
            return {}
        finally:
            if close_session:
                session.close()

    def print_report(self) -> None:
        stats = self.get_performance_stats()
        print("\n" + "=" * 55)
        print("  PAPER TRADING PERFORMANCE AUDIT")
        print("=" * 55)
        for k, v in stats.items():
            label = k.replace("_", " ").title()
            print(f"  {label:<35} {v}")
        print("=" * 55 + "\n")

"""
QuantKnight Portfolio Manager — $150k paper-trading budget tracker.

Manages open positions, sizes new trades by ATR-based risk, records P&L,
and persists state to data/portfolio.json (committed to GitHub after each run).

Position sizing:
  risk_amount = equity × RISK_PCT_PER_TRADE   (default 1%)
  shares      = risk_amount / ATR              (1 ATR = max risk per share)
  capped at   MAX_POSITION_PCT of equity       (default 5%)

Direction:
  LONG  — APPROVED or WATCH verdicts
  SHORT — REJECTED verdict; intraday only (auto-closed at preclose run)
"""
import json
import os
from datetime import date, datetime
from typing import Dict, List, Optional
from loguru import logger

from config.settings import (
    TOTAL_CAPITAL, RISK_PCT_PER_TRADE, MAX_POSITION_PCT,
)

_PORTFOLIO_PATH = "data/portfolio.json"
_TRADES_DIR     = "data/trades"


# ── Portfolio state helpers ───────────────────────────────────────────────────

def _default_portfolio() -> Dict:
    return {
        "total_capital":   TOTAL_CAPITAL,
        "cash":            TOTAL_CAPITAL,
        "total_equity":    TOTAL_CAPITAL,
        "cumulative_pnl":  0.0,
        "as_of":           datetime.utcnow().isoformat(),
        "positions":       [],
    }


def load_portfolio() -> Dict:
    """Load portfolio state from disk; return default $150k state if not found."""
    os.makedirs("data", exist_ok=True)
    if os.path.exists(_PORTFOLIO_PATH):
        try:
            with open(_PORTFOLIO_PATH) as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"Could not load portfolio.json: {exc} — starting fresh")
    p = _default_portfolio()
    save_portfolio(p)
    return p


def save_portfolio(portfolio: Dict) -> None:
    """Persist portfolio state to disk."""
    os.makedirs("data", exist_ok=True)
    portfolio["as_of"] = datetime.utcnow().isoformat()
    with open(_PORTFOLIO_PATH, "w") as f:
        json.dump(portfolio, f, indent=2, default=str)


def load_today_trades() -> Dict:
    """Load or create today's trade log."""
    os.makedirs(_TRADES_DIR, exist_ok=True)
    today = str(date.today())
    path  = f"{_TRADES_DIR}/{today}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "date":           today,
        "runs":           [],
        "entries":        [],
        "exits":          [],
        "realized_pnl":   0.0,
        "unrealized_pnl": 0.0,
        "daily_pnl":      0.0,
        "cumulative_pnl": 0.0,
    }


def save_today_trades(trades: Dict) -> None:
    os.makedirs(_TRADES_DIR, exist_ok=True)
    today = str(date.today())
    with open(f"{_TRADES_DIR}/{today}.json", "w") as f:
        json.dump(trades, f, indent=2, default=str)


# ── PortfolioManager ──────────────────────────────────────────────────────────

class PortfolioManager:
    """
    Stateful wrapper around portfolio.json.
    Each method loads the latest state, applies changes, and saves.
    """

    def __init__(self):
        self.portfolio = load_portfolio()
        self.today_trades = load_today_trades()

    # ── Sizing ────────────────────────────────────────────────────────────────

    def get_position_size(self, atr: float, verdict: str = "APPROVED") -> int:
        """
        Return number of shares to buy/short.

        Risk budget:
          base risk_amount = equity × 1%
          shares = risk_amount / ATR
          capped at: equity × 5% / entry_price   (applied by caller)
          WATCH verdicts get 60% of standard size.
        """
        if atr is None or atr <= 0:
            return 0
        equity      = self.portfolio["total_equity"]
        risk_amount = equity * RISK_PCT_PER_TRADE
        if verdict == "WATCH":
            risk_amount *= 0.6
        shares = int(risk_amount / atr)
        return max(shares, 1)

    def get_max_shares(self, price: float) -> int:
        """Hard cap: max 5% of equity in one position."""
        if not price or price <= 0:
            return 0
        equity = self.portfolio["total_equity"]
        return int((equity * MAX_POSITION_PCT) / price)

    def has_position(self, symbol: str) -> bool:
        return any(p["symbol"] == symbol for p in self.portfolio["positions"])

    def available_cash(self) -> float:
        return self.portfolio["cash"]

    # ── Open position ─────────────────────────────────────────────────────────

    def open_position(
        self,
        rec: Dict,
        mode: str = "morning",
    ) -> Optional[Dict]:
        """
        Open a new paper position based on a recommendation dict.
        Returns the position dict if opened, None if skipped.

        rec keys used: symbol, close (entry price), stop_loss, target_short,
                       atr, verdict, composite_score, entry_setup, bull_thesis
        """
        symbol    = rec.get("symbol", "")
        verdict   = rec.get("verdict", "WATCH")
        price     = rec.get("close") or rec.get("entry_price")
        stop      = rec.get("stop_loss")
        target    = rec.get("target_short")
        atr       = rec.get("atr")

        if not price or not atr:
            logger.debug(f"Skipping {symbol}: missing price or ATR")
            return None

        if self.has_position(symbol):
            logger.debug(f"Skipping {symbol}: already have open position")
            return None

        direction = "SHORT" if verdict == "REJECTED" else "LONG"
        trade_type = "intraday" if verdict == "REJECTED" else "swing"

        # For shorts: flip stop/target
        if direction == "SHORT":
            stop   = price + (atr * 2)   # stop above entry for shorts
            target = price - (atr * 3)   # target below entry

        shares = min(
            self.get_position_size(atr, verdict),
            self.get_max_shares(price)
        )
        if shares == 0:
            return None

        cost = shares * price
        if cost > self.portfolio["cash"]:
            shares = int(self.portfolio["cash"] / price)
            cost   = shares * price
            if shares == 0:
                logger.warning(f"Insufficient cash for {symbol} — skipping")
                return None

        now = datetime.utcnow()
        position = {
            "symbol":          symbol,
            "direction":       direction,
            "trade_type":      trade_type,
            "shares":          shares,
            "entry_price":     round(price, 4),
            "stop_loss":       round(stop, 4) if stop else None,
            "target":          round(target, 4) if target else None,
            "entry_date":      str(date.today()),
            "entry_time":      now.strftime("%H:%M"),
            "composite_score": rec.get("composite_score"),
            "verdict":         verdict,
            "reason": (
                f"{rec.get('entry_setup','?')} setup, "
                f"score {rec.get('composite_score', 0):.1f}, "
                f"RSI {rec.get('rsi', 0):.0f}, "
                f"EPS growth {rec.get('eps_growth_yoy', 0):.0f}%"
            ),
        }

        self.portfolio["positions"].append(position)
        self.portfolio["cash"] = round(self.portfolio["cash"] - cost, 2)

        self.today_trades["entries"].append({**position, "cost": round(cost, 2)})
        self.today_trades["runs"].append(mode)

        logger.info(
            f"  OPENED {direction} {symbol}: {shares} shares @ ${price:.2f} "
            f"| stop ${stop:.2f} | target ${target:.2f} | cost ${cost:,.0f}"
        )
        self._save()
        return position

    # ── Close position ────────────────────────────────────────────────────────

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "target/stop",
    ) -> Optional[Dict]:
        """
        Close an open position and record P&L.
        Returns exit summary dict or None if not found.
        """
        pos = next((p for p in self.portfolio["positions"] if p["symbol"] == symbol), None)
        if pos is None:
            return None

        shares     = pos["shares"]
        entry      = pos["entry_price"]
        direction  = pos["direction"]

        if direction == "LONG":
            pnl_dollars = (exit_price - entry) * shares
            pnl_pct     = (exit_price - entry) / entry * 100
        else:  # SHORT
            pnl_dollars = (entry - exit_price) * shares
            pnl_pct     = (entry - exit_price) / entry * 100

        proceeds = shares * exit_price

        self.portfolio["positions"] = [
            p for p in self.portfolio["positions"] if p["symbol"] != symbol
        ]
        self.portfolio["cash"]          = round(self.portfolio["cash"] + proceeds, 2)
        self.portfolio["cumulative_pnl"] = round(
            self.portfolio["cumulative_pnl"] + pnl_dollars, 2
        )

        exit_rec = {
            "symbol":      symbol,
            "direction":   direction,
            "shares":      shares,
            "entry_price": entry,
            "exit_price":  round(exit_price, 4),
            "pnl_dollars": round(pnl_dollars, 2),
            "pnl_pct":     round(pnl_pct, 2),
            "reason":      reason,
            "close_date":  str(date.today()),
        }
        self.today_trades["exits"].append(exit_rec)
        self.today_trades["realized_pnl"] = round(
            self.today_trades.get("realized_pnl", 0) + pnl_dollars, 2
        )

        sign = "+" if pnl_dollars >= 0 else ""
        logger.info(
            f"  CLOSED {direction} {symbol}: {shares} shares @ ${exit_price:.2f} "
            f"| P&L {sign}${pnl_dollars:,.0f} ({sign}{pnl_pct:.1f}%) — {reason}"
        )
        self._save()
        return exit_rec

    # ── Close all intraday shorts ─────────────────────────────────────────────

    def close_all_intraday(self, client) -> List[Dict]:
        """
        Fetch current prices and close every intraday SHORT position.
        Called at preclose (3 PM ET).
        """
        closed = []
        intraday = [p for p in self.portfolio["positions"] if p.get("trade_type") == "intraday"]
        if not intraday:
            logger.info("No intraday positions to close.")
            return []

        for pos in intraday:
            sym = pos["symbol"]
            try:
                df = client.get_daily_ohlcv(sym, days=5)
                current = float(df.iloc[-1]["Close"]) if not df.empty else pos["entry_price"]
            except Exception:
                current = pos["entry_price"]  # flat exit if no price

            rec = self.close_position(sym, current, reason="preclose auto-exit")
            if rec:
                closed.append(rec)

        return closed

    # ── Mark-to-market ────────────────────────────────────────────────────────

    def mark_to_market(self, client) -> float:
        """
        Fetch latest prices for all open positions, compute unrealized P&L,
        and update total_equity. Returns total unrealized P&L.
        """
        unrealized = 0.0
        invested   = 0.0

        for pos in self.portfolio["positions"]:
            sym     = pos["symbol"]
            shares  = pos["shares"]
            entry   = pos["entry_price"]
            try:
                df      = client.get_daily_ohlcv(sym, days=5)
                current = float(df.iloc[-1]["Close"]) if not df.empty else entry
            except Exception:
                current = entry

            if pos["direction"] == "LONG":
                unreal = (current - entry) * shares
            else:
                unreal = (entry - current) * shares

            unrealized += unreal
            invested   += shares * current

        self.today_trades["unrealized_pnl"] = round(unrealized, 2)
        self.portfolio["total_equity"] = round(
            self.portfolio["cash"] + invested, 2
        )
        self.portfolio["cumulative_pnl"] = round(
            self.portfolio["total_equity"] - TOTAL_CAPITAL, 2
        )
        self._save()
        return unrealized

    # ── Resolve stops & targets ───────────────────────────────────────────────

    def resolve_stops_and_targets(self, client) -> List[Dict]:
        """
        Check each swing position against stop/target; auto-close if hit.
        Called at every run.
        """
        closed = []
        swing = [p for p in self.portfolio["positions"] if p.get("trade_type") != "intraday"]

        for pos in swing:
            sym  = pos["symbol"]
            try:
                df   = client.get_daily_ohlcv(sym, days=5)
                if df.empty:
                    continue
                current = float(df.iloc[-1]["Close"])
            except Exception:
                continue

            stop   = pos.get("stop_loss")
            target = pos.get("target")

            if pos["direction"] == "LONG":
                hit_stop   = stop   is not None and current <= stop
                hit_target = target is not None and current >= target
            else:
                hit_stop   = stop   is not None and current >= stop
                hit_target = target is not None and current <= target

            if hit_target:
                rec = self.close_position(sym, current, reason="target hit")
                if rec:
                    closed.append(rec)
            elif hit_stop:
                rec = self.close_position(sym, current, reason="stop-loss hit")
                if rec:
                    closed.append(rec)

        return closed

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> Dict:
        p = self.portfolio
        return {
            "equity":         p["total_equity"],
            "cash":           p["cash"],
            "cumulative_pnl": p["cumulative_pnl"],
            "open_positions": len(p["positions"]),
            "today_entries":  len(self.today_trades.get("entries", [])),
            "today_exits":    len(self.today_trades.get("exits", [])),
            "today_realized": self.today_trades.get("realized_pnl", 0.0),
        }

    def _save(self):
        save_portfolio(self.portfolio)
        save_today_trades(self.today_trades)

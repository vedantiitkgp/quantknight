"""
Telegram notification delivery.

Formats and sends the nightly recommendation memo to your Telegram chat.
Messages are split if they exceed Telegram's 4096-character limit.

Setup:
  1. Message @BotFather on Telegram → /newbot → get your BOT_TOKEN
  2. Message @userinfobot to get your CHAT_ID
  3. Add both to .env
"""
import requests
from datetime import date
from typing import Dict, List
from loguru import logger

from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

_TELEGRAM_MAX_LEN = 4000   # safe limit (Telegram caps at 4096)


def _send(text: str, parse_mode: str = "HTML") -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured — skipping notification.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": parse_mode,
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        return True
    except Exception as exc:
        logger.error(f"Telegram send failed: {exc}")
        return False


def _split_message(text: str, max_len: int = _TELEGRAM_MAX_LEN) -> List[str]:
    """Split long messages at paragraph boundaries."""
    if len(text) <= max_len:
        return [text]
    chunks, current = [], ""
    for para in text.split("\n\n"):
        if len(current) + len(para) + 2 <= max_len:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks


def send_pipeline_header(n_picks: int) -> None:
    """Send the opening header for tonight's scan."""
    msg = (
        f"<b>📊 Stock Engine — {date.today()}</b>\n\n"
        f"Evening pipeline complete.\n"
        f"<b>{n_picks}</b> high-conviction picks identified.\n"
        f"Full analysis follows below…\n"
        f"{'─' * 35}"
    )
    _send(msg)


def send_recommendation(rec: Dict, rank: int) -> None:
    """
    Format and send a single stock recommendation.

    rec fields used:
      symbol, close, entry_setup, composite_score, sentiment_label,
      sentiment_score, verdict, stop_loss, target_short, target_long,
      risk_reward, atr, full_memo
    """
    sym      = rec.get("symbol", "")
    verdict  = rec.get("verdict", "WATCH")
    score    = rec.get("composite_score", 0)
    setup    = rec.get("entry_setup", "N/A")
    close    = rec.get("close", 0)
    stop     = rec.get("stop_loss")
    t_short  = rec.get("target_short")
    t_long   = rec.get("target_long")
    rr       = rec.get("risk_reward")
    senti    = rec.get("sentiment_label", "Neutral")
    senti_sc = rec.get("sentiment_score", 0)

    verdict_emoji = {"APPROVED": "✅", "WATCH": "👀", "REJECTED": "❌"}.get(verdict, "")

    header = (
        f"{verdict_emoji} <b>#{rank}  {sym}</b>  |  Score: {score:.1f}/100\n"
        f"Setup: <code>{setup}</code>  |  Sentiment: {senti} ({senti_sc:+.2f})\n\n"
        f"<b>Price Levels</b>\n"
        f"  Current Price : ${close:.2f}\n"
        f"  Entry Zone    : ${close:.2f} (market close)\n"
        f"  Stop-Loss     : ${stop:.2f}\n"
        f"  Short Target  : ${t_short:.2f}  ({_pct(close, t_short):+.1f}%)\n"
        f"  Long Target   : ${t_long:.2f}  ({_pct(close, t_long):+.1f}%)\n"
        f"  Risk/Reward   : 1 : {rr:.1f}\n\n"
        f"<b>Full Analysis</b>\n"
    ) if all(v is not None for v in [stop, t_short, t_long, rr]) else (
        f"{verdict_emoji} <b>#{rank}  {sym}</b>  |  Score: {score:.1f}/100\n"
        f"Setup: <code>{setup}</code>  |  Sentiment: {senti}\n"
        f"Current Price: ${close:.2f}\n\n"
        f"<b>Full Analysis</b>\n"
    )

    memo = rec.get("full_memo", "No agent memo generated.")
    full_msg = header + memo

    # Send in chunks if needed
    chunks = _split_message(full_msg)
    for i, chunk in enumerate(chunks):
        if i > 0:
            chunk = f"<b>{sym}</b> (continued)\n\n" + chunk
        _send(chunk)

    logger.info(f"Telegram notification sent for {sym}")


def send_performance_update(stats: Dict) -> None:
    """Send weekly paper-trading performance stats."""
    lines = [
        f"<b>📈 Paper Trading Accuracy Report</b>\n",
        f"Total Trades:    {stats.get('total_trades', 0)}",
        f"Open Trades:     {stats.get('open_trades', 0)}",
        f"Win Rate:        {stats.get('win_rate_pct', 0):.1f}%",
        f"Avg Win:         +{stats.get('avg_win_pct', 0):.1f}%",
        f"Avg Loss:        {stats.get('avg_loss_pct', 0):.1f}%",
        f"Profit Factor:   {stats.get('profit_factor', 0):.2f}",
        f"Expectancy:      {stats.get('expectancy_pct', 0):+.2f}% per trade",
    ]
    _send("\n".join(lines))


def send_error_alert(stage: str, error: str) -> None:
    """Alert on pipeline failure."""
    _send(
        f"⚠️ <b>Pipeline Error — {stage}</b>\n\n"
        f"<code>{error[:400]}</code>"
    )


def _pct(base: float, target: float) -> float:
    if base and base != 0:
        return ((target - base) / base) * 100
    return 0.0

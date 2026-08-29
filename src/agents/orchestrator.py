"""
Multi-Agent Debate Orchestrator.

Simulates a real institutional investment committee:

  1. BULL ANALYST   — builds the strongest upside case
  2. BEAR ANALYST   — attacks the thesis, identifies risks
  3. RISK MANAGER   — adjudicates, enforces R/R discipline, issues verdict

The LLM is given ONLY pre-computed numerical data (no internet access,
no hallucination of prices or metrics).  Every number in the final memo
comes directly from the quantitative engine.

Supports both Anthropic Claude and OpenAI (configurable via settings).
"""
import json
from datetime import date
from typing import Dict, Optional
from loguru import logger

from config.settings import (
    LLM_PROVIDER,
    CLAUDE_MODEL_BULL, CLAUDE_MODEL_BEAR, CLAUDE_MODEL_RISK,
    OPENAI_MODEL,
    ANTHROPIC_API_KEY, ANTHROPIC_WORKSPACE_ID, OPENAI_API_KEY,
)


# ── LLM Wrappers ──────────────────────────────────────────────────────────────

def _call_anthropic(model: str, system: str, user: str, max_tokens: int = 1500) -> str:
    import anthropic
    extra_headers = {}
    if ANTHROPIC_WORKSPACE_ID:
        extra_headers["anthropic-workspace-id"] = ANTHROPIC_WORKSPACE_ID
    client = anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY,
        default_headers=extra_headers if extra_headers else None,
    )
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text.strip()


def _call_openai(system: str, user: str, max_tokens: int = 1500) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def _llm(system: str, user: str, model: str = CLAUDE_MODEL_RISK, max_tokens: int = 1500) -> str:
    """
    Route to the correct model.  Each agent passes its own model constant
    so Haiku handles formatting tasks and Sonnet handles synthesis.
    """
    if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
        return _call_anthropic(model, system, user, max_tokens)
    elif OPENAI_API_KEY:
        return _call_openai(system, user, max_tokens)
    else:
        raise RuntimeError("No LLM API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")


# ── Prompt Templates ──────────────────────────────────────────────────────────

_SYSTEM_STRICT = """
You are a quantitative analyst at a tier-1 institutional investment firm.
You have been given a structured JSON data packet containing ONLY pre-computed,
verified financial metrics for a stock.

STRICT RULES — YOU MUST FOLLOW THESE WITHOUT EXCEPTION:
1. You MUST only use the numbers present in the JSON data provided to you.
2. You MUST NOT invent, guess, or extrapolate any financial figure.
3. If a metric is missing (null), acknowledge it as "data unavailable" — do NOT substitute a value.
4. You MUST cite specific numbers from the JSON in every paragraph.
5. Write in the style of a professional equity research note — clear, concise, evidence-based.
6. Do NOT use vague phrases like "the stock could potentially" without citing data evidence.
"""

_BULL_PROMPT = """
You are the BULL ANALYST. Your job is to make the strongest INVESTMENT CASE for {symbol}.

STOCK DATA PACKET:
{data_json}

Write a structured BULL CASE covering:
1. **Fundamental Strength** (cite ROIC, FCF yield, EPS growth from the data)
2. **Technical Setup** (cite exact EMA alignment, RSI, entry setup, ATR-based entry zone)
3. **Momentum & Catalyst** (cite price momentum, earnings surprise, analyst targets if present)
4. **Short-Term Entry Thesis** (cite entry price, stop-loss, and short-term target from data)
5. **Long-Term Thesis** (cite long-term target and fundamental justification)

Keep each section to 2–3 sentences. Use professional financial language.
"""

_BEAR_PROMPT = """
You are the BEAR ANALYST. Your job is to identify ALL material RISKS for {symbol}.

STOCK DATA PACKET:
{data_json}

Write a structured BEAR CASE covering:
1. **Valuation Risk** (cite PEG, EV/EBITDA, P/FCF from the data — flag if elevated)
2. **Fundamental Weakness** (cite any negative: high debt, weak FCF, slowing EPS, etc.)
3. **Technical Risk** (cite RSI if overbought, poor EMA alignment, or low volume)
4. **Downside Scenario** (cite the stop-loss price from data; what triggers it?)
5. **Macro / Sector Risk** (flag any relevant concern from the data — insider selling, low institutional count, etc.)

Be honest and rigorous. If fundamentals are strong, say so but still enumerate risks.
"""

_RISK_PROMPT = """
You are the RISK MANAGER and final decision authority.

You have reviewed the BULL and BEAR cases below. Your job is to:
1. Adjudicate the debate objectively.
2. Issue a VERDICT: APPROVED / WATCH / REJECTED.
3. Write the final consolidated TRADE MEMO for the investment committee.

BULL CASE:
{bull_case}

BEAR CASE:
{bear_case}

STOCK DATA PACKET:
{data_json}

Your final TRADE MEMO must include:
- **VERDICT**: APPROVED / WATCH / REJECTED (with one-sentence rationale)
- **Short-Term Trade Setup** (entry range, stop-loss, target — must match data exactly)
- **Long-Term Investment Thesis** (3–6 month outlook)
- **Key Risk to Monitor** (single most important risk from the bear case)
- **Position Sizing Note**: Given the ATR and stop-loss in the data, suggest relative position sizing (e.g., "smaller than usual given wide ATR")

VERDICT CRITERIA:
- APPROVED:  Bull factors clearly outweigh bear risks; setup is technically clean.
- WATCH:     Mixed signals — worth monitoring but no clean entry now.
- REJECTED:  Bear risks dominate or no valid technical entry setup detected.
"""


# ── Orchestrator ──────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    Runs the full Bull → Bear → Risk Manager pipeline for a single stock.
    """

    def run(self, candidate: Dict) -> Dict:
        """
        Parameters
        ----------
        candidate : dict
            Must contain 'symbol' and all pre-computed factor fields.

        Returns
        -------
        dict with keys: symbol, bull_thesis, bear_risks, verdict, full_memo
        """
        symbol = candidate.get("symbol", "UNKNOWN")
        logger.info(f"  Running agent debate for {symbol} …")

        # Prepare a clean JSON snapshot — only include non-null numeric facts
        data_snapshot = _clean_snapshot(candidate)
        data_json = json.dumps(data_snapshot, indent=2)

        result = {
            "symbol":      symbol,
            "bull_thesis": "",
            "bear_risks":  "",
            "verdict":     "WATCH",
            "full_memo":   "",
            "error":       None,
        }

        try:
            # ── Step 1: Bull Agent — Haiku (structured formatting only) ───────
            bull = _llm(
                system=_SYSTEM_STRICT,
                user=_BULL_PROMPT.format(symbol=symbol, data_json=data_json),
                model=CLAUDE_MODEL_BULL,
                max_tokens=900,
            )
            result["bull_thesis"] = bull
            logger.debug(f"  {symbol} — Bull case complete (Haiku)")

            # ── Step 2: Bear Agent — Haiku (structured formatting only) ───────
            bear = _llm(
                system=_SYSTEM_STRICT,
                user=_BEAR_PROMPT.format(symbol=symbol, data_json=data_json),
                model=CLAUDE_MODEL_BEAR,
                max_tokens=900,
            )
            result["bear_risks"] = bear
            logger.debug(f"  {symbol} — Bear case complete (Haiku)")

            # ── Step 3: Risk Manager — Sonnet (synthesis + verdict) ────────────
            memo = _llm(
                system=_SYSTEM_STRICT,
                user=_RISK_PROMPT.format(
                    bull_case=bull,
                    bear_case=bear,
                    data_json=data_json,
                ),
                model=CLAUDE_MODEL_RISK,
                max_tokens=1400,
            )
            result["full_memo"] = memo
            result["verdict"]   = _extract_verdict(memo)
            logger.info(f"  {symbol} — Verdict: {result['verdict']}")

        except Exception as exc:
            logger.error(f"Agent debate failed for {symbol}: {exc}")
            result["error"]   = str(exc)
            result["verdict"] = "WATCH"   # safe default on error

        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_snapshot(candidate: Dict) -> Dict:
    """
    Build a compact, LLM-readable snapshot from the candidate dict.
    Excludes raw list/object fields and null values to keep token count low.
    """
    INCLUDE_KEYS = {
        "symbol", "close", "entry_setup", "setup_confidence",
        "ema_alignment", "rsi", "atr", "macd_hist",
        "stop_loss", "target_short", "target_long", "risk_reward",
        "momentum_12_1", "momentum_3m",
        "vol_ratio", "high_52w", "low_52w",
        "roic", "roe", "fcf_yield", "debt_equity", "net_margin",
        "gross_margin", "current_ratio",
        "eps_growth_yoy", "eps_growth_qoq", "eps_acceleration",
        "annual_eps_growth", "rev_growth_yoy", "eps_surprise_pct",
        "peg_ratio", "ev_ebitda", "p_fcf", "pe_ratio",
        "analyst_target_mean", "analyst_target_high", "analyst_count",
        "insider_net", "institutional_holders_count",
        "sentiment_score", "sentiment_label",
        "composite_score", "quality_score", "momentum_score",
        "technical_score", "value_score",
    }

    snap = {}
    for k in INCLUDE_KEYS:
        v = candidate.get(k)
        if v is not None:
            # Round floats for readability
            snap[k] = round(v, 4) if isinstance(v, float) else v

    snap["analysis_date"] = str(date.today())
    return snap


def _extract_verdict(memo: str) -> str:
    """Parse APPROVED / WATCH / REJECTED from the Risk Manager output."""
    upper = memo.upper()
    for verdict in ("APPROVED", "REJECTED", "WATCH"):
        if verdict in upper:
            return verdict
    return "WATCH"

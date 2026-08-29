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

Write a structured BULL CASE covering ALL of the following sections.
For each section, you MUST cite specific numbers or quotes from the data packet above.

1. **Fundamental Strength**
   - Cite ROIC, ROE, FCF yield, net margin, and gross margin.
   - Cite EPS growth (YoY and QoQ) and whether there is acceleration.
   - Cite revenue growth trend. Is the business compounding?

2. **Valuation**
   - Cite PEG, EV/EBITDA, P/FCF, and P/E from the data.
   - Are these metrics cheap, fair, or expensive vs. typical quality thresholds (PEG < 1.5, EV/EBITDA < 15)?

3. **Technical Setup**
   - Cite the exact entry_setup name and setup_confidence.
   - Cite EMA alignment, RSI level, MACD histogram direction, and volume ratio.
   - Describe what the setup means: e.g., "RSI 48 on a pullback to rising 21-EMA signals a healthy reset before continuation."

4. **News Catalysts** (use the recent_news headlines from the data packet)
   - Quote or paraphrase the 2–3 most bullish headlines by name.
   - Explain HOW each piece of news supports the investment case (earnings beat → guidance raise → re-rating potential, etc.).
   - Cite the FinBERT sentiment score (e.g., +0.72 = strongly positive).

5. **Analyst & Institutional Conviction**
   - Cite analyst_target_mean / analyst_target_high and the analyst_count.
   - Cite institutional_holders_count and insider_net direction (positive = buying).

6. **Entry / Exit Plan**
   - Cite entry price (close), stop_loss, target_short, and target_long from the data.
   - Cite the risk_reward ratio.

Keep each section to 2–4 sentences. Use professional financial language.
"""

_BEAR_PROMPT = """
You are the BEAR ANALYST. Your job is to identify ALL material RISKS for {symbol}.

STOCK DATA PACKET:
{data_json}

Write a structured BEAR CASE covering ALL of the following sections.
For each section, you MUST cite specific numbers or facts from the data packet above.

1. **Valuation Risk**
   - Cite PEG, EV/EBITDA, P/FCF, and P/E from the data.
   - Are any elevated vs. historical norms or sector peers? Flag explicitly.
   - At what price level would valuation become unjustifiable given current earnings?

2. **Fundamental Weaknesses**
   - Cite debt_equity and current_ratio. Is leverage a concern?
   - Are there signs of slowing EPS (low eps_acceleration, negative eps_growth_qoq)?
   - Cite any weak FCF yield or compressing margins.

3. **Technical & Momentum Risk**
   - Is RSI overbought (> 70)? Is volume below average (vol_ratio < 1)?
   - Cite MACD histogram trend — is momentum fading?
   - What technical break would invalidate the bull setup?

4. **News & Sentiment Risks** (use the recent_news headlines from the data packet)
   - Quote or paraphrase the 1–2 most bearish or concerning headlines by name.
   - Explain the risk they represent: regulatory pressure, margin squeeze, industry headwind, etc.
   - If all headlines are positive, note that the stock may already have priced in the good news (mean reversion risk).

5. **Macro / Sector / Structural Risks**
   - Based on the sector/industry field, identify macro risks relevant to this sector
     (e.g., rising rates → insurance liabilities; energy price drop → E&P revenue; FDA approvals → biotech binary risk).
   - Cite insider_net if negative (selling pressure) or institutional_holders_count if low.

6. **Downside Scenario**
   - Cite the stop_loss price from the data. What price action triggers it?
   - Quantify the downside from current close to stop_loss as a percentage.

Be honest and rigorous. If fundamentals are genuinely strong, say so — but still enumerate every material risk.
"""

_RISK_PROMPT = """
You are the RISK MANAGER and final decision authority for the investment committee.

You have reviewed the BULL and BEAR analyst cases below. Your job is to:
1. Adjudicate the debate objectively and issue a VERDICT.
2. Write the final consolidated TRADE MEMO for the investment committee.
3. Include all context a portfolio manager needs to monitor the position.

BULL CASE:
{bull_case}

BEAR CASE:
{bear_case}

STOCK DATA PACKET:
{data_json}

Your final TRADE MEMO must include ALL of the following sections:

**VERDICT**: APPROVED / WATCH / REJECTED
- One sentence explaining the decisive factor that drove the verdict.

**Why This Stock, Why Now**
- 2–3 sentences summarising the core investment thesis in plain English.
- Reference the specific news catalyst(s) and fundamental driver(s) that make the timing relevant TODAY.
- Mention the sector/industry and whether sector tailwinds or headwinds apply.

**Trade Setup (Exact Numbers)**
- Entry: [close from data]
- Stop-loss: [stop_loss from data] — explain what technical level it represents
- Short-term target: [target_short from data] | Risk/Reward: [risk_reward from data]
- Long-term target: [target_long from data]

**Fundamental Snapshot**
- List 4–6 key metrics from the data (ROIC, EPS growth, FCF yield, PEG, debt_equity, rev_growth_yoy).
- One sentence interpreting whether the fundamentals support the entry price.

**News Context**
- Summarise the 2–3 most relevant recent headlines from the data packet.
- Explain how each headline affects the investment thesis (positive catalyst, priced-in risk, ongoing uncertainty, etc.).

**Key Risk to Monitor**
- Single most important risk from the bear case — with a specific price level or data trigger that would prompt an early exit.

**Position Sizing Note**
- Based on ATR and stop_loss distance, recommend sizing (full / 60% / 40%) with rationale.

VERDICT CRITERIA:
- APPROVED:  Bull factors clearly outweigh bear risks; setup is technically clean; news is supportive.
- WATCH:     Mixed signals — fundamentals or news are good but entry setup is not ideal; monitor for cleaner entry.
- REJECTED:  Bear risks dominate, news is negative/concerning, or no valid technical entry setup detected.
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
                max_tokens=1400,
            )
            result["bull_thesis"] = bull
            logger.debug(f"  {symbol} — Bull case complete (Haiku)")

            # ── Step 2: Bear Agent — Haiku (structured formatting only) ───────
            bear = _llm(
                system=_SYSTEM_STRICT,
                user=_BEAR_PROMPT.format(symbol=symbol, data_json=data_json),
                model=CLAUDE_MODEL_BEAR,
                max_tokens=1400,
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
                max_tokens=2000,
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
        "sector", "industry",
    }

    snap = {}
    for k in INCLUDE_KEYS:
        v = candidate.get(k)
        if v is not None:
            # Round floats for readability
            snap[k] = round(v, 4) if isinstance(v, float) else v

    # Include top news headlines so agents can cite specific catalysts
    headlines = candidate.get("top_headlines", [])
    if headlines:
        snap["recent_news"] = [
            {
                "headline": h.get("headline", ""),
                "sentiment": round(h.get("sentiment", 0.0), 3),
                "source": h.get("source", ""),
            }
            for h in headlines[:5]
        ]

    snap["analysis_date"] = str(date.today())
    return snap


def _extract_verdict(memo: str) -> str:
    """Parse APPROVED / WATCH / REJECTED from the Risk Manager output."""
    upper = memo.upper()
    for verdict in ("APPROVED", "REJECTED", "WATCH"):
        if verdict in upper:
            return verdict
    return "WATCH"

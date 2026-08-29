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
You are a senior portfolio manager at a tier-1 hedge fund writing internal research notes.
You have a structured JSON data packet: pre-computed financial metrics AND recent news articles
(headline + body snippet + FinBERT sentiment score) for a stock.

STRICT RULES — NON-NEGOTIABLE:
1. Only use numbers and facts present in the JSON. Do NOT invent or estimate any figure.
2. If a metric is missing, say "data unavailable" — never substitute a value.
3. Cite specific numbers from the JSON in every paragraph.
4. CROSS-CORRELATE news with fundamentals. When a news article says something, tie it to a
   specific metric in the data. Example: "Reuters reports record Q2 revenue — this directly
   CONFIRMS our data showing rev_growth_yoy=+47%; the news is corroboration, not speculation."
5. You MUST reach a clear, decisive opinion. Neutral or balanced conclusions are NOT acceptable.
   If evidence is mixed, weigh it and come down clearly on one side with a reason.
   "On one hand… on the other hand…" is not a conclusion — it is an abdication of analysis.
6. Write with conviction. Use language like "This IS a high-quality setup", "The risk IS
   manageable", "This position SHOULD NOT be entered" — not "might", "could", "potentially".
7. Short-circuit clichés: no "investors should be cautious", no "time will tell", no "monitoring
   is advised". Make a call.
"""

_BULL_PROMPT = """
You are the BULL ANALYST making the strongest possible INVESTMENT CASE for {symbol}.

STOCK DATA PACKET (metrics + recent news with FinBERT sentiment scores):
{data_json}

Write a structured BULL CASE. Be OPINIONATED and SPECIFIC. Do NOT hedge. Cite exact numbers.
For the news sections, quote the actual headline text and link it to a specific metric.

─────────────────────────────────────────────────────────────
1. WHAT THE BUSINESS DOES AND WHY IT'S WINNING RIGHT NOW
─────────────────────────────────────────────────────────────
Use sector/industry from the data. In 2–3 sentences explain the business model and
what structural advantage is driving the numbers. Be concrete, not generic.

─────────────────────────────────────────────────────────────
2. FUNDAMENTAL QUALITY (cite every number)
─────────────────────────────────────────────────────────────
- ROIC: [value] — above/below 15% quality threshold?
- FCF yield: [value] — does free cash generation support valuation?
- EPS growth YoY: [value], QoQ: [value], acceleration: [value] — accelerating or decelerating?
- Revenue growth YoY: [value] — is the top line expanding?
- Net margin / Gross margin: [values] — are margins expanding, stable, or contracting?
State clearly: "The fundamentals ARE / ARE NOT high-quality because [specific reason]."

─────────────────────────────────────────────────────────────
3. VALUATION — IS THE PRICE RIGHT?
─────────────────────────────────────────────────────────────
- PEG: [value] (cheap if < 1.0, fair if 1.0–1.5, stretched if > 2.0)
- EV/EBITDA: [value] (reasonable < 15x for quality businesses)
- P/FCF: [value], P/E: [value]
State clearly: "At these multiples, the stock IS / IS NOT attractively valued because [reason]."

─────────────────────────────────────────────────────────────
4. WHAT THE NEWS SAYS — AND WHAT IT MEANS FOR THE THESIS
─────────────────────────────────────────────────────────────
For EACH of the top 3 most positive recent_news articles (sentiment > 0), do ALL of the following:
  a) Quote the headline exactly.
  b) State what event or action the article describes (earnings beat, contract win, guidance raise, etc.).
  c) Cross-correlate: tie this news event to a SPECIFIC metric in the data.
     Example: "This earnings beat DIRECTLY CONFIRMS our data showing eps_surprise_pct=+18.4% —
     the market is now validating what the fundamentals already showed."
  d) Explain the investment implication: does this re-rate the stock, expand the moat, or de-risk the thesis?
If no clearly positive news exists, say so and explain what the news SILENCE implies.

─────────────────────────────────────────────────────────────
5. TECHNICAL ENTRY — WHY NOW IS THE RIGHT TIME
─────────────────────────────────────────────────────────────
- Entry setup: [entry_setup] at [setup_confidence]% confidence
- EMA alignment: [ema_alignment] — are all major EMAs stacked bullishly?
- RSI: [rsi] — is this a reset/pullback entry or a momentum breakout?
- Volume ratio: [vol_ratio] — is the move confirmed by volume?
- Entry: [close], Stop: [stop_loss], Target (ST): [target_short], Target (LT): [target_long]
- Risk/Reward: [risk_reward]x
Describe in one sentence WHY the technical setup timing is compelling TODAY.

─────────────────────────────────────────────────────────────
6. ANALYST CONVICTION
─────────────────────────────────────────────────────────────
- [analyst_count] analysts with mean target [analyst_target_mean], high target [analyst_target_high]
- Institutional holders: [institutional_holders_count], Insider net: [insider_net]
Is smart money aligned with this thesis?

END with ONE bold sentence that is your BULL VERDICT: "I rate {symbol} a STRONG BUY / BUY / HOLD here because [single decisive reason]."
"""

_BEAR_PROMPT = """
You are the BEAR ANALYST tasked with identifying EVERY material risk for {symbol}.
Your job is to CHALLENGE the bull case aggressively. Be OPINIONATED and SPECIFIC.

STOCK DATA PACKET (metrics + recent news with FinBERT sentiment scores):
{data_json}

Write a structured BEAR CASE. Reach a clear conclusion — do NOT give neutral analysis.
For news, quote actual headlines and tie them to specific risks or data metrics.

─────────────────────────────────────────────────────────────
1. VALUATION RISK — IS THE STOCK TOO EXPENSIVE?
─────────────────────────────────────────────────────────────
- PEG: [value] — does this justify the growth rate?
- EV/EBITDA: [value] — how many years of EBITDA is the market pricing in?
- P/FCF: [value], P/E: [value]
At what price does valuation become indefensible? Cite exact numbers.
State clearly: "Valuation IS / IS NOT a material risk because [reason]."

─────────────────────────────────────────────────────────────
2. FUNDAMENTAL WEAKNESSES — WHERE DOES THE BUSINESS CRACK?
─────────────────────────────────────────────────────────────
- Debt/Equity: [value] — is leverage excessive (> 1.5 is elevated)?
- Current ratio: [value] — liquidity risk?
- EPS growth decelerating? (eps_acceleration < 0 = warning)
- FCF yield vs. P/FCF: is free cash generation actually supporting the price?
- Any margin compression? (cite gross_margin, net_margin)
Be direct: "The biggest fundamental weakness IS [specific metric + value]."

─────────────────────────────────────────────────────────────
3. WHAT THE NEWS SAYS — THE RISKS HIDDEN IN THE HEADLINES
─────────────────────────────────────────────────────────────
For EACH of the top 2 most negative or cautionary recent_news articles, do ALL of the following:
  a) Quote the headline exactly.
  b) Describe what risk the article represents (regulatory, competitive, macro, execution, etc.).
  c) Cross-correlate: tie this news to a SPECIFIC metric in the data.
     Example: "Reuters reports pricing pressure from competition — this IS already showing up
     in our data: gross_margin has compressed to [X]%, validating the concern."
  d) Explain the investment risk: does this shrink the moat, threaten the revenue line, or indicate
     management credibility issues?
If all headlines are positive: "With uniformly positive coverage (avg sentiment +[X]), the stock
may already have priced in good news. Mean reversion risk is elevated — latecomers rarely profit."

─────────────────────────────────────────────────────────────
4. SECTOR & MACRO RISKS SPECIFIC TO THIS BUSINESS
─────────────────────────────────────────────────────────────
Based on sector/industry in the data, name the 2–3 macro or sector risks most relevant to
THIS company's business model (not generic risks). Examples:
- Insurance: rising catastrophe costs, reserve adequacy, interest rate sensitivity on float
- Energy/E&P: commodity price cycles, capex discipline, OPEC supply decisions
- Biotech/Pharma: FDA binary events, patent cliffs, pipeline concentration
- Semis: inventory cycles, geopolitical supply chain, customer concentration
Tie each macro risk to a number in the data where possible.
Also cite insider_net if negative, institutional_holders_count if low.

─────────────────────────────────────────────────────────────
5. TECHNICAL BREAKDOWN SCENARIO
─────────────────────────────────────────────────────────────
- Stop-loss: [stop_loss] — that is [X]% below current price of [close]
- What specific price action (volume spike, EMA cross, RSI collapse) would signal the thesis is wrong?
- If RSI > 70: the stock is overbought and vulnerable to a technical flush.
- If vol_ratio < 1: the move lacks conviction.

END with ONE bold sentence that is your BEAR VERDICT: "The most dangerous risk for {symbol} IS [specific risk] — here's why it matters more than the bulls acknowledge: [one-sentence counter]."
"""

_RISK_PROMPT = """
You are the RISK MANAGER and final decision authority for the investment committee.
You have reviewed the BULL and BEAR cases. You must issue a VERDICT and write a TRADE MEMO.

IMPORTANT: Do NOT summarise the bull and bear cases. ADJUDICATE them.
Come down clearly on one side. State which analyst made the stronger case and why.
Wishy-washy memos waste the committee's time. Make a call.

BULL CASE:
{bull_case}

BEAR CASE:
{bear_case}

STOCK DATA PACKET:
{data_json}

─────────────────────────────────────────────────────────────
VERDICT: [APPROVED / WATCH / REJECTED]
─────────────────────────────────────────────────────────────
One sentence. Name the single decisive factor that tipped the verdict.
Example: "APPROVED — EPS acceleration of +[X]% confirmed by two independent news sources
outweighs the valuation premium; the setup IS clean."

─────────────────────────────────────────────────────────────
WHY THIS STOCK, WHY TODAY
─────────────────────────────────────────────────────────────
2–3 sentences in plain English. Answer:
- What does this company do and why is it in a position of strength / weakness RIGHT NOW?
- What specific news event or fundamental catalyst is creating the opportunity TODAY?
- Is the sector tailwind or headwind? Cite sector/industry from the data.
Do NOT say "could benefit" — say "IS benefiting from [X] as evidenced by [metric]."

─────────────────────────────────────────────────────────────
NEWS ↔ FUNDAMENTALS CORRELATION
─────────────────────────────────────────────────────────────
Take the 2–3 most impactful recent_news items and for EACH:
- Quote the headline.
- State what it confirms or contradicts in the quantitative data.
- Make a judgment: does this news ADD TO or DETRACT FROM conviction?
Example: "Headline: '[X]' — This CONFIRMS rev_growth_yoy=+[Y]%; it adds to conviction
because management is publicly guiding higher while the quant data already shows acceleration."

─────────────────────────────────────────────────────────────
TRADE SETUP
─────────────────────────────────────────────────────────────
- Entry:            [close]
- Stop-loss:        [stop_loss]  (represents [explain the technical level — ATR stop, prior low, EMA])
- Short-term target:[target_short]  |  R/R: [risk_reward]x
- Long-term target: [target_long]
- Position sizing:  FULL / 60% / 40% — state why (e.g., "wide ATR of [X] relative to price warrants 60% sizing")

─────────────────────────────────────────────────────────────
FUNDAMENTAL VERDICT (6 key metrics)
─────────────────────────────────────────────────────────────
| Metric           | Value  | Interpretation          |
|------------------|--------|-------------------------|
| ROIC             | [val]  | [strong/weak/avg]       |
| EPS growth YoY   | [val]  | [accelerating/slowing]  |
| FCF yield        | [val]  | [cheap/fair/expensive]  |
| PEG ratio        | [val]  | [cheap/fair/stretched]  |
| Debt/Equity      | [val]  | [safe/elevated/risky]   |
| Rev growth YoY   | [val]  | [expanding/flat/shrinking] |
One sentence: "The fundamentals [DO / DO NOT] justify the current price of [close] because [reason]."

─────────────────────────────────────────────────────────────
THE ONE RISK THAT COULD KILL THIS TRADE
─────────────────────────────────────────────────────────────
State the single most important risk. Give the exact price level or data threshold that
would signal the thesis is broken and the position must be exited immediately.
Do not list multiple risks here — pick ONE and defend why it is the most dangerous.

VERDICT CRITERIA:
- APPROVED:  Fundamentals confirmed by news, technical setup clean, risk/reward > 2x. Enter now.
- WATCH:     Thesis intact but entry timing poor (overbought, volume missing, news unclear). Monitor.
- REJECTED:  News contradicts fundamentals, technicals broken, or risk clearly dominates reward.
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

    # Prefer full articles (headline + body snippet) over bare headlines.
    # full_news is set in Stage 6 of the pipeline; top_headlines is the fallback.
    full_news = candidate.get("full_news", [])
    if full_news:
        snap["recent_news"] = full_news  # already formatted + sorted by |sentiment|
    else:
        headlines = candidate.get("top_headlines", [])
        if headlines:
            snap["recent_news"] = [
                {
                    "headline":  h.get("headline", ""),
                    "summary":   "",
                    "sentiment": round(h.get("sentiment", 0.0), 3),
                    "source":    h.get("source", ""),
                    "published": "",
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

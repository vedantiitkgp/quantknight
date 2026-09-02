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
You have a JSON data packet with pre-computed financial metrics, positioning data, and recent
news articles (headline + body + FinBERT sentiment score) for a stock.

═══════════════════════════════════════════════════════════
ABSOLUTE RULES — VIOLATING THESE MAKES THE MEMO WORTHLESS
═══════════════════════════════════════════════════════════
1. ONLY use numbers/facts in the JSON. Never invent or estimate any figure.
2. If a metric is missing, say "data unavailable" — do not substitute.
3. REASON THROUGH MECHANISMS, not just observations. For every data point, follow:
      Observation → Mechanism → Market Reaction → Price Impact
   Examples:
   • eps_surprise_pct=+18% → Earnings beat → Quant funds trigger buy programs, analysts revise
     targets upward → P/E multiple expands → bullish price action in 1–5 days
   • short_pct_float=22% + positive catalyst → High short interest squeezed → Covering buyers
     amplify price move → upside can be 2–3× what fundamentals alone would suggest
   • days_to_earnings=6 → Binary event in 6 days → If beat, gap up; if miss, crash → reduces
     position certainty → size accordingly
   • Government rate cut in news → Insurance float income falls → But discount rate drops too →
     equity risk premium narrows → net effect depends on ROIC vs. cost of capital
4. CROSS-CORRELATE news with fundamentals. Quote the headline, then tie it to a metric:
   "Reuters: '[headline]' — This CONFIRMS/CONTRADICTS our data showing [metric=value]."
5. For each news event about another company or government action, reason through the
   TRANSMISSION MECHANISM to THIS stock's price:
   • Competitor news → market share, pricing power, margins
   • Sector ETF flow → institutional buy/sell pressure on this name
   • Government action → which revenue line, which cost line, net EPS impact
   • Fed/macro → discount rate, consumer spending, credit conditions
6. You MUST reach a DECISIVE OPINION. No balanced conclusions.
   Weigh the evidence and come down on one side. Use conviction language:
   "This IS", "The thesis IS broken", "ENTER NOW", "DO NOT enter" — not "might/could/may".
7. No clichés: never write "investors should monitor", "time will tell", "proceed with caution".
   Make a call. Defend it with data.
"""

_BULL_PROMPT = """
You are the BULL ANALYST. Build the strongest possible investment case for {symbol}.
Be decisive. Cite exact numbers. Reason through mechanisms, not just observations.

STOCK DATA PACKET:
{data_json}

══════════════════════════════════════════════════════════════════
SECTION 1 — WHAT THIS BUSINESS DOES AND WHY IT IS WINNING NOW
══════════════════════════════════════════════════════════════════
Use company_description, sector, industry from the data.
2–3 sentences: What structural advantage explains the financial metrics below?
Be concrete — name the product, service, or market position. No generic statements.

══════════════════════════════════════════════════════════════════
SECTION 2 — FUNDAMENTAL QUALITY VERDICT
══════════════════════════════════════════════════════════════════
Output a markdown table, then one concluding sentence:

| Metric | Value | Signal | Why It Matters |
|--------|-------|--------|----------------|
| ROIC | X% | MOAT / COMMODITY | Threshold 15%; above = pricing power |
| FCF Yield | X% | CHEAP / REINVESTING / COSTLY | >4% = cheap on cash |
| EPS Growth YoY | X% | ACCELERATING / DECELERATING | Growth rate direction = multiple re-rating |
| EPS Growth QoQ | X% | ACCELERATING / DECELERATING | Quarter trend |
| Revenue Growth YoY | X% | ORGANIC / COST-DRIVEN | Is top-line driving EPS? |
| Net Margin | X% | EXPANDING / STABLE / COMPRESSED | Pricing power |
| Gross Margin | X% | EXPANDING / STABLE / COMPRESSED | Moat signal |

Fill every X with the exact data value. Signal must be one of the options shown.
CONCLUDE (1 sentence): "The fundamentals ARE [high/average/weak] quality. Here is why: [data]."

══════════════════════════════════════════════════════════════════
SECTION 3 — VALUATION: IS THE PRICE JUSTIFIED?
══════════════════════════════════════════════════════════════════
Output a markdown table, then one concluding sentence:

| Metric | Value | Signal | Interpretation |
|--------|-------|--------|----------------|
| PEG | X | CHEAP / FAIR / EXPENSIVE | <1.0 cheap; >2.0 fully priced |
| EV/EBITDA | Xx | CHEAP / FAIR / STRETCHED | Quality biz 12–18x; stretched >20x |
| P/FCF | Xx | LOW / FAIR / HIGH | Paying Xx per $1 of free cash |
| Trailing P/E | Xx | — | Current earnings multiple |
| Forward P/E | Xx | EXPANDING / CONTRACTING | vs trailing = multiple direction |
| Analyst Target | $X | +X% upside | X analysts; high target $X |

Fill every X with exact data. Signal must be one of the options shown.
CONCLUDE (1 sentence): "At [close], the stock IS [cheap/fairly valued/expensive] because [reason]."

══════════════════════════════════════════════════════════════════
SECTION 4 — EARNINGS REPORT ANALYSIS & PRICE IMPACT
══════════════════════════════════════════════════════════════════
• EPS surprise: [eps_surprise_pct]%
  Mechanism: A beat of this magnitude → analyst estimates revised upward → quant models rebalance
  → institutional buy programs triggered within 1–5 trading sessions → price re-rating.
  Quantify: "At current P/E of [pe_ratio], a 10% EPS revision implies a price target of approximately
  [close * (1 + 0.10)] — this is the mechanism, not speculation."
• Next earnings: [next_earnings_date] ([days_to_earnings] days away)
  If < 21 days: "This is a near-term binary event. A beat would [X]; a miss would [Y]."
  If > 60 days: "No near-term earnings overhang — clean entry window for the thesis to develop."
• EPS acceleration: if positive, say "The acceleration in earnings growth IS the catalyst for
  multiple re-rating. When growth RATES increase, the market reprices the stock to a higher P/E."

══════════════════════════════════════════════════════════════════
SECTION 5 — NEWS CATALYST ANALYSIS (all headlines)
══════════════════════════════════════════════════════════════════
For EACH article in recent_news, one compact line:
"[source] '[headline]' (sent=[val]) → [event type] → [1-sentence price mechanism] → [ADD TO CONVICTION / DETRACTS / NEUTRAL]"

Cover all articles. Then pick the SINGLE most bullish headline and write 2-3 sentences:
why it specifically CONFIRMS a data metric (cite the exact metric=value) and what price move it implies.

══════════════════════════════════════════════════════════════════
SECTION 6 — POSITIONING & SHORT SQUEEZE POTENTIAL
══════════════════════════════════════════════════════════════════
• Short interest: [short_pct_float]% of float, [short_ratio] days to cover.
  If short_pct_float > 10%: "High short interest IS a coiled spring. A positive catalyst would force
  short covering + new long buying simultaneously = amplified move. At [short_pct_float]% shorted,
  covering alone could add [short_pct_float/10]–[short_pct_float/5]% buying pressure."
  If short_pct_float < 5%: "Low short interest means the move is not amplified by covering — the
  upside is clean, driven by genuine new buyers."
• Insider net: [insider_net] (positive = net buying, negative = net selling)
  If positive: "Insiders ARE buying with own money — the strongest alignment signal possible."
  If negative: "Insider selling is a warning. Management knows more than the market."
• Institutional holders: [institutional_holders_count]
  If increasing: implies accumulation phase — institutional flows create sustained buying pressure.
• Beta: [beta] — at current beta, a 1% S&P 500 move implies approximately [beta]% move in {symbol}.
  In a bull market environment, high beta amplifies the upside case.

══════════════════════════════════════════════════════════════════
SECTION 7 — SECTOR ROTATION & OTHER-STOCK IMPACT
══════════════════════════════════════════════════════════════════
Based on sector/industry in the data:
• Is money flowing INTO this sector? (Look at news for sector ETF flows, sector peer earnings)
• If sector peers reported strong results: "Sector-wide strength IS a tailwind — institutional
  rotations into [sector] ETFs lift all names including {symbol}."
• If a sector peer mentioned in news is winning: state the read-through to {symbol}.
• If macro is supportive (rates, commodities, policy): "The macro backdrop IS [supportive/neutral/
  headwind] for the [sector] sector because [mechanism]."

══════════════════════════════════════════════════════════════════
SECTION 8 — TECHNICAL ENTRY: WHY THE TIMING IS RIGHT TODAY
══════════════════════════════════════════════════════════════════
• Entry setup: [entry_setup] at [setup_confidence]% confidence
• EMA alignment: [ema_alignment] — all EMAs stacked bullishly = trend is UP
• RSI [rsi]: below 50 = pullback entry (low risk); 50–65 = momentum entry; above 70 = extended
• MACD histogram [macd_hist]: positive + rising = momentum building; negative = avoid
• Volume ratio [vol_ratio]: above 1 = institutional participation confirming the move
• Entry [close] | Stop [stop_loss] | ST Target [target_short] | LT Target [target_long] | R/R [risk_reward]x
Explain in 1 sentence why the technical structure says "enter NOW not later."

END YOUR CASE with this sentence — fill in ALL brackets with actual data:
"**BULL VERDICT: I rate {symbol} a [STRONG BUY/BUY] at [close] with [risk_reward]x R/R because [the single most decisive quantitative reason — cite a number].**"
"""

_BEAR_PROMPT = """
You are the BEAR ANALYST. Find every material reason {symbol} should NOT be held.
Be aggressive. Cite exact numbers. Reason through mechanisms, not just risks.

STOCK DATA PACKET:
{data_json}

══════════════════════════════════════════════════════════════════
SECTION 1 — VALUATION: IS THE PRICE ALREADY WRONG?
══════════════════════════════════════════════════════════════════
Output a markdown table, then one concluding sentence:

| Metric | Value | Risk Level | Bear Interpretation |
|--------|-------|------------|---------------------|
| PEG | X | LOW / MODERATE / HIGH | >2.0 = paying 2× growth rate |
| EV/EBITDA | Xx | CHEAP / FAIR / STRETCHED | Prices in X years of EBITDA |
| P/FCF | Xx | LOW / FAIR / EXCESSIVE | Paying Xx per $1 FCF |
| Trailing P/E | Xx | — | Current multiple |
| Forward P/E | Xx | EXPANDING / CONTRACTING | Acceleration priced in? |
| Analyst Target | $X | +X% / -X% | At target, implied P/E = X |

Fill every X with exact data. Risk Level must be one of the options shown.
CONCLUDE (1 sentence): "Valuation IS [a major risk / manageable / cheap] because [number + reason]."

══════════════════════════════════════════════════════════════════
SECTION 2 — FUNDAMENTAL CRACKS: WHERE DOES THE BUSINESS BREAK?
══════════════════════════════════════════════════════════════════
Output a markdown table, then one concluding sentence:

| Metric | Value | Risk Level | Bear Interpretation |
|--------|-------|------------|---------------------|
| Debt/Equity | X | LOW / ELEVATED / DANGEROUS | >1.5 elevated; model rate-rise impact |
| EPS Acceleration | X | POSITIVE / DECELERATING | Negative = multiple contraction risk |
| FCF Yield | X% | ADEQUATE / THIN / PRICED-FOR-PERFECTION | <2% = no margin for error |
| Gross Margin | X% | STABLE / COMPRESSING | Compression = pricing power loss |
| Net Margin | X% | STABLE / COMPRESSING | Compression = earnings risk |

Fill every X with exact data. Risk Level must be one of the options shown.
CONCLUDE (1 sentence): "The biggest fundamental weakness IS [metric=val] because [mechanism]."

══════════════════════════════════════════════════════════════════
SECTION 3 — NEWS RISK ANALYSIS (all headlines)
══════════════════════════════════════════════════════════════════
For EACH article in recent_news, one compact line:
"[source] '[headline]' (sent=[val]) → [risk type] → [1-sentence price mechanism] → [REDUCES / UNDERMINES / NEUTRAL]"

Cover all articles. Then pick the SINGLE most bearish headline and write 2-3 sentences:
why it specifically CONTRADICTS or threatens a data metric (cite exact metric=value) and what downside it implies.
If all headlines are positive: note this explicitly — "Uniformly bullish coverage IS itself a contrarian risk signal — good news is priced in, bad news is not."

══════════════════════════════════════════════════════════════════
SECTION 4 — GOVERNMENT & REGULATORY ACTION IMPACT
══════════════════════════════════════════════════════════════════
Scan the recent_news for any government actions (policy changes, regulations, tariffs, rate decisions,
FDA decisions, antitrust, sanctions, elections, spending bills). For each one found:
  a) Name the action and the government body.
  b) Identify which specific revenue line or cost line of {symbol} is affected.
  c) Model the mechanism: "A [X]% tariff on [input] → COGS rises by ~[Y]% → gross margin compresses
     from [val]% to approximately [val-Y]% → EPS impact of [direction]."
  d) Timeline: is this already in the stock or forward-looking?
If no government actions in news: state this explicitly and note the policy risk relevant to
the sector (see below).

══════════════════════════════════════════════════════════════════
SECTION 5 — SECTOR-SPECIFIC MACRO RISKS (not generic — be exact)
══════════════════════════════════════════════════════════════════
Based on sector/industry, name the 2–3 risks SPECIFIC to this business model:
• Financial Services / Insurance: interest rate impact on float income, catastrophe reserve
  adequacy, regulatory capital requirements, credit cycle exposure
• Energy / E&P / Chemicals: commodity price cycles (WTI, natural gas, fertilizer), OPEC decisions,
  capex discipline, pipeline/export capacity constraints
• Healthcare / Pharma / Biotech: FDA approval binary events, drug pricing legislation (IRA/Medicare),
  patent cliff timing, pipeline concentration risk
• Technology / Semis / Software: inventory correction cycles, export controls (CHIPS Act, Entity List),
  customer concentration (top 3 customers as % of revenue), AI capex cycle sustainability
• Industrials / Defense: government contract award cycles, supply chain tariff exposure, capex
  spend correlation to GDP
• Consumer Discretionary / Retail: interest rate impact on credit spending, consumer confidence,
  inventory obsolescence
For EACH sector risk: tie it to a number from the data. "If [sector risk], it would impact
[metric=val] because [mechanism]."

══════════════════════════════════════════════════════════════════
SECTION 6 — POSITIONING: IS SMART MONEY ALREADY LEAVING?
══════════════════════════════════════════════════════════════════
• Short interest: [short_pct_float]% of float short, [short_ratio] days to cover.
  High short interest IS a double-edged sword — but also means professionals are betting AGAINST.
  Thesis: Why do short sellers (who do extensive research) believe this stock IS overvalued?
  Their thesis might be: [derive from bear fundamentals above].
• Insider net: [insider_net]. If negative: "Management IS selling into strength — the people who
  know most about the business ARE reducing exposure. This IS a material warning."
• Beta [val]: At this beta, a 5% market correction would imply approximately [5*beta]% drawdown in
  {symbol}. In risk-off environments, high-beta names sell off disproportionately.
• Days to earnings: [days_to_earnings] days.
  If < 21 days: "Earnings are [days_to_earnings] days away — this IS a binary event. If the company
  misses estimates by even 5%, the stock IS vulnerable to a gap down of 10–20%. Size accordingly."

══════════════════════════════════════════════════════════════════
SECTION 7 — TECHNICAL BREAKDOWN SCENARIO
══════════════════════════════════════════════════════════════════
• Stop-loss [stop_loss] = [((close-stop_loss)/close)*100]% below entry at [close].
• If RSI > 70: "The stock IS overbought. RSI mean-reversion from >70 to <50 historically accompanies
  a 5–15% price correction. The technical entry IS poor timing."
• If vol_ratio < 1.0: "Volume IS below average — the move lacks institutional confirmation.
  Low-volume moves fail more often than high-volume moves."
• What price action INVALIDATES the bull thesis: "A close below [stop_loss] on above-average volume
  would signal institutional distribution — EXIT immediately, no averaging down."

END YOUR CASE with this sentence — fill in ALL brackets with actual data:
"**BEAR VERDICT: The most dangerous risk for {symbol} IS [specific risk] at [metric=val],
because [1-sentence mechanism of how it flows to a lower price]. The bull case breaks if [specific trigger].**"
"""

_RISK_PROMPT = """
You are the RISK MANAGER. You have the full bull and bear cases below. ADJUDICATE them.
Do NOT summarise — state which case is STRONGER and WHY. Then issue the VERDICT.
This memo goes to the portfolio manager who will execute. Make every word count.

BULL CASE:
{bull_case}

BEAR CASE:
{bear_case}

STOCK DATA PACKET:
{data_json}

══════════════════════════════════════════════════════════════════
VERDICT: [APPROVED / WATCH / REJECTED]
══════════════════════════════════════════════════════════════════
State which analyst made the stronger case and the single factor that decided it.
Format: "[VERDICT] — The [bull/bear] case IS stronger because [decisive factor with data].
The [other] analyst's [strongest point] is [acknowledged but outweighed / a real risk that
prevents APPROVED / the reason for REJECTED]."

══════════════════════════════════════════════════════════════════
WHY THIS STOCK, WHY TODAY — THE ONE-PARAGRAPH INVESTMENT CASE
══════════════════════════════════════════════════════════════════
Write as if briefing a PM who has 30 seconds to decide. Include:
1. What the company does (use company_description / sector / industry)
2. The PRIMARY catalyst that makes TODAY the right entry point — cite a specific news event OR
   earnings data OR technical setup as the timing trigger
3. The PRIMARY risk that could make this wrong — name it, give the price level
4. Net conclusion: "This IS / IS NOT a high-conviction trade because [reason]."
Do NOT say "could" or "may". State what IS happening and what WOULD happen in each scenario.

══════════════════════════════════════════════════════════════════
NEWS ↔ FUNDAMENTALS SUMMARY (cover ALL headlines)
══════════════════════════════════════════════════════════════════
For EACH article in recent_news, one compact line:
"[source] '[headline]' (sent=[val]) → [event type] → [1-sentence price mechanism] → [ADDS TO / DETRACTS / NEUTRAL]"

Cover every article — do not skip any. Keep each line to 1 sentence after the arrows.
Close with: "Net news IS [bullish/neutral/bearish]: [X]/[total] articles support the thesis; dominant theme: [Z]."

══════════════════════════════════════════════════════════════════
EARNINGS CALENDAR & BINARY EVENT RISK
══════════════════════════════════════════════════════════════════
Next earnings: [next_earnings_date] ([days_to_earnings] days)
• If < 14 days: "BINARY EVENT RISK IS HIGH. Do not enter full position before earnings.
  Recommended: 40–50% position now; add after earnings confirm the thesis. If the company
  misses by [X]%, the stock IS likely to gap down [Y]% — this IS the primary exit risk."
• If 15–45 days: "Moderate catalyst proximity. Enter full position if setup is clean.
  Earnings in [days_to_earnings] days means the thesis has [days_to_earnings] days to develop
  before the next binary test."
• If > 45 days: "Clean entry window. No near-term earnings overhang. The technical setup
  IS the primary driver of near-term price action."
EPS surprise history: [eps_surprise_pct]% most recent beat/miss.
"A company that beat by [eps_surprise_pct]% last quarter IS likely to have revised estimates
that ARE/ARE NOT achievable — [explain based on whether growth is accelerating or decelerating]."

══════════════════════════════════════════════════════════════════
MARKET MICROSTRUCTURE ASSESSMENT
══════════════════════════════════════════════════════════════════
Net buyer or net seller pressure from all signals:
• Short interest [short_pct_float]%: [HIGH=squeeze risk / LOW=clean setup / MODERATE=watch]
• Short ratio [short_ratio] days to cover: "At current volume, shorts need [val] days to exit —
  any positive catalyst creates sustained mechanical buying pressure for [val] days."
• Insider net [insider_net]: [buying=strongly bullish / selling=distribution warning]
• Institutional count [institutional_holders_count]: [high=stable / low=under-owned=re-rating potential]
• Beta [beta]: "In current market conditions, [symbol] moves [beta]× the S&P 500.
  If S&P rallies 2%, expect approximately [2*beta]% in {symbol} due to market correlation alone."
CONCLUDE: "Net positioning IS [bullish / neutral / bearish] for {symbol} because [1 sentence]."

══════════════════════════════════════════════════════════════════
TRADE SETUP (exact numbers — no approximations)
══════════════════════════════════════════════════════════════════
• Entry:          [close]
• Stop-loss:      [stop_loss]  → [explain: ATR-based / prior support / EMA level]
  Downside if stopped: [((close-stop_loss)/close)*100]%
• ST Target:      [target_short]  |  R/R: [risk_reward]×
• LT Target:      [target_long]
• Position size:  [FULL 100% / 60% / 40%] of standard allocation
  Rationale: [cite the specific reason — earnings proximity / wide ATR / high short interest / clean setup]

══════════════════════════════════════════════════════════════════
FUNDAMENTAL SCORECARD
══════════════════════════════════════════════════════════════════
| Metric           | Value           | Grade        | Why It Matters                    |
|------------------|-----------------|--------------|-----------------------------------|
| ROIC             | [val]%          | [A/B/C/D]    | Moat / pricing power proxy        |
| EPS growth YoY   | [val]%          | [A/B/C/D]    | Earnings momentum                 |
| EPS acceleration | [val]           | [+/-]        | Growth rate speeding up/slowing   |
| FCF yield        | [val]%          | [A/B/C/D]    | Cash return vs. price paid        |
| PEG ratio        | [val]           | [cheap/fair/stretched] | Growth vs. price paid    |
| Debt/Equity      | [val]           | [safe/ok/risky] | Balance sheet risk             |
| Revenue growth   | [val]%          | [A/B/C/D]    | Top-line momentum                 |
One sentence: "The fundamentals [DO / DO NOT] support paying [close] for {symbol} because [reason]."

══════════════════════════════════════════════════════════════════
THE ONE RISK THAT KILLS THIS TRADE
══════════════════════════════════════════════════════════════════
Name the SINGLE most dangerous risk. Give:
1. The exact trigger (price level, news event, metric threshold)
2. The mechanism by which it causes loss
3. The exit action: "If [trigger], EXIT immediately — do not wait for confirmation."
No hedging. No second risk. Pick one. Defend the choice.

VERDICT CRITERIA (for reference):
APPROVED:  Fundamentals confirmed by news. Technical setup clean. R/R > 2×. Enter now.
WATCH:     Thesis intact but entry timing poor (earnings too close / overbought / news mixed). Wait.
REJECTED:  News contradicts fundamentals, technicals broken, risk > reward, or positioning warns.
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
                max_tokens=6000,
            )
            result["bull_thesis"] = bull
            logger.debug(f"  {symbol} — Bull case complete (Haiku)")

            # ── Step 2: Bear Agent — Haiku (structured formatting only) ───────
            bear = _llm(
                system=_SYSTEM_STRICT,
                user=_BEAR_PROMPT.format(symbol=symbol, data_json=data_json),
                model=CLAUDE_MODEL_BEAR,
                max_tokens=6000,
            )
            result["bear_risks"] = bear
            logger.debug(f"  {symbol} — Bear case complete (Haiku)")

            # ── Step 3: Risk Manager — Sonnet (synthesis + verdict) ────────────
            memo = _llm(
                system=_SYSTEM_STRICT,
                user=_RISK_PROMPT.format(
                    symbol=symbol,
                    bull_case=bull,
                    bear_case=bear,
                    data_json=data_json,
                ),
                model=CLAUDE_MODEL_RISK,
                max_tokens=8000,
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
        "sector", "industry", "company_description",
        "beta", "forward_pe",
        "short_pct_float", "short_ratio",
        "next_earnings_date", "days_to_earnings",
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
    """
    Parse APPROVED / WATCH / REJECTED from the Risk Manager output.

    Strategy (in priority order):
    1. Look for the explicit "VERDICT: WATCH" header line the prompt instructs.
    2. Look for a line that STARTS with the verdict word (e.g. "WATCH — The bull…").
    3. Find whichever verdict word appears FIRST in the text (position-based,
       avoids false positives like "prevents APPROVED status").
    """
    import re
    upper = memo.upper()

    # 1. Explicit header: "VERDICT: WATCH" or "VERDICT: **WATCH**"
    m = re.search(r'VERDICT\s*:\s*\**(APPROVED|REJECTED|WATCH)\**', upper)
    if m:
        return m.group(1)

    # 2. Line starts with the verdict word followed by space/dash/newline
    m = re.search(r'^\s*(APPROVED|REJECTED|WATCH)\s*[—–\-\s]', upper, re.MULTILINE)
    if m:
        return m.group(1)

    # 3. Earliest position in text (avoids picking up a word that appears
    #    in a different context later, e.g. "prevents APPROVED status")
    positions = {v: upper.find(v) for v in ("APPROVED", "REJECTED", "WATCH") if v in upper}
    if positions:
        return min(positions, key=positions.get)

    return "WATCH"

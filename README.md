# Stock Engine — Professional US Equity Recommendation System

A production-grade, quant-style stock recommendation engine that runs as a
nightly batch job on your Mac. Every weekday at 7 PM it scans the full US
equity market, applies multi-factor scoring, runs a multi-agent LLM debate,
and delivers ranked trade ideas to your phone via Telegram.

---

## Architecture

```
 7 PM Trigger (macOS launchd)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1 — Universe Builder                                          │
│  S&P 500 + NASDAQ-100 + FMP Screener → ~500–1500 US equity tickers │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 2 — Hard Filter Gate                                          │
│  Minervini Stage-2 Trend Template + Liquidity (MCap > $1B)         │
│  Passes ~15–25% of universe                                         │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 3 — Fundamental Analysis (per-stock)                          │
│  ROIC · ROE · FCF Yield · D/E · EPS Growth · EPS Acceleration      │
│  PEG · EV/EBITDA · Insider Transactions · Analyst Targets           │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 4 — Technical Analysis (per-stock)                            │
│  EMA 20/50/150/200 · RSI · ATR · MACD · OBV · Volume               │
│  Entry Setup Detection:                                             │
│    RSI_PULLBACK · BREAKOUT · MACD_CROSSOVER · SEPA_PULLBACK         │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 5 — Composite Scorer (cross-sectional percentile ranking)     │
│  Quality 30% · Technical 28% · Momentum 25% · Value 17%            │
│  Ranks all filtered stocks against each other (no absolute cutoffs) │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 6 — FinBERT Sentiment (top 15 candidates only)               │
│  News from FMP + Finnhub + SEC 8-K filings                          │
│  Scored via HuggingFace ProsusAI/finbert → −1.0 to +1.0            │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 7 — Multi-Agent Debate (top 15 candidates)                    │
│  BULL ANALYST  → builds strongest upside case                       │
│  BEAR ANALYST  → attacks thesis, identifies all risks               │
│  RISK MANAGER  → adjudicates, issues APPROVED/WATCH/REJECTED        │
│                  calculates ATR-based stop-loss + targets            │
│  Powered by Claude Opus 4.6 (or GPT-4o as fallback)                 │
│  Zero hallucination: LLM only uses pre-computed numbers             │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 8 — Persist + Notify                                          │
│  PostgreSQL: saves recommendations, factor scores, agent memos      │
│  Telegram: sends full trade memo to your phone                      │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 9 — Paper Trade Resolution                                    │
│  Checks all open paper trades against latest prices                 │
│  Closes trades that hit target or stop-loss                         │
│  Computes running accuracy metrics (win rate, profit factor, etc.)  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Strategies Implemented

### Entry Setups (Technical)

| Setup | Logic | Inspired By |
|---|---|---|
| `SEPA_PULLBACK` | Price pulls back to 20 EMA while all 4 EMAs aligned | Mark Minervini |
| `RSI_PULLBACK`  | RSI resets to 38–55 in established uptrend | O'Neil / Elder |
| `BREAKOUT`      | 20-day high break with 150%+ volume | Turtle Trading / Darvas |
| `MACD_CROSSOVER`| MACD histogram turns positive in uptrend | Elder Triple Screen |

### Factor Scoring (Fundamental)

| Factor | Metrics | Inspired By |
|---|---|---|
| Quality (30%) | ROIC, ROE, FCF Yield, Net Margin, D/E | AQR QMJ, Greenblatt |
| Momentum (25%) | 12-1 month return, EPS acceleration | Jegadeesh-Titman, O'Neil |
| Technical (28%) | EMA alignment, RSI zone, volume | Minervini, Elder |
| Value (17%) | PEG, EV/EBITDA, P/FCF | Graham, Greenblatt |

### Hard Filters (Before Scoring)

- **Minervini Stage-2 Trend Template**: Price > 200d SMA, 150d SMA > 200d SMA,
  Price > 50d SMA, within 25% of 52-week high, 25%+ above 52-week low
- **Liquidity**: Market Cap > $1B, Volume > 500K shares/day, Price > $10
- **Exchange**: NYSE and NASDAQ only

---

## Quick Start

### 1. One-Click Mac Setup

```bash
cd /Users/vedantsaraswat/Documents/Vedant/Stock_Engine
chmod +x setup_mac.sh
./setup_mac.sh
```

This installs PostgreSQL, creates the database, sets up the Python virtual
environment, and optionally installs the launchd scheduler.

### 2. Configure API Keys

```bash
cp .env.example .env
open .env   # fill in your keys
```

| Key | Where to Get It | Cost |
|---|---|---|
| `FMP_API_KEY` | financialmodelingprep.com | Free tier available |
| `ANTHROPIC_API_KEY` | console.anthropic.com | Pay-per-use |
| `HF_API_TOKEN` | huggingface.co/settings/tokens | Free |
| `TELEGRAM_BOT_TOKEN` | @BotFather on Telegram | Free |
| `TELEGRAM_CHAT_ID` | @userinfobot on Telegram | Free |

### 3. Test on a Single Stock

```bash
source .venv/bin/activate
python scripts/test_single_stock.py AAPL
python scripts/test_single_stock.py NVDA --no-agent   # skip LLM (faster)
```

### 4. Run the Backtest

```bash
python scripts/run_backtest.py
python scripts/run_backtest.py --strategy RSI_PULLBACK --symbols AAPL MSFT NVDA
```

### 5. Run the Full Pipeline Manually

```bash
python -m pipeline.run_pipeline --dry-run   # no DB writes, no Telegram
python -m pipeline.run_pipeline             # full production run
```

### 6. Check Paper Trading Accuracy

```bash
python scripts/check_performance.py
python scripts/check_performance.py --days 30 --resolve
```

---

## Project Structure

```
Stock_Engine/
├── config/
│   └── settings.py              ← All constants + API key loading
├── src/
│   ├── data/
│   │   ├── fmp_client.py        ← Financial Modeling Prep API wrapper
│   │   └── news_client.py       ← Multi-source news aggregation
│   ├── models/
│   │   └── database.py          ← SQLAlchemy ORM (PostgreSQL)
│   ├── screener/
│   │   ├── universe.py          ← US equity universe builder
│   │   └── filter_engine.py     ← Minervini SEPA + liquidity gates
│   ├── factors/
│   │   ├── fundamental.py       ← ROIC, FCF, EPS, valuation metrics
│   │   ├── technical.py         ← Indicators + entry setup detection
│   │   ├── sentiment.py         ← FinBERT news sentiment engine
│   │   └── composite_scorer.py  ← Percentile-ranked composite score
│   ├── agents/
│   │   └── orchestrator.py      ← Bull / Bear / Risk Manager debate
│   ├── backtest/
│   │   ├── engine.py            ← Walk-forward simulation
│   │   └── metrics.py           ← Sharpe, drawdown, win rate
│   ├── paper_trading/
│   │   └── tracker.py           ← Log + auto-resolve paper trades
│   └── notifications/
│       └── telegram_bot.py      ← Telegram delivery
├── pipeline/
│   └── run_pipeline.py          ← Main 9-stage nightly orchestrator
├── scripts/
│   ├── init_db.py               ← Create database schema
│   ├── run_backtest.py          ← Standalone backtest runner
│   ├── test_single_stock.py     ← Single-ticker analysis
│   ├── check_performance.py     ← Paper trading dashboard
│   └── install_scheduler.sh     ← macOS launchd installer
├── setup_mac.sh                 ← One-click Mac setup
├── docker-compose.yml           ← PostgreSQL + Redis containers
├── Dockerfile                   ← Engine container
├── requirements.txt
└── .env.example
```

---

## Telegram Output Format

Each evening you receive one message per pick:

```
✅ #1  NVDA  |  Score: 87.3/100
Setup: SEPA_PULLBACK  |  Sentiment: Positive (+0.71)

Price Levels
  Current Price :  $134.20
  Entry Zone    :  $134.20
  Stop-Loss     :  $128.40   (2 × ATR below support)
  Short Target  :  $152.40   (+13.5%)
  Long Target   :  $170.60   (+27.2%)
  Risk/Reward   :  1 : 3.0

Full Analysis
[Full institutional-quality memo from Risk Manager Agent]
```

---

## Measuring Accuracy

The engine automatically tracks every recommendation as a paper trade.
Check your accuracy at any time:

```bash
python scripts/check_performance.py
```

Key metrics to monitor:
- **Win Rate > 50%** — decent; most pros run 45–65%
- **Profit Factor > 1.5** — strong edge (gross wins ÷ gross losses)
- **Expectancy > 0** — positive means the strategy makes money on average

---

## Performance Expectations (Backtest Targets)

Based on the implemented strategies (RSI Pullback + SEPA on quality stocks):

| Metric | Target | Industry Benchmark |
|---|---|---|
| CAGR | 15–25% | S&P 500: ~10% |
| Sharpe Ratio | > 1.0 | Hedge funds: ~0.7–1.5 |
| Max Drawdown | < 20% | S&P 500: −34% in 2020 |
| Win Rate | 50–60% | — |
| Profit Factor | > 1.5 | Break-even = 1.0 |

> These are backtested targets. Past performance is not indicative of future results.
> Always use this system as a research tool, not as sole investment advice.

---

## Extending the Engine

- **Add a new entry setup**: edit `src/factors/technical.py → _detect_entry_setup()`
- **Change factor weights**: edit `config/settings.py → FACTOR_WEIGHTS`
- **Add a new data source**: add methods to `src/data/fmp_client.py` or create a new client
- **Change LLM provider**: set `LLM_PROVIDER=openai` in `.env`
- **Run intraday**: reduce `StartCalendarInterval` in the launchd plist to hourly

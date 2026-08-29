"""
Central configuration module.
All constants live here so the rest of the codebase imports from one place.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ─────────────────────────────────────────────────────────────────
FMP_API_KEY        = os.getenv("FMP_API_KEY", "")
FINNHUB_API_KEY    = os.getenv("FINNHUB_API_KEY", "")
ANTHROPIC_API_KEY      = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_WORKSPACE_ID = os.getenv("ANTHROPIC_WORKSPACE_ID", "")   # required for identity-linked keys
OPENAI_API_KEY         = os.getenv("OPENAI_API_KEY", "")
HF_API_TOKEN       = os.getenv("HF_API_TOKEN", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Environment detection ─────────────────────────────────────────────────────
IS_CI = bool(os.getenv("GITHUB_ACTIONS"))   # True when running in GitHub Actions

# ── Database ──────────────────────────────────────────────────────────────────
_default_db = (
    "sqlite:///data/quant_engine.db"           # GitHub Actions: file-based
    if IS_CI else
    "postgresql://engine_admin@localhost:5432/quant_engine"   # local Mac
)
DATABASE_URL = os.getenv("DATABASE_URL", _default_db)

# ── Portfolio / Paper-Trading Budget ─────────────────────────────────────────
TOTAL_CAPITAL       = float(os.getenv("TOTAL_CAPITAL", "150000"))  # $150k
RISK_PCT_PER_TRADE  = 0.01    # risk 1% of equity per trade
MAX_POSITION_PCT    = 0.05    # max 5% of equity in any single position
INTRADAY_CLOSE_HOUR = 15      # 3 PM ET — auto-close all intraday shorts

# ── LLM — Smart Model Routing ────────────────────────────────────────────────
# Each agent uses the cheapest model capable of its specific task.
# Bull + Bear agents format pre-computed data → Haiku is sufficient.
# Risk Manager synthesises the full debate → Sonnet for stronger reasoning.
# No task in this pipeline requires Opus.
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "anthropic")   # "anthropic" | "openai"

# Bull Agent: just formats provided JSON into a structured upside case
CLAUDE_MODEL_BULL   = os.getenv("CLAUDE_MODEL_BULL",   "claude-haiku-4-5")

# Bear Agent: same — structured formatting of risk factors from JSON
CLAUDE_MODEL_BEAR   = os.getenv("CLAUDE_MODEL_BEAR",   "claude-haiku-4-5")

# Risk Manager: adjudicates debate, writes final memo — needs stronger reasoning
CLAUDE_MODEL_RISK   = os.getenv("CLAUDE_MODEL_RISK",   "claude-sonnet-4-5")

OPENAI_MODEL        = "gpt-4o-mini"   # OpenAI fallback (cheaper equivalent)

# ── Pipeline Sizing ───────────────────────────────────────────────────────────
UNIVERSE_SIZE          = int(os.getenv("UNIVERSE_SIZE", "500"))
MAX_AGENT_CANDIDATES   = int(os.getenv("MAX_AGENT_CANDIDATES", "15"))
MAX_FINAL_PICKS        = int(os.getenv("MAX_FINAL_PICKS", "10"))

# ── Hard Liquidity / Universe Filters ────────────────────────────────────────
MIN_MARKET_CAP   = 1_000_000_000   # $1 billion
MIN_DAILY_VOLUME = 500_000          # 500 K shares/day
MIN_PRICE        = 10.0             # avoids micro-cap noise

# ── Multi-Factor Weights (must sum to 1.0) ───────────────────────────────────
FACTOR_WEIGHTS = {
    "quality":   0.30,   # ROIC, ROE, FCF yield, D/E, margins
    "technical": 0.28,   # EMA trend, RSI setup, volume, ATR
    "momentum":  0.25,   # 12-1 month price momentum, EPS revisions
    "value":     0.17,   # PEG, EV/EBITDA vs sector, P/FCF
}

# ── Fundamental Thresholds ────────────────────────────────────────────────────
MIN_ROIC           = 10.0   # % – minimum return on invested capital
MIN_ROE            = 8.0    # % – minimum return on equity
MAX_DEBT_TO_EQUITY = 2.5    # hard ceiling
MIN_FCF_YIELD      = 0.01   # 1 % minimum free cash flow yield

# ── EPS Acceleration (CAN SLIM C & A) ────────────────────────────────────────
MIN_QUARTERLY_EPS_GROWTH = 15.0   # % YoY for most recent quarter
MIN_ANNUAL_EPS_GROWTH    = 15.0   # % for trailing 12 months

# ── Technical Settings ────────────────────────────────────────────────────────
EMA_FAST          = 20
EMA_MID           = 50
EMA_SLOW          = 150
EMA_TREND         = 200
RSI_PERIOD        = 14
ATR_PERIOD        = 14
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9
BB_PERIOD         = 20
BB_STD            = 2.0
OBV_PERIOD        = 20      # OBV EMA smoothing

# ── Entry Signal Thresholds ───────────────────────────────────────────────────
# Setup 1: Pullback in established uptrend
RSI_PULLBACK_LOW    = 35
RSI_PULLBACK_HIGH   = 55

# Setup 2: Momentum breakout
RSI_BREAKOUT_MIN       = 55
VOLUME_BREAKOUT_MULT   = 1.5    # 150 % of 20-day avg volume

# ── Risk Management ───────────────────────────────────────────────────────────
ATR_STOP_MULTIPLIER   = 2.0    # stop-loss = entry - (2 × ATR)
ATR_TARGET_MULT_SHORT = 3.0    # short-term target (1:3 R/R minimum)
ATR_TARGET_MULT_LONG  = 6.0    # long-term target

# ── Minervini SEPA Trend Template ────────────────────────────────────────────
SEPA_MAX_FROM_52W_HIGH = 0.25   # price must be within 25 % of 52-week high
SEPA_MIN_FROM_52W_LOW  = 0.25   # price must be at least 25 % above 52-week low

# ── Momentum Lookback ─────────────────────────────────────────────────────────
MOMENTUM_LOOKBACK  = 252    # 12 months
MOMENTUM_SKIP      = 21     # skip most recent month (reversion avoidance)

# ── Backtesting ──────────────────────────────────────────────────────────────
BACKTEST_START_DATE = "2019-01-01"
BACKTEST_END_DATE   = "2024-12-31"
INITIAL_CAPITAL     = 100_000.0
COMMISSION_PCT      = 0.001     # 0.1 % per side (realistic for retail)
SLIPPAGE_PCT        = 0.001     # 0.1 % market impact

# Walk-forward windows (years)
WF_TRAIN_YEARS = 3
WF_TEST_YEARS  = 1

# ── Sector ETF Benchmarks ─────────────────────────────────────────────────────
SECTOR_ETFS = {
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Financials":             "XLF",
    "Consumer Discretionary": "XLY",
    "Industrials":            "XLI",
    "Energy":                 "XLE",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
    "Consumer Staples":       "XLP",
    "Communication Services": "XLC",
}
MARKET_BENCHMARK = "SPY"

# ── HuggingFace FinBERT Inference ────────────────────────────────────────────
HF_FINBERT_URL = (
    "https://api-inference.huggingface.co/models/ProsusAI/finbert"
)

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE  = "logs/engine.log"

"""
SQLAlchemy ORM models for persistent storage.

Tables:
  - tickers          — master list of US equities in universe
  - daily_price      — adjusted daily OHLCV
  - factor_scores    — per-stock computed factor scores
  - pipeline_runs    — one row per pipeline execution
  - recommendations  — final ranked picks with entry levels
  - paper_trades     — simulated trade log (nightly pipeline)
  - performance_stats— weekly accuracy snapshots
  - positions        — live paper positions (intraday engine, $150k budget)
  - daily_pnl        — one row per trading day: P&L summary
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Text, Boolean,
    DateTime, Date, JSON, ForeignKey, UniqueConstraint, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from config.settings import DATABASE_URL, IS_CI

Base = declarative_base()

# ── ORM Models ────────────────────────────────────────────────────────────────

class Ticker(Base):
    __tablename__ = "tickers"
    id          = Column(Integer, primary_key=True)
    symbol      = Column(String(12), unique=True, nullable=False, index=True)
    name        = Column(String(200))
    sector      = Column(String(100))
    industry    = Column(String(150))
    exchange    = Column(String(20))
    market_cap  = Column(Float)
    active      = Column(Boolean, default=True)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    factor_scores   = relationship("FactorScore", back_populates="ticker_ref",
                                   cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="ticker_ref",
                                   cascade="all, delete-orphan")
    paper_trades    = relationship("PaperTrade", back_populates="ticker_ref",
                                   cascade="all, delete-orphan")
    positions       = relationship("Position", back_populates="ticker_ref",
                                   cascade="all, delete-orphan")


class DailyPrice(Base):
    __tablename__ = "daily_price"
    __table_args__ = (UniqueConstraint("symbol", "date"),)
    id         = Column(Integer, primary_key=True)
    symbol     = Column(String(12), nullable=False, index=True)
    date       = Column(Date, nullable=False, index=True)
    open       = Column(Float)
    high       = Column(Float)
    low        = Column(Float)
    close      = Column(Float)
    adj_close  = Column(Float)
    volume     = Column(Float)


class FactorScore(Base):
    __tablename__ = "factor_scores"
    __table_args__ = (UniqueConstraint("symbol", "score_date"),)
    id              = Column(Integer, primary_key=True)
    symbol          = Column(String(12), ForeignKey("tickers.symbol"), nullable=False)
    score_date      = Column(Date, nullable=False)

    roic            = Column(Float)
    roe             = Column(Float)
    fcf_yield       = Column(Float)
    debt_to_equity  = Column(Float)
    net_margin      = Column(Float)
    eps_growth_qoq  = Column(Float)
    eps_growth_yoy  = Column(Float)
    rev_growth_yoy  = Column(Float)
    peg_ratio       = Column(Float)
    ev_ebitda       = Column(Float)
    price_momentum  = Column(Float)
    rsi             = Column(Float)
    atr             = Column(Float)
    ema_alignment   = Column(Float)

    quality_score   = Column(Float)
    momentum_score  = Column(Float)
    value_score     = Column(Float)
    technical_score = Column(Float)
    composite_score = Column(Float)

    entry_setup     = Column(String(50))
    sentiment_score = Column(Float)

    ticker_ref = relationship("Ticker", back_populates="factor_scores")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    id              = Column(Integer, primary_key=True)
    run_at          = Column(DateTime, default=datetime.utcnow)
    run_mode        = Column(String(20), default="nightly")  # morning/midday/eod/nightly
    universe_size   = Column(Integer)
    candidates_out  = Column(Integer)
    final_picks     = Column(Integer)
    duration_sec    = Column(Float)
    errors          = Column(Text)
    status          = Column(String(20), default="ok")


class Recommendation(Base):
    __tablename__ = "recommendations"
    id              = Column(Integer, primary_key=True)
    run_id          = Column(Integer, ForeignKey("pipeline_runs.id"))
    symbol          = Column(String(12), ForeignKey("tickers.symbol"), nullable=False)
    rec_date        = Column(Date, nullable=False)

    entry_price     = Column(Float)
    stop_loss       = Column(Float)
    target_short    = Column(Float)
    target_long     = Column(Float)
    risk_reward     = Column(Float)
    atr             = Column(Float)

    composite_score = Column(Float)
    sentiment_score = Column(Float)
    bull_thesis     = Column(Text)
    bear_risks      = Column(Text)
    risk_verdict    = Column(String(20))
    full_memo       = Column(Text)

    horizon_short   = Column(String(30))
    horizon_long    = Column(String(30))
    created_at      = Column(DateTime, default=datetime.utcnow)

    ticker_ref = relationship("Ticker", back_populates="recommendations")


class PaperTrade(Base):
    """Nightly pipeline paper trades (original tracker)."""
    __tablename__ = "paper_trades"
    id              = Column(Integer, primary_key=True)
    symbol          = Column(String(12), ForeignKey("tickers.symbol"), nullable=False)
    rec_id          = Column(Integer, ForeignKey("recommendations.id"))
    open_date       = Column(Date, nullable=False)
    close_date      = Column(Date)
    entry_price     = Column(Float)
    stop_loss       = Column(Float)
    target_short    = Column(Float)
    exit_price      = Column(Float)
    pnl_pct         = Column(Float)
    hit_target      = Column(Boolean)
    hit_stop        = Column(Boolean)
    status          = Column(String(20), default="OPEN")

    ticker_ref = relationship("Ticker", back_populates="paper_trades")


class PerformanceStats(Base):
    __tablename__ = "performance_stats"
    id              = Column(Integer, primary_key=True)
    computed_at     = Column(DateTime, default=datetime.utcnow)
    total_trades    = Column(Integer)
    open_trades     = Column(Integer)
    win_rate        = Column(Float)
    avg_win_pct     = Column(Float)
    avg_loss_pct    = Column(Float)
    profit_factor   = Column(Float)
    sharpe_ratio    = Column(Float)
    max_drawdown    = Column(Float)
    expectancy      = Column(Float)
    notes           = Column(Text)


class Position(Base):
    """
    Live paper positions managed by the intraday engine.
    One row per open position; closed rows are kept for audit.
    """
    __tablename__ = "positions"
    id              = Column(Integer, primary_key=True)
    symbol          = Column(String(12), ForeignKey("tickers.symbol"), nullable=False)
    direction       = Column(String(5), nullable=False)   # "LONG" | "SHORT"
    trade_type      = Column(String(10), default="swing") # "swing" | "intraday"
    shares          = Column(Float, nullable=False)
    entry_price     = Column(Float, nullable=False)
    stop_loss       = Column(Float)
    target          = Column(Float)
    entry_date      = Column(Date, nullable=False)
    entry_time      = Column(String(8))                   # HH:MM ET
    close_date      = Column(Date)
    close_price     = Column(Float)
    pnl_dollars     = Column(Float)
    pnl_pct         = Column(Float)
    is_open         = Column(Boolean, default=True, index=True)
    composite_score = Column(Float)
    verdict         = Column(String(20))
    reason          = Column(Text)                        # summary of why we entered

    ticker_ref = relationship("Ticker", back_populates="positions")


class DailyPnL(Base):
    """
    One row per trading day — tracks running budget and P&L.
    """
    __tablename__ = "daily_pnl"
    __table_args__ = (UniqueConstraint("trade_date"),)
    id              = Column(Integer, primary_key=True)
    trade_date      = Column(Date, nullable=False)
    realized_pnl    = Column(Float, default=0.0)   # closed trades today
    unrealized_pnl  = Column(Float, default=0.0)   # open positions MTM
    total_pnl       = Column(Float, default=0.0)   # realized + unrealized
    cumulative_pnl  = Column(Float, default=0.0)   # all-time realized P&L
    cash            = Column(Float)                 # available cash
    equity          = Column(Float)                 # total portfolio value
    positions_count = Column(Integer, default=0)
    trades_entered  = Column(Integer, default=0)
    trades_exited   = Column(Integer, default=0)
    notes           = Column(Text)


# ── Session Factory ────────────────────────────────────────────────────────────

def get_engine():
    kwargs = {"pool_pre_ping": True}
    if DATABASE_URL.startswith("sqlite"):
        # SQLite doesn't support pool_pre_ping the same way; use StaticPool
        from sqlalchemy.pool import StaticPool
        kwargs = {
            "connect_args": {"check_same_thread": False},
            "poolclass": StaticPool,
        }
    return create_engine(DATABASE_URL, **kwargs)


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    """Create all tables (idempotent)."""
    import os
    if DATABASE_URL.startswith("sqlite"):
        os.makedirs("data", exist_ok=True)
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("Database schema initialised successfully.")

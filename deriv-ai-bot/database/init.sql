-- PostgreSQL Database Schema for Deriv AI Trading Bot
-- Target: DigitalOcean VPS (PostgreSQL 16)

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Bot Global Configuration
CREATE TABLE IF NOT EXISTS bot_configuration (
    key VARCHAR(64) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Seed default configuration if empty
INSERT INTO bot_configuration (key, value, description)
VALUES 
    ('mode', '"demo"', 'Current trading mode: demo or real'),
    ('trade_cycle_seconds', '45', 'Frequency of main scanning loop in seconds'),
    ('stake_mode', '"flat"', 'Staking mode: flat or martingale'),
    ('min_confidence', '0.80', 'Minimum signal confidence threshold for execution')
ON CONFLICT (key) DO NOTHING;

-- 2. Strategy Definitions
CREATE TABLE IF NOT EXISTS strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(64) UNIQUE NOT NULL,
    version VARCHAR(16) NOT NULL DEFAULT '1.0.0',
    enabled BOOLEAN DEFAULT TRUE,
    account_mode VARCHAR(16) NOT NULL CHECK (account_mode IN ('demo', 'real')),
    min_confidence NUMERIC(5, 4) DEFAULT 0.8000,
    max_stake NUMERIC(10, 2) DEFAULT 10.00,
    params JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO strategies (name, version, enabled, account_mode, min_confidence, max_stake)
VALUES 
    ('Digits_Default', '1.0.0', true, 'demo', 0.8000, 10.00),
    ('RiseFall_Default', '1.0.0', true, 'demo', 0.8000, 10.00)
ON CONFLICT (name) DO NOTHING;

-- 3. Signal History Log
CREATE TABLE IF NOT EXISTS signals (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    contract_type VARCHAR(32) NOT NULL,
    barrier VARCHAR(16),
    confidence NUMERIC(5, 4) NOT NULL,
    raw_confidence NUMERIC(5, 4) NOT NULL,
    mor_score NUMERIC(5, 2),
    trend_strength NUMERIC(5, 4),
    ev NUMERIC(6, 4),
    family VARCHAR(32),
    horizon VARCHAR(16),
    status VARCHAR(32) DEFAULT 'generated', -- generated, selected, filtered_ev, filtered_corr, executed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_created ON signals (symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status);

-- 4. Trades Execution Log
CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    contract_id BIGINT UNIQUE NOT NULL,
    account_id VARCHAR(64) NOT NULL DEFAULT 'unknown',
    account_mode VARCHAR(16) NOT NULL CHECK (account_mode IN ('demo', 'real')),
    symbol VARCHAR(32) NOT NULL,
    contract_type VARCHAR(32) NOT NULL,
    barrier VARCHAR(16),
    stake NUMERIC(10, 2) NOT NULL,
    pnl NUMERIC(10, 2),
    profit NUMERIC(10, 2),
    buy_price NUMERIC(10, 2),
    sell_price NUMERIC(10, 2),
    confidence NUMERIC(5, 4) NOT NULL,
    confidence_level VARCHAR(16) CHECK (confidence_level IN ('LOW', 'MEDIUM', 'HIGH')),
    mor_score NUMERIC(5, 2),
    family VARCHAR(32),
    horizon VARCHAR(16),
    opened_at TIMESTAMP WITH TIME ZONE NOT NULL,
    closed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(32) NOT NULL, -- open, win, loss, failed, stale_reaped
    strategy_name VARCHAR(64) REFERENCES strategies(name),
    raw_payload JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_status ON trades (symbol, status);
CREATE INDEX IF NOT EXISTS idx_trades_closed_at ON trades (closed_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_opened_at ON trades (opened_at DESC);

-- 5. Learning Cycles & AI Audit Reports
CREATE TABLE IF NOT EXISTS learning_cycles (
    id BIGSERIAL PRIMARY KEY,
    cycle_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    trades_analyzed INT NOT NULL,
    overall_win_rate NUMERIC(5, 2),
    overall_calibration_error NUMERIC(5, 4),
    top_patterns JSONB DEFAULT '[]',
    banned_setups JSONB DEFAULT '[]',
    boosted_setups JSONB DEFAULT '[]',
    confidence_adjustments JSONB DEFAULT '{}',
    audit_report JSONB DEFAULT '{}',
    deepseek_summary TEXT
);

CREATE INDEX IF NOT EXISTS idx_learning_cycles_ts ON learning_cycles (cycle_timestamp DESC);

-- 6. Market Data Snapshots (1-minute aggregates)
CREATE TABLE IF NOT EXISTS market_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    epoch BIGINT NOT NULL,
    open NUMERIC(14, 6),
    high NUMERIC(14, 6),
    low NUMERIC(14, 6),
    close NUMERIC(14, 6),
    tick_count INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_market_data_sym_epoch ON market_data (symbol, epoch DESC);

-- 7. Performance Metrics Rollup
CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    period VARCHAR(16) NOT NULL CHECK (period IN ('hourly', 'daily', 'weekly')),
    account_mode VARCHAR(16) NOT NULL,
    total_trades INT NOT NULL,
    win_rate NUMERIC(5, 2) NOT NULL,
    net_profit NUMERIC(10, 2) NOT NULL,
    max_drawdown NUMERIC(5, 2) NOT NULL,
    sharpe_ratio NUMERIC(6, 3),
    consecutive_losses INT NOT NULL,
    active_symbols INT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_perf_metrics_ts ON performance_metrics (timestamp DESC);

-- 8. Live Agent Health & Runtime States
CREATE TABLE IF NOT EXISTS agent_states (
    agent_name VARCHAR(64) PRIMARY KEY,
    enabled BOOLEAN DEFAULT TRUE,
    status VARCHAR(32) DEFAULT 'stopped', -- starting, running, paused, stopped, error
    weight NUMERIC(5, 4) DEFAULT 1.0000,
    win_rate NUMERIC(5, 2) DEFAULT 0.00,
    total_signals INT DEFAULT 0,
    successful_signals INT DEFAULT 0,
    last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Seed core default agents into agent_states
INSERT INTO agent_states (agent_name, enabled, status, weight)
VALUES 
    ('TrendAgent', true, 'running', 1.2000),
    ('VolatilityAgent', true, 'running', 1.0000),
    ('PatternAgent', true, 'running', 1.1000),
    ('ScalpingAgent', true, 'running', 1.0000),
    ('RiskAgent', true, 'running', 1.5000),
    ('ExecutionAgent', true, 'running', 1.0000),
    ('LearningAgent', true, 'running', 1.0000)
ON CONFLICT (agent_name) DO NOTHING;

-- 9. Agent Decision Votes Audit Log
CREATE TABLE IF NOT EXISTS agent_votes (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    contract_type VARCHAR(32) NOT NULL,
    agent_name VARCHAR(64) NOT NULL,
    confidence NUMERIC(5, 4) NOT NULL,
    weight NUMERIC(5, 4) NOT NULL,
    weighted_score NUMERIC(5, 4) NOT NULL,
    rationale TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_agent_votes_symbol_agent ON agent_votes (symbol, agent_name, created_at DESC);

-- 10. Multi-Account / Multi-Tenant Profiles
CREATE TABLE IF NOT EXISTS user_accounts (
    id SERIAL PRIMARY KEY,
    account_id VARCHAR(64) UNIQUE NOT NULL,
    user_id VARCHAR(64) NOT NULL DEFAULT 'default_user',
    account_mode VARCHAR(16) NOT NULL CHECK (account_mode IN ('demo', 'real')),
    currency VARCHAR(8) DEFAULT 'USD',
    balance NUMERIC(12, 2) DEFAULT 0.00,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO user_accounts (account_id, account_mode, currency, balance)
VALUES ('CR_DEMO_01', 'demo', 'USD', 10000.00)
ON CONFLICT (account_id) DO NOTHING;

-- 11. Deep Trade Memory & Feature Store (Dataset Generator)
CREATE TABLE IF NOT EXISTS trade_memory (
    id BIGSERIAL PRIMARY KEY,
    contract_id BIGINT UNIQUE NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    contract_type VARCHAR(32) NOT NULL,
    stake NUMERIC(10, 2) NOT NULL,
    pnl NUMERIC(10, 2),
    profit NUMERIC(10, 2),
    status VARCHAR(32) NOT NULL,
    confidence NUMERIC(5, 4) NOT NULL,
    ensemble_score NUMERIC(5, 4) NOT NULL,
    tick_history JSONB NOT NULL DEFAULT '[]',
    indicators_snapshot JSONB NOT NULL DEFAULT '{}',
    agent_votes JSONB NOT NULL DEFAULT '[]',
    feature_vector JSONB NOT NULL DEFAULT '{}',
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_trade_memory_sym_status ON trade_memory (symbol, status);
CREATE INDEX IF NOT EXISTS idx_trade_memory_opened ON trade_memory (opened_at DESC);



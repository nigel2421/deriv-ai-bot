# 🤖 Deriv AI Trading Bot — Evolutionary Multi-Agent System

An enterprise-grade, **self-improving** autonomous trading system for **Deriv.com** that watches **20 synthetic & FX markets** simultaneously using a **swarm of specialist AI agents** that compete, evolve, breed, and adapt in real-time.

> [!WARNING]
> Trading financial and synthetic instruments involves significant risk of capital loss. Always start in **demo** mode (`MODE=demo`) and thoroughly validate strategy performance before deploying real capital.

---

## 🧠 How The System Thinks

Rather than a single bot making decisions, this system runs a **council of specialist agents** that each analyse the market from a different perspective. They vote, get scored on their accuracy, and the best agents gain more influence over time.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    20 Market Watcher Sub-Agents                      │
│  R_10  R_25  R_50  R_75  R_100  1HZ10V … BOOM1000  CRASH500  JD50  │
└────────────────────────────┬─────────────────────────────────────────┘
                             │  live tick feeds
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  Specialist Agent Council (votes)                    │
│                                                                      │
│  TrendAgent  ·  VolatilityAgent  ·  PatternAgent  ·  RiskAgent      │
│  LearningAgent  ·  RLAgent (Q-Learning)  ·  ConsensusAgent          │
└────────────────────────────┬─────────────────────────────────────────┘
                             │  weighted signals
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│              Reputation Engine  (who do we trust most?)              │
│   Tracks last 100 trades per agent → dynamic vote weight (0.5x–1.8x)│
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│           Chief Strategy Agent  (Meta-Agent / The Boss)              │
│   Ranks agents · Disables underperformers · Promotes high-flyers     │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│              Market Regime Agent  (Is now a good time?)              │
│   TRENDING · SIDEWAYS_CHOP · HIGH_VOLATILITY · SLOW                 │
│   Gates which strategies are allowed to trade right now              │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│            Portfolio Manager Agent  (Hard Financial Rules)           │
│   1% risk per trade · Max daily loss veto · Max 3 concurrent trades  │
│   Min stake $1.00 · Dynamic position sizing from balance             │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ approved trade
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     Execution Agent → Deriv API                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Evolutionary Stages (What's Built)

### Stage 1 — Agent Reputation Engine ✅
Each specialist agent has a **rolling 100-trade win-rate** that directly controls how much weight their vote carries in the consensus.

| Accuracy (last 100 trades) | Vote Multiplier |
|---|---|
| ≥ 75% | **1.8x** — Elite performer |
| ≥ 70% | **1.5x** — Strong performer |
| ≥ 55% | **1.2x** — Solid performer |
| ≥ 45% | **0.9x** — Weak performer |
| < 45% | **0.5x** — On notice |

File: [`src/agents/reputation.py`](src/agents/reputation.py)

---

### Stage 2 — Chief Strategy Agent (Meta-Agent) ✅
The **Meta-Agent** monitors all specialist agents and autonomously manages the team:

- 🏆 **Promotes** agents with win-rate > 70% → boosts their weight
- 🔕 **Quarantines** agents with win-rate < 35% over 20+ trades → disabled
- 📋 **Live leaderboard** available on the dashboard

File: [`src/agents/chief_strategy_agent.py`](src/agents/chief_strategy_agent.py)

---

### Stage 3 — Trade Memory Database ✅
Every trade is stored in **PostgreSQL** with full context for learning:

```
Trade Record includes:
  ├── market / symbol
  ├── tick history snapshot
  ├── technical indicators (EMA, RSI, MACD)
  ├── every agent's vote & confidence level
  ├── consensus confidence score
  ├── market regime at execution time
  ├── stake size & position sizing rationale
  └── final result (profit/loss)
```

Table: `trade_memory` in [`database/init.sql`](database/init.sql)

---

### Stage 4 — Reinforcement Learning Agent ✅
A **Q-Learning agent** that learns from every trade outcome in real-time:

- State: `(market_regime, trend_direction, volatility_band)`
- Actions: CALL / PUT / SKIP
- Reward: profit/loss PnL signal
- Updates Q-table after every closed trade
- Gradually shifts from exploration → exploitation

File: [`src/agents/rl_agent.py`](src/agents/rl_agent.py)

---

### Stage 5 — Portfolio Manager Agent ✅
**Hard financial guardrails that no agent can override:**

| Rule | Limit |
|---|---|
| Min stake | $1.00 USD |
| Risk per trade | 1% of account balance |
| Max concurrent trades | 3 |
| Max daily loss | 3% of session start balance |
| Exposure control | If daily loss limit hit → full trading halt |

File: [`src/agents/portfolio_manager_agent.py`](src/agents/portfolio_manager_agent.py)

---

### Stage 6 — Market Regime Agent ✅
Classifies the current market environment and **gates strategies accordingly**:

| Regime | Detected When | Action |
|---|---|---|
| `TRENDING` | ADX high, EMA aligned | Enable trend strategies, disable range agents |
| `SIDEWAYS_CHOP` | Low ADX, tight range | Enable range/digit strategies |
| `HIGH_VOLATILITY` | ATR spike > 2σ | Reduce stake size, tighten filters |
| `SLOW` | Low tick velocity | Skip or wait |

File: [`src/agents/market_regime_agent.py`](src/agents/market_regime_agent.py)

---

### Stage 7 — Agent Breeding System (Genetic Algorithm) ✅
Agents don't just learn — they **breed**. The best-performing strategy parameters are combined to create offspring strategies:

```
Parent A (TrendAgent DNA)    +    Parent B (PatternAgent DNA)
    high EMA weight               strong RSI filter
              ↓  crossover  +  mutation
         Child Strategy DNA
             ↓
    New specialist agent spawned automatically
```

- Fitness score = win-rate × average PnL
- Top performers breed every N cycles
- Mutations add random parameter variance
- Poor offspring are culled automatically

File: [`src/agents/breeding_system.py`](src/agents/breeding_system.py)

---

### Stage 8 — Infrastructure Agent ✅
Continuously monitors the health of the **entire system**:

| Monitored Resource | Alert Threshold | Action |
|---|---|---|
| CPU usage | > 85% | Alert + throttle |
| RAM usage | > 90% | Alert |
| Disk space | > 90% | Alert |
| Redis connection | Ping fail | Auto-reconnect |
| PostgreSQL connection | Query fail | Auto-reconnect |

File: [`src/agents/infrastructure_agent.py`](src/agents/infrastructure_agent.py)

---

### Stage 9 — Knowledge Graph (Neo4j) 🔜 Planned
The next evolution. Instead of just storing trade rows, the system will build a **relationship graph**:

```
Trade_1254
     ├── executed_on      → R_100
     ├── detected_by      → TrendAgent
     ├── detected_by      → PatternAgent
     ├── used_strategy    → StrategyDNA_89
     ├── market_regime    → TRENDING
     └── outcome          → profit $3.20
```

This enables pattern queries like:
> *"Show me all trades where TrendAgent + PatternAgent agreed during a TRENDING regime on R_100 — what was the historical win rate?"*

---

## 🌐 Monitored Markets (20 Total)

| Family | Symbols | Notes |
|---|---|---|
| Classic Synthetics | `R_10`, `R_25`, `R_50`, `R_75`, `R_100` | Core volatility indices |
| 1-Second (1Hz) | `1HZ10V`, `1HZ25V`, `1HZ50V`, `1HZ75V`, `1HZ100V` | High-frequency scans |
| Boom Indices | `BOOM1000`, `BOOM500` | CALL only — spike capture |
| Crash Indices | `CRASH1000`, `CRASH500` | PUT only — crash capture |
| Jump Indices | `JD10`, `JD25`, `JD50` | High-volatility jumps |
| FX Pairs | `frxEURUSD`, `frxGBPUSD` | London/NY session gated |
| Step Index | `STPIDX` | Fixed 0.1-pip steps |

Each market has its own dedicated **MarketWatcherSubAgent** streaming live ticks.

File: [`src/agents/market_subagent.py`](src/agents/market_subagent.py)

---

## ⚡ Quick Start (Local)

### Prerequisites
- Python 3.10+
- Deriv API Token — [get one here](https://app.deriv.com/account/api-token)

### Install & Run

```powershell
# 1. Clone the repo
git clone https://github.com/nigel2421/deriv-ai-bot.git
cd deriv-ai-bot

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and fill in your .env file
cp .env.example .env
# Edit .env — add DERIV_API_TOKEN, DERIV_APP_ID at minimum

# 5. Start the trading engine
python -m src.main

# 6. Start the live dashboard (separate terminal)
python -m uvicorn src.cloud_app:app --host 0.0.0.0 --port 8080
```

Then open **http://localhost:8080** in your browser.

---

## 📊 Live Dashboard (http://localhost:8080)

The dashboard shows a real-time card for every active component:

| Panel | What it shows |
|---|---|
| 💰 **Account** | Live balance, open trades, daily PnL |
| 🤖 **Agent Cards** | One card per agent — status, accuracy, vote weight |
| 👑 **Meta-Agent Leaderboard** | Chief Strategy Agent rankings & actions |
| 📈 **Market Regimes** | Current regime classification per symbol family |
| 🧬 **Breeding Lab** | Active StrategyDNA pool, fitness scores, generations |
| 🛡️ **Portfolio Manager** | Current risk exposure, daily loss tracker |
| 🖥️ **Infrastructure Health** | CPU, RAM, Redis, Postgres — live telemetry |

---

## 🗂️ Project Structure

```text
deriv-ai-bot/
├── config/
│   └── settings.py              Global env vars, MIN_STAKE=$1.00, risk rules
├── database/
│   └── init.sql                 PostgreSQL schema (trade_memory table + indexes)
├── docs/
│   ├── ARCHITECTURE.md          Full system architecture deep-dive
│   ├── AGENTS.md                Per-agent reference guide
│   └── IMPLEMENTATION_PLAN.md  Original phased delivery plan
├── src/
│   ├── cloud_app.py             Starlette dashboard + REST API
│   ├── orchestrator.py          Main trade loop coordinator
│   ├── main.py                  Entry point
│   ├── agents/
│   │   ├── manager.py           AgentManager — coordinates all agents
│   │   ├── base_agent.py        BaseAgent interface
│   │   ├── bus.py               Redis event bus (pub/sub)
│   │   ├── market_subagent.py   20 Market Watcher Sub-Agents
│   │   ├── reputation.py        Stage 1: Rolling accuracy → vote weight
│   │   ├── chief_strategy_agent.py  Stage 2: Meta-Agent leaderboard
│   │   ├── rl_agent.py          Stage 4: Q-Learning reinforcement agent
│   │   ├── portfolio_manager_agent.py  Stage 5: Hard financial rules
│   │   ├── market_regime_agent.py      Stage 6: Regime classification
│   │   ├── breeding_system.py   Stage 7: Genetic Algorithm breeding
│   │   ├── infrastructure_agent.py     Stage 8: System health monitor
│   │   ├── trend_agent.py       Specialist: EMA/MACD trend signals
│   │   ├── volatility_agent.py  Specialist: ATR/Bollinger signals
│   │   ├── pattern_agent.py     Specialist: Candlestick pattern signals
│   │   ├── risk_agent.py        Specialist: Risk-weighted veto signals
│   │   ├── learning_agent.py    Specialist: Historical win-rate signals
│   │   ├── consensus_agent.py   Aggregates all votes into a decision
│   │   └── execution_agent.py   Sends approved trades to Deriv API
│   ├── api/
│   │   ├── deriv_client.py      WebSocket connection to Deriv
│   │   └── trade_executor.py    Proposal + Buy pipeline
│   └── database/
│       └── db.py                PostgreSQL async client
├── tests/
│   ├── test_evolutionary_system.py   Stages 1-4 unit tests
│   └── test_advanced_stages.py       Stages 5-8 unit tests
├── docker-compose.yml           PostgreSQL + Redis + Bot services
├── requirements.txt             Python dependencies
└── .env.example                 Environment variable template
```

---

## 🔑 Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `DERIV_APP_ID` | `33R2Z6MTElnIWrId8aH3m` | Your Deriv App ID |
| `DERIV_API_TOKEN` | *(required)* | Demo or real API token |
| `MODE` | `demo` | `demo` or `real` |
| `EXECUTE_TRADES` | `true` | Set `false` for paper-only proposals |
| `MIN_STAKE` | `1.00` | Minimum trade stake in USD |
| `MAX_OPEN_TRADES` | `20` | Max concurrent positions (1 per market sub-agent) |
| `ENABLE_POSTGRES` | `true` | Enable trade memory storage |
| `ENABLE_REDIS` | `true` | Enable Redis agent event bus |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `DATABASE_URL` | `postgresql://...` | Postgres connection string |
| `TELEGRAM_BOT_TOKEN` | *(optional)* | For mobile alerts |
| `TELEGRAM_CHAT_ID` | *(optional)* | Your Telegram chat ID |

---

## 🧪 Running Tests

```powershell
# All tests
pytest

# Specific stage tests
pytest tests/test_evolutionary_system.py -v   # Stages 1-4
pytest tests/test_advanced_stages.py -v       # Stages 5-8
```

---

## 🛡️ Safety Rules

1. **Always start in demo mode** — `MODE=demo` in `.env`
2. **$1.00 minimum stake** — hard-coded floor, Deriv minimum is $0.50
3. **1% risk per trade** — Portfolio Manager enforces this mathematically
4. **Max 3% daily loss** — full halt if breached, no override possible
5. **No martingale by default** — flat stake sizing is the default

---

## 📜 License & Disclaimer

Open-source for educational and research purposes. You are solely responsible for managing your financial risk, API tokens, and compliance with Deriv's terms of service.

# 🤖 Agent Reference Guide

This document describes every agent in the system — what it does, what inputs it takes, what signals it outputs, and how it interacts with other agents.

---

## Agent Lifecycle Overview

```
Boot
 │
 ├─► AgentManager registers all agents
 │
 ├─► MarketWatcherSubAgents start streaming 20 live tick feeds
 │
 ├─► On each tick cycle:
 │       1. Market data pushed to all enabled specialist agents
 │       2. Each agent evaluates → emits AgentSignal (CALL / PUT / SKIP)
 │       3. ReputationEngine applies dynamic vote weights
 │       4. ConsensusAgent aggregates weighted votes
 │       5. MarketRegimeAgent checks if regime allows this trade type
 │       6. PortfolioManagerAgent checks financial guardrails
 │       7. ExecutionAgent sends to Deriv API if approved
 │       8. Trade outcome recorded → reputation updated → RL agent learns
 │
 └─► ChiefStrategyAgent runs periodic review:
         - Ranks agents by performance
         - Promotes / quarantines as needed
         - Triggers AgentBreedingSystem to evolve new DNA
```

---

## 🏗️ Infrastructure Agents

### AgentManager — *The CEO*
**File:** [`src/agents/manager.py`](../src/agents/manager.py)

Coordinates the entire agent ecosystem. Owns the agent registry, dispatches market context to all enabled agents, collects signals, and manages agent lifecycle (enable/disable/weight).

| Property | Value |
|---|---|
| Role | Orchestrator |
| Publishes to | `TOPIC_SIGNALS`, `TOPIC_CONSENSUS` |
| Subscribes to | `TOPIC_CONTROL` |
| Dynamic control | Yes — enable/disable/reweight any agent at runtime |

**API control commands** (sent via Redis `TOPIC_CONTROL`):
```json
{ "action": "enable_agent",  "target_agent": "TrendAgent" }
{ "action": "disable_agent", "target_agent": "VolatilityAgent" }
{ "action": "set_weight",    "target_agent": "PatternAgent", "weight": 1.5 }
{ "action": "stop_all" }
```

---

### AgentEventBus — *The Nervous System*
**File:** [`src/agents/bus.py`](../src/agents/bus.py)

Redis pub/sub backbone connecting all agents. Every signal, tick, alert, and control message flows through this bus.

| Topic | Purpose |
|---|---|
| `TOPIC_TICKS` | Raw market tick data from sub-agents |
| `TOPIC_SIGNALS` | Individual agent vote signals |
| `TOPIC_CONSENSUS` | Final aggregated decision |
| `TOPIC_CONTROL` | Runtime control commands |
| `TOPIC_ALERTS` | Infrastructure & risk alerts |

---

## 👁️ Market Watchers (20 Sub-Agents)

### MarketWatcherSubAgent
**File:** [`src/agents/market_subagent.py`](../src/agents/market_subagent.py)

One dedicated sub-agent per market symbol, streaming live ticks and publishing them to the event bus. There are **20 instances** running simultaneously.

| Symbol Family | Count | Symbols |
|---|---|---|
| Classic Volatility | 5 | R_10, R_25, R_50, R_75, R_100 |
| 1Hz Volatility | 5 | 1HZ10V, 1HZ25V, 1HZ50V, 1HZ75V, 1HZ100V |
| Boom | 2 | BOOM1000, BOOM500 |
| Crash | 2 | CRASH1000, CRASH500 |
| Jump | 3 | JD10, JD25, JD50 |
| FX | 2 | frxEURUSD, frxGBPUSD |
| Step | 1 | STPIDX |

Each sub-agent:
1. Connects to Deriv WebSocket API
2. Subscribes to tick feed for its symbol
3. Publishes enriched tick data to `TOPIC_TICKS`
4. Maintains a rolling tick window for analysis

---

## 🧠 Specialist Analysis Agents

All specialist agents inherit from `BaseAgent` and follow this interface:
```python
async def evaluate(context: Dict[str, Any]) -> List[AgentSignal]
```

Each `AgentSignal` contains:
- `direction`: `"CALL"` | `"PUT"` | `"SKIP"`
- `confidence`: float `0.0 – 1.0`
- `weight`: float (set by ReputationEngine)
- `reasoning`: str (human-readable explanation)

---

### TrendAgent
**File:** [`src/agents/trend_agent.py`](../src/agents/trend_agent.py)

Analyses directional market momentum using EMA crossovers and MACD.

| Input | Output |
|---|---|
| 50-tick EMA, 200-tick EMA, MACD signal | CALL (bullish), PUT (bearish), SKIP (no trend) |

**Best regime:** `TRENDING`
**Worst regime:** `SIDEWAYS_CHOP`

---

### VolatilityAgent
**File:** [`src/agents/volatility_agent.py`](../src/agents/volatility_agent.py)

Monitors market volatility expansion/contraction using ATR and Bollinger Bands.

| Input | Output |
|---|---|
| ATR(14), Bollinger Band width, tick velocity | CALL/PUT on breakout, SKIP in compression |

**Best regime:** `HIGH_VOLATILITY` breakouts
**Worst regime:** `SLOW` markets

---

### PatternAgent
**File:** [`src/agents/pattern_agent.py`](../src/agents/pattern_agent.py)

Detects repeating price action patterns in the tick/candle stream.

| Input | Output |
|---|---|
| OHLC candle data, tick sequences | Signal based on recognised pattern confidence |

**Patterns recognised:** Engulfing, hammer, doji, three-bar reversal, trend continuation sequences

---

### RiskAgent
**File:** [`src/agents/risk_agent.py`](../src/agents/risk_agent.py)

Acts as a **veto agent** — can cancel proposed trades based on risk conditions.

| Condition | Action |
|---|---|
| Recent loss streak ≥ 3 | SKIP for low-confidence setups |
| Current market in drawdown | Reduce confidence weight |
| Duplicate symbol/type in last 2 trades | SKIP (diversity rule) |

---

### LearningAgent
**File:** [`src/agents/learning_agent.py`](../src/agents/learning_agent.py)

Uses **historical win-rate data per setup** to adjust signal confidence.

- Tracks `symbol|contract_type|barrier` win-rate over time
- Boosts confidence on setups with > 60% historical win-rate
- Suppresses signals for setups with < 40% historical win-rate
- Integrates with PostgreSQL `trade_memory` table

---

### RLAgent — Q-Learning Agent
**File:** [`src/agents/rl_agent.py`](../src/agents/rl_agent.py)  *(Stage 4)*

A **Reinforcement Learning agent** that learns from every completed trade.

```
State Space:
  (market_regime, trend_direction, volatility_band)
  → ~48 discrete states

Action Space:
  CALL | PUT | SKIP

Reward Function:
  +profit  on win
  -loss    on loss
  -0.1     on SKIP (opportunity cost penalty)

Q-Table update:
  Q(s, a) ← Q(s, a) + α[r + γ·max Q(s', a') − Q(s, a)]
```

- Starts with high exploration (ε=0.9), decays to exploitation (ε=0.1) over time
- Q-table persisted to disk — survives restarts
- Publishes `rl_boost` modifier that adjusts final consensus confidence

---

### ConsensusAgent
**File:** [`src/agents/consensus_agent.py`](../src/agents/consensus_agent.py)

Aggregates all specialist agent signals into a single trade decision.

```
Weighted Vote Formula:
  total_call_weight  = Σ (signal.confidence × signal.weight) for CALL votes
  total_put_weight   = Σ (signal.confidence × signal.weight) for PUT votes

  consensus_conf = max(total_call_weight, total_put_weight) / total_weight

Decision: CALL if call_weight > put_weight AND consensus_conf ≥ MIN_CONFIDENCE
           PUT  if put_weight > call_weight AND consensus_conf ≥ MIN_CONFIDENCE
           SKIP otherwise
```

---

### ExecutionAgent
**File:** [`src/agents/execution_agent.py`](../src/agents/execution_agent.py)

Sends approved trade proposals to the Deriv API.

1. Receives approved `ConsensusDecision`
2. Requests proposal from Deriv (validates stake, checks barrier)
3. Executes buy if proposal is valid
4. Monitors trade until expiry
5. Publishes outcome to `TOPIC_SIGNALS` for reputation update

---

## 🎖️ Evolutionary Agents

### AgentReputationEngine
**File:** [`src/agents/reputation.py`](../src/agents/reputation.py)  *(Stage 1)*

Tracks each agent's **rolling win-rate over the last 100 trades** and assigns a dynamic vote multiplier.

```
Track: history[agent_name] = deque(maxlen=100)  # True/False per trade
       pnl_history[agent_name] = float           # cumulative PnL

Calculate weight:
  accuracy ≥ 0.75 → weight = 1.80x
  accuracy ≥ 0.70 → weight = 1.50x
  accuracy ≥ 0.55 → weight = 1.20x
  accuracy ≥ 0.45 → weight = 0.90x
  accuracy  < 0.45 → weight = 0.50x
```

State is persisted to `data/agent_reputation.json` and survives restarts.

---

### ChiefStrategyAgent — Meta-Agent
**File:** [`src/agents/chief_strategy_agent.py`](../src/agents/chief_strategy_agent.py)  *(Stage 2)*

The **Meta-Agent** that manages the entire agent team autonomously.

| Trigger | Condition | Action |
|---|---|---|
| Periodic review | Agent win-rate > 70% for ≥ 20 trades | Promote → increase weight cap |
| Periodic review | Agent win-rate < 35% for ≥ 20 trades | Quarantine → disable agent |
| On quarantine | — | Log + alert via `TOPIC_ALERTS` |
| On promotion | — | Log + increase weight multiplier |

Produces a **leaderboard** visible on the dashboard showing all agents ranked by performance.

---

### MarketRegimeAgent
**File:** [`src/agents/market_regime_agent.py`](../src/agents/market_regime_agent.py)  *(Stage 6)*

Classifies the current market environment and **gates strategy selection**.

```
Regime Detection Logic:
  HIGH_VOLATILITY  if ATR > ATR_mean + 2σ
  TRENDING         if ADX > 25 AND EMA slope strong
  SIDEWAYS_CHOP    if ADX < 20 AND tight range
  SLOW             if tick velocity < threshold

Gating Rules:
  TRENDING       → enable TrendAgent, disable VolatilityAgent breakout mode
  SIDEWAYS_CHOP  → enable PatternAgent/digit strategies, disable TrendAgent
  HIGH_VOLATILITY → all agents active but reduce stake size
  SLOW           → skip cycle or wait
```

---

### PortfolioManagerAgent
**File:** [`src/agents/portfolio_manager_agent.py`](../src/agents/portfolio_manager_agent.py)  *(Stage 5)*

**Hard financial guardrails** — the last gate before any trade executes. No agent can override it.

```
Position Sizing:
  risk_amount = account_balance × 0.01   (1% risk per trade)
  stake = min(risk_amount, MAX_STAKE_USD)
  stake = max(stake, MIN_STAKE_USD=1.00)

Hard Vetoes:
  ✗ daily_loss ≥ 3% of session_start_balance  → FULL HALT
  ✗ open_trades ≥ MAX_OPEN_TRADES (20)         → SKIP new trade (1 per market max)
  ✗ stake < MIN_STAKE (1.00 USD)               → reject (Deriv min is $0.50)
```

---

### AgentBreedingSystem
**File:** [`src/agents/breeding_system.py`](../src/agents/breeding_system.py)  *(Stage 7)*

Implements a **Genetic Algorithm** to evolve strategy parameters over time.

```
StrategyDNA contains:
  ├── ema_period: int
  ├── rsi_threshold: float
  ├── confidence_threshold: float
  ├── volatility_filter: float
  └── regime_preference: str

Evolution Cycle:
  1. Score all DNA by fitness = win_rate × avg_pnl
  2. Select top 50% as parents
  3. Crossover: mix parameter genes from two parents
  4. Mutate: small random perturbation on each gene
  5. Spawn offspring as new specialist agents
  6. Cull bottom 25% of the population
```

---

### InfrastructureAgent
**File:** [`src/agents/infrastructure_agent.py`](../src/agents/infrastructure_agent.py)  *(Stage 8)*

System health monitor that publishes real-time telemetry and triggers self-healing.

```
Monitored:
  CPU usage (%)       → alert if > 85%
  RAM usage (%)       → alert if > 90%
  Disk usage (%)      → alert if > 90%
  Redis ping          → reconnect if fail
  PostgreSQL query    → reconnect if fail

Telemetry published to TOPIC_ALERTS every 30 seconds.
Dashboard reads telemetry via /status API.
```

---

## 📐 Agent Signal Flow (Detailed)

```
MarketWatcherSubAgent (R_100)
        │
        │  tick { price, time, symbol }
        ▼
AgentManager.evaluate_market_context(context)
        │
        ├──► TrendAgent.evaluate()       → AgentSignal(CALL, conf=0.82, weight=1.5)
        ├──► VolatilityAgent.evaluate()  → AgentSignal(CALL, conf=0.71, weight=1.2)
        ├──► PatternAgent.evaluate()     → AgentSignal(SKIP, conf=0.45, weight=0.9)
        ├──► RiskAgent.evaluate()        → AgentSignal(CALL, conf=0.90, weight=1.0)
        ├──► LearningAgent.evaluate()    → AgentSignal(CALL, conf=0.77, weight=1.2)
        └──► RLAgent.evaluate()          → rl_boost = +0.05
        │
        ▼
ReputationEngine.apply_weights(signals)
        │  Each signal.weight updated dynamically from rolling accuracy
        ▼
ConsensusAgent.aggregate(signals)
        │  CALL vote: 0.82×1.5 + 0.71×1.2 + 0.90×1.0 + 0.77×1.2 = 3.987
        │  total weight: 5.8
        │  consensus_conf = 3.987 / 5.8 = 0.687 (+0.05 RL boost) = 0.737
        ▼
MarketRegimeAgent.approve(decision, regime="TRENDING")
        │  TRENDING → TrendAgent strategies allowed ✓
        ▼
PortfolioManagerAgent.approve(decision, balance=500.00)
        │  risk_amount = 500 × 0.01 = $5.00
        │  stake = max(1.00, min(5.00, MAX_STAKE)) = $5.00
        │  open_trades=1 < 3 ✓ | daily_loss=0.5% < 3% ✓
        ▼
ExecutionAgent.execute(symbol="R_100", direction=CALL, stake=5.00)
        │
        ├──► Deriv proposal request
        ├──► Deriv buy → Trade ID: 1254
        └──► Monitor until expiry
                │
                ▼
         Trade wins → profit $4.50
                │
                ├──► ReputationEngine.record_outcome(agent, won=True, profit=4.50)
                ├──► RLAgent.update_q_table(state, action, reward=4.50, next_state)
                └──► TradeMemory.save(full trade record to PostgreSQL)
```

---

## 🔮 Planned: Stage 9 — KnowledgeGraphAgent

Will connect to **Neo4j** to store trade entities as a graph:

- **Nodes**: Trade, Agent, Market, Strategy, Regime, Outcome
- **Edges**: `executed_on`, `detected_by`, `used_strategy`, `market_regime`, `resulted_in`

Enables relationship queries:
- Which agent combinations have the best synergy on R_100?
- What strategy DNA performs best during HIGH_VOLATILITY?
- Which regimes precede the most profitable outcomes?

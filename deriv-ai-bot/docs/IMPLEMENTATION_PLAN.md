# Implementation Plan — Deriv AI Evolutionary Trading Bot

**Last Updated:** 2026-09-17  
**Current Status:** Stages 1–8 complete and running. Stage 9 (Knowledge Graph) next.

---

## 📍 Where We Are

```
Stage 1  ✅  Agent Reputation Engine         (rolling win-rate → dynamic vote weight)
Stage 2  ✅  Chief Strategy Agent            (Meta-Agent — promotes/quarantines)
Stage 3  ✅  Trade Memory Database           (full trade context in PostgreSQL)
Stage 4  ✅  Reinforcement Learning Agent    (Q-Learning from live trade outcomes)
Stage 5  ✅  Portfolio Manager Agent         (hard financial rules, 1% risk, $1 min)
Stage 6  ✅  Market Regime Agent             (TRENDING/CHOP/VOLATILE gating)
Stage 7  ✅  Agent Breeding System           (Genetic Algorithm — evolving DNA)
Stage 8  ✅  Infrastructure Agent            (CPU/RAM/Redis/Postgres health monitor)
Stage 9  🔜  Knowledge Graph (Neo4j)         (relationship pattern discovery)
```

---

## ✅ Completed Work (Stages 1–8)

### Stage 1 — Agent Reputation Engine
**File:** `src/agents/reputation.py`

Each agent has a rolling window of its last 100 trade outcomes. The win-rate in that window directly controls its vote multiplier in the next consensus round.

```
accuracy ≥ 75% → 1.80x weight  (elite)
accuracy ≥ 70% → 1.50x weight  (strong)
accuracy ≥ 55% → 1.20x weight  (solid)
accuracy ≥ 45% → 0.90x weight  (weak)
accuracy  < 45% → 0.50x weight  (on notice)
```

State persisted to `data/agent_reputation.json`.

---

### Stage 2 — Chief Strategy Agent (Meta-Agent)
**File:** `src/agents/chief_strategy_agent.py`

The Meta-Agent runs periodic reviews over all specialist agents:

- Win-rate > 70% (≥20 trades sample) → **Promote** (increase weight cap)
- Win-rate < 35% (≥20 trades sample) → **Quarantine** (disable the agent)
- Publishes leaderboard to the dashboard via `/status` API

---

### Stage 3 — Trade Memory Database
**File:** `database/init.sql` table `trade_memory`

Every trade is stored with:

| Column | Description |
|---|---|
| `symbol` | Market traded |
| `contract_type` | CALL / PUT / DIGITOVER etc |
| `stake` | Stake amount |
| `duration` | Contract duration |
| `agent_votes` | JSON blob — all agent signals with confidence |
| `consensus_conf` | Final confidence score |
| `market_regime` | Regime at execution time |
| `rl_boost` | RL agent modifier applied |
| `result` | win / loss |
| `profit` | Actual PnL |
| `created_at` | Timestamp |

---

### Stage 4 — Reinforcement Learning Agent
**File:** `src/agents/rl_agent.py`

Q-Learning agent using trade outcome as reward signal.

```
State:  (regime, trend_dir, volatility_band)  → ~48 states
Action: CALL | PUT | SKIP
Reward: +profit on win, -loss on loss, -0.1 on skip

Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]
  α = 0.1 (learning rate)
  γ = 0.95 (discount factor)
  ε = 0.9 → 0.1 (exploration decay over time)
```

Q-table persisted to `data/rl_qtable.json`.

---

### Stage 5 — Portfolio Manager Agent
**File:** `src/agents/portfolio_manager_agent.py`

**Hard rules — no agent can override:**

```python
MIN_STAKE        = 1.00   # USD — Deriv min is 0.50
RISK_PER_TRADE   = 0.01   # 1% of balance
MAX_OPEN_TRADES  = 20   # 1 per market sub-agent (20 markets watched)
MAX_DAILY_LOSS   = 0.03   # 3% of session start balance → HALT
```

Position sizing formula:
```
stake = balance × RISK_PER_TRADE
stake = max(MIN_STAKE, min(stake, MAX_STAKE))
```

If `daily_loss ≥ 3%` → trading halted until next session.

---

### Stage 6 — Market Regime Agent
**File:** `src/agents/market_regime_agent.py`

```
TRENDING        → ADX > 25, EMA slope strong
SIDEWAYS_CHOP   → ADX < 20, tight Bollinger range
HIGH_VOLATILITY → ATR > mean + 2σ
SLOW            → tick velocity < threshold
```

Regime controls which agent strategies are gated in/out per cycle.

---

### Stage 7 — Agent Breeding System
**File:** `src/agents/breeding_system.py`

Genetic Algorithm over `StrategyDNA` objects:

```
DNA genes: ema_period, rsi_threshold, confidence_threshold,
           volatility_filter, regime_preference

Fitness   = win_rate × avg_pnl

Lifecycle:
  1. Score population by fitness
  2. Select top 50% as parents
  3. Crossover (gene-level mix from two parents)
  4. Mutate (Gaussian noise on continuous genes)
  5. Spawn offspring as new specialist agents
  6. Cull bottom 25%
```

Evolution triggers every N completed trades or on a timer cycle.

---

### Stage 8 — Infrastructure Agent
**File:** `src/agents/infrastructure_agent.py`

| Resource | Monitoring | Alert Threshold |
|---|---|---|
| CPU | `psutil.cpu_percent()` | > 85% |
| RAM | `psutil.virtual_memory()` | > 90% |
| Disk | `psutil.disk_usage('/')` | > 90% |
| Redis | PING | fail → reconnect |
| PostgreSQL | SELECT 1 | fail → reconnect |

Publishes telemetry every 30 seconds to `TOPIC_ALERTS`. Dashboard reads it via `/status`.

---

## 🔜 Stage 9 — Knowledge Graph (Neo4j) [NEXT]

### Goal
Replace rows-only thinking with **relationship-aware pattern discovery**. Instead of asking "which trades won?" we ask "which *combinations* of agents, markets, and regimes produce the best outcomes?"

### Architecture

```
Neo4j Graph Schema
───────────────────
Nodes:
  (:Trade   { id, timestamp, profit, result })
  (:Agent   { name, type, current_weight })
  (:Market  { symbol, family })
  (:Strategy{ dna_id, fitness_score })
  (:Regime  { type })               # TRENDING | CHOP | etc
  (:Outcome { result, profit })

Edges:
  (Trade)-[:EXECUTED_ON]  →(Market)
  (Trade)-[:DETECTED_BY]  →(Agent)
  (Trade)-[:USED_STRATEGY]→(Strategy)
  (Trade)-[:DURING_REGIME]→(Regime)
  (Trade)-[:RESULTED_IN]  →(Outcome)
  (Agent)-[:BRED_FROM]    →(Agent)   # lineage tracking
```

### Example Graph Queries (Cypher)

```cypher
-- Find best agent combinations on R_100
MATCH (t:Trade)-[:EXECUTED_ON]->(m:Market {symbol:'R_100'}),
      (t)-[:DETECTED_BY]->(a:Agent),
      (t)-[:RESULTED_IN]->(o:Outcome)
RETURN a.name, count(t), avg(o.profit)
ORDER BY avg(o.profit) DESC

-- Which strategy DNA wins most during TRENDING regime?
MATCH (t:Trade)-[:DURING_REGIME]->(r:Regime {type:'TRENDING'}),
      (t)-[:USED_STRATEGY]->(s:Strategy),
      (t)-[:RESULTED_IN]->(o:Outcome {result:'win'})
RETURN s.dna_id, count(t) as wins
ORDER BY wins DESC LIMIT 10
```

### New Files Required

| File | Purpose |
|---|---|
| `src/agents/knowledge_graph_agent.py` | Ingests trade data into Neo4j after each trade |
| `src/agents/graph_intelligence_agent.py` | Runs Cypher pattern queries, outputs recommendations |

### Docker Changes

```yaml
# Add to docker-compose.yml
neo4j:
  image: neo4j:5-community
  container_name: bot-neo4j
  ports:
    - "127.0.0.1:7474:7474"   # Browser UI
    - "127.0.0.1:7687:7687"   # Bolt protocol
  environment:
    NEO4J_AUTH: "neo4j/derivbot2026"
  volumes:
    - neo4j_data:/data
```

### New Dependencies

```
neo4j>=5.0.0        # Neo4j Python driver
```

---

## 📅 Roadmap Summary

| Stage | Feature | Status | File |
|---|---|---|---|
| 1 | Agent Reputation Engine | ✅ Done | `agents/reputation.py` |
| 2 | Chief Strategy Agent (Meta-Agent) | ✅ Done | `agents/chief_strategy_agent.py` |
| 3 | Trade Memory Database | ✅ Done | `database/init.sql` |
| 4 | RL Agent (Q-Learning) | ✅ Done | `agents/rl_agent.py` |
| 5 | Portfolio Manager Agent | ✅ Done | `agents/portfolio_manager_agent.py` |
| 6 | Market Regime Agent | ✅ Done | `agents/market_regime_agent.py` |
| 7 | Agent Breeding System (GA) | ✅ Done | `agents/breeding_system.py` |
| 8 | Infrastructure Agent | ✅ Done | `agents/infrastructure_agent.py` |
| 9 | Knowledge Graph (Neo4j) | 🔜 Next | `agents/knowledge_graph_agent.py` |
| 10 | GraphIntelligence + Recommendations | 🔜 Future | `agents/graph_intelligence_agent.py` |
| 11 | Agent Lineage Tracking (breeding family tree) | 🔜 Future | Graph extension |
| 12 | Strategy Market Condition Matching | 🔜 Future | Uses graph queries |
| 13 | Self-Evolving Cypher Query Generator | 🔜 Future | LLM + Graph |

---

## 🗂️ Original Phase Plan (Tick Engine vs Candle Engine)

The original two-engine plan from 2026-07-17 remains partially open:

### Phase A — Anti-spiral Hardening ✅ Done
Flat stake, cooldowns, daily loss hard stop, min confidence gates.

### Phase B — Intelligent Trade-Type Router 🔜 Partial
The regime agent handles rough gating. A dedicated `TradeTypeRouter` that scores DIGIT* vs CALL/PUT from regime hasn't been built as a standalone module yet.

### Phase C — Minute/Candle Engine 🔜 Planned
1m/2m OHLC candle builder from ticks, EMA 12/26, RSI, port from XML bots.

### Phase D — Extended Market Coverage ✅ Done via Sub-Agents
All 20 markets now have dedicated sub-agents streaming live.

### Phase E — XML Import Pipeline 🔜 Optional
Parser for Binary Bot Blockly → JSON recipe for our signal plugins.

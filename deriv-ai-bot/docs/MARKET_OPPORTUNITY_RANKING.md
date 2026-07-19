# Market Opportunity Ranking (MOR)

Transforms the bot from a **signal engine** into a **market-selection engine**.

```text
Old:  Is this a good trade?
New:  Across all markets, where is the BEST edge right now?
```

## Production formula

```text
Opportunity =
  20% Pattern Strength
+ 15% Pattern Clarity
+ 15% HPP
+ 10% HPP Velocity (mapped 0–100)
+ 15% Momentum Persistence
+ 10% Regime Match
+ 10% Expected Value (mapped)
+  5% Confidence

Final Score = Opportunity − Risk Penalties
Rank Score  = blend(Final, Final × confidence)
```

### Penalties

| Condition | Penalty |
|-----------|---------|
| High drawdown | −10 |
| Unstable HPP | −5 |
| Low sample confidence (&lt;20 trades) | −15 |

### Tiers

| Score | Tier |
|-------|------|
| 90–100 | **ELITE** |
| 80–89 | **STRONG** |
| 70–79 | **WATCHLIST** |
| &lt;70 | **IGNORE** |

Only **ELITE / STRONG** that also pass scanner gates are `tradeable`.

### Opportunity velocity & acceleration

```text
Velocity     = current − previous score
Acceleration = Δ velocity
```

Rising path 79→82→85→90 = emerging edge.

### Multi-horizon

```text
Short / Medium / Long opportunity
Short high + medium lower → fresh opportunity emerging
```

### Correlation filter

Highly related markets (e.g. V75 + V100) — keep one cluster leader to avoid duplicate risk.

### Self-optimizing priority

Every closed trade updates market priority (PF, HPP vel, DD, clarity).  
Every **500** trades → Top/Worst PF report.  
Every **1000** trades → MOR validation (elite vs ignore, velocity, confidence).

## Trade gates (still required)

```text
Edge ≥ 80 · Clarity ≥ 75 · HPP ≥ 75 · MP ≥ 70 · EV > 0
```

## Code

```text
src/analytics/market_opportunity_ranking.py
src/analytics/market_scanner.py   # rank_markets uses MOR
```

## Verification checks

1. Top-ranked markets → higher WR / PF / lower DD than bottom  
2. Elite (90+) beats Strong (80–89)  
3. Opportunity velocity &gt; 0 beats velocity &lt; 0  
4. Confidence &gt; 80 beats confidence &lt; 50  

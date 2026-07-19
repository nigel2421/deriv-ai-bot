# Multi-Market Category Architecture

When scanning **all Deriv markets**, do **not** use one scoring engine for everything.

Assign each symbol a **category**, then run the matching engine.

## Categories

| Category | Examples | Scoring engine | Contracts |
|----------|----------|----------------|-----------|
| **Forex** | EUR/USD, GBP/USD, USD/JPY… | Momentum, Persistence, Transition, Vol regime, HPP, HPP velocity | CALL/PUT |
| **Stocks** | AAPL, TSLA, NVDA… | Momentum, Persistence, Trend, HPP | CALL/PUT |
| **Indices** | US500, NAS100, DAX… | Momentum, Persistence, Trend, Dir. entropy | CALL/PUT |
| **Commodities** | Gold, Oil, Sugar… | Momentum, Volatility, Persistence | CALL/PUT |
| **Crypto** | BTC, ETH… | Momentum, Persistence, Acceleration, Vol regime | CALL/PUT |
| **Synthetic Vol** | R_10…R_100, 1HZ* | **Entropy**, Pattern strength/clarity, Momentum, Persistence | Digits + CALL/PUT |
| **Boom** | Boom 500/1000… | Spike detection, Persistence, Pattern | CALL/PUT |
| **Crash** | Crash 500/1000… | Spike detection, Persistence, Pattern | CALL/PUT |

## Scoring paths

```text
synthetic_vol  → digits_and_rf   (entropy + RF engines)
boom / crash   → spike           (spike + directional)
everything else → directional    (momentum/persistence/HPP)
```

## Code

```text
src/strategy/market_categories.py
  classify_market(symbol)
  market_profile(symbol)
  scoring_engine(category)
  filter_allowed_for_symbol(symbol, types)
```

Wired into:

- `orchestrator.scan_markets` — category filters digit vs RF allow-lists  
- `trade_filter.evaluate_setup` — attaches `market_category` + engine weights  
- `trade_selector` — mild path/family alignment bias  
- dashboard analytics → `market_book`

## Extended synthetic / derived classes

| Category | Examples | Metrics |
|----------|----------|---------|
| **Boom / Crash** | Boom 500, Crash 1000 | Spike analysis, Persistence, Pattern strength |
| **Step** | Step 0.1–0.5 | Momentum, Persistence, Transition |
| **DSI** | DSI10/20/30 | Regime detection, Persistence, Momentum |
| **Vol Switch** | Vol switch indices | Regime classification, Volatility engine |
| **Jump** | Jump 10–100 | Jump probability, Entropy, Persistence |
| **DEX** | DEX 600 UP/DN… | Spike prediction, Momentum, Persistence |
| **Trek** | Trek indices | Direction persistence, Trend strength |
| **Skew Step** | Skew step | Bias detection, Entropy, Persistence |
| **Daily Reset** | Bull/Bear reset | Trend following, Momentum |
| **Derived FX** | EURUSD DFX10/20… | Momentum, Persistence, Regime |

## Market Scanner (self-optimizing)

```text
Market Scanner rank example
V75           Score 91
V50           Score 88
Boom 500      Score 83
EURUSD DFX20  Score 80
Crash 300     Score 76
```

**Trade only if:**

```text
Edge Score >= 80
Pattern Clarity >= 75
HPP >= 75
Momentum Persistence >= 70
EV > 0
```

**Every 500 trades** → PF report (Top / Worst markets).

**Priority auto-adjust:**

| Reduce scan priority | Boost scan priority |
|----------------------|---------------------|
| PF &lt; 1 | PF &gt; 1.5 |
| HPP Velocity &lt; 0 | HPP Velocity &gt; 0 |
| Drawdown rising | Clarity improving |

Code: `src/analytics/market_scanner.py`

## Expanding SYMBOLS

Current Cloud focus (learning density):

```text
SYMBOLS=R_25,R_50,1HZ50V,R_10
```

Presets:

```text
PRESET_SYNTHETIC_FOCUS | PRESET_SYNTHETIC_FULL
PRESET_BOOM_CRASH | PRESET_JUMP | PRESET_DSI
PRESET_FOREX_MAJORS | PRESET_CRYPTO
```

**Do not enable all markets at once** until GCS learning + HPP have depth.

Rollout:

1. Synthetics (current) until ≥100 closed trades  
2. Boom/Crash + Jump  
3. DSI / Step / DFX  
4. Real forex / stocks / crypto last  

## Digit contracts

Digits only on **synthetic vol** and **jump** (last-digit stream).  
All other categories: **CALL/PUT only**.

# No-Trade / EV Decision Engine

Elite systems optimize **when not to trade**, not only when to fire.

## Hard blocks

| Gate | Threshold | Meaning |
|------|-----------|---------|
| Pattern Clarity | &lt; 75 | Structure unstable |
| HPP Velocity | &lt; −5 | Pattern degrading |
| Entropy Stability | &lt; 60 | Unstable entropy |
| Trade Quality | &lt; 80 | Composite too weak |
| Expected Value | ≤ 0 | No positive expectancy |
| Ensemble | not 4/4 BUY | Engines disagree |
| Regime | contract not allowed | e.g. RANDOM → none |
| Edge decay | ≥ 20% from peak HPP | Retire strategy |

Cold-start softens thresholds so learning can collect samples (still requires **EV &gt; 0** and high confidence).

## Trade Quality Score

```
30% Pattern Strength
+ 25% Pattern Clarity
+ 20% HPP
+ 15% HPP Velocity (mapped 0–100)
+ 10% Confidence
```

Trade only when **≥ 80**.

## Expected Value

```
EV = (Pwin × Reward) − (Ploss × Risk)
```

- **EV &gt; 0** → may trade  
- **EV ≤ 0** → always block  

## Risk sizing from quality

| Decision quality | Risk of balance |
|------------------|-----------------|
| 90+ | 1.0% |
| 80–90 | 0.5% |
| &lt; 80 | 0% (no trade) |

Combined with loss-adaptive step-down in `adaptive_stake`.

## Regime allowlist

- **RANDOM** → no contracts  
- **BALANCED** → EVEN/ODD, CALL/PUT  
- **EMERGING / STRONG PATTERN** → differ/over/under + parity + R/F  
- **HIGH CLUSTERING** → match/differ + parity  

## Calibration & drift

`src/analytics/calibration.py` records every closed trade:

```json
{
  "contract": "DIGITDIFF",
  "entropy": 83,
  "clarity": 87,
  "hpp": 79,
  "velocity": 11,
  "quality": 84,
  "predicted_p": 0.75,
  "result": "WIN"
}
```

Tracks:

- Score-band calibration (80–90 → ~80% WR?)
- Wilson confidence intervals  
- Prediction drift over rolling 500 trades (alert if error &gt; 10%)  
- Peak HPP / edge decay  
- Validation checklist (HPP / clarity / regime / monotonic score bands)

## Feature success criteria

A change is **successful** only if:

1. Win rate improves **or** stays stable  
2. Profit factor improves  
3. Max drawdown decreases  
4. Calibration error decreases  
5. High-score trades outperform low-score  
6. Backtest **and** forward-test both improve  

## Key modules

| Module | Role |
|--------|------|
| `no_trade_engine.py` | Central ALLOW / REJECT + EV + quality |
| `calibration.py` | Bands, CI, drift, attribution, peaks |
| `trade_filter.py` | Wires classic gates **and** no-trade |
| `orchestrator.py` | Quality risk %, outcome → calibration |

## Tests

```bash
python -m pytest tests/test_no_trade_engine.py -q
```

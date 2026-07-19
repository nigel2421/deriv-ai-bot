# Momentum + Persistence Engines (Second System)

Runs **alongside** Entropy / Pattern / HPP — does not replace them.

```text
Current System              New System
--------------              ----------
Entropy Engine              Momentum Engine
Pattern Strength            Persistence Engine
Pattern Clarity             Transition Engine
HPP
HPP Velocity
Adaptive Weights
```

## Formulas

### Momentum

```text
raw = (Up − Down) / N
score = 50 + (raw × 50)
```

Example: 14 up, 6 down → raw 0.40 → **score 70** (Bullish).

### Persistence

From transition matrix `UP→UP` / `DOWN→DOWN` (0–100).  
Effective score shrinks toward 50 when sample confidence is low.

### Momentum Persistence

```text
MP = 60% Persistence + 40% Momentum
```

### Final Trade Quality

```text
30% Pattern Strength
+ 20% Pattern Clarity
+ 15% HPP
+ 10% HPP Velocity
+ 15% Momentum Persistence
+ 10% Confidence
```

Execute only when:

```text
Final Quality ≥ 80
AND EV > 0
AND Regime compatible
AND Confidence ≥ 70
```

### Dual blend (informational)

```text
50% Existing Edge + 30% Momentum + 20% Persistence
```

## Rise/Fall gates

**Rise (CALL):** Momentum > 65 · Persistence > 60 · TQ > 80 · HPP Vel > 0  
**Fall (PUT):** Momentum < 35 · Persistence > 60 · TQ > 80 · HPP Vel > 0  

Also blocks late entries when **Persistence Velocity** is declining.

## Digit confirmation

Low MP (e.g. 28) with strong entropy → reduce confidence.  
High MP (e.g. 75) → increase confidence.

## Persistence velocity & acceleration

A persistence of **60% is bullish if rising, dangerous if falling**.

### Steps

1. **Persistence** from transition matrix (`UP→UP / (UP→UP+UP→DOWN)`)
2. **History** stored every calculation
3. **Raw velocity** = current − previous  
4. **Smoothed** = current − mean(prior readings)  
5. **Multi-TF**: fast(20) / medium(100) / slow(500)  
6. **Velocity score** = `50 + velocity×3` (clamp 0–100)  
7. **Acceleration** = Δ velocity (e.g. +2,+4,+7,+12 → accel +5)  
8. **Confidence** = `min(transitions/500, 1)` → adjusted vel = vel × conf  

### Persistence Engine composite

```text
50% Persistence + 30% Velocity score + 20% Acceleration score
```

Example: 70 / 85 / 90 → **78.5 → 79**

### RF filter

Base Rise:

```text
Persistence > 55% AND Velocity > 0 AND Acceleration > 0
```

Premium:

```text
Persistence > 60 AND Velocity > +5 AND Acceleration > +2
```

### Validation (auto)

Group A (P-Vel > 0) must beat Group B (P-Vel < 0) on WR and PF after ~1000 trades.  
If not → **automatically reduce velocity weight**.

## Feature contribution (every 500 trades)

`CalibrationTracker.feature_contribution_report()` ranks lift of high vs low terciles for entropy, clarity, persistence, momentum, HPP velocity, etc.

## Modules

| File | Role |
|------|------|
| `momentum_persistence_engine.py` | Engines + Final Quality + RF gates |
| `no_trade_engine.py` | Final Quality weights with MP |
| `meta_validator.py` | RF MP gates + conf ≥ 70 |
| `trade_filter.py` | Dual-system wiring |
| `calibration.py` | Feature contribution report |

## Tests

```bash
python -m pytest tests/test_momentum_persistence.py -q
```

# Rise/Fall Directional Engine

Digit entropy / HPP / clarity remain **excellent for digit contracts**.  
For CALL/PUT, the model shifts to **momentum, persistence, transitions, and directional entropy**.

## Why

| Family | Primary drivers |
|--------|-----------------|
| Digit Differ / Match / Even-Odd / Over-Under | Digit distribution, entropy, concentration, streaks |
| Rise / Fall | Direction, momentum, trend persistence, volatility microstructure |

## RF composite weights

```text
Momentum            35%
Trend Strength      25%
Volatility Regime   20%
HPP                 10%
Directional Entropy 10%
```

## Metrics

### 1. Tick momentum
Last 20 non-flat ticks → up share (e.g. 15/5 → 75% Strong Bullish).  
Oriented per side: CALL uses bullish score, PUT uses bearish.

### 2. Persistence
`P(UP→UP)`, `P(DOWN→DOWN)` from the transition matrix.  
Continuation after a move feeds HPP-style predictive power.

### 3. Transition matrix
```text
UP→UP · UP→DOWN · DOWN→UP · DOWN→DOWN
```

### 4. Volatility regime
`CALM | NORMAL | EXPANDING | CHAOTIC`  
EXPANDING / CHAOTIC → RF trades blocked (many RF strategies fail there).

### 5. Directional entropy
Binary entropy on {UP, DOWN}.  
50/50 → high entropy, no edge. 72/28 → low entropy, possible edge.

## Meta-validator

Before every trade, **all** must agree:

- Pattern Strength  
- Pattern Clarity  
- HPP  
- Velocity (≥ −5; negative = decaying edge → **BLOCK**)  
- EV > 0  
- Confidence  
- Regime allows contract  
- (RF) RF score + vol tradeable  

Example: strength 88 / clarity 82 / HPP 84 / velocity **−15** → **BLOCKED**.

## Contract profiles (CALL/PUT)

```text
momentum            0.35
trend_strength      0.25
volatility_score    0.20
persistence         0.10
directional_entropy 0.10
```

Digit entropy is **not** a primary RF weight.

## Digit contracts (unchanged intent)

- **Differ**: high digit entropy / diversity / low concentration  
- **Match**: low entropy / high concentration / repetition  
- Even/Odd & Over/Under: parity / threshold entropy + momentum  

## Automated validation checklist

1. Score vs WR — higher score → higher WR  
2. HPP vs WR  
3. Clarity vs WR  
4. Velocity vs WR — positive velocity outperforms negative  
5. Contract-specific — digit metrics help digits; directional metrics help RF  
6. Calibration — predicted 80% ≈ actual ~80%  
7. Profit factor — release must improve or hold  
8. Drawdown — release must reduce or hold  

See `CalibrationTracker.validation_checklist()`.

## Modules

| File | Role |
|------|------|
| `rise_fall_engine.py` | Directional metrics + RF score |
| `meta_validator.py` | All-gates agreement |
| `contract_profiles.py` | CALL/PUT weights |
| `trade_filter.py` | RF reweight + meta wiring |

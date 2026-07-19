# Win-edge audit (post analytics stack)

## Honest expectation

Deriv **synthetic digit** contracts (Even/Odd, Over/Under, Differ) are designed so fair odds sit near **theoretical probability**. There is **no free lunch**. A good bot:

1. **Trades less** (skips noise)
2. **Risks less** after losses
3. **Stops** when session target or stop-loss hits
4. **Learns** which symbol×type combos are cold and avoids them

It does **not** guarantee >50% on fair coins. Target is **positive expectancy after costs/filters**, not magic WR.

## What improves odds of *keeping* money

| Control | Setting (Cloud demo) | Effect |
|---------|----------------------|--------|
| Analytics gate | on | Skip weak setups |
| Min sample | **80** (prod 500) | Need history before full auto |
| Pattern / clarity / edge | ≥72 | Quality filter |
| Flat stake | flat | No martingale blowups |
| Max stake % | 1.0% | 1% risk band |
| Session stop | 5% | Hard loss cap |
| Session target | 1:3 | Lock gains |
| Pause after 3 losses | 10 min | Stop spiral |
| Adaptive learner | always | De-weight losers |
| HPP self-learning | on | Metric weights evolve |
| DeepSeek | optional | Trade-type advice |

## Realistic phases

1. **Cold start (0–80 samples/type):** few trades; bootstrap softens scores; learning accumulates  
2. **Learning (80–500):** filters active; skip hours/symbols; HPP/profile weights update  
3. **Mature (500+):** production sample gate; only high-clarity setups  

## Cloud Run caveat

Disk is **ephemeral**. `learning_state.json` / HPP files survive only while the **same instance** stays warm (`min-instances=1`). Redeploy resets local learning unless you later attach a volume or external store.

## DeepSeek

Requires Secret Manager secret `deepseek-api-key` → env `DEEPSEEK_API_KEY`. Without it, advisor is off; local filters still run.

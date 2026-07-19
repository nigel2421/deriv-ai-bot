# Strategies from community XML bots

**Local folder:** `xml bots/` (31 Binary Bot XML files scanned 2026-07-17)

## Scan summary

| Contract | # of XML files |
|----------|----------------|
| CALL / PUT (Rise/Fall) | 12–13 |
| DIGITOVER | 11 |
| DIGITUNDER | 6 |
| DIGITDIFF | 4 |
| DIGITEVEN / ODD | 2–3 |

Heavy use of **martingale**, **RSI/EMA/SMA**, **last_digit**, **candles**. Markets mostly **R_100 / R_10**.

See also: `docs/IMPLEMENTATION_PLAN.md` for phased build.

Source Drive folders (original share):

1. **BOTS 4 TRADING** — trend / candle / EMA / RSI / martingale packs  
2. **Over Under Bots** — digit OVER/UNDER queues, sure-bet styles, even/odd  
3. **Binary Bot - Strategies**

## Patterns we can employ (mapped to this app)

| Community pattern | Idea | Fit for our bot | Recommendation |
|-------------------|------|-----------------|----------------|
| **Over 1–2 / Over 2 / Over 2–3** | Prefer lower OVER barriers for higher hit-rate | Already: **adaptive barriers** pick among viable OVER levels by recent digits | Keep adaptive; optional fixed `over_barrier` only for A/B tests |
| **Under 7 / Under 8** | High UNDER barriers (win 0–6 or 0–7) | Adaptive UNDER already raises barrier when low digits dominate | Keep; learner de-prioritizes if WR collapses |
| **Digit Queue / SureBet queue** | Wait for a sequence (e.g. two highs) then trade opposite/over | Not fully coded | **Next:** add “digit queue” gate: e.g. last 2 digits ≥7 → UNDER, or last 3 even → ODD fade |
| **Even/Odd bots** | Trade parity only | Supported; parity streak boosts confidence | Prefer EVEN/ODD when parity conf ≥ 0.62 (already) |
| **Digit Differ / Match** | High payout, low probability | Allowed types can include MATCH/DIFF | **Caution:** only with high conf; default off |
| **EMA 12 & 26** | Trend from MA cross | Rise/Fall uses EMA 8/21 + MACD | Align durations; good for CALL/PUT |
| **RSI cool kid** | RSI filter for entries | RSI in chart tools | Keep RSI dampen on extremes |
| **Candle 123 / five-candle** | Multi-candle pattern | Partial via structure HH/HL | Optional: 3-tick micro-candle OHLC pattern |
| **Martingale** | Double after loss | Present with **hard stake cap** | Keep max_steps ≤ 3 and MAX_STAKE |
| **Trend lover / bulls-bears** | Only trade with trend | Regime + rise/fall agreement | Keep chop filter (skip whipsaw) |
| **Roll-over Over/Under** | Switch type after win/loss | Zuno-style switch | Optional hybrid martingale_zuno |

## Highest-value upgrades (from these packs)

1. **Digit queue triggers** (SureBet / Queue 2 & 7 style) — wait for pattern, then one trade.  
2. **Asymmetric OVER/UNDER lists** — e.g. prefer OVER@1–3 and UNDER@7–8 only (higher hit rate).  
3. **RSI + EMA stack only for Rise/Fall** — already close; require agreement (done).  
4. **No martingale on digits** after N losses — flat stake after streak (safer than 2^n).  

## What not to copy blindly

- Names like “100%”, “Sure Bet”, “Verified Hack” are marketing.  
- XML bots often martingale into ruin; we cap stake for that reason.  
- Digits on fair synthetics have no free lunch — edge is filter quality + bankroll rules.

## How to import more XMLs later

Drop files under `data/xml_bots/` and/or paste key rules into `config/strategy.xml`.  
Full Binary Bot XML parse is out of scope; we port **rules**, not Blockly graphs.

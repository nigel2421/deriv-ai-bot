# Implementation Plan — Intelligent Multi-Horizon Trading Bot

**Date:** 2026-07-17  
**Repo assets:** 31 Binary Bot XML files under `xml bots/`  
**Goal:** Tick + minute horizons, smart trade-type selection, stop loss spirals, grow from community bot ideas without copying reckless martingale.

---

## 0. Inventory of `xml bots/` (what the community bots actually do)

| Category | Files (examples) | Contracts | Notes for us |
|----------|------------------|-----------|--------------|
| **Digit OVER** | Over 1-2, Over 2, Over 2-3, Convoy, Ricochet, Solitary, Smart Brain, Queue 2 | DIGITOVER | Prefer lower barriers (1–3) for higher hit-rate |
| **Digit UNDER** | Under-7, Under8, Queue 7, Mount Anda | DIGITUNDER | Prefer high barriers (7–8) |
| **Even/Odd** | Binary Bots Africa EO, Everest EO, martingale.xml | DIGITEVEN/ODD | Parity-only path |
| **Differ** | Digit Differ, Dollar path, SM, Kenya | DIGITDIFF | High payout / low hit-rate — optional only |
| **Rise/Fall (CALL/PUT)** | EMA 12&26, bulls/bears, five candle, candle oscillator 2m, Trend Lover, dream, ICEBOX, Bronze | CALL/PUT | Many use **candles + RSI/EMA/MACD**, often **minutes** |
| **Martingale** | Almost all (326 refs) | sizing | Dangerous; we must **cap or replace** |

**Dominant markets in XMLs:** `R_100`, `R_10`, some 1HZ. Almost no Boom/Crash in these files.

**Implication:** Build **two engines**:
1. **Tick engine** — digits + short CALL/PUT (current).  
2. **Minute/candle engine** — Rise/Fall with EMA/RSI/candle patterns (from XML packs).

---

## 1. Problem: “Same down pits” / losses pile up

### Root causes today
1. **Martingale** doubles stake after losses → one bad streak digs a hole.  
2. **Same symbol + same contract type** re-selected after losses (learner not strict enough).  
3. **force_resume / martingale reset** can re-open deactivated markets.  
4. **High trade frequency** (many symbols × 45s cycles) increases exposure to noise.  
5. **No “cooldown per setup”** — only global consecutive-loss pause.

### Immediate fixes (Phase A — do first)
| Fix | Behavior |
|-----|----------|
| **A1 Flat stake default** | `stake_mode=flat` (no double); martingale optional via config |
| **A2 Setup cooldown** | After 2 losses on `symbol\|type`, ban that setup 15–30 min |
| **A3 Symbol cooldown** | After 3 losses on a symbol, skip symbol for N minutes |
| **A4 Stricter learner skip** | Skip if WR < 40% after 5 samples; cold streak 2 (not 4) |
| **A5 Soft landing** | After global 3 losses, only take conf ≥ 0.88 for 20 min |
| **A6 No martingale reset on resume** | Resume clears pause only; does **not** reset all martingales |
| **A7 Daily loss hard stop** | Already have %; enforce and surface on dashboard |

---

## 2. Architecture target

```
                    ┌─────────────────────┐
                    │   Market Scanner    │
                    │  (symbols + ticks)  │
                    └──────────┬──────────┘
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
   ┌───────────────┐  ┌────────────────┐  ┌─────────────────┐
   │ Tick Digit    │  │ Tick Rise/Fall │  │ Minute Candle   │
   │ OVER/UNDER/EO │  │ CALL/PUT       │  │ CALL/PUT (OHLC) │
   └───────┬───────┘  └───────┬────────┘  └────────┬────────┘
           │                  │                    │
           └──────────────────┼────────────────────┘
                              ▼
                 ┌────────────────────────┐
                 │ Intelligent Selector   │
                 │ score = conf + learn   │
                 │       - risk penalty   │
                 │ + horizon fit          │
                 └───────────┬────────────┘
                             ▼
                 ┌────────────────────────┐
                 │ Risk / Anti-spiral     │
                 │ stake · cooldowns ·    │
                 │ daily cap · pause      │
                 └───────────┬────────────┘
                             ▼
                      Proposal → Buy
```

---

## 3. Phased delivery

### Phase A — Anti-spiral + XML learnings (1–2 days) ✅ partially done / ship next
- [x] Adaptive barriers, digit queue, regime filter, resume controls  
- [ ] Flat stake default; martingale opt-in  
- [ ] Setup + symbol cooldowns  
- [ ] Stricter learner bans  
- [ ] Don’t reset martingale on force_resume  
- [ ] Port **SureBet queue 2/7** fully (already partial in `digit_queue.py`)  
- [ ] Port **Over 1–2 / Under 7–8** preference bands into barrier picker  

### Phase B — Intelligent trade-type selection (2–3 days)
- [ ] **TradeTypeRouter**: scores DIGIT* vs CALL/PUT from regime  
  - Chop → prefer EVEN/ODD or skip  
  - Strong trend → CALL/PUT only  
  - Digit queue fire → digits only  
- [ ] **Diversity rule**: max 2 trades in a row same `symbol|type`  
- [ ] **Expected-value gate**: skip if learner EV < 0 for that setup  
- [ ] Dashboard: show last 10 trades + cooldowns  

### Phase C — Minute / candle engine (3–5 days)
- [ ] Candle builder from ticks (1m / 2m OHLC)  
- [ ] Port EMA 12/26, RSI, candle color, 5-candle pattern from XMLs  
- [ ] Duration: `duration_unit=m`, duration 1–5 minutes for CALL/PUT  
- [ ] Separate min_confidence for minutes (e.g. 0.75–0.82)  
- [ ] Only one open minute trade at a time (longer hold)  

### Phase D — Markets beyond volatility (2–3 days, careful)
| Priority | Markets | Why |
|----------|---------|-----|
| P0 | Keep R_* + 1HZ | Proven on our stack |
| P1 | Step indices (if options support) | Cleaner for Rise/Fall |
| P2 | Jump (limited) | Needs event filter |
| P3 | Boom/Crash | **Not** for digit martingale; separate spike module later |
| Avoid for now | Forex multi-pair dump | Different session/duration model |

Each new symbol: dry-run proposal only → then small stake.

### Phase E — XML import pipeline (optional, 2 days)
- [ ] Parser for Binary Bot Blockly fields → JSON recipe  
- [ ] Map recipe → our signal plugins  
- [ ] Do **not** auto-enable martingale from XML  

---

## 4. Intelligent selection rules (product spec)

```
IF chop_score high:
    prefer DIGITEVEN/ODD if parity_conf >= 0.75 else SKIP
ELIF digit_queue triggered:
    trade queue type only (digits)
ELIF trend strong AND tools agree:
    trade CALL/PUT (tick or minute depending on candle strength)
ELIF digit model conf >= min_conf:
    trade OVER/UNDER with adaptive barrier
ELSE:
    SKIP  (no force trade)
```

**Never:**
- Increase stake after loss by default  
- Re-enter same setup within cooldown  
- Trade if daily loss limit hit  
- Trade if symbol on ban list  

---

## 5. Success metrics

| Metric | Target |
|--------|--------|
| Max single-trade stake | ≤ MAX_STAKE (e.g. $5–8) |
| Max daily drawdown | ≤ 2–3% session start |
| Same setup losses before ban | 2 |
| Trades/hour | lower than today; quality > quantity |
| Win rate by family | tracked in learner; auto-skip < 40% |
| Minute engine | ≥ paper accuracy before live execute |

---

## 6. Suggested build order (next PRs)

1. **PR1 Anti-spiral** — cooldowns, flat stake, stricter bans, resume fix  
2. **PR2 Router** — intelligent family selection  
3. **PR3 Candles** — 1m OHLC + EMA/RSI CALL/PUT minutes  
4. **PR4 Markets** — Step pilot + proposal dry-run  
5. **PR5 XML recipes** — import top 5 bots as config recipes  

---

## 7. Config knobs (strategy.xml / env)

```xml
<global>
  <min_confidence>0.80</min_confidence>
  <stake_mode>flat</stake_mode>          <!-- flat | martingale -->
  <setup_cooldown_minutes>20</setup_cooldown_minutes>
  <symbol_cooldown_losses>3</symbol_cooldown_losses>
  <soft_landing_losses>3</soft_landing_losses>
  <soft_landing_min_conf>0.88</soft_landing_min_conf>
  <enable_minute_engine>true</enable_minute_engine>
  <minute_duration>2</minute_duration>
</global>
```

---

## 8. Risk disclaimer

Community XMLs heavily use martingale and “sure bet” branding. Our edge is **filters + bankroll**, not doubling down. Expanding to minutes and more markets only helps if anti-spiral stays strict.

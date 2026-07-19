# Deriv AI Trading Bot

Automated **Deriv.com** digits trading bot (Over/Under/Even/Odd) with hybrid **XGBoost (+ optional LSTM)** signals, Martingale/Zuno strategies, risk controls, Telegram ops, backtesting, and Docker deploy.

**WARNING:** Trading bots can lose money quickly. Start in **demo** mode. Paper-trade and backtest before any real account use.

---

## Features

| Area | Capability |
|------|------------|
| **API** | WebSocket client, `req_id` RPC, ticks, history, proposal→buy→monitor |
| **AI** | Feature schema, train/evaluate gates, live inference + heuristic fallback |
| **Strategy** | XML config, Martingale stakes, Zuno type switching |
| **Risk** | Live balance, session stop-loss 5–10%, 1:3 profit target, 1–2% stake, consecutive losses |
| **DeepSeek** | Optional advisor analyzes trade types + feeds learning multipliers (`DEEPSEEK_API_KEY`) |
| **Analytics** | Digit heatmap, patterns, edge/pattern scores, AI filter (Skip/Trade), edge scanner, adaptive stake |
| **Gates** | Auto-trade only if Strength≥75 · **Clarity≥80** · Edge≥80 · Live≥80 · Sample≥500 · Quality≥80 |
| **Ops** | Telegram `/status` `/pause` `/resume` `/stats`, structured logs |
| **Research** | Data collector, tick backtest, Monte Carlo path analysis |
| **Deploy** | Docker Compose, optional Streamlit dashboard |

---

## Quick start

```bash
# 1. Environment
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate
pip install -r requirements.txt

# 2. Config
cp .env.example .env
# Set DERIV_API_TOKEN (demo). Optional: TELEGRAM_*, risk knobs.

# 3. Train model (offline OK — creates sample data if needed)
python scripts/train_model.py

# 4. Connection smoke test
python scripts/test_connection.py

# 5. Collect live ticks + bootstrap
python scripts/data_collector.py --count 500

# 6. Backtest
python scripts/backtest.py --data data/historical/ticks.csv --symbol R_100 --no-model

# 7. Run bot (demo)
python src/main.py --mode demo
```

### Docker (local)

```bash
cp .env.example .env   # set secrets
docker compose up --build -d bot
# → http://localhost:8080  (status page)  /health  /status
docker compose logs -f bot

# Optional Streamlit dashboard (profile)
docker compose --profile dashboard up --build -d dashboard
# → http://localhost:8501
```

### Google Cloud Run

See full guide: **[docs/CLOUD_RUN.md](docs/CLOUD_RUN.md)**

You need: GCP project, `gcloud` CLI, Secret Manager secrets for `DERIV_API_TOKEN` / `DERIV_APP_ID`, and **min instances = 1** (always-on).

```powershell
# After secrets exist:
.\scripts\deploy_cloud_run.ps1 -ProjectId YOUR_GCP_PROJECT -Region us-central1
```

Then open the service URL → `/` for live status, `/status` for JSON.

Or: `bash scripts/deploy.sh`

---

## Project layout

```text
config/           settings + strategy.xml
src/
  main.py         entrypoint
  orchestrator.py trade loop
  api/            Deriv WS, executor, monitor
  ai/             train / predict / schema / metrics
  strategy/       martingale, zuno, signals, risk
  backtest/       offline engine
  dashboard/      Streamlit UI
  models/         trained artifacts (gitignored weights)
scripts/          train, collect, backtest, smoke tests
tests/            unit + integration
data/             logs, historical ticks, training exports
Dockerfile
docker-compose.yml
```

---

## Configuration

Copy `.env.example` → `.env`. Important keys:

| Variable | Purpose |
|----------|---------|
| `DERIV_APP_ID` | App id (default `1089` if placeholder) |
| `DERIV_API_TOKEN` | **Required** demo/real token |
| `MODE` | `demo` / `real` |
| `SYMBOLS` | e.g. `R_100,R_75` |
| `EXECUTE_TRADES` | `false` = proposal only |
| `MIN_BALANCE` / `MAX_OPEN_TRADES` / `MAX_STAKE_PCT` | Risk |
| `TICK_HISTORY_COUNT` | Warmup ticks on connect |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Alerts + remote pause |
| `MIN_MODEL_ACCURACY` | Train save gate |

Strategy per market: `config/strategy.xml`.

---

## Common commands

```bash
# Tests
pytest

# Train / evaluate models
python scripts/train_model.py
python scripts/train_model.py --lstm
python scripts/evaluate_model.py

# Live smokes
python scripts/test_connection.py
python scripts/test_tick_history.py --count 100
python scripts/test_trade_loop.py --execute false
python scripts/test_risk_balance.py

# Data + research
python scripts/data_collector.py --symbols R_100,R_75 --count 1000
python scripts/backtest.py --export data/training/backtest_trades.csv
python scripts/monte_carlo_backtest.py --trades data/training/backtest_trades.csv

# Dashboard (local)
streamlit run src/dashboard/app.py
```

---

## Safety checklist

1. Use a **demo** token first (`MODE=demo`).
2. Keep `EXECUTE_TRADES=false` until proposal path looks correct in logs.
3. Fund demo only after risk gates and backtests look sane.
4. For `real`, set `EXECUTE_TRADES=true` explicitly and start with tiny stakes.
5. Prefer Telegram `/pause` when away from the machine.

---

## Logs & artifacts

- Runtime: `data/logs/bot.log`
- Models: `src/models/` (`xgboost_model.pkl`, `feature_schema.json`, `model_meta.json`)
- Ticks: `data/historical/`
- Backtests: `data/training/`

---

## License / disclaimer

Educational / research software. **You** are responsible for compliance with Deriv terms and for all financial outcomes. No warranty.

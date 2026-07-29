# Deriv AI Trading Bot

An enterprise-grade, autonomous multi-market trading system for **Deriv.com** supporting **20 synthetic & FX markets**, real-time **Expected Value (EV)** ranking, continuous adaptive learning, persistent GCS storage, and automated **DeepSeek AI per-market analysis**.

Built with Python, FastAPI/Starlette, WebSockets, XGBoost, TensorFlow, and integrated with Google Cloud Run and GCP Secret Manager.

> [!WARNING]
> Trading financial and synthetic instruments involves significant risk of capital loss. Always start in **demo** mode (`MODE=demo`) and thoroughly validate strategy performance before deploying real capital.

---

## Key Capabilities

| Component | Architecture & Features |
| --- | --- |
| **Asset Pool (20 Markets)** | Volatility Indices (`R_10`–`R_100`, `1HZ10V`–`1HZ100V`), Major FX (`frxEURUSD`, `frxGBPUSD`), Spike Indices (`BOOM1000`, `BOOM500`, `CRASH1000`, `CRASH500`), Jump Indices (`JD10`, `JD25`, `JD50`), Step Index (`STPIDX`). |
| **DeepSeek AI Advisor** | Performs automated LLM-powered strategic audits every 100 closed trades per symbol, querying DeepSeek Chat API with complete GCS historical data to generate setup bans/boosts, confidence adjustments, and Telegram alerts. |
| **Persistent Learning (GCS)** | Google Cloud Storage volume mounted directly at `/app/data` on Cloud Run. Win rates, calibration error, MOR rankings, and trade history survive container restarts permanently. |
| **Signal & Strategy Engine** | 11 analytical engines (Entropy, Hurst HPP Foundation & Velocity, Pattern Strength & Clarity, Momentum, Persistence, Regime Filter, EV Engine, Transition Matrix, MOR Tracker). |
| **Multi-Trade Concurrency** | Executes up to 3 concurrent trades (`MAX_OPEN_TRADES=3`) across uncorrelated assets with dynamic stake clamping (`MAX_STAKE_PCT=0.4%`) to preserve capital. |
| **Cloud Run Native** | Single-instance, non-throttled continuous container (`min-instances=1`, `no-cpu-throttling`, `cpu-boost`, 2 vCPU / 2Gi RAM) with automated PowerShell and Bash deployment pipelines. |
| **Live Control & Ops** | Modern web dashboard (`/`), JSON status API (`/status`), health check endpoints (`/health`, `/ready`), and Telegram bot commands (`/status`, `/pause`, `/resume`, `/stats`). |

---

## Supported Markets & Strategy Suite

| Market Family | Symbols | Default Duration | Allowed Contract Types | Strategy Notes |
| --- | --- | --- | --- | --- |
| **Classic Synthetics** | `R_10`, `R_25`, `R_50`, `R_75`, `R_100` | 5 ticks | `DIGITOVER`, `DIGITUNDER`, `DIGITEVEN`, `DIGITODD`, `CALL`, `PUT` | Adaptive digit barriers + trend momentum |
| **1-Second (1Hz)** | `1HZ10V`, `1HZ25V`, `1HZ50V`, `1HZ75V`, `1HZ100V` | 5 ticks | `DIGITOVER`, `DIGITUNDER`, `DIGITEVEN`, `DIGITODD`, `CALL`, `PUT` | High-frequency tick scans |
| **Major FX** | `frxEURUSD`, `frxGBPUSD` | 30 minutes | `CALL`, `PUT` | Session-gated (London/NY hours) |
| **Boom Indices** | `BOOM1000`, `BOOM500` | 10–15 ticks | `CALL` only | Captures upward spikes |
| **Crash Indices** | `CRASH1000`, `CRASH500` | 10–15 ticks | `PUT` only | Captures downward crashes |
| **Jump Indices** | `JD10`, `JD25`, `JD50` | 5 ticks | `DIGITOVER`, `DIGITUNDER`, `DIGITEVEN`, `DIGITODD`, `CALL`, `PUT` | Handles high volatility jumps |
| **Step Index** | `STPIDX` | 5 ticks | `DIGITOVER`, `DIGITUNDER`, `DIGITEVEN`, `DIGITODD` | Fixed 0.1 pip steps; low-entropy pattern |

---

## Quick Start (Local Setup)

### 1. Prerequisites

- Python 3.10+
- Deriv API Token (Demo token from [Deriv API Token Settings](https://app.deriv.com/account/api-token))
- Deriv App ID (Default `33R2Z6MTElnIWrId8aH3m` or your own from developers.deriv.com)

### 2. Installation

```bash
# Clone repository
git clone https://github.com/nigel2421/deriv-ai-bot.git
cd deriv-ai-bot

# Create and activate virtual environment
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and populate your credentials:

```bash
cp .env.example .env
```

Key environment variables:

| Variable | Default / Example | Purpose |
| --- | --- | --- |
| `DERIV_APP_ID` | `33R2Z6MTElnIWrId8aH3m` | Deriv App ID |
| `DERIV_API_TOKEN` | `your_api_token_here` | Deriv API PAT or Bearer token |
| `MODE` | `demo` | Trading mode (`demo` or `real`) |
| `EXECUTE_TRADES` | `true` | `true` to execute live buys; `false` for paper proposals |
| `MAX_OPEN_TRADES` | `3` | Maximum simultaneous open positions |
| `MAX_STAKE_PCT` | `0.4` | Max stake as % of account balance |
| `TRADE_CYCLE_SECONDS` | `45` | Scan cycle frequency |
| `DEEPSEEK_ENABLED` | `true` | Enable DeepSeek LLM advisor |
| `DEEPSEEK_API_KEY` | `sk-...` | DeepSeek API Key |
| `DEEPSEEK_ANALYZE_EVERY` | `100` | Trades per market before triggering DeepSeek audit |
| `TELEGRAM_BOT_TOKEN` | `optional` | Telegram bot token for mobile notifications |
| `TELEGRAM_CHAT_ID` | `optional` | Telegram chat ID |

### 4. Run Locally

```bash
# Run connection smoke test
python scripts/test_connection.py

# Run unit tests
pytest

# Start local trading orchestrator (HTTP dashboard at http://localhost:8080)
uvicorn src.cloud_app:app --host 0.0.0.0 --port 8080
```

---

## Google Cloud Run Deployment Guide

### Architecture Overview

```text
               +-------------------------------------------------+
               |             Google Cloud Run                    |
               |                                                 |
               |   +-----------------+     +-----------------+   |
 Deriv WS <====>   | Trading Loop    |     | Web Server      |   | <== Browser / Health
  (API v2)     |   | (Orchestrator)  | <==>| (Starlette/Uvi) |   |
               |   +--------+--------+     +--------+--------+   |
               |            |                       |            |
               +------------|-----------------------|------------+
                            |                       |
               +------------v-----------------------v------------+
               |  GCS Persistent Volume Mount (/app/data)        |
               |  gs://<project-id>-deriv-bot-data               |
               +-------------------------------------------------+
```

### Step-by-Step Replication for GCP Deployment

#### Step 1: Install GCP CLI & Authenticate

```powershell
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID
```

#### Step 2: Create Secrets in Secret Manager

Store sensitive keys safely in GCP Secret Manager:

```powershell
# Create secrets
echo "YOUR_DERIV_TOKEN"    | gcloud secrets create deriv-api-token --data-file=-
echo "YOUR_DERIV_APP_ID"   | gcloud secrets create deriv-app-id --data-file=-
echo "YOUR_TELEGRAM_TOKEN" | gcloud secrets create telegram-bot-token --data-file=-
echo "YOUR_TELEGRAM_CHAT"  | gcloud secrets create telegram-chat-id --data-file=-
echo "YOUR_DEEPSEEK_KEY"   | gcloud secrets create deepseek-api-key --data-file=-
```

#### Step 3: Run Automated Deployment Script

Execute the provided automated deployment script. It will enable required GCP APIs, create the Artifact Registry repository, provision the GCS persistent volume bucket, submit the container build, and deploy to Cloud Run with mounted storage.

**On Windows (PowerShell):**

```powershell
.\scripts\deploy_cloud_run.ps1 -ProjectId YOUR_GCP_PROJECT_ID -Region us-central1
```

**On Linux / macOS (Bash):**

```bash
chmod +x scripts/deploy_cloud_run.sh
./scripts/deploy_cloud_run.sh YOUR_GCP_PROJECT_ID us-central1
```

---

## Monitoring & Operations

### Endpoints

Once deployed, access your Cloud Run URL:

- **Web Dashboard**: `https://<your-cloud-run-url>/` — Live balance, active trades, probability panel, MOR rankings, transition matrix, calibration stats, DeepSeek progress, and controls.
- **Status API**: `https://<your-cloud-run-url>/status` — Detailed JSON telemetry for monitoring.
- **Health Check**: `https://<your-cloud-run-url>/health` — Returns `ok` when container is healthy.
- **Ready Probe**: `https://<your-cloud-run-url>/ready` — Returns 200 when trading loop is actively running.

### Telegram Controls

If Telegram tokens are configured, control your bot remotely via chat:

- `/status` — View current balance, open trades, daily PnL, and win rates.
- `/pause` — Pause trading activity safely.
- `/resume` — Resume trading and clear loss streak cooldowns.
- `/stats` — View cumulative learning statistics.

---

## Project Structure

```text
config/
  settings.py             Global environment variables & defaults
  strategy.xml            XML-based strategy definition per market
src/
  cloud_app.py            Starlette Web App entrypoint (Dashboard, REST API, OAuth)
  orchestrator.py         Core trade loop coordinator & multi-market scanner
  api/                    Deriv WebSocket client, execution pipeline, trade monitor
  ai/                     XGBoost/LSTM predictor, schema, feature engineering
  strategy/
    deepseek_advisor.py   DeepSeek LLM per-market analysis engine
    ai_auditor.py         Feature quartile attribution & audit reporter
    ev_engine.py          Expected Value calculation & candidate ranking
    mor_tracker.py        Market Opportunity Ranking & velocity tracker
    trade_selector.py     Multi-trade candidate selection & risk filter
    risk_manager.py       Stake clamping, daily loss limit & drawdown protection
    trend_analyzer.py     EMA, RSI, MACD & market structure trend analysis
    regime_filter.py      Chop/efficiency filter to skip noisy markets
scripts/
  deploy_cloud_run.ps1    Windows PowerShell deployment pipeline with GCS mount
  deploy_cloud_run.sh     Linux/macOS Bash deployment pipeline
  audit_live_status.py    CLI script to fetch and format live Cloud Run status
tests/                    Pytest unit & integration test suite
```

---

## Safety & Best Practices

1. **Demo First**: Keep `MODE=demo` until your strategy demonstrates consistent positive Expected Value over at least 500 trades.
2. **Fixed Stake Sizing**: Use `STAKE_MODE=flat` (default) to avoid exponential loss escalation.
3. **Persistent Volume**: Ensure the GCS bucket mount at `/app/data` is active so learning states are preserved across deployments.
4. **Cloud Run Instance Limit**: Always keep `--min-instances 1` and `--max-instances 1` to prevent multi-instance trade duplication.

---

## License & Disclaimer

This project is open-source and intended for educational and research purposes. You are solely responsible for managing your financial risk, API tokens, and compliance with Deriv terms of service.

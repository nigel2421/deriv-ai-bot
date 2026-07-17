# Deploy Deriv AI Bot to Google Cloud Run

This bot is a **long-running WebSocket trader**. Cloud Run is HTTP-based, so the container runs:

- **uvicorn** HTTP server on `$PORT` (`/`, `/health`, `/status`)
- **trading loop** in the background (same logic as `python src/main.py`)

## What you need

| Item | Notes |
|------|--------|
| **GCP project** | Billing enabled |
| **gcloud CLI** | [Install](https://cloud.google.com/sdk/docs/install) + `gcloud auth login` |
| **APIs** | Cloud Run, Artifact Registry, Cloud Build, Secret Manager |
| **Secrets** | At least `DERIV_API_TOKEN`, `DERIV_APP_ID` |
| **Demo token** | From [Deriv API token](https://app.deriv.com/account/api-token) |
| **Demo balance** | Top up virtual funds so trades can execute |
| **Optional Telegram** | Bot token + chat id for remote `/pause` `/resume` |

### Critical Cloud Run settings (already in deploy scripts)

| Setting | Value | Why |
|---------|--------|-----|
| `--min-instances` | **1** | Avoid scale-to-zero (bot would stop) |
| `--max-instances` | **1** | One trader; avoid duplicate positions |
| `--no-cpu-throttling` | on | WebSocket/background work needs CPU when idle |
| `--memory` | **2Gi** | TensorFlow/XGBoost + buffers |
| `--cpu` | **2** | Comfortable for ML + WS |
| `--concurrency` | **1** | Single process / one loop |

**Cost note:** min-instances=1 with always-on CPU bills continuously. Expect roughly tens of USD/month depending on region/size—not free tier friendly.

---

## 1. One-time GCP setup

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com
```

### Store secrets

```bash
# PowerShell: use without trailing newline
echo -n "YOUR_DERIV_TOKEN" | gcloud secrets create deriv-api-token --data-file=-
echo -n "YOUR_APP_ID"      | gcloud secrets create deriv-app-id --data-file=-

# Or add a new version if secret exists:
# echo -n "TOKEN" | gcloud secrets versions add deriv-api-token --data-file=-

# Optional Telegram
echo -n "BOT_TOKEN" | gcloud secrets create telegram-bot-token --data-file=-
echo -n "CHAT_ID"   | gcloud secrets create telegram-chat-id --data-file=-
```

Grant the Cloud Run runtime service account access to secrets (replace PROJECT_NUMBER):

```bash
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)')
SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

for S in deriv-api-token deriv-app-id; do
  gcloud secrets add-iam-policy-binding $S \
    --member="serviceAccount:${SA}" \
    --role="roles/secretmanager.secretAccessor"
done
```

---

## 2. Local smoke test of Cloud entrypoint

```bash
# From project root, venv active
set PORT=8080
uvicorn src.cloud_app:app --host 0.0.0.0 --port 8080
```

Open:

- http://localhost:8080/ — status page  
- http://localhost:8080/health — `ok`  
- http://localhost:8080/status — JSON  

---

## 3. Deploy

### Windows PowerShell

```powershell
.\scripts\deploy_cloud_run.ps1 -ProjectId YOUR_PROJECT_ID -Region us-central1
```

### Bash

```bash
chmod +x scripts/deploy_cloud_run.sh
./scripts/deploy_cloud_run.sh YOUR_PROJECT_ID us-central1
```

### Manual `gcloud` (equivalent)

```bash
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT/deriv-ai/deriv-ai-bot:latest .

gcloud run deploy deriv-ai-bot \
  --image REGION-docker.pkg.dev/PROJECT/deriv-ai/deriv-ai-bot:latest \
  --region REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi --cpu 2 \
  --min-instances 1 --max-instances 1 \
  --no-cpu-throttling \
  --concurrency 1 \
  --set-env-vars "MODE=demo,EXECUTE_TRADES=true,SAVE_TICK_HISTORY=false" \
  --set-secrets "DERIV_API_TOKEN=deriv-api-token:latest,DERIV_APP_ID=deriv-app-id:latest"
```

---

## 4. Verify online

```bash
URL=$(gcloud run services describe deriv-ai-bot --region REGION --format='value(status.url)')
curl -s "$URL/health"
curl -s "$URL/status" | jq .
# Open $URL in browser
```

Logs:

```bash
gcloud run services logs read deriv-ai-bot --region REGION --limit 50
```

---

## 5. What “online” looks like

| URL | Purpose |
|-----|---------|
| `/` | Simple live dashboard (balance, open trades, last cycle) |
| `/status` | Full JSON for monitoring |
| `/health` | Liveness for Cloud Run |
| `/ready` | 200 only when bot status is `running` |

Trading control remains via **Telegram** (`/pause` `/resume` `/status`) if tokens are set.

---

## 6. Important limitations on Cloud Run

1. **Ephemeral disk** — logs/ticks vanish on redeploy; set `SAVE_TICK_HISTORY=false` (default in deploy script). Use Telegram + Cloud Logging for ops.
2. **No Streamlit on the trading service** — Streamlit is optional separately; the trading service uses the lightweight `/` page.
3. **Image size** — TensorFlow makes a large image; first build is slow.
4. **Secrets** — never bake `.env` into the image; use Secret Manager only.
5. **Real money** — keep `MODE=demo` until you intentionally change it.

---

## 7. Env vars to set on the service

| Name | Example | Required |
|------|---------|----------|
| `MODE` | `demo` | yes |
| `EXECUTE_TRADES` | `true` | yes for live buys |
| `SYMBOLS` | `R_100,R_75` | optional |
| `DERIV_API_TOKEN` | secret | yes |
| `DERIV_APP_ID` | secret or env | yes |
| `TELEGRAM_*` | secrets | optional |
| `TRADE_CYCLE_SECONDS` | `60` | optional |
| `MIN_BALANCE` / risk knobs | as local | optional |

---

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `/status` shows `error` Missing token | Secret not mounted / wrong name |
| Always restarting | Need min-instances=1 + no CPU throttling |
| No trades | Demo balance 0, or confidence gate |
| Huge cold start | Normal with TF; min-instances=1 avoids cold path |
| 403 on secrets | IAM `secretAccessor` on runtime SA |

#!/usr/bin/env bash
# Build + deploy Deriv AI Bot to Google Cloud Run
#
# Usage:
#   ./scripts/deploy_cloud_run.sh PROJECT_ID [REGION] [MODE]
#
# Flags (set via environment variables before running):
#   COST_OPTIMIZED=1   min-instances=0, CPU throttled. ~$0 when idle.
#                      WARNING: Cold starts ~10-15s. NOT for live trading.
#   CPU_THROTTLED=1    min-instances=1, CPU throttled between cycles.
#                      Saves ~15-20% on CPU. Slight latency between 45s trade ticks.
#
# COST BREAKDOWN (July 2026 actuals):
#   Cloud Run CPU (always-allocated, min=1):  $46.08
#   Cloud Run Memory:                          $5.11
#   Artifact Registry (18 GiB):               $1.87
#   Secret Manager:                            $0.16
#   ~$48.44/month after spending-based discount
#
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT_ID="${1:?Usage: $0 PROJECT_ID [REGION] [MODE]}"
REGION="${2:-us-central1}"
SERVICE="deriv-ai-bot"
REPO="deriv-ai"
MODE="${MODE:-demo}"
COST_OPTIMIZED="${COST_OPTIMIZED:-0}"
CPU_THROTTLED="${CPU_THROTTLED:-0}"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com secretmanager.googleapis.com

if ! gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPO" \
    --repository-format=docker --location="$REGION" \
    --description="Deriv AI Bot images"
fi

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest"
gcloud builds submit --tag "$IMAGE" .

# ── Compute profile ───────────────────────────────────────────────────────────
if [[ "$COST_OPTIMIZED" == "1" ]]; then
  MIN_INSTANCES=0
  CPU_FLAG="--cpu-throttling"
  echo "[COST-OPTIMIZED] min-instances=0, CPU throttled. Cold starts expected. NOT for live trading."
elif [[ "$CPU_THROTTLED" == "1" ]]; then
  MIN_INSTANCES=1
  CPU_FLAG="--cpu-throttling"
  echo "[CPU-THROTTLED] min-instances=1, CPU throttled between trade cycles. Est. saving: ~15-20% on CPU."
else
  MIN_INSTANCES=1
  CPU_FLAG="--no-cpu-throttling"
  echo "Deploying Cloud Run service (min-instances=1, CPU always allocated - standard trading mode)..."
fi

gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 2 \
  --timeout 3600 \
  --concurrency 1 \
  --min-instances "$MIN_INSTANCES" \
  --max-instances 1 \
  --cpu-boost \
  "$CPU_FLAG" \
  --env-vars-file "scripts/cloudrun-env.yaml" \
  --set-secrets "DERIV_API_TOKEN=deriv-api-token:latest,DERIV_APP_ID=deriv-app-id:latest,TELEGRAM_BOT_TOKEN=telegram-bot-token:latest,TELEGRAM_CHAT_ID=telegram-chat-id:latest,DEEPSEEK_API_KEY=deepseek-api-key:latest"

# ── Artifact Registry cleanup policy ─────────────────────────────────────────
# Keeps the 3 most recent images; deletes anything older than 7 days.
# Runs after every deploy so the registry stays lean (currently 18 GiB / $1.87/mo).
echo "Applying Artifact Registry cleanup policy (keep 3 latest, delete >7 days)..."
POLICY_FILE="$(mktemp /tmp/ar-cleanup-XXXXXX.json)"
cat > "$POLICY_FILE" << 'EOF'
[
  {
    "name": "delete-old-images",
    "action": {"type": "Delete"},
    "condition": {
      "olderThan": "604800s",
      "tagState": "tagged"
    }
  },
  {
    "name": "keep-minimum-3",
    "action": {"type": "Keep"},
    "mostRecentVersions": {
      "keepCount": 3
    }
  }
]
EOF

gcloud artifacts repositories set-cleanup-policies "$REPO" \
  --location="$REGION" \
  --policy="$POLICY_FILE" \
  --no-dry-run

rm -f "$POLICY_FILE"
echo "Cleanup policy applied. Old images will be pruned automatically."

echo "URL: $(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"

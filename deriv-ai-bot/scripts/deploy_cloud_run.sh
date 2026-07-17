#!/usr/bin/env bash
# Build + deploy to Google Cloud Run
# Usage: ./scripts/deploy_cloud_run.sh PROJECT_ID [REGION]
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT_ID="${1:?Usage: $0 PROJECT_ID [REGION]}"
REGION="${2:-us-central1}"
SERVICE="deriv-ai-bot"
REPO="deriv-ai"
MODE="${MODE:-demo}"

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

gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --concurrency 1 \
  --min-instances 1 \
  --max-instances 1 \
  --cpu-boost \
  --no-cpu-throttling \
  --set-env-vars "MODE=${MODE},EXECUTE_TRADES=true,PYTHONUNBUFFERED=1,TF_CPP_MIN_LOG_LEVEL=2,SAVE_TICK_HISTORY=false,TICK_HISTORY_COUNT=200" \
  --set-secrets "DERIV_API_TOKEN=deriv-api-token:latest,DERIV_APP_ID=deriv-app-id:latest"

echo "URL: $(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"

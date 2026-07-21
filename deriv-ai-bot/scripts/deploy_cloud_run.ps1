<#
.SYNOPSIS
  Build and deploy Deriv AI Bot to Google Cloud Run.

.EXAMPLE
  .\scripts\deploy_cloud_run.ps1 -ProjectId my-gcp-project -Region us-central1

.NOTES
  Prerequisites: gcloud CLI logged in, APIs enabled, Artifact Registry repo.
#>
param(
  [Parameter(Mandatory = $true)][string]$ProjectId,
  [string]$Region = "us-central1",
  [string]$Service = "deriv-ai-bot",
  [string]$Repo = "deriv-ai",
  [string]$Mode = "demo"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "Project=$ProjectId Region=$Region Service=$Service"

gcloud config set project $ProjectId

# Enable required APIs (idempotent)
gcloud services enable `
  run.googleapis.com `
  artifactregistry.googleapis.com `
  cloudbuild.googleapis.com `
  secretmanager.googleapis.com

# Artifact Registry repo
$repoExists = gcloud artifacts repositories describe $Repo --location=$Region 2>$null
if (-not $repoExists) {
  gcloud artifacts repositories create $Repo `
    --repository-format=docker `
    --location=$Region `
    --description="Deriv AI Bot images"
}

$Image = "$Region-docker.pkg.dev/$ProjectId/$Repo/${Service}:latest"
Write-Host "Building $Image ..."

gcloud builds submit --tag $Image .

Write-Host "Deploying Cloud Run service (min instances=1, CPU always allocated)..."

# Secrets must already exist: deriv-api-token, optional telegram secrets
# Create with:
#   echo -n "TOKEN" | gcloud secrets create deriv-api-token --data-file=-
#   echo -n "TOKEN" | gcloud secrets versions add deriv-api-token --data-file=-

gcloud run deploy $Service `
  --image $Image `
  --region $Region `
  --platform managed `
  --allow-unauthenticated `
  --port 8080 `
  --memory 2Gi `
  --cpu 2 `
  --timeout 3600 `
  --concurrency 1 `
  --min-instances 1 `
  --max-instances 1 `
  --cpu-boost `
  --no-cpu-throttling `
  --env-vars-file "scripts/cloudrun-env.yaml" `
  --set-secrets "DERIV_API_TOKEN=deriv-api-token:latest,DERIV_APP_ID=deriv-app-id:latest,TELEGRAM_BOT_TOKEN=telegram-bot-token:latest,TELEGRAM_CHAT_ID=telegram-chat-id:latest"

Write-Host ""
Write-Host "Telegram secrets (create/update from .env):"
Write-Host "  echo TOKEN | gcloud secrets versions add telegram-bot-token --data-file=-"
Write-Host "  echo CHAT  | gcloud secrets versions add telegram-chat-id --data-file=-"
Write-Host ""
Write-Host "Service URL:"
gcloud run services describe $Service --region $Region --format="value(status.url)"

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

$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "Project=$ProjectId Region=$Region Service=$Service"

gcloud config set project $ProjectId

# Enable required APIs (idempotent)
gcloud services enable `
  run.googleapis.com `
  artifactregistry.googleapis.com `
  cloudbuild.googleapis.com `
  secretmanager.googleapis.com

# Artifact Registry repo check
$repoExists = $false
$null = gcloud artifacts repositories describe $Repo --location=$Region 2>&1
if ($LASTEXITCODE -eq 0) { $repoExists = $true }

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

# Ensure GCS bucket for persistent learning state exists
$Bucket = "$ProjectId-deriv-bot-data"
$bucketExists = $false
$null = gcloud storage buckets describe "gs://$Bucket" 2>&1
if ($LASTEXITCODE -eq 0) { $bucketExists = $true }

if (-not $bucketExists) {
  Write-Host "Creating GCS bucket gs://$Bucket for persistent learning data..."
  gcloud storage buckets create "gs://$Bucket" --location=$Region --project=$ProjectId
}

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
  --add-volume "name=gcs-data,type=cloud-storage,bucket=$Bucket" `
  --add-volume-mount "volume=gcs-data,mount-path=/app/data" `
  --env-vars-file "scripts/cloudrun-env.yaml" `
  --set-secrets "DERIV_API_TOKEN=deriv-api-token:latest,DERIV_APP_ID=deriv-app-id:latest,TELEGRAM_BOT_TOKEN=telegram-bot-token:latest,TELEGRAM_CHAT_ID=telegram-chat-id:latest,DEEPSEEK_API_KEY=deepseek-api-key:latest"

Write-Host ""
Write-Host "Telegram secrets (create/update from .env):"
Write-Host "  echo TOKEN | gcloud secrets versions add telegram-bot-token --data-file=-"
Write-Host "  echo CHAT  | gcloud secrets versions add telegram-chat-id --data-file=-"
Write-Host ""
Write-Host "Service URL:"
gcloud run services describe $Service --region $Region --format="value(status.url)"

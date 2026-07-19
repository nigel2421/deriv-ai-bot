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

# gcloud writes warnings to stderr; don't treat as terminating errors
$ErrorActionPreference = "Continue"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "Project=$ProjectId Region=$Region Service=$Service"

gcloud config set project $ProjectId
if ($LASTEXITCODE -ne 0) { throw "gcloud config set project failed" }

# Enable required APIs (idempotent)
gcloud services enable `
  run.googleapis.com `
  artifactregistry.googleapis.com `
  cloudbuild.googleapis.com `
  secretmanager.googleapis.com
if ($LASTEXITCODE -ne 0) { throw "gcloud services enable failed" }

# Artifact Registry repo
$repoExists = $null
gcloud artifacts repositories describe $Repo --location=$Region 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
  gcloud artifacts repositories create $Repo `
    --repository-format=docker `
    --location=$Region `
    --description="Deriv AI Bot images"
  if ($LASTEXITCODE -ne 0) { throw "create artifact repo failed" }
}

$Image = "$Region-docker.pkg.dev/$ProjectId/$Repo/${Service}:latest"
Write-Host "Building $Image ..."

gcloud builds submit --tag $Image .
if ($LASTEXITCODE -ne 0) { throw "gcloud builds submit failed" }

Write-Host "Deploying Cloud Run service (min instances=1, CPU always allocated)..."

# Secrets must already exist: deriv-api-token, optional telegram secrets
# Create with:
#   echo -n "TOKEN" | gcloud secrets create deriv-api-token --data-file=-
#   echo -n "TOKEN" | gcloud secrets versions add deriv-api-token --data-file=-

# Build secrets list — only include secrets that exist (DeepSeek optional)
$secretMap = @(
  "DERIV_API_TOKEN=deriv-api-token:latest",
  "DERIV_APP_ID=deriv-app-id:latest",
  "TELEGRAM_BOT_TOKEN=telegram-bot-token:latest",
  "TELEGRAM_CHAT_ID=telegram-chat-id:latest"
)
$ds = gcloud secrets describe deepseek-api-key 2>$null
if ($LASTEXITCODE -eq 0) {
  $secretMap += "DEEPSEEK_API_KEY=deepseek-api-key:latest"
  Write-Host "DeepSeek secret found — will mount DEEPSEEK_API_KEY"
} else {
  Write-Host "WARNING: secret deepseek-api-key missing — DeepSeek advisor disabled until created."
  Write-Host "  Create:  echo -n YOUR_KEY | gcloud secrets create deepseek-api-key --data-file=-"
  Write-Host "  Then grant runtime SA secretAccessor and redeploy."
}
$secretsCsv = ($secretMap -join ",")

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
  --set-secrets $secretsCsv

Write-Host ""
Write-Host "Service URL:"
gcloud run services describe $Service --region $Region --format="value(status.url)"

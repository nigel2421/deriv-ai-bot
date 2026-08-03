<#
.SYNOPSIS
  Build and deploy Deriv AI Bot to Google Cloud Run.

.EXAMPLE
  # Standard (always-on trading bot, ~$51/month)
  .\scripts\deploy_cloud_run.ps1 -ProjectId my-gcp-project -Region us-central1

  # Cost-optimized (cold-start on first request, ~$0 when idle – NOT for live trading)
  .\scripts\deploy_cloud_run.ps1 -ProjectId my-gcp-project -CostOptimized

.NOTES
  Prerequisites: gcloud CLI logged in, APIs enabled, Artifact Registry repo.

  COST BREAKDOWN (July 2026 actuals):
    Cloud Run CPU (always-allocated, min=1):  $46.08
    Cloud Run Memory (always-allocated):       $5.11
    Artifact Registry storage (18 GiB):        $1.87
    Secret Manager:                            $0.16
    Network egress:                            $0.05
    ────────────────────────────────────────────────
    Total (before discount):                  ~$53.27
    Spending-based discount applied:           -$4.83
    Approximate monthly bill:                 ~$48.44

  OPTIMIZATION OPTIONS:
    Option A – Keep min-instances=1 but switch to CPU throttling:
      Savings: ~15-20%% on CPU. Side-effect: 250-400ms latency spike between
      45-second trade cycles while CPU wakes. Acceptable for this bot since
      TRADE_CYCLE_SECONDS=45 and the process stays in RAM.
      Use: add -CpuThrottled switch.

    Option B – Set min-instances=0 (scale to zero):
      Savings: ~100%% when idle (cold start ~10-15s). NOT suitable for live
      trading – use only for demo/testing where missed ticks are acceptable.
      Use: add -CostOptimized switch.
#>
param(
  [Parameter(Mandatory = $true)][string]$ProjectId,
  [string]$Region = "us-central1",
  [string]$Service = "deriv-ai-bot",
  [string]$Repo = "deriv-ai",
  [string]$Mode = "demo",
  # Scale to zero when idle. Saves ~$51/mo but causes cold starts. Use for demo/testing only.
  [switch]$CostOptimized,
  # CPU throttled between requests. Saves ~15-20%% on CPU. Slight latency between cycles.
  [switch]$CpuThrottled
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

# ── Compute profile ─────────────────────────────────────────────────────────
if ($CostOptimized) {
  $MinInstances = 0
  $CpuThrottlingFlag = "--cpu-throttling"
  Write-Host "[COST-OPTIMIZED] min-instances=0, CPU throttled. Cold starts expected. NOT for live trading." -ForegroundColor Yellow
} elseif ($CpuThrottled) {
  $MinInstances = 1
  $CpuThrottlingFlag = "--cpu-throttling"
  Write-Host "[CPU-THROTTLED] min-instances=1, CPU throttled between trade cycles. Est. saving: ~15-20%% on CPU." -ForegroundColor Cyan
} else {
  $MinInstances = 1
  $CpuThrottlingFlag = "--no-cpu-throttling"
  Write-Host "Deploying Cloud Run service (min-instances=1, CPU always allocated - standard trading mode)..."
}

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
  --memory 1Gi `
  --cpu 2 `
  --timeout 3600 `
  --concurrency 1 `
  --min-instances $MinInstances `
  --max-instances 1 `
  --cpu-boost `
  $CpuThrottlingFlag `
  --add-volume "name=gcs-data,type=cloud-storage,bucket=$Bucket" `
  --add-volume-mount "volume=gcs-data,mount-path=/app/data" `
  --env-vars-file "scripts/cloudrun-env.yaml" `
  --set-secrets "DERIV_API_TOKEN=deriv-api-token:latest,DERIV_APP_ID=deriv-app-id:latest,TELEGRAM_BOT_TOKEN=telegram-bot-token:latest,TELEGRAM_CHAT_ID=telegram-chat-id:latest,DEEPSEEK_API_KEY=deepseek-api-key:latest"

# ── Artifact Registry cleanup policy ────────────────────────────────────────
# Keeps the 3 most recent images; deletes anything older than 7 days.
# Runs after every deploy so the registry stays lean (currently 18 GiB / $1.87/mo).
Write-Host "Applying Artifact Registry cleanup policy (keep 3 latest, delete >7 days)..."
$CleanupPolicy = @'
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
'@

$PolicyFile = Join-Path $env:TEMP "ar-cleanup-policy.json"
# Note: Set-Content -Encoding UTF8 adds a BOM that gcloud rejects — use WriteAllText instead
[System.IO.File]::WriteAllText($PolicyFile, $CleanupPolicy.Trim(), [System.Text.UTF8Encoding]::new($false))

gcloud artifacts repositories set-cleanup-policies $Repo `
  --location=$Region `
  --policy=$PolicyFile `
  --no-dry-run

Remove-Item $PolicyFile -Force
Write-Host "Cleanup policy applied. Old images will be pruned automatically." -ForegroundColor Green

Write-Host ""
Write-Host "Telegram secrets (create/update from .env):"
Write-Host "  echo TOKEN | gcloud secrets versions add telegram-bot-token --data-file=-"
Write-Host "  echo CHAT  | gcloud secrets versions add telegram-chat-id --data-file=-"
Write-Host ""
Write-Host "Service URL:"
gcloud run services describe $Service --region $Region --format="value(status.url)"

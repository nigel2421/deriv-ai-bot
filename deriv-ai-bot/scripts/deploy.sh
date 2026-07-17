#!/usr/bin/env bash
# Build and run the bot with Docker Compose (demo by default).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "Missing .env — copy from .env.example and set DERIV_API_TOKEN"
  exit 1
fi

export MODE="${MODE:-demo}"
echo "Building and starting bot (MODE=$MODE)..."
docker compose up --build -d bot
docker compose ps
echo "Logs: docker compose logs -f bot"

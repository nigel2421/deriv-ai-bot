#!/bin/bash
echo "🚀 Deploying Deriv AI Bot..."

# Build and run Docker
docker-compose build
docker-compose up -d

echo "✅ Deployment complete! Check logs with: docker-compose logs -f"
echo "Monitor health: docker-compose ps"

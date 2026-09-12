#!/usr/bin/env bash
# ==============================================================================
# Deriv AI Trading Bot - Automated DigitalOcean VPS Provisioning & Hardening Script
# Target OS: Ubuntu 24.04 LTS / Ubuntu 22.04 LTS
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo "Starting VPS Provisioning for Deriv AI Trading Bot..."
echo "======================================================================"

# 1. Update system packages
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get upgrade -y
apt-get install -y \
    curl \
    git \
    ufw \
    fail2ban \
    htop \
    unzip \
    ca-certificates \
    gnupg \
    lsb-release \
    postgresql-client \
    nginx \
    certbot \
    python3-certbot-nginx

# 2. Configure UFW Firewall
echo "Configuring UFW Firewall..."
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp       # SSH
ufw allow 80/tcp       # HTTP
ufw allow 443/tcp      # HTTPS
ufw --force enable

# 3. Create non-root system user 'derivbot'
if ! id "derivbot" &>/dev/null; then
    echo "Creating system user 'derivbot'..."
    useradd -m -s /bin/bash derivbot
    usermod -aG sudo derivbot
fi

# 4. Install Docker and Docker Compose Plugin
echo "Installing Docker Engine & Compose..."
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg --yes
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
usermod -aG docker derivbot

# 5. Create Directory Structure under /bot
echo "Creating application directory structure under /bot..."
mkdir -p /bot/{config,engine,learning,database,strategies,logs,monitoring,api,dashboard,backups}
chown -R derivbot:derivbot /bot

echo "======================================================================"
echo "VPS Provisioning Complete!"
echo "User 'derivbot' created. Docker installed. Firewall enabled."
echo "Next step: Copy project files to /bot and run: docker compose up -d"
echo "======================================================================"

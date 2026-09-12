#!/usr/bin/env bash
# ==============================================================================
# Deriv AI Trading Bot - PostgreSQL Daily Automated Backup Script
# Performs pg_dump inside Docker container, compresses, and rotates backups.
# ==============================================================================

set -euo pipefail

BACKUP_DIR="/bot/backups/daily"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/deriv_trading_db_${TIMESTAMP}.sql.gz"
RETENTION_DAYS=7

mkdir -p "${BACKUP_DIR}"

echo "Starting PostgreSQL backup: ${BACKUP_FILE}..."

# Execute pg_dump inside bot-postgres container
docker exec bot-postgres pg_dump -U deriv_user -d deriv_trading_db | gzip > "${BACKUP_FILE}"

echo "Backup successful: ${BACKUP_FILE} (Size: $(du -sh "${BACKUP_FILE}" | cut -f1))"

# Delete backups older than RETENTION_DAYS
echo "Cleaning up backups older than ${RETENTION_DAYS} days..."
find "${BACKUP_DIR}" -name "deriv_trading_db_*.sql.gz" -mtime +${RETENTION_DAYS} -delete

echo "Backup rotation complete."

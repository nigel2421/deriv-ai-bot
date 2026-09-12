"""
Data Migration Script: JSONL Trade History -> PostgreSQL Database.
Reads existing data/trade_history.jsonl records and imports them into PostgreSQL.
"""
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.db import db_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migration")


def migrate_history(jsonl_path: Path):
    if not jsonl_path.exists():
        logger.warning("Trade history file %s not found. Skipping migration.", jsonl_path)
        return

    if not db_manager.enabled:
        logger.error("PostgreSQL connection is not enabled or available. Aborting migration.")
        sys.exit(1)

    logger.info("Starting migration from %s to PostgreSQL...", jsonl_path)
    migrated = 0
    failed = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # Ensure contract_id exists
                cid = record.get("contract_id") or record.get("id") or (100000000 + line_num)
                record["contract_id"] = int(cid)

                success = db_manager.record_trade(record)
                if success:
                    migrated += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error("Error migrating line %d: %s", line_num, e)
                failed += 1

    logger.info("Migration complete: %d trades migrated, %d failed/skipped.", migrated, failed)


if __name__ == "__main__":
    history_file = Path(os.getenv("TRADE_HISTORY_PATH", "data/trade_history.jsonl"))
    migrate_history(history_file)

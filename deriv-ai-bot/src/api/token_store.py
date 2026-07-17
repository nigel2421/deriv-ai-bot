"""Persist OAuth access tokens between restarts (ephemeral on Cloud Run unless volume)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

TOKEN_PATH = Path("data/oauth_token.json")


def save_token_payload(payload: Dict[str, Any]) -> Path:
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["saved_at"] = time.time()
    # normalize expiry
    if "expires_in" in data and "expires_at" not in data:
        try:
            data["expires_at"] = time.time() + float(data["expires_in"])
        except (TypeError, ValueError):
            pass
    TOKEN_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved OAuth token payload → %s", TOKEN_PATH)
    return TOKEN_PATH


def load_access_token() -> Optional[str]:
    if not TOKEN_PATH.is_file():
        return None
    try:
        data = json.loads(TOKEN_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to read token store: %s", e)
        return None
    exp = data.get("expires_at")
    if exp is not None:
        try:
            if time.time() > float(exp) - 60:
                logger.warning("Stored OAuth token expired")
                return None
        except (TypeError, ValueError):
            pass
    token = data.get("access_token")
    return str(token) if token else None


def clear_token() -> None:
    if TOKEN_PATH.is_file():
        TOKEN_PATH.unlink(missing_ok=True)

"""
Persist learning artifacts beyond Cloud Run ephemeral disk.

When LEARNING_GCS_URI (or HPP_GCS_URI) is set, sync JSON state files
to/from Google Cloud Storage on load and after saves.

Example:
  LEARNING_GCS_URI=gs://my-bucket/deriv-bot/learning/
  Files: learning_state.json, hpp_outcomes.json, calibration_outcomes.json, …

Without GCS, local data/ paths work as today (fine for local; ephemeral on Cloud Run).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

# Local relative paths that hold learning / attribution state
DEFAULT_LEARNING_FILES = (
    "data/learning_state.json",
    "data/hpp_outcomes.json",
    "data/hpp_peaks.json",
    "data/calibration_outcomes.json",
    "data/persistence_history.json",
    "data/contract_profile_learning.json",
    "data/deepseek_recommendations.json",
)


def gcs_uri_prefix() -> Optional[str]:
    raw = (
        os.getenv("LEARNING_GCS_URI")
        or os.getenv("LEARNING_GCS_PREFIX")
        or os.getenv("HPP_GCS_URI")
        or ""
    ).strip()
    if not raw:
        return None
    if not raw.startswith("gs://"):
        logger.warning("LEARNING_GCS_URI must start with gs:// (got %s)", raw[:40])
        return None
    return raw.rstrip("/") + "/"


def _parse_gs(uri: str) -> tuple:
    # gs://bucket/path/
    body = uri[5:]
    parts = body.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def pull_from_gcs(local_files: Optional[Iterable[str]] = None) -> List[str]:
    """Download learning files from GCS if configured. Returns list of restored paths."""
    prefix_uri = gcs_uri_prefix()
    if not prefix_uri:
        return []
    files = list(local_files or DEFAULT_LEARNING_FILES)
    restored: List[str] = []
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        logger.warning(
            "LEARNING_GCS_URI set but google-cloud-storage not installed — "
            "add google-cloud-storage to requirements-cloud.txt"
        )
        return []
    try:
        bucket_name, prefix = _parse_gs(prefix_uri)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        for rel in files:
            name = Path(rel).name
            blob = bucket.blob(f"{prefix}{name}")
            if not blob.exists():
                continue
            dest = Path(rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest))
            restored.append(str(dest))
            logger.info("Restored learning file from GCS: %s", dest)
    except Exception as e:
        logger.warning("GCS pull failed: %s", e)
    return restored


def push_to_gcs(local_files: Optional[Iterable[str]] = None) -> List[str]:
    """Upload learning files to GCS if configured."""
    prefix_uri = gcs_uri_prefix()
    if not prefix_uri:
        return []
    files = list(local_files or DEFAULT_LEARNING_FILES)
    uploaded: List[str] = []
    try:
        from google.cloud import storage  # type: ignore
    except ImportError:
        return []
    try:
        bucket_name, prefix = _parse_gs(prefix_uri)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        for rel in files:
            path = Path(rel)
            if not path.is_file():
                continue
            blob = bucket.blob(f"{prefix}{path.name}")
            blob.upload_from_filename(str(path))
            uploaded.append(str(path))
        if uploaded:
            logger.info("Pushed %d learning files to %s", len(uploaded), prefix_uri)
    except Exception as e:
        logger.warning("GCS push failed: %s", e)
    return uploaded


def bootstrap_learning_from_gcs() -> None:
    """Call once at process start (before AdaptiveLearner load)."""
    restored = pull_from_gcs()
    if restored:
        logger.info("Learning bootstrap: restored %d files from GCS", len(restored))

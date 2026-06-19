"""Durable on-disk job store for the Streamlit ICF app.

Jobs are keyed by authenticated user email so results survive browser
disconnects and page reloads (as long as the user returns to the same
Container Apps replica via sticky-session cookie).

Outputs are deleted after JOB_RETENTION_HOURS or when the user starts a
new generation for the same account.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JOBS_ROOT = Path(os.environ.get("ICF_JOBS_DIR", "/tmp/icf-jobs"))
JOB_RETENTION_HOURS = int(os.environ.get("ICF_JOB_RETENTION_HOURS", "3"))


def _safe_user_key(email: str) -> str:
    normalized = email.strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def user_jobs_dir(email: str) -> Path:
    return JOBS_ROOT / _safe_user_key(email)


def create_job(email: str, job_id: str, meta: dict[str, Any]) -> Path:
    """Create a job directory and write initial metadata."""
    job_dir = user_jobs_dir(email) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "job_id": job_id, "job_dir": str(job_dir)}
    _write_meta(job_dir, meta)
    return job_dir


def _write_meta(job_dir: Path, meta: dict[str, Any]) -> None:
    (job_dir / "meta.json").write_text(
        json.dumps(meta, default=str, indent=2),
        encoding="utf-8",
    )


def update_job_meta(job_dir: Path, **fields: Any) -> None:
    meta = json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))
    meta.update(fields)
    _write_meta(job_dir, meta)


def load_job_meta(job_dir: Path) -> dict[str, Any]:
    return json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))


def list_user_jobs(email: str) -> list[Path]:
    root = user_jobs_dir(email)
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / "meta.json").is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


_ACTIVE_JOB_STATUSES = frozenset({"running", "queued"})


def find_running_job(email: str) -> dict[str, Any] | None:
    """Return the user's in-flight job (queued or actively processing)."""
    for job_dir in list_user_jobs(email):
        meta = load_job_meta(job_dir)
        if meta.get("status") in _ACTIVE_JOB_STATUSES:
            meta["job_dir"] = str(job_dir)
            meta["job_id"] = job_dir.name
            return meta
    return None


def find_latest_completed_job(email: str) -> dict[str, Any] | None:
    for job_dir in list_user_jobs(email):
        meta = load_job_meta(job_dir)
        if meta.get("status") == "complete":
            meta["job_dir"] = str(job_dir)
            meta["job_id"] = job_dir.name
            return meta
    return None


def abandon_stale_running_jobs(max_age_hours: int = 3) -> None:
    """Mark orphaned 'running' jobs as failed (e.g. after container restart)."""
    cutoff = time.time() - max_age_hours * 3600
    if not JOBS_ROOT.is_dir():
        return
    for user_dir in JOBS_ROOT.iterdir():
        if not user_dir.is_dir():
            continue
        for job_dir in user_dir.iterdir():
            meta_path = job_dir / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = load_job_meta(job_dir)
            except (json.JSONDecodeError, OSError):
                continue
            if meta.get("status") not in _ACTIVE_JOB_STATUSES:
                continue
            if job_dir.stat().st_mtime >= cutoff:
                continue
            update_job_meta(
                job_dir,
                status="failed",
                error=(
                    "Job was interrupted (container restart or connection loss). "
                    "Please start a new generation."
                ),
                completed_at=datetime.now(timezone.utc).isoformat(),
            )


def load_outputs_from_job(job_dir: Path) -> list[tuple[str, bytes, str]]:
    meta = load_job_meta(job_dir)
    outputs: list[tuple[str, bytes, str]] = []
    for entry in meta.get("files", []):
        path = job_dir / entry["name"]
        if path.is_file():
            outputs.append((entry["name"], path.read_bytes(), entry["mime"]))
    return outputs


def purge_old_jobs() -> None:
    """Best-effort cleanup of expired job directories."""
    if not JOBS_ROOT.is_dir():
        return
    cutoff = time.time() - JOB_RETENTION_HOURS * 3600
    for user_dir in JOBS_ROOT.iterdir():
        if not user_dir.is_dir():
            continue
        for job_dir in user_dir.iterdir():
            if not job_dir.is_dir():
                continue
            try:
                if job_dir.stat().st_mtime < cutoff:
                    shutil.rmtree(job_dir, ignore_errors=True)
            except OSError:
                logger.warning("Could not purge job dir %s", job_dir)


def clear_user_jobs(email: str) -> None:
    shutil.rmtree(user_jobs_dir(email), ignore_errors=True)


def new_job_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

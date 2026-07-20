"""
Azure-backed job store, queue, and transient blob storage for the UHN AI-ICF tool.

All state lives in the ``aiicfstorage`` account so the UI container and the
worker job share one source of truth:

  - Storage Queue  ``icf-jobs``     : one small JSON message per submitted job
  - Blob container ``icf-input``    : uploaded protocol, deleted once processed
  - Blob container ``icf-output``   : generated DOCX, deleted after retention
  - Table          ``jobs``         : per-job status / metadata

Auth is keyless: every client uses ``DefaultAzureCredential`` (the container's
managed identity in Azure, your ``az login`` locally). No account keys or
connection strings for storage anywhere in the app.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from azure.core.exceptions import (
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
)
from azure.data.tables import TableServiceClient, UpdateMode
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.storage.queue import QueueServiceClient

log = logging.getLogger(__name__)


class ActiveJobExistsError(RuntimeError):
    """Raised when the user already has a queued or running job."""


# ---------------------------------------------------------------------------
# Configuration (all overridable via env; sensible defaults for this deployment)
# ---------------------------------------------------------------------------

ACCOUNT = os.environ.get("AZURE_STORAGE_ACCOUNT", "aiicfstorage")
QUEUE_NAME = os.environ.get("ICF_QUEUE_NAME", "icf-jobs")
INPUT_CONTAINER = os.environ.get("ICF_INPUT_CONTAINER", "icf-input")
OUTPUT_CONTAINER = os.environ.get("ICF_OUTPUT_CONTAINER", "icf-output")
JOBS_TABLE = os.environ.get("ICF_JOBS_TABLE", "jobs")
# Opportunistic deletion window for finished outputs / job rows (hours).
RETENTION_HOURS = int(os.environ.get("ICF_RETENTION_HOURS", "3"))
# Active (queued/running) jobs older than this are treated as abandoned.
STALE_HOURS = int(os.environ.get("ICF_STALE_HOURS", "12"))

_BLOB_EP = f"https://{ACCOUNT}.blob.core.windows.net"
_QUEUE_EP = f"https://{ACCOUNT}.queue.core.windows.net"
_TABLE_EP = f"https://{ACCOUNT}.table.core.windows.net"

# Study label -> registry filename under ``data/``. Shared by the UI (which
# offers the choice) and the worker (which resolves the actual path).
REGISTRY_FILES: dict[str, str] = {
    "Full Informed Consent Form": "UHN_standard_ICF_template_breakdown_new.json",
    "Minimal Risk Informed Consent Form": "minimal_risk_ICF_template_breakdown.json",
}

# ---------------------------------------------------------------------------
# Lazy, process-wide clients
# ---------------------------------------------------------------------------

_cred: DefaultAzureCredential | None = None
_blob_svc: BlobServiceClient | None = None
_queue_svc: QueueServiceClient | None = None
_table_svc: TableServiceClient | None = None
_infra_ready = False


def _credential() -> DefaultAzureCredential:
    global _cred
    if _cred is None:
        _cred = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    return _cred


def _blob_service() -> BlobServiceClient:
    global _blob_svc
    if _blob_svc is None:
        _blob_svc = BlobServiceClient(_BLOB_EP, credential=_credential())
    return _blob_svc


def _queue():
    global _queue_svc
    if _queue_svc is None:
        _queue_svc = QueueServiceClient(_QUEUE_EP, credential=_credential())
    return _queue_svc.get_queue_client(QUEUE_NAME)


def _table_service() -> TableServiceClient:
    global _table_svc
    if _table_svc is None:
        _table_svc = TableServiceClient(_TABLE_EP, credential=_credential())
    return _table_svc


def _container(name: str):
    return _blob_service().get_container_client(name)


def _jobs_table():
    return _table_service().get_table_client(JOBS_TABLE)


def ensure_infra() -> None:
    """Create the queue, containers, and tables if missing. Cheap, idempotent."""
    global _infra_ready
    if _infra_ready:
        return
    try:
        _queue_svc_local = _queue()
        try:
            _queue_svc_local.create_queue()
        except ResourceExistsError:
            pass
        for c in (INPUT_CONTAINER, OUTPUT_CONTAINER):
            try:
                _blob_service().create_container(c)
            except ResourceExistsError:
                pass
        for t in (JOBS_TABLE,):
            try:
                _table_service().create_table(t)
            except (ResourceExistsError, HttpResponseError):
                pass
        _infra_ready = True
    except Exception as ex:  # noqa: BLE001
        # Don't hard-fail the page on a transient control-plane hiccup; the
        # next call will retry. Surfaced in logs for diagnosis.
        log.warning("ensure_infra: %s: %s", type(ex).__name__, ex)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def new_job_id() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _q(s: str) -> str:
    """Escape a value for an OData filter literal."""
    return s.replace("'", "''")


def mime_for(suffix: str) -> str:
    return _mime_for(suffix)


def _mime_for(suffix: str) -> str:
    return {
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".json": "application/json",
        ".pdf": "application/pdf",
        ".txt": "text/plain",
    }.get(suffix.lower(), "application/octet-stream")


def _entity_to_job(e: Any) -> dict[str, Any]:
    def _jload(key: str, default: Any) -> Any:
        raw = e.get(key)
        if not raw:
            return default
        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            return default

    return {
        "owner": e.get("PartitionKey"),
        "job_id": e.get("RowKey"),
        "status": e.get("status"),
        "created_at": e.get("created_at"),
        "started_at": e.get("started_at"),
        "completed_at": e.get("completed_at"),
        "user_email": e.get("user_email"),
        "study_label": e.get("study_label"),
        "protocol_name": e.get("protocol_name"),
        "input_blob": e.get("input_blob"),
        "us_funded": bool(e.get("us_funded", False)),
        "sdm": bool(e.get("sdm", False)),
        "summary": _jload("summary_json", {}),
        "output_blobs": _jload("output_blobs_json", []),
        "error": e.get("error"),
        "email_status": _jload("email_status_json", None),
    }


def _merge(owner: str, job_id: str, fields: dict[str, Any]) -> None:
    entity = {"PartitionKey": owner.lower(), "RowKey": job_id, **fields}
    _jobs_table().update_entity(entity, mode=UpdateMode.MERGE)


# ---------------------------------------------------------------------------
# Producer side (UI): enqueue + read status + load outputs
# ---------------------------------------------------------------------------


def enqueue_job(
    owner: str,
    *,
    study_label: str,
    protocol_name: str,
    protocol_bytes: bytes,
    us_funded: bool,
    sdm: bool,
) -> str:
    """Persist the protocol + a job row, then enqueue a message. Returns job_id."""
    owner = owner.lower()
    if find_active_job(owner):
        raise ActiveJobExistsError(
            "You already have a generation in progress. Please wait for it to finish."
        )

    job_id = new_job_id()
    input_blob = f"{job_id}/{protocol_name}"
    blob_uploaded = False
    row_written = False
    try:
        _container(INPUT_CONTAINER).upload_blob(input_blob, protocol_bytes, overwrite=True)
        blob_uploaded = True

        _jobs_table().upsert_entity(
            {
                "PartitionKey": owner,
                "RowKey": job_id,
                "status": "queued",
                "created_at": now_iso(),
                "user_email": owner,
                "study_label": study_label,
                "protocol_name": protocol_name,
                "input_blob": input_blob,
                "us_funded": bool(us_funded),
                "sdm": bool(sdm),
            }
        )
        row_written = True

        _queue().send_message(
            json.dumps(
                {
                    "job_id": job_id,
                    "owner": owner,
                    "input_blob": input_blob,
                    "protocol_name": protocol_name,
                    "study_label": study_label,
                    "us_funded": bool(us_funded),
                    "sdm": bool(sdm),
                }
            )
        )
        return job_id
    except Exception:
        if row_written:
            try:
                _jobs_table().delete_entity(owner, job_id)
            except Exception:  # noqa: BLE001
                pass
        if blob_uploaded:
            try:
                _container(INPUT_CONTAINER).delete_blob(input_blob)
            except Exception:  # noqa: BLE001
                pass
        raise


def get_job(owner: str, job_id: str) -> dict[str, Any] | None:
    try:
        return _entity_to_job(_jobs_table().get_entity(owner.lower(), job_id))
    except ResourceNotFoundError:
        return None


def find_active_job(owner: str) -> dict[str, Any] | None:
    f = (
        f"PartitionKey eq '{_q(owner.lower())}' "
        "and (status eq 'queued' or status eq 'running')"
    )
    for e in _jobs_table().query_entities(f):
        return _entity_to_job(e)
    return None


def find_latest_completed_job(owner: str) -> dict[str, Any] | None:
    f = f"PartitionKey eq '{_q(owner.lower())}' and status eq 'complete'"
    rows = [_entity_to_job(e) for e in _jobs_table().query_entities(f)]
    if not rows:
        return None
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows[0]


def load_outputs(owner: str, job_id: str) -> list[tuple[str, bytes, str]]:
    """Download the generated files for a job from the output container."""
    cc = _container(OUTPUT_CONTAINER)
    out: list[tuple[str, bytes, str]] = []
    for b in cc.list_blobs(name_starts_with=f"{job_id}/"):
        name = b.name.split("/", 1)[1]
        data = cc.get_blob_client(b.name).download_blob().readall()
        out.append((name, data, _mime_for(Path(name).suffix)))
    return out


# ---------------------------------------------------------------------------
# Consumer side (worker): receive + status transitions + I/O
# ---------------------------------------------------------------------------


def receive_message(visibility_timeout: int = 5400):
    """Receive at most one message, hidden for ``visibility_timeout`` seconds.

    Returns ``(message, job_dict)`` or ``(None, None)`` if the queue is empty.
    """
    for m in _queue().receive_messages(
        max_messages=1, visibility_timeout=visibility_timeout
    ):
        try:
            return m, json.loads(m.content)
        except Exception:  # noqa: BLE001
            return m, {}
    return None, None


def delete_message(msg) -> None:
    _queue().delete_message(msg)


def mark_running(owner: str, job_id: str) -> None:
    _merge(owner, job_id, {"status": "running", "started_at": now_iso()})


def download_input(input_blob: str) -> tuple[str, bytes]:
    data = _container(INPUT_CONTAINER).get_blob_client(input_blob).download_blob().readall()
    name = input_blob.split("/", 1)[1] if "/" in input_blob else input_blob
    return name, data


def _delete_input(owner: str, job_id: str) -> None:
    job = get_job(owner, job_id)
    if job and job.get("input_blob"):
        try:
            _container(INPUT_CONTAINER).delete_blob(job["input_blob"])
        except Exception:  # noqa: BLE001
            pass


def complete_job(
    owner: str,
    job_id: str,
    outputs: list[tuple[str, bytes, str]],
    summary: dict | None,
) -> None:
    cc = _container(OUTPUT_CONTAINER)
    names: list[str] = []
    for name, data, _mime in outputs:
        cc.upload_blob(f"{job_id}/{name}", data, overwrite=True)
        names.append(name)
    _merge(
        owner,
        job_id,
        {
            "status": "complete",
            "completed_at": now_iso(),
            "summary_json": json.dumps(summary or {}),
            "output_blobs_json": json.dumps(names),
        },
    )
    _delete_input(owner, job_id)


def fail_job(owner: str, job_id: str, error: str) -> None:
    _merge(
        owner,
        job_id,
        {"status": "failed", "completed_at": now_iso(), "error": str(error)[:30000]},
    )
    _delete_input(owner, job_id)


def set_email_status(owner: str, job_id: str, ok: bool, message: str) -> None:
    _merge(owner, job_id, {"email_status_json": json.dumps([bool(ok), str(message)])})


# ---------------------------------------------------------------------------
# Retention / cleanup (opportunistic; called on UI load)
# ---------------------------------------------------------------------------


def purge_expired() -> None:
    """Delete finished outputs + job rows past retention; abandon stale actives.

    A 1-day Storage lifecycle policy is the guaranteed backstop (see setup
    commands); this keeps the typical case close to the stated retention window.
    """
    now = datetime.now(timezone.utc)
    retain_cutoff = now - timedelta(hours=RETENTION_HOURS)
    stale_cutoff = now - timedelta(hours=STALE_HOURS)

    # Output blobs past retention; input blobs past the stale window (the worker
    # normally deletes inputs on completion — this is a safety net).
    try:
        oc = _container(OUTPUT_CONTAINER)
        for b in oc.list_blobs():
            ts = getattr(b, "creation_time", None) or b.last_modified
            if ts and ts < retain_cutoff:
                try:
                    oc.delete_blob(b.name)
                except Exception:  # noqa: BLE001
                    pass
    except Exception as ex:  # noqa: BLE001
        log.warning("purge outputs: %s: %s", type(ex).__name__, ex)

    try:
        ic = _container(INPUT_CONTAINER)
        for b in ic.list_blobs():
            ts = getattr(b, "creation_time", None) or b.last_modified
            if ts and ts < stale_cutoff:
                try:
                    ic.delete_blob(b.name)
                except Exception:  # noqa: BLE001
                    pass
    except Exception as ex:  # noqa: BLE001
        log.warning("purge inputs: %s: %s", type(ex).__name__, ex)

    # Job rows: drop terminal rows past retention; mark very old actives failed.
    try:
        t = _jobs_table()
        for e in t.list_entities():
            created = e.get("created_at")
            try:
                dt = datetime.fromisoformat(created)
            except Exception:  # noqa: BLE001
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            status = e.get("status")
            if status in ("complete", "failed") and dt < retain_cutoff:
                try:
                    t.delete_entity(e["PartitionKey"], e["RowKey"])
                except Exception:  # noqa: BLE001
                    pass
            elif status in ("queued", "running") and dt < stale_cutoff:
                try:
                    _merge(
                        e["PartitionKey"],
                        e["RowKey"],
                        {
                            "status": "failed",
                            "error": "Abandoned (exceeded stale window).",
                            "completed_at": now_iso(),
                        },
                    )
                except Exception:  # noqa: BLE001
                    pass
    except Exception as ex:  # noqa: BLE001
        log.warning("purge rows: %s: %s", type(ex).__name__, ex)

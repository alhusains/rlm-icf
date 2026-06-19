"""
UHN ICF Automation — Streamlit web wrapper around the rlm-icf pipeline.

In production (Azure Container Apps) the same file runs as the entrypoint —
no code changes between local and deployed.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import tempfile
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from icf.job_store import (
    abandon_stale_running_jobs,
    create_job,
    find_latest_completed_job,
    find_running_job,
    load_job_meta,
    load_outputs_from_job,
    new_job_id,
    purge_old_jobs,
    update_job_meta,
)
from icf.pipeline import ICFPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
LOGO_PATH = REPO_ROOT / "data" / "UHN_logo.png"
# Load once at import — byte payloads are more reliable than filesystem paths
# inside Streamlit, especially behind Container Apps ingress.
LOGO_BYTES: bytes | None = LOGO_PATH.read_bytes() if LOGO_PATH.is_file() else None

REGISTRIES: dict[str, Path] = {
    "Full Informed Consent Form": REPO_ROOT / "data" / "UHN_standard_ICF_template_breakdown_new.json",
    "Minimal Risk Informed Consent Form": REPO_ROOT / "data" / "minimal_risk_ICF_template_breakdown.json",
}

REQUIRED_ENV = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
]

# ---------------------------------------------------------------------------
# Email delivery (Azure Communication Services)
# ---------------------------------------------------------------------------
# Optional feature. When both vars are present, generated ICFs are emailed to
# the user. If they're absent, the app still works fully via download — email
# delivery just degrades silently so a transient ACS issue can't block the tool.

ACS_CONNECTION_STRING = os.environ.get("ACS_CONNECTION_STRING")
ACS_SENDER_ADDRESS = os.environ.get("ACS_SENDER_ADDRESS")
EMAIL_ENABLED = bool(ACS_CONNECTION_STRING and ACS_SENDER_ADDRESS)

# Only one pipeline per container replica — concurrent runs on 1 CPU / 2 GiB
# cause API contention and section failures when colleagues test together.
_PIPELINE_SLOT = threading.Semaphore(1)


def get_user_email() -> str | None:
    """Best-effort read of the signed-in user's email from Entra Easy Auth.

    Azure Container Apps Easy Auth injects the authenticated principal into the
    request headers. Requires Streamlit >= 1.37 for ``st.context.headers``.
    Returns None if no email can be determined (e.g. running locally without auth).
    """
    try:
        headers = dict(st.context.headers or {})
    except Exception:
        return None

    # Normalise to lowercase keys (header lookups should be case-insensitive).
    lower = {k.lower(): v for k, v in headers.items()}

    name = lower.get("x-ms-client-principal-name")
    if name and "@" in name:
        return name

    raw = lower.get("x-ms-client-principal")
    if raw:
        try:
            decoded = json.loads(base64.b64decode(raw).decode("utf-8"))
            for claim in decoded.get("claims", []):
                if claim.get("typ", "").lower() in (
                    "preferred_username",
                    "email",
                    "emails",
                    "upn",
                    "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
                ):
                    val = claim.get("val")
                    if val and "@" in val:
                        return val
        except Exception:
            return None
    return None


def send_output_email(
    to_email: str,
    outputs: list[tuple[str, bytes, str]],
    protocol_name: str,
) -> tuple[bool, str]:
    """Email the generated ICF documents as attachments. Returns (ok, message).

    Only the two primary deliverables (draft + marked-up) are attached. The bytes
    come straight from memory — nothing is written to durable storage server-side.
    """
    if not EMAIL_ENABLED:
        return False, "Email delivery is not configured on this deployment."

    attachments = [
        {
            "name": name,
            "contentType": mime,
            "contentInBase64": base64.b64encode(data).decode("utf-8"),
        }
        for name, data, mime in outputs
        if name.startswith(("draft_icf_", "marked_up_icf_"))
    ]
    if not attachments:
        return False, "No ICF documents were produced to email."

    try:
        from azure.communication.email import EmailClient

        client = EmailClient.from_connection_string(ACS_CONNECTION_STRING)
        message = {
            "senderAddress": ACS_SENDER_ADDRESS,
            "recipients": {"to": [{"address": to_email}]},
            "content": {
                "subject": "Your AI-ICF draft is ready",
                "plainText": (
                    "Your AI-generated draft Informed Consent Form is attached.\n\n"
                    f"Source protocol: {protocol_name}\n\n"
                    "Two documents are attached:\n"
                    "  - Draft version: your working copy to review and update prior "
                    "to CAPCR submission.\n"
                    "  - Marked-up version: traceability to the protocol content used "
                    "by the AI, plus an appendix of suggested plain-language "
                    "improvements.\n\n"
                    "This is an AI-generated draft. All content must be carefully "
                    "reviewed and verified by qualified research personnel before "
                    "submission to the REB or use with study participants.\n\n"
                    "- UHN AI-ICF Tool (internal use only)"
                ),
            },
            "attachments": attachments,
        }
        poller = client.begin_send(message)
        result = poller.result()
        status = str((result or {}).get("status", "Unknown"))
        if status.lower() == "succeeded":
            return True, f"A copy was emailed to {to_email}."
        return False, f"Email send finished with status: {status}."
    except Exception as ex:  # noqa: BLE001
        return False, f"Could not send email: {type(ex).__name__}: {ex}"

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="UHN ICF Generator",
    page_icon="📋",
    layout="centered",
)

# Minimal custom CSS: tighten spacing and add subtle card styling
st.markdown(
    """
    <style>
    .stAlert > div { border-radius: 8px; }
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    .step-label {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6c757d;
        margin-bottom: 0.25rem;
    }
    .beta-badge {
        display: inline-block;
        background-color: #ffc107;
        color: #000;
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        padding: 2px 7px;
        border-radius: 4px;
        vertical-align: middle;
        margin-left: 10px;
        text-transform: uppercase;
    }
    .recommended-use-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-left: 4px solid #0d6efd;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 1rem 0 0.25rem 0;
    }
    .recommended-use-box h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: #212529;
    }
    .recommended-use-box ul {
        margin: 0;
        padding-left: 1.25rem;
    }
    .recommended-use-box li {
        margin-bottom: 0.35rem;
        font-size: 0.92rem;
        line-height: 1.45;
        color: #495057;
    }
    .recommended-use-box li:last-child {
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def check_environment() -> list[str]:
    """Return a list of human-readable problems with the runtime environment."""
    problems: list[str] = []
    missing = [v for v in REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        problems.append("Missing environment variable(s): " + ", ".join(missing))
    for label, path in REGISTRIES.items():
        if not path.exists():
            problems.append(f"{label} template registry not found: {path}")
    return problems


env_problems = check_environment()
if env_problems:
    st.error("This deployment is misconfigured — please contact your administrator:")
    for p in env_problems:
        st.write(f"• {p}")
    st.stop()

# ---------------------------------------------------------------------------
# First-visit disclaimer (modal gate)
# ---------------------------------------------------------------------------

if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False


@st.dialog("Important Notice", width="large")
def show_disclaimer() -> None:
    st.markdown(
        "Before using the **AI-ICF Tool**, please read and acknowledge the following:"
    )
    st.markdown(
        "- This tool generates an **AI-assisted draft** Informed Consent Form (ICF) from "
        "study protocols. Output may contain errors, omissions, or inaccuracies.\n"
        "- Full responsibility for reviewing, verifying, revising, and approving all "
        "consent documentation rests with the study team. Generated drafts must not "
        "be submitted to the REB or shared with study participants without thorough "
        "human review and approval.\n"
        "- This tool does not replace legal, regulatory, ethical, or clinical review."
    )
    st.divider()
    accepted = st.checkbox(
        "I understand and agree to review the AI-generated draft ICF before submitting it to the REB",
        value=False,
    )
    if st.button(
        "Continue to AI-ICF Tool",
        type="primary",
        use_container_width=True,
        disabled=not accepted,
    ):
        st.session_state.disclaimer_accepted = True
        st.rerun()


if not st.session_state.disclaimer_accepted:
    show_disclaimer()
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — About
# ---------------------------------------------------------------------------

with st.sidebar:
    if LOGO_BYTES:
        st.image(LOGO_BYTES, width="stretch")
        st.markdown("")


    st.warning(
        "**Beta Version**\n\n"
        "This tool is under active development. AI-generated output may contain "
        "errors, omissions, or inaccuracies, particularly for complex or "
        "non-standard study designs. As a trial deployment participant, we ask that "
        "you please provide the AI-ICF team with feedback to enable continuous "
        "improvement of the tool."
    )

    st.markdown("## About this tool")
    st.markdown(
        "**AI-ICF** is an AI-assisted tool that helps research study teams at the "
        "University Health Network (UHN) generate structured draft Informed Consent "
        "Forms (ICFs) from study protocols."
    )

    st.markdown("**How it works**")
    st.markdown(
        "1. Select the consent form template\n"
        "2. Upload your study protocol (PDF or DOCX)\n"
        "3. The tool reads the protocol and extracts relevant information "
        "for each ICF section\n"
        "4. A structured draft ICF is generated and ready for your review"
    )

    st.markdown("**What you get**")
    st.markdown(
        "Two versions of the consent form are generated:\n\n"
        "1. **Draft version:** your working copy to review and update prior to "
        "CAPCR submission.\n\n"
        "2. **Marked-up version:** provides traceability to protocol content used "
        "by the AI and includes an appendix with suggested plain language improvements."
    )

    st.markdown("**Privacy and Confidentiality**")
    st.markdown(
        "- All processing runs entirely within UHN's secure environment. No data "
        "ever leaves UHN's managed environment.\n"
        "- Protocols and draft ICFs are kept only during processing and up to 3 hours "
        "after generation for download, then deleted automatically. "
        "Only limited metadata (usage data) is retained for audit purposes.\n"
        "- No AI model training/learning occurs with submitted protocols or generated "
        "draft consent forms. Submitted protocols are used solely for consent form "
        "generation.\n"
        "- For more information, please email [agata.misiura@uhn.ca](mailto:agata.misiura@uhn.ca)"
    )

    st.divider()

    st.error(
        "**Important Disclaimers**\n\n"
        "- All generated drafts **must be carefully reviewed** by the study team "
        "prior to submission to the REB. Do not submit AI-generated content directly "
        "to the REB or to study participants without thorough human review and approval.\n"
        "- This tool does **not replace** legal, regulatory, ethical, or clinical "
        "review of consent documentation.\n"
        "- For **internal UHN use only**."
    )

    st.divider()
    st.caption(
        f"Model: `{os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'unknown')}`\n\n"
        "For questions, please contact the project team: "
        "[agata.misiura@uhn.ca](mailto:agata.misiura@uhn.ca)"
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    '<h1 style="margin-bottom:0.1rem">AI-ICF Tool '
    '<span class="beta-badge">Beta</span></h1>',
    unsafe_allow_html=True,
)
st.markdown(
    "Generate a structured draft Informed Consent Form (ICF) from a study protocol "
    "in minutes. Complete the steps below to get started."
)

st.markdown(
    """
    <div class="recommended-use-box">
        <h4>Recommended Use</h4>
        <ul>
            <li>Limited to main ICFs (minimal risk and above minimal risk studies)</li>
            <li>At this point, only protocols can be used as the source document</li>
            <li>Currently optimized for investigator-initiated, non-complex, single-arm/cohort studies. Support
                for more complex study designs is under development.</li>
            <li>Excludes: CTO studies, the input of sponsor/CRO provided consent forms,
                and the generation of other consent forms (e.g. optional consent forms,
                pregnancy follow-up, etc.)</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

for key in (
    "run_outputs",
    "run_summary",
    "run_error",
    "email_status",
    "protocol_bytes",
    "protocol_name",
    "pipeline_job",
):
    if key not in st.session_state:
        st.session_state[key] = None


def _pipeline_is_running() -> bool:
    job = st.session_state.get("pipeline_job")
    if isinstance(job, dict) and job.get("status") in ("running", "queued"):
        return True
    email = get_user_email()
    if email and find_running_job(email):
        return True
    return False


def _meta_to_pipeline_job(meta: dict[str, Any]) -> dict[str, Any]:
    started_raw = meta.get("started_at")
    started_at = (
        datetime.fromisoformat(started_raw)
        if isinstance(started_raw, str)
        else datetime.now()
    )
    return {
        "status": meta.get("status"),
        "job_id": meta.get("job_id"),
        "job_dir": meta.get("job_dir"),
        "started_at": started_at,
        "study_label": meta.get("study_label"),
        "protocol_name": meta.get("protocol_name"),
        "us_funded": meta.get("us_funded", False),
        "sdm": meta.get("sdm", False),
    }


def _restore_session_from_disk() -> None:
    """Re-attach an in-progress job after Streamlit session TTL or page reload."""
    email = get_user_email()
    if not email:
        return

    if isinstance(st.session_state.get("pipeline_job"), dict):
        return

    running = find_running_job(email)
    if running:
        st.session_state.pipeline_job = _meta_to_pipeline_job(running)


purge_old_jobs()
abandon_stale_running_jobs()
_restore_session_from_disk()


# ---------------------------------------------------------------------------
# Step 1 — Study type
# ---------------------------------------------------------------------------

st.markdown('<p class="step-label">Step 1 of 4</p>', unsafe_allow_html=True)
st.subheader("Select consent form template")
st.markdown("Choose the consent form template for your study.")

STUDY_OPTIONS: dict[str, str | None] = {
    "— Select a study type —": None,
    "Above Minimal Risk Informed Consent Form": "Full Informed Consent Form",
    "Minimal Risk Informed Consent Form": "Minimal Risk Informed Consent Form",
}

selected_study_label = st.selectbox(
    "Study type",
    options=list(STUDY_OPTIONS.keys()),
    index=0,
    label_visibility="collapsed",
)
selected_study: str | None = STUDY_OPTIONS[selected_study_label]  # type: ignore[assignment]

if selected_study == "Full Informed Consent Form":
    st.success(
        "**Above Minimal Risk Informed Consent Form** selected. This template is suitable for studies involving more than minimal risk."
    )
elif selected_study == "Minimal Risk Informed Consent Form":
    st.success(
        "**Minimal Risk Informed Consent Form** selected. The simplified ICF template will be used, suitable for studies where risks are no greater than those of everyday life."
    )

if selected_study is not None:
    st.checkbox(
        "Is this study funded or supported by a US federal funding agency "
        "(e.g., NIH, DHHS, etc.)?",
        value=False,
        key="us_federal_funding",
    )
    st.caption(
        "Select this field if your study is US federally funded or supported to generate "
        "the **Summary of Informed Consent Form**, which is required by US federal "
        "regulations."
    )
    st.checkbox(
        "Is this form intended to be completed by a substitute decision maker (SDM)?",
        value=False,
        key="sdm_form",
    )
    st.caption(
        "If you select this, the generated consent form will include language and signature space for the SDM."
    )

st.markdown(
    "Unsure which template to use for your study? "
    "[Click here](https://intranet.uhnresearch.ca/service/documents-and-forms)"
)

st.divider()

# ---------------------------------------------------------------------------
# Step 2 — Upload protocol
# ---------------------------------------------------------------------------

st.markdown('<p class="step-label">Step 2 of 4</p>', unsafe_allow_html=True)
st.subheader("Upload protocol")

if selected_study is None:
    st.info("Complete Step 1 to enable protocol upload.")
else:
    st.markdown(
        "Upload your study protocol. Accepted formats: **PDF** or **DOCX** only.\n\n"
        "> **Note:** Legacy `.doc` files are **not** accepted. If your protocol is in "
        "`.doc` format, please do the following before uploading: "
        "(1) Open the file in Word; "
        "(2) Click **File**; "
        "(3) Select **Save As**; "
        "(4) Select the folder to save your file in; "
        '(5) Select **Word Document (*.docx)** in the **Save as type** field; '
        "(6) Click **Save**."
    )
    uploaded = st.file_uploader(
        "Clinical protocol",
        type=["pdf", "docx"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key="protocol_uploader",
    )
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        if file_bytes:
            st.session_state.protocol_bytes = file_bytes
            st.session_state.protocol_name = uploaded.name
        elif st.session_state.protocol_bytes is None:
            st.warning(
                "The file is still loading. Wait a moment, or upload again if this message persists."
            )
    else:
        st.session_state.protocol_bytes = None
        st.session_state.protocol_name = None

    if st.session_state.protocol_bytes and st.session_state.protocol_name:
        size_kb = len(st.session_state.protocol_bytes) / 1024
        st.success(
            f"Protocol uploaded: **{st.session_state.protocol_name}** ({size_kb:.1f} KB)"
        )

st.divider()

# ---------------------------------------------------------------------------
# Step 3 — Generate
# ---------------------------------------------------------------------------

st.markdown('<p class="step-label">Step 3 of 4</p>', unsafe_allow_html=True)
st.subheader("Generate ICF")

protocol_ready = bool(st.session_state.protocol_bytes and st.session_state.protocol_name)

if selected_study is None or not protocol_ready:
    st.info(
        "Complete Steps 1 and 2 to start generating your ICF."
    )
    generate_clicked = False
else:
    study_display = (
        "Above Minimal Risk Informed Consent Form"
        if selected_study == "Full Informed Consent Form"
        else selected_study
    )
    st.markdown(
        f"Everything is ready. Click **Generate ICF** to begin processing "
        f"the **{st.session_state.protocol_name}** protocol as a **{study_display}**.\n\n"
        "This typically takes **20–30 minutes**. Processing time may vary based on "
        "protocol length and level of detail.\n\n"
        "Once ready, your draft consent form will be available for download on this "
        "page. A copy will also be emailed to you. Before submitting to CAPCR, please "
        "review, revise and modify the form as required."
    )

    # The finished ICF is always sent to the verified UHN sign-in address.
    # The field is read-only by design: a user can only ever receive their own ICF.
    detected_email = get_user_email()
    if EMAIL_ENABLED:
        if detected_email:
            st.text_input(
                "Email address for delivery",
                value=detected_email,
                disabled=True,
                help="Your finished ICF is sent to your verified UHN address. "
                "This cannot be changed.",
            )
        else:
            st.warning(
                "We couldn't read your UHN email from your sign-in, so a copy "
                "can't be emailed automatically. You'll still be able to download "
                "your ICF on this page once it's ready."
            )
    else:
        st.caption(
            "Email delivery isn't configured on this deployment — your ICF will "
            "still be available to download on this page once it's ready."
        )

    generate_clicked = st.button(
        "Generate ICF",
        type="primary",
        use_container_width=True,
        disabled=_pipeline_is_running(),
    )

# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def _mime_for(suffix: str) -> str:
    return {
        ".docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ".json": "application/json",
        ".pdf": "application/pdf",
        ".txt": "text/plain",
    }.get(suffix.lower(), "application/octet-stream")


def _pipeline_worker(job: dict[str, Any], job_dir: Path) -> None:
    """Run the ICF pipeline in a background thread (20–30 min). Mutates *job* and disk meta."""
    job["status"] = "queued"
    update_job_meta(job_dir, status="queued")
    _PIPELINE_SLOT.acquire()  # wait for this replica's single processing slot
    job["status"] = "running"
    update_job_meta(job_dir, status="running", processing_started_at=datetime.now().isoformat())
    workdir = Path(tempfile.mkdtemp(prefix="icfrun_"))
    try:
        protocol_path = workdir / job["protocol_name"]
        protocol_path.write_bytes(job["protocol_bytes"])

        out_dir = job_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        pipeline = ICFPipeline(
            protocol_path=str(protocol_path),
            template_path=job["registry_path"],
            output_dir=str(out_dir),
            model_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            backend="azure_openai",
            backend_kwargs={
                "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
                "azure_deployment": os.environ["AZURE_OPENAI_DEPLOYMENT"],
            },
            extraction_backend="rlm",
            verbose=False,
            skip_review=False,
            us_funded=job["us_funded"],
            sdm=job["sdm"],
            debug_log_dir=None,
        )

        result = pipeline.run()

        files_meta: list[dict[str, str]] = []
        outputs: list[tuple[str, bytes, str]] = []
        for f in sorted(out_dir.iterdir()):
            if not f.is_file():
                continue
            dest = job_dir / f.name
            shutil.copy2(f, dest)
            mime = _mime_for(f.suffix)
            files_meta.append({"name": f.name, "mime": mime})
            outputs.append((f.name, dest.read_bytes(), mime))

        summary = getattr(result, "summary", None)
        job["outputs"] = outputs
        job["summary"] = summary
        job["status"] = "complete"
        update_job_meta(
            job_dir,
            status="complete",
            summary=summary,
            files=files_meta,
            completed_at=datetime.now().isoformat(),
        )
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        job["error"] = err
        job["status"] = "failed"
        update_job_meta(job_dir, status="failed", error=err, completed_at=datetime.now().isoformat())
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        _PIPELINE_SLOT.release()


def _job_status_from_disk(job: dict[str, Any]) -> str | None:
    job_dir = job.get("job_dir")
    if not job_dir:
        return job.get("status")
    meta_path = Path(job_dir) / "meta.json"
    if not meta_path.is_file():
        return job.get("status")
    return load_job_meta(Path(job_dir)).get("status")


def _finish_pipeline_job(job: dict[str, Any]) -> None:
    """Copy a finished background job into session state and send email once."""
    job_dir = Path(job["job_dir"]) if job.get("job_dir") else None
    if job_dir and job_dir.is_dir():
        disk_outputs = load_outputs_from_job(job_dir)
        if disk_outputs:
            job["outputs"] = disk_outputs
        disk_meta = load_job_meta(job_dir)
        job["summary"] = disk_meta.get("summary", job.get("summary"))
        if disk_meta.get("status") == "failed":
            job["status"] = "failed"
            job["error"] = disk_meta.get("error", job.get("error"))

    if job.get("status") == "complete":
        st.session_state.run_outputs = job.get("outputs")
        st.session_state.run_summary = job.get("summary")
        st.session_state.run_error = None
        st.session_state.email_status = None
        email_status: tuple[bool, str] | None = None
        if EMAIL_ENABLED:
            recipient = (get_user_email() or "").strip()
            if recipient and "@" in recipient:
                email_status = send_output_email(
                    recipient,
                    st.session_state.run_outputs or [],
                    job["protocol_name"],
                )
                st.session_state.email_status = email_status
            else:
                email_status = (
                    False,
                    "We couldn't determine your UHN email from your sign-in, so no "
                    "copy was emailed. You can still download below.",
                )
                st.session_state.email_status = email_status
        if job_dir and email_status is not None:
            update_job_meta(job_dir, email_status=list(email_status))
    elif job.get("status") == "failed":
        st.session_state.run_error = job.get("error")
        st.session_state.run_outputs = None
        st.session_state.run_summary = None
    st.session_state.pipeline_job = None


def _format_elapsed(started_at: datetime) -> str:
    total_seconds = int((datetime.now() - started_at).total_seconds())
    mins, secs = divmod(total_seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}h {mins:02d}m {secs:02d}s"
    return f"{mins}m {secs:02d}s"


if (
    generate_clicked
    and protocol_ready
    and selected_study is not None
    and not _pipeline_is_running()
):
    user_email = get_user_email()
    if user_email and find_running_job(user_email):
        st.warning(
            "You already have a generation in progress for your account. "
            "Please wait for it to finish, or reload this page to reconnect to it."
        )
    else:
        job_id = new_job_id()
        owner = user_email or "anonymous"
        job_dir = create_job(
            owner,
            job_id,
            {
                "status": "running",
                "started_at": datetime.now().isoformat(),
                "study_label": selected_study,
                "protocol_name": st.session_state.protocol_name,
                "registry_path": str(REGISTRIES[selected_study]),
                "us_funded": bool(st.session_state.get("us_federal_funding", False)),
                "sdm": bool(st.session_state.get("sdm_form", False)),
            },
        )
        st.session_state.pipeline_job = {
            "status": "running",
            "job_id": job_id,
            "job_dir": str(job_dir),
            "started_at": datetime.now(),
            "study_label": selected_study,
            "protocol_name": st.session_state.protocol_name,
            "protocol_bytes": st.session_state.protocol_bytes,
            "registry_path": str(REGISTRIES[selected_study]),
            "us_funded": bool(st.session_state.get("us_federal_funding", False)),
            "sdm": bool(st.session_state.get("sdm_form", False)),
        }
        st.session_state.run_outputs = None
        st.session_state.run_summary = None
        st.session_state.run_error = None
        st.session_state.email_status = None
        threading.Thread(
            target=_pipeline_worker,
            args=(st.session_state.pipeline_job, job_dir),
            daemon=True,
        ).start()
        st.rerun()

_progress_interval = timedelta(seconds=5) if _pipeline_is_running() else None


@st.fragment(run_every=_progress_interval)
def _render_generation_progress() -> None:
    """Heartbeat UI while the pipeline runs — keeps the browser WebSocket alive."""
    job = st.session_state.get("pipeline_job")
    if not isinstance(job, dict):
        return

    status = _job_status_from_disk(job) or job.get("status")
    if status == "queued":
        started_at: datetime = job["started_at"]
        with st.status("Queued — waiting for a processing slot…", expanded=True) as status_box:
            status_box.write(f"Template: **{job['study_label']}**")
            status_box.write(f"Protocol: **{job['protocol_name']}**")
            status_box.write(f"Waiting: **{_format_elapsed(started_at)}**")
            status_box.write(
                "Another generation may be finishing on this server. "
                "Your job will start automatically — no need to refresh."
            )
    elif status == "running":
        started_at: datetime = job["started_at"]
        with st.status("Generating ICF — please wait…", expanded=True) as status_box:
            status_box.write(f"Template: **{job['study_label']}**")
            if job.get("us_funded"):
                status_box.write(
                    "US federal funding: **yes** (Summary of ICF sections will be included)"
                )
            if job.get("sdm"):
                status_box.write("Substitute decision maker (SDM): **yes**")
            status_box.write(f"Protocol: **{job['protocol_name']}**")
            status_box.write(f"Elapsed: **{_format_elapsed(started_at)}**")
            status_box.write(
                "Extracting and synthesising information from the protocol. "
                "This typically takes 20–30 minutes. Processing time may vary based on "
                "protocol length and level of detail."
            )
            status_box.write(
                "**You can switch to other tabs** while this runs. If this page reloads, "
                "open it again with the same browser and your job will reconnect automatically. "
                "Finished files are also emailed to you."
            )
    elif status in ("complete", "failed"):
        job["status"] = status
        if status == "failed" and job.get("job_dir"):
            job["error"] = load_job_meta(Path(job["job_dir"])).get("error")
        _finish_pipeline_job(job)
        st.rerun(scope="app")


_render_generation_progress()

# ---------------------------------------------------------------------------
# Optional: reload a previous completed run (not automatic — avoids stale UI)
# ---------------------------------------------------------------------------

_user_email = get_user_email()
if (
    _user_email
    and not st.session_state.run_outputs
    and not st.session_state.run_error
    and not _pipeline_is_running()
):
    _prior = find_latest_completed_job(_user_email)
    if _prior:
        _prior_summary = _prior.get("summary") or {}
        _prior_errors = _prior_summary.get("errors", 0)
        _prior_label = _prior.get("protocol_name", "unknown file")
        if st.button(
            f"Load results from previous run ({_prior_label})",
            help="Loads your most recent completed generation from the last 3 hours.",
        ):
            st.session_state.run_outputs = load_outputs_from_job(Path(_prior["job_dir"]))
            st.session_state.run_summary = _prior_summary
            st.session_state.run_error = None
            if _prior.get("email_status") is not None:
                raw = _prior["email_status"]
                st.session_state.email_status = (bool(raw[0]), str(raw[1]))
            st.rerun()
        if _prior_errors:
            st.caption(
                f"Note: that run reported **{_prior_errors}** section error(s). "
                "Start a new generation if you intended to process a different protocol."
            )

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if st.session_state.run_error:
    st.error("The pipeline encountered an error.")
    with st.expander("Error details", expanded=False):
        st.code(st.session_state.run_error)

if st.session_state.run_outputs:
    st.divider()
    st.subheader("Download results")

    email_status = st.session_state.get("email_status")
    if email_status:
        ok, msg = email_status
        (st.success if ok else st.warning)(msg)

    summary = st.session_state.run_summary or {}
    if summary:
        cols = st.columns(4)
        cols[0].metric("Sections found", summary.get("found", "—"))
        cols[1].metric("Partial", summary.get("partial", 0))
        cols[2].metric("Not found", summary.get("not_found", 0))
        cols[3].metric("Errors", summary.get("errors", 0))
        st.markdown("")

    st.warning(
        "**Review reminder:** This is an AI-generated draft. All content must be "
        "carefully reviewed and verified by qualified research personnel before "
        "submission to the REB or use with study participants."
    )

    st.markdown(
        "Two versions of the consent form are generated:\n\n"
        "1. **Draft version:** your working copy to review and update prior to "
        "CAPCR submission.\n\n"
        "2. **Marked-up version:** provides traceability to protocol content used "
        "by the AI and includes an appendix with suggested plain language improvements."
    )

    primary_by_prefix = {
        o[0]: o
        for o in st.session_state.run_outputs
        if o[0].startswith(("draft_icf_", "marked_up_icf_"))
    }
    download_order = [
        ("draft_icf_", "Download the draft version of the consent form"),
        ("marked_up_icf_", "Download the marked up version that includes evidence annotations"),
    ]
    for prefix, label in download_order:
        match = next((item for name, item in primary_by_prefix.items() if name.startswith(prefix)), None)
        if match is None:
            continue
        name, data, mime = match
        st.download_button(
            label=label,
            data=data,
            file_name=name,
            mime=mime,
            use_container_width=True,
            key=f"download_{name}",
        )

    st.info(
        "Files are available on this page for 3 hours after generation, and a copy is "
        "emailed to your UHN address. Refreshing the page will restore completed downloads "
        "for your signed-in account."
    )

    st.divider()
    st.markdown('<p class="step-label">Step 4 of 4</p>', unsafe_allow_html=True)
    st.subheader("Review and revise before submitting to CAPCR")
    st.markdown(
        "The AI-generated draft is a starting point — not a finished document. "
        "Before submitting to CAPCR / the REB, your study team should:\n\n"
        "- Read the full draft carefully and verify all extracted information against the protocol\n"
        "- Fill in any sections marked **[PLEASE COMPLETE]** or highlighted in yellow\n"
        "- Revise language, formatting, and study-specific details as needed"
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "UHN ICF Automation · Internal use only · "
    "Generated drafts must be reviewed and approved before REB submission."
)
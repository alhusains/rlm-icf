"""
UHN ICF Automation - Streamlit web wrapper around the rlm-icf pipeline.

This is the UI tier only. It authenticates the user (Entra Easy Auth), takes
the upload, and enqueues a job onto the shared storage queue. The heavy 20-30 min pipeline runs in a separate worker job
(worker.py); this process only enqueues, polls job status, and serves the
finished files for download. All shared state lives in the aiicfstorage
account via icf.jobs (queue + blob + table).
"""

from __future__ import annotations

import base64
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from icf import jobs  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
LOGO_PATH = REPO_ROOT / "data" / "UHN_logo.png"
LOGO_BYTES: bytes | None = LOGO_PATH.read_bytes() if LOGO_PATH.is_file() else None

# Display label -> registry key understood by the worker (see icf.jobs).
STUDY_OPTIONS: dict[str, str | None] = {
    "- Select a study type -": None,
    "Above Minimal Risk Informed Consent Form": "Full Informed Consent Form",
    "Minimal Risk Informed Consent Form": "Minimal Risk Informed Consent Form",
}

# The UI tier only needs to reach storage; the worker owns OpenAI/ACS config.
REQUIRED_ENV = ["AZURE_STORAGE_ACCOUNT"]


def get_user_email() -> str | None:
    """Best-effort read of the signed-in user's email from Entra Easy Auth."""
    try:
        headers = dict(st.context.headers or {})
    except Exception:
        return None

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


def recommended_use_box_html() -> str:
    """Recommended Use callout — light-dark() tracks Streamlit theme without Python detection."""
    return """
    <style>
    .recommended-use-box {
        background-color: light-dark(#f8f9fa, #262730);
        border: 1px solid light-dark(#dee2e6, #464b5d);
        border-left: 4px solid light-dark(#1f77b4, #4dabf7);
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 1rem 0 0.25rem 0;
    }
    .recommended-use-box h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: light-dark(#31333f, #fafafa);
    }
    .recommended-use-box ul { margin: 0; padding-left: 1.25rem; }
    .recommended-use-box li {
        margin-bottom: 0.35rem;
        font-size: 0.92rem;
        line-height: 1.45;
        color: light-dark(#31333f, #d1d5db);
        opacity: 0.9;
    }
    .recommended-use-box li:last-child { margin-bottom: 0; }
    </style>
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
    """


def inject_app_css() -> None:
    """Inject app CSS. Metrics use light-dark() so light mode stays unchanged."""
    st.markdown(
        """
        <style>
        .stAlert > div { border-radius: 8px; }

        div[data-testid="stMetric"],
        div[data-testid="metric-container"] {
            background-color: light-dark(#f8f9fa, #262730) !important;
            border: 1px solid light-dark(#dee2e6, #464b5d) !important;
            border-radius: 8px;
            padding: 0.75rem 1rem;
        }
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricLabel"] div {
            color: light-dark(#6c757d, #adb5bd) !important;
            opacity: 1 !important;
        }
        [data-testid="stMetricValue"] p,
        [data-testid="stMetricValue"] div {
            color: light-dark(#212529, #fafafa) !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }
        [data-testid="stMetricDelta"] {
            color: light-dark(#212529, #fafafa) !important;
            opacity: 0.85;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="UHN ICF Generator",
    page_icon="📋",
    layout="centered",
)

inject_app_css()

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def check_environment() -> list[str]:
    problems: list[str] = []
    missing = [v for v in REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        problems.append("Missing environment variable(s): " + ", ".join(missing))
    return problems


env_problems = check_environment()
if env_problems:
    st.error("This deployment is misconfigured - please contact your administrator:")
    for p in env_problems:
        st.write(f"• {p}")
    st.stop()

# Provision queue/containers/tables on first run (idempotent, cheap).
jobs.ensure_infra()

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
# Sidebar - About
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
        "- There is no long-term storage of protocols or AI-generated consent forms. "
        "AI-generated consent forms are retained for 24 hours and automatically deleted.\n"
        "- No AI model training/learning occurs with submitted protocols or generated "
        "draft consent forms. Submitted protocols are used solely for consent form "
        "generation.\n"
        "- For more information, please email [AIHub@uhn.ca](mailto:AIHub@uhn.ca)"
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
        f"Model: `{os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'gpt-5.4')}`\n\n"
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

st.markdown(recommended_use_box_html(), unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Session state + housekeeping
# ---------------------------------------------------------------------------

for key in ("run_outputs", "run_summary", "run_error", "email_status", "protocol_bytes",
            "protocol_name", "pipeline_job"):
    if key not in st.session_state:
        st.session_state[key] = None

_USER_EMAIL = get_user_email()


def _pipeline_is_running() -> bool:
    job = st.session_state.get("pipeline_job")
    if isinstance(job, dict) and job.get("status") in ("running", "queued"):
        return True
    if _USER_EMAIL and jobs.find_active_job(_USER_EMAIL):
        return True
    return False


def _restore_session() -> None:
    """Re-attach an in-progress job after a page reload, using shared state."""
    if not _USER_EMAIL or isinstance(st.session_state.get("pipeline_job"), dict):
        return
    active = jobs.find_active_job(_USER_EMAIL)
    if active:
        st.session_state.pipeline_job = {
            "status": active.get("status"),
            "job_id": active.get("job_id"),
            "owner": active.get("owner"),
            "study_label": active.get("study_label"),
            "protocol_name": active.get("protocol_name"),
            "us_funded": active.get("us_funded", False),
            "sdm": active.get("sdm", False),
            "created_at": active.get("created_at"),
        }


jobs.purge_expired()
_restore_session()

# ---------------------------------------------------------------------------
# Step 1 - Study type
# ---------------------------------------------------------------------------

st.markdown('<p class="step-label">Step 1 of 4</p>', unsafe_allow_html=True)
st.subheader("Select consent form template")
st.markdown("Choose the consent form template for your study.")

selected_study_label = st.selectbox(
    "Study type",
    options=list(STUDY_OPTIONS.keys()),
    index=0,
    label_visibility="collapsed",
)
selected_study: str | None = STUDY_OPTIONS[selected_study_label]

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
# Step 2 - Upload protocol
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
# Step 3 - Generate (enqueue)
# ---------------------------------------------------------------------------

st.markdown('<p class="step-label">Step 3 of 4</p>', unsafe_allow_html=True)
st.subheader("Generate ICF")

protocol_ready = bool(st.session_state.protocol_bytes and st.session_state.protocol_name)

generate_clicked = False

if selected_study is None or not protocol_ready:
    st.info("Complete Steps 1 and 2 to start generating your ICF.")
elif not _USER_EMAIL:
    st.warning(
        "We couldn't read your UHN identity from your sign-in, so generation is "
        "disabled. Please reload the page; if this persists, contact the project team."
    )
else:
    study_display = (
        "Above Minimal Risk Informed Consent Form"
        if selected_study == "Full Informed Consent Form"
        else selected_study
    )
    st.markdown(
        f"Everything is ready. Click **Generate ICF** to begin processing "
        f"the **{st.session_state.protocol_name}** protocol as a **{study_display}**.\n\n"
        "Processing time may vary based on demand and can take up to 24 hours. "
        "You will receive an email once your AI-ICF is ready."
    )

    st.text_input(
        "Email address for delivery",
        value=_USER_EMAIL,
        disabled=True,
        help="Your finished ICF is sent to your verified UHN address. This cannot be changed.",
    )

    generate_clicked = st.button(
        "Generate ICF",
        type="primary",
        use_container_width=True,
        disabled=_pipeline_is_running(),
    )

# ---------------------------------------------------------------------------
# Enqueue handler
# ---------------------------------------------------------------------------

if (
    generate_clicked
    and protocol_ready
    and selected_study is not None
    and _USER_EMAIL
    and not _pipeline_is_running()
):
    if jobs.find_active_job(_USER_EMAIL):
        st.warning(
            "You already have a generation in progress for your account. "
            "Please wait for it to finish, or reload this page to reconnect to it."
        )
    else:
        try:
            job_id = jobs.enqueue_job(
                _USER_EMAIL,
                study_label=selected_study,
                protocol_name=st.session_state.protocol_name,
                protocol_bytes=st.session_state.protocol_bytes,
                us_funded=bool(st.session_state.get("us_federal_funding", False)),
                sdm=bool(st.session_state.get("sdm_form", False)),
            )
            st.session_state.pipeline_job = {
                "status": "queued",
                "job_id": job_id,
                "owner": _USER_EMAIL.lower(),
                "study_label": selected_study,
                "protocol_name": st.session_state.protocol_name,
                "us_funded": bool(st.session_state.get("us_federal_funding", False)),
                "sdm": bool(st.session_state.get("sdm_form", False)),
                "created_at": jobs.now_iso(),
            }
            st.session_state.run_outputs = None
            st.session_state.run_summary = None
            st.session_state.run_error = None
            st.session_state.email_status = None
            st.rerun()
        except jobs.ActiveJobExistsError as ex:
            st.warning(str(ex))
        except Exception as ex:  # noqa: BLE001
            st.error(f"Could not submit your job: {type(ex).__name__}: {ex}")

# ---------------------------------------------------------------------------
# Progress (polls shared state)
# ---------------------------------------------------------------------------


def _format_elapsed(created_at: str | None) -> str:
    if not created_at:
        return "0m 00s"
    try:
        started = datetime.fromisoformat(created_at)
    except Exception:
        return "0m 00s"
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    total = int((datetime.now(timezone.utc) - started).total_seconds())
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins:02d}m {secs:02d}s" if hours else f"{mins}m {secs:02d}s"


def _load_finished_job(job_row: dict[str, Any]) -> None:
    st.session_state.run_outputs = jobs.load_outputs(job_row["owner"], job_row["job_id"])
    st.session_state.run_summary = job_row.get("summary") or {}
    st.session_state.run_error = None
    es = job_row.get("email_status")
    st.session_state.email_status = (bool(es[0]), str(es[1])) if es else None
    st.session_state.pipeline_job = None


_active = st.session_state.get("pipeline_job")
_poll_interval = 5 if (isinstance(_active, dict) and _USER_EMAIL) else None


@st.fragment(run_every=_poll_interval)
def _render_generation_progress() -> None:
    job = st.session_state.get("pipeline_job")
    if not isinstance(job, dict):
        return

    row = jobs.get_job(job.get("owner"), job.get("job_id"))
    status = (row or {}).get("status") or job.get("status")

    if status == "queued":
        with st.status("Queued - waiting for an available worker...", expanded=True) as box:
            box.write(f"Template: **{job['study_label']}**")
            box.write(f"Protocol: **{job['protocol_name']}**")
            box.write(f"Waiting: **{_format_elapsed(job.get('created_at'))}**")
            box.write(
                "Your job is in the queue and will start automatically when a worker "
                "is free - no need to refresh. You'll be emailed when it's done."
            )
    elif status == "running":
        with st.status("Generating ICF - please wait...", expanded=True) as box:
            box.write(f"Template: **{job['study_label']}**")
            if job.get("us_funded"):
                box.write("US federal funding: **yes** (Summary of ICF sections will be included)")
            if job.get("sdm"):
                box.write("Substitute decision maker (SDM): **yes**")
            box.write(f"Protocol: **{job['protocol_name']}**")
            box.write(f"Elapsed: **{_format_elapsed(job.get('created_at'))}**")
            box.write(
                "Processing your request. This may take up to 24 hours under load. "
                "You will be notified by email when complete. You can close this tab; "
                "returning later with the same account reconnects to the job."
            )
    elif status in ("complete", "failed"):
        if status == "complete" and row:
            _load_finished_job(row)
        elif status == "failed":
            st.session_state.run_error = (row or {}).get("error") or "The pipeline encountered an error."
            st.session_state.run_outputs = None
            st.session_state.run_summary = None
            st.session_state.pipeline_job = None
        st.rerun(scope="app")


_render_generation_progress()

# ---------------------------------------------------------------------------
# Optional: reload a previous completed run
# ---------------------------------------------------------------------------

if (
    _USER_EMAIL
    and not st.session_state.run_outputs
    and not st.session_state.run_error
    and not _pipeline_is_running()
):
    _prior = jobs.find_latest_completed_job(_USER_EMAIL)
    if _prior:
        _prior_summary = _prior.get("summary") or {}
        _prior_errors = _prior_summary.get("errors", 0)
        _prior_label = _prior.get("protocol_name", "unknown file")
        if st.button(
            f"Load results from previous run ({_prior_label})",
            help="Loads your most recent completed generation still within the retention window.",
        ):
            _load_finished_job(_prior)
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
        cols[0].metric("Sections found", summary.get("found", "-"))
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
        "The AI-generated draft is a starting point - not a finished document. "
        "Before submitting to CAPCR / the REB, your study team should:\n\n"
        "- Read the full draft carefully and verify all extracted information against the protocol\n"
        "- Fill in any sections marked **[PLEASE COMPLETE]** or highlighted in yellow\n"
        "- Revise language, formatting, and study-specific details as needed\n"
        "- Ensure all instructional and placeholder text (highlighted text and text in grey "
        "italics) is removed before submitting into CAPCR for REB approval"
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "UHN ICF Automation · Internal use only · "
    "Generated drafts must be reviewed and approved before REB submission."
)

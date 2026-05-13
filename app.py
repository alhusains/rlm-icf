"""
UHN ICF Automation — Streamlit web wrapper around the rlm-icf pipeline.

In production (Azure Container Apps) the same file runs as the entrypoint —
no code changes between local and deployed.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load .env early so os.environ is populated before we touch the pipeline.
load_dotenv()

# The pipeline module logs a fair amount; keep it at WARNING in the UI by
# default so the Streamlit log isn't noisy. Flip to INFO if debugging.
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from icf.pipeline import ICFPipeline  # noqa: E402  (import after env load)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY = REPO_ROOT / "data" / "standard_ICF_template_breakdown.json"

# Required environment variables. We fail fast at app start if any are missing
# rather than failing mid-pipeline 5 minutes into a user's run.
REQUIRED_ENV = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
]

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="UHN ICF Generator",
    page_icon="📄",
    layout="centered",
)

st.title("UHN ICF Automation")
st.caption(
    "Upload a clinical study protocol — get a draft Informed Consent Form "
    "aligned with the UHN ICF template."
)

# ---------------------------------------------------------------------------
# Pre-flight checks (visible to user so misconfigurations are obvious)
# ---------------------------------------------------------------------------


def check_environment() -> list[str]:
    """Return a list of human-readable problems with the runtime environment."""
    problems: list[str] = []

    missing = [v for v in REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        problems.append(
            "Missing environment variable(s): " + ", ".join(missing)
        )

    if not DEFAULT_REGISTRY.exists():
        problems.append(
            f"ICF template registry not found at {DEFAULT_REGISTRY}. "
            "The container build is missing the data/ directory."
        )

    return problems


env_problems = check_environment()
if env_problems:
    st.error("This deployment is misconfigured:")
    for p in env_problems:
        st.write(f"• {p}")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — non-essential controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    extraction_backend = st.selectbox(
        "Extraction backend",
        options=["rlm", "naive"],
        index=0,
        help=(
            "RLM (default): iterative, most accurate. "
            "Naive: single LLM call per section, faster but less precise."
        ),
    )

    skip_review = st.checkbox(
        "Skip plain-language review pass",
        value=False,
        help=(
            "The review pass annotates terminology, passive voice, and other "
            "plain-language issues. Skipping it makes runs ~30%% faster."
        ),
    )

    st.divider()
    st.caption(
        f"Model: `{os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'unknown')}` "
        f"@ `{os.environ.get('AZURE_OPENAI_ENDPOINT', 'unknown')}`"
    )

# ---------------------------------------------------------------------------
# Main form
# ---------------------------------------------------------------------------

st.subheader("1. Upload protocol")
uploaded = st.file_uploader(
    "Clinical protocol (PDF or DOCX)",
    type=["pdf", "docx"],
    help="The full clinical study protocol document.",
    accept_multiple_files=False,
)

st.subheader("2. Generate")

# We track run state in session_state so the result persists across reruns
# (e.g. when the user clicks a download button, Streamlit re-executes the
# whole script — without this, the outputs would disappear).
if "run_outputs" not in st.session_state:
    st.session_state.run_outputs = None
if "run_summary" not in st.session_state:
    st.session_state.run_summary = None
if "run_error" not in st.session_state:
    st.session_state.run_error = None

generate_clicked = st.button(
    "Generate ICF",
    type="primary",
    disabled=uploaded is None,
    use_container_width=True,
)


def run_pipeline(protocol_bytes: bytes, protocol_name: str) -> None:
    """Run the ICF pipeline and stash results on session_state.

    We use a *persistent* temp directory across the run (created in this
    function, copied out at the end, deleted before returning). The pipeline
    needs file paths, not in-memory buffers.
    """
    # Reset any prior run.
    st.session_state.run_outputs = None
    st.session_state.run_summary = None
    st.session_state.run_error = None

    # tempfile.mkdtemp() guarantees a unique dir per concurrent user.
    workdir = Path(tempfile.mkdtemp(prefix="icfrun_"))
    try:
        protocol_path = workdir / protocol_name
        protocol_path.write_bytes(protocol_bytes)

        out_dir = workdir / "output"
        out_dir.mkdir()

        pipeline = ICFPipeline(
            protocol_path=str(protocol_path),
            template_path=str(DEFAULT_REGISTRY),
            output_dir=str(out_dir),
            model_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            backend="azure_openai",
            backend_kwargs={
                "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
                "azure_deployment": os.environ["AZURE_OPENAI_DEPLOYMENT"],
                # api_key is read from AZURE_OPENAI_API_KEY by the SDK
            },
            extraction_backend=extraction_backend,
            verbose=False,
            skip_review=skip_review,
            # Never enable debug log dir in production — it would write
            # protocol content (PHI-adjacent) to disk.
            debug_log_dir=None,
        )

        result = pipeline.run()

        # Read every output file into memory so we can let Streamlit serve
        # them via download_button after we delete the temp dir.
        outputs: list[tuple[str, bytes, str]] = []
        for f in sorted(out_dir.iterdir()):
            if f.is_file():
                mime = _mime_for(f.suffix)
                outputs.append((f.name, f.read_bytes(), mime))

        st.session_state.run_outputs = outputs
        st.session_state.run_summary = getattr(result, "summary", None)

    except Exception as e:  # noqa: BLE001 — we want to show any error to user
        st.session_state.run_error = (
            f"{type(e).__name__}: {e}\n\n"
            f"{traceback.format_exc()}"
        )
    finally:
        # Always clean up the temp dir — uploaded protocol bytes never
        # persist after the run completes.
        shutil.rmtree(workdir, ignore_errors=True)


def _mime_for(suffix: str) -> str:
    return {
        ".docx": (
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        ".json": "application/json",
        ".pdf": "application/pdf",
        ".txt": "text/plain",
    }.get(suffix.lower(), "application/octet-stream")


if generate_clicked and uploaded is not None:
    with st.status(
        "Generating ICF — this typically takes 3–10 minutes…",
        expanded=True,
    ) as status:
        status.write("Reading protocol…")
        status.write(
            f"Backend: **{extraction_backend}** • "
            f"Started: {datetime.now().strftime('%H:%M:%S')}"
        )
        status.write("Calling Azure OpenAI for each ICF section…")

        run_pipeline(uploaded.getvalue(), uploaded.name)

        if st.session_state.run_error:
            status.update(label="Generation failed", state="error")
        else:
            status.update(label="Generation complete", state="complete")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if st.session_state.run_error:
    st.error("The pipeline failed. Details below.")
    with st.expander("Error trace", expanded=False):
        st.code(st.session_state.run_error)

if st.session_state.run_outputs:
    st.subheader("3. Download")

    summary = st.session_state.run_summary or {}
    if summary:
        cols = st.columns(3)
        cols[0].metric(
            "Sections extracted",
            summary.get("extracted", summary.get("found", "—")),
        )
        cols[1].metric("Errors", summary.get("errors", 0))
        cols[2].metric(
            "Not found",
            summary.get("not_found", summary.get("missing", 0)),
        )

    # Surface the two ICF DOCX files first; everything else (extraction
    # report JSON, adapted registry) goes underneath.
    primary = [
        o for o in st.session_state.run_outputs
        if o[0].startswith(("final_icf_", "draft_icf_"))
    ]
    secondary = [
        o for o in st.session_state.run_outputs if o not in primary
    ]

    for name, data, mime in primary:
        label = (
            "📘 Download FINAL ICF (publication-quality)"
            if name.startswith("final_icf_")
            else "📝 Download DRAFT ICF (annotated with evidence)"
        )
        st.download_button(
            label=label,
            data=data,
            file_name=name,
            mime=mime,
            use_container_width=True,
        )

    if secondary:
        with st.expander("Other artifacts (extraction report, etc.)"):
            for name, data, mime in secondary:
                st.download_button(
                    label=name,
                    data=data,
                    file_name=name,
                    mime=mime,
                    key=f"dl_{name}",
                )

    st.info(
        "These files exist only in your browser session. "
        "Refresh or close the tab to discard them. "
        "Nothing is stored on the server."
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "UHN ICF Automation · Internal use only · "
    "Generated drafts must be reviewed before submission to REB."
)
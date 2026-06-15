"""
UHN ICF Automation — Streamlit web wrapper around the rlm-icf pipeline.

In production (Azure Container Apps) the same file runs as the entrypoint —
no code changes between local and deployed.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from icf.pipeline import ICFPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent

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
    logo_path = REPO_ROOT / "data" / "UHN_logo.png"
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
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
        "- Protocols and draft ICFs are kept only during processing, and then deleted. "
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
            <li>Currently optimized for non-complex, single-arm/cohort studies. Support
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
    uploaded = None
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
    )
    if uploaded is not None:
        st.success(f"Protocol uploaded: **{uploaded.name}** ({uploaded.size / 1024:.1f} KB)")

st.divider()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

for key in ("run_outputs", "run_summary", "run_error"):
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------------------------
# Step 3 — Generate
# ---------------------------------------------------------------------------

st.markdown('<p class="step-label">Step 3 of 4</p>', unsafe_allow_html=True)
st.subheader("Generate ICF")

if selected_study is None or uploaded is None:
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
        f"the **{uploaded.name}** protocol as a **{study_display}**.\n\n"
        "This typically takes **20–30 minutes**. Processing time may vary based on "
        "protocol length and level of detail.\n\n"
        "Once ready, your draft consent form will be available for download on this "
        "page. A copy will also be emailed to you. Before submitting to CAPCR, please "
        "review, revise and modify the form as required."
    )
    generate_clicked = st.button(
        "Generate ICF",
        type="primary",
        use_container_width=True,
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


def run_pipeline(
    protocol_bytes: bytes,
    protocol_name: str,
    registry_path: str,
) -> None:
    """Run the ICF pipeline and stash results on session_state."""
    st.session_state.run_outputs = None
    st.session_state.run_summary = None
    st.session_state.run_error = None

    workdir = Path(tempfile.mkdtemp(prefix="icfrun_"))
    try:
        protocol_path = workdir / protocol_name
        protocol_path.write_bytes(protocol_bytes)

        out_dir = workdir / "output"
        out_dir.mkdir()

        us_funded = bool(st.session_state.get("us_federal_funding", False))
        sdm = bool(st.session_state.get("sdm_form", False))

        pipeline = ICFPipeline(
            protocol_path=str(protocol_path),
            template_path=registry_path,
            output_dir=str(out_dir),
            model_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            backend="azure_openai",
            backend_kwargs={
                "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
                "azure_deployment": os.environ["AZURE_OPENAI_DEPLOYMENT"],
                # api_key is read from AZURE_OPENAI_API_KEY by the Azure SDK
            },
            extraction_backend="rlm",
            verbose=False,
            skip_review=False,
            us_funded=us_funded,
            sdm=sdm,
            # Never write a debug log dir in production (protocol text is sensitive).
            debug_log_dir=None,
        )

        result = pipeline.run()

        outputs: list[tuple[str, bytes, str]] = []
        for f in sorted(out_dir.iterdir()):
            if f.is_file():
                outputs.append((f.name, f.read_bytes(), _mime_for(f.suffix)))

        st.session_state.run_outputs = outputs
        st.session_state.run_summary = getattr(result, "summary", None)

    except Exception as e:  # noqa: BLE001
        st.session_state.run_error = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if generate_clicked and uploaded is not None and selected_study is not None:
    registry_path = str(REGISTRIES[selected_study])
    with st.status(
        "Generating ICF — please wait…",
        expanded=True,
    ) as status:
        status.write(f"Template: **{selected_study}**")
        if st.session_state.get("us_federal_funding"):
            status.write("US federal funding: **yes** (Summary of ICF sections will be included)")
        if st.session_state.get("sdm_form"):
            status.write("Substitute decision maker (SDM): **yes**")
        status.write(f"Protocol: **{uploaded.name}**")
        status.write(f"Started: {datetime.now().strftime('%H:%M:%S')}")
        status.write(
            "Extracting and synthesising information from the protocol. "
            "This typically takes 20–30 minutes. Processing time may vary based on "
            "protocol length and level of detail."
        )

        run_pipeline(uploaded.getvalue(), uploaded.name, registry_path)

        if st.session_state.run_error:
            status.update(label="Generation failed — see error details below", state="error")
        else:
            status.update(label="ICF generated successfully", state="complete")

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
        )

    st.info(
        "Files are available only in your current browser session. "
        "Refresh or close this tab to discard them. "
        "No protocol data is retained on the server after your session ends."
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

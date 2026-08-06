# AI-ICF — UHN Informed Consent Form Pipeline

AI-assisted drafting of **Informed Consent Forms (ICFs)** from clinical study protocols (PDF/DOCX), aligned with UHN ICF templates.

Built on [Recursive Language Models (RLMs)](https://arxiv.org/abs/2512.24601). Production uses the **RLM extraction backend** with **Azure OpenAI**.

> **Beta.** Output is a draft for study-team review. It does **not** replace REB, legal, regulatory, ethical, or clinical review. Do not submit AI-generated content to CAPCR/REB or share it with participants without thorough human verification.

## What it does

1. Study team selects an ICF template (above minimal risk or minimal risk) and uploads a protocol.
2. Optional study flags (US federal funding, substitute decision maker) adjust which sections run and how wording is injected.
3. The pipeline extracts each template section from the protocol using RLM, with evidence quotes and confidence.
4. Post-extraction passes harmonize related sub-sections, validate quote grounding, review for plain language, and auto-remediate eligible flags.
5. Outputs: a UHN-branded **draft** DOCX, a **marked-up** DOCX (traceability + review appendix), and a full **JSON** audit report.

**Intended for:** UHN investigator-initiated, non-complex studies (typically single-arm / cohort).

**Not intended for:** CTO studies, sponsor/CRO-provided consent forms, optional/pregnancy follow-up forms, or legacy `.doc` uploads.

---

## Architecture

Production splits the UI from the long-running pipeline:

```
┌─────────────────────┐     enqueue      ┌──────────────────┐
│  Streamlit UI       │ ───────────────► │  Azure Storage   │
│  (app.py)           │                  │  Queue: icf-jobs │
│  Entra Easy Auth    │ ◄── poll/status ─│  Blob + Table    │
└─────────────────────┘                  └────────┬─────────┘
                                                  │
                                         KEDA trigger
                                                  │
                                                  ▼
                                         ┌──────────────────┐
                                         │  Worker job      │
                                         │  (worker.py)     │
                                         │  ICFPipeline     │
                                         │  Azure OpenAI    │
                                         └────────┬─────────┘
                                                  │
                              outputs + optional ACS email
```

| Tier | Entry point | Role |
|------|-------------|------|
| **UI** | `app.py` | Auth, upload, enqueue job, poll status, serve downloads. Needs storage only. |
| **Worker** | `worker.py` | One queue message → one full RLM pipeline run (~20–30 min) → blob upload → optional email. |
| **Job store** | `icf/jobs.py` | Queue + blobs (`icf-input` / `icf-output`) + `jobs` table via `DefaultAzureCredential`. |

The same Docker image runs both tiers (`CMD` defaults to Streamlit; the worker job overrides the command to `python worker.py`).

Local/CLI runs bypass the queue and call `ICFPipeline` directly via `run_pipeline.py`.

---

## Pipeline stages

```
Protocol (PDF/DOCX)  +  ICF Template Registry (JSON)
         |                        |
         v                        v
    [1] Ingest              [2] Load registry
         |                        |
         +--------+---------------+
                  v
        [2.5] Runtime injections (US funding, SDM, …)
                  |
                  v
        [3] Extract (RLM, one fresh instance per section)
                  |
                  v
        [5.5] Harmonize related sub-sections
                  |
                  v
        [6] Validate (quote grounding + meta-commentary checks)
                  |
                  v
        [8] Plain-language review
                  |
                  v
        [9] Remediate eligible review flags
                  |
                  v
        Assemble outputs
              ├── draft_icf_*.docx
              ├── marked_up_icf_*.docx
              └── extraction_report_*.json
```

| Stage | Module | Description |
|-------|--------|-------------|
| **1 Ingest** | `icf/ingest.py` | Parse PDF/DOCX; page markers for evidence traceability. |
| **2 Registry** | `icf/registry.py` | Load JSON template sections (CSV legacy still supported). |
| **2.5 Injections** | `icf/runtime_injections.py` | Apply US-funding / SDM notes onto sections before extraction. |
| **3 Extract** | `icf/extract.py` | RLM per section: search protocol via REPL, return structured JSON. |
| **5.5 Harmonize** | `icf/harmonize.py` | De-dupe / redistribute content across related sub-sections (on by default). |
| **6 Validate** | `icf/validate.py` | Verify cited quotes appear in the protocol; flag meta-commentary. |
| **8 Review** | `icf/review.py` | Plain-language flags (terminology, passive voice, repetition, etc.). |
| **9 Remediate** | `icf/remediate.py` | Auto-patch HIGH (and eligible MEDIUM) flags. |
| **Assemble** | `icf/assemble.py`, `icf/clean_icf.py` | Marked-up DOCX, draft DOCX, JSON report. |

Production worker settings: `extraction_backend="hybrid"`, `backend="azure_openai"`, review/harmonize/remediation **enabled**.

CLI skip flags: `--skip-harmonize`, `--skip-review`, `--skip-remediation`, `--remediate-high-only`.

---

## RLM extraction

For each template section, the pipeline starts a **fresh** `RLM(environment="local")` instance:

1. Full protocol text is loaded as `context_0`.
2. The model writes `` ```repl `` blocks to search/chunk the protocol and call `llm_query` / `llm_query_batched`.
3. It returns structured JSON: `status`, `answer`, `filled_template`, `evidence[]`, `confidence`, `notes`.
4. Routing before RLM:
   - **Standard boilerplate** → use required text as-is (`STANDARD_TEXT`)
   - **Not in protocol** → `SKIPPED` (manual study-team entry)
   - **Otherwise** → RLM extraction with an iteration budget by complexity (Easy 10 / Moderate 15 / Complex 20)

Quality loop: invalid/garbage JSON is retried; unfilled placeholders or meta-commentary can trigger a refinement pass.

---

## Templates and study options

### Registries

| Study type | Registry file |
|------------|---------------|
| Above minimal risk (full ICF) | `data/UHN_standard_ICF_template_breakdown_new.json` |
| Minimal risk | `data/minimal_risk_ICF_template_breakdown.json` |

Each registry is a JSON document with a `schema` (symbol/field guide) and a `sections[]` array. Section fields include `section_id`, `heading`, `instructions`, `required_text` / `suggested_text`, complexity, and availability flags (`is_in_protocol`, `partially_in_protocol`, `is_standard_text`).

Symbol conventions: `{{placeholders}}`, `<conditions>`, `<<blocks>>`, and `OR` alternatives (see `schema.symbol_guide` in the registry).

### Runtime flags

| Flag | CLI | Effect |
|------|-----|--------|
| US federal funding | `--us-funded` | Include Summary of ICF sections (`1.x`) and related wording injections. |
| Substitute decision maker | `--sdm` | SDM intro on §3 and SDM signature-page wording. |

In the web UI these appear as checkboxes on the generation form.

---

## Outputs

Files are named `{artifact}_{backend}_{protocol_stem}.{ext}` (production backend stem is `hybrid`):

| Artifact | Description |
|----------|-------------|
| `draft_icf_*.docx` | Working copy for the study team (UHN layout, grey annotations, text and section headings colour-coded black/blue/yellow by required/suggested/AI-generated provenance). |
| `marked_up_icf_*.docx` | Traceability: evidence, status, validation, review appendix. |
| `extraction_report_*.json` | Full audit trail (extractions, validations, review, remediation). |

Completed web jobs retain downloadable outputs for a short window (default **3 hours**, `ICF_RETENTION_HOURS`), then are purged from blob/table storage. When Azure Communication Services is configured on the worker, the draft and marked-up DOCX are also emailed to the requester.

---

## Quick start (local CLI)

```bash
# From repo root
pip install -e .
pip install -r requirements.txt   # or at least: python-docx pypdf

cp .env.example .env
# Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
```

```bash
# Standard (above minimal risk) template — default when --study-type standard
python run_pipeline.py \
    --protocol path/to/protocol.pdf \
    --study-type standard \
    --verbose

# Minimal risk
python run_pipeline.py \
    --protocol path/to/protocol.pdf \
    --study-type minimal_risk

# Explicit registry + study flags + section filter
python run_pipeline.py \
    --protocol path/to/protocol.pdf \
    --registry data/UHN_standard_ICF_template_breakdown_new.json \
    --us-funded \
    --sdm \
    --sections 2.1 3 6 8 \
    --backend azure_openai \
    --azure-deployment gpt-5.4
```

Defaults: `--extraction-backend rlm`, `--backend azure_openai`, `--model gpt-5.4`.

Useful options:

| Flag | Purpose |
|------|---------|
| `--output-dir` | Output directory (default: `output`) |
| `--max-iterations` | Cap RLM iterations per section (default: 20) |
| `--debug-log-dir` | Write JSONL RLM iteration traces |
| `--skip-harmonize` / `--skip-review` / `--skip-remediation` | Skip post-extraction stages |
| `--convert-registry` | One-shot CSV → JSON registry conversion |

Re-run post-processing on an existing report without re-extraction:

```bash
python run_remediation_only.py --report output/extraction_report_rlm_Prot_000.json
```

### Consistency evaluation

Two local CLIs measure how stable outputs are across repeated runs of the same protocol (same seed / model). Neither is used by the production UI/worker.

| Script | What it tests | Cost / runtime |
|--------|---------------|----------------|
| `run_consistency_eval.py` | Raw extraction only, **one section** at a time | Faster / cheaper |
| `run_full_consistency_eval.py` | Full pipeline (extract → harmonize → review → remediate), all or selected sections | Slow / expensive; meant for unattended runs |

Both write a JSON report; the full-pipeline script also writes a reviewer-friendly Word report.

```bash
# Single-section extraction consistency (e.g. section 3, 3 runs)
python run_consistency_eval.py \
    --protocol path/to/protocol.pdf \
    --study-type standard \
    --section 3 \
    --runs 3 \
    --backend azure_openai

# Full-pipeline consistency across the whole protocol (or selected sections)
python run_full_consistency_eval.py \
    --protocol path/to/protocol.pdf \
    --study-type standard \
    --runs 3 \
    --backend azure_openai

# Background run with logs (recommended for full-pipeline eval)
nohup python run_full_consistency_eval.py \
    --protocol path/to/protocol.pdf \
    --study-type standard \
    --sections 3 5 8 \
    > consistency_eval.log 2>&1 &
```

Useful flags (both scripts unless noted): `--registry` (instead of `--study-type`), `--model`, `--seed`, `--output-dir`, `--judge-model`. Full-pipeline only: `--skip-harmonize` / `--skip-review` / `--skip-remediation`, `--us-funded`, `--sdm`.

Default output directory for the full-pipeline script is `<protocol_dir>/consistency_eval_<protocol_stem>/` (Word + JSON reports, plus per-run pipeline outputs under `pipeline_runs/`).

---

## Local web stack (UI + worker)

The UI only talks to Azure Storage. The worker needs OpenAI (and optionally ACS email).

```bash
# Terminal 1 — UI
export AZURE_STORAGE_ACCOUNT=aiicfstorage
# Use `az login` (or managed identity) with RBAC on the storage account
streamlit run app.py

# Terminal 2 — worker
export AZURE_STORAGE_ACCOUNT=aiicfstorage
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_DEPLOYMENT=gpt-5.4
python worker.py
```

Upload a protocol in the UI; the worker picks up the queue message and writes results back to blob storage for download (and email, if configured).

---

## Azure deployment

Typical layout (Canada Central, resource group `rgUHN-aihub`):

| Component | Name / notes |
|-----------|----------------|
| UI Container App | `ca-uhn-icf` — Streamlit on port 8000, Entra Easy Auth, sticky sessions |
| Worker Container Apps Job | `ca-uhn-aiicf-worker` — KEDA on queue `icf-jobs` |
| Storage | `aiicfstorage` — queue, input/output blobs, jobs table |
| Azure OpenAI | Resource `rebicf`, deployment e.g. `gpt-5.4` (worker only) |
| Email (optional) | Azure Communication Services on the worker |
| ACR | Image `rlm-icf:$TAG` built with `az acr build` (not local Docker) |

Deploy sequence:

```bash
# 1. Initial UI / infra (see deploy.sh for resource names and sizing)
export AZURE_OPENAI_API_KEY='...'
./deploy.sh

# 2. Storage + worker job; strip OpenAI secrets from the UI tier
IMAGE_TAG=vX ./scripts/setup_azure_storage_worker.sh

# 3. Streamlit ingress / sticky sessions
./scripts/configure_azure_app.sh
```

Images are built in ACR via `az acr build`. The Dockerfile serves both UI and worker.

---

## Environment variables

### UI (`app.py`)

| Variable | Required | Purpose |
|----------|----------|---------|
| `AZURE_STORAGE_ACCOUNT` | Yes | Storage account name (default `aiicfstorage`) |

Auth is keyless via managed identity / `DefaultAzureCredential`.

### Worker (`worker.py`)

| Variable | Required | Purpose |
|----------|----------|---------|
| `AZURE_STORAGE_ACCOUNT` | Yes | Same storage account |
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI endpoint |
| `AZURE_OPENAI_API_KEY` | Yes | API key |
| `AZURE_OPENAI_DEPLOYMENT` | Yes | Model deployment (e.g. `gpt-5.4`) |
| `AZURE_OPENAI_API_VERSION` | No | API version override |
| `ACS_CONNECTION_STRING` | No | Enable completion email |
| `ACS_SENDER_ADDRESS` | No | Verified sender address |
| `ACS_SENDER_NAME` | No | Display name (default `UHN AI-Hub`) |
| `ACS_REPLY_TO` | No | Reply-to (default `AIHub@uhn.ca`) |
| `ICF_MAX_DEQUEUE` | No | Poison-message threshold (default `3`) |
| `ICF_VISIBILITY_TIMEOUT` | No | Queue visibility seconds (default `5400`) |

### Job store overrides (`icf/jobs.py`)

| Variable | Default |
|----------|---------|
| `ICF_QUEUE_NAME` | `icf-jobs` |
| `ICF_INPUT_CONTAINER` | `icf-input` |
| `ICF_OUTPUT_CONTAINER` | `icf-output` |
| `ICF_JOBS_TABLE` | `jobs` |
| `ICF_RETENTION_HOURS` | `3` |
| `ICF_STALE_HOURS` | `12` |

### Local CLI (`.env`)

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=gpt-5.4
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

---

## Project structure

```
app.py                       # Streamlit UI (enqueue / poll / download)
worker.py                    # Queue consumer — runs ICFPipeline
run_pipeline.py              # Local/CLI pipeline entry point
run_remediation_only.py      # Re-run harmonize/review/remediate from a JSON report
run_consistency_eval.py      # Single-section extraction consistency eval
run_full_consistency_eval.py # Full-pipeline consistency eval (Word + JSON report)
deploy.sh                    # Initial Azure Container Apps deploy
scripts/
  setup_azure_storage_worker.sh   # Queue, worker job, storage RBAC
  configure_azure_app.sh          # Streamlit ingress tuning

icf/
  pipeline.py                # Orchestrator (all stages)
  jobs.py                    # Azure queue + blob + table job store
  ingest.py                  # Protocol PDF/DOCX loader
  registry.py                # Template registry loader
  runtime_injections.py      # US-funding / SDM injections
  extract.py                 # RLM extraction engine
  prompts.py                 # Extraction prompts
  harmonize.py               # Section-group harmonization
  validate.py                # Quote / meta-commentary validation
  review.py                  # Plain-language review
  remediate.py               # Auto-remediation of review flags
  consistency_eval.py        # Consistency metrics + Word report helpers
  assemble.py                # Marked-up DOCX + JSON report
  clean_icf.py               # Draft DOCX (UHN publication layout)
  types.py                   # Shared data types
  plain_language.py          # Shared plain-language guidelines

rlm/                         # Recursive Language Models library
data/
  UHN_standard_ICF_template_breakdown_new.json
  minimal_risk_ICF_template_breakdown.json
  UHN_logo.png
```

---

## Based on

This project builds on the [Recursive Language Models (RLM)](https://arxiv.org/abs/2512.24601) framework. If you use this work, please cite:

```bibtex
@misc{zhang2025recursivelanguagemodels,
      title={Recursive Language Models},
      author={Alex L. Zhang and Tim Kraska and Omar Khattab},
      year={2025},
      eprint={2512.24601},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24601},
}
```

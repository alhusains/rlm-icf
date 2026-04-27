# UHN ICF Automation Pipeline

Automatically extracts information from clinical study protocols (PDF/DOCX) and generates Informed Consent Form (ICF) documents aligned with the UHN ICF template.

Built on [Recursive Language Models (RLMs)](https://arxiv.org/abs/2512.24601) with support for multiple extraction backends and automated quality evaluation.

## Overview

The pipeline reads a clinical study protocol, matches it against a structured ICF template registry, and uses LLMs to extract and draft each ICF section. It produces publication-quality DOCX documents with UHN branding, evidence citations, and confidence scoring.

```
Protocol (PDF/DOCX)  +  ICF Template Registry (JSON)
         |                        |
         v                        v
    [1] Ingest              [2] Load Registry
         |                        |
         +--------+---------------+
                  v
        [3] Phase A: Extract trigger sections (Introduction, Purpose)
                  |
                  v
        [4] Adapt: LLM decides which optional sections to skip
                  |
                  v
        [5] Phase B: Extract remaining sections
                  |
                  v
        [6] Validate (quote grounding + reading level)
                  |
                  v
        [7] Assemble outputs
              +-- draft_icf_*.docx      (annotated with evidence & status)
              +-- final_icf_*.docx      (clean, publication-quality, UHN-branded)
              +-- extraction_report_*.json  (full structured data)
```

## Quick Setup

```bash
# Install dependencies
pip install -e .

# For RAG backend (optional)
pip install -e ".[rag]"

# For Azure AI Search backend (optional)
pip install azure-search-documents

# For evaluation framework (optional)
pip install deepeval textstat
```

Copy `.env.example` to `.env` and fill in your values. A single `.env` file drives all 4 extraction backends **and** the evaluation judge:

```bash
cp .env.example .env
```

```env
# Azure OpenAI (used by all backends + evaluation judge)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=gpt-5.1
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# OpenAI (alternative — used when --backend openai)
# OPENAI_API_KEY=sk-your-key

# Azure AI Search (used by azure_ai_search backend)
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_KEY=your-key
AZURE_SEARCH_INDEX=your-index-name

# Evaluation judge model override (optional, default: gpt-4o)
# EVAL_JUDGE_MODEL=gpt-4o
```

## Extraction Backends

The pipeline supports **4 extraction backends**, selectable via `--extraction-backend`:

| Backend | Flag | How it works | LLM calls/section |
|---------|------|--------------|--------------------|
| **RLM** (default) | `rlm` | Iterative REPL: LLM writes Python code to search the protocol, executes it, refines, calls `FINAL_VAR()` | 1 (multi-iteration) |
| **Naive** | `naive` | Full protocol text sent in a single LLM call per section | 1 |
| **RAG** (local) | `rag` | BM25 + dense embeddings + cross-encoder reranking, then single LLM call | 1 |
| **Azure AI Search** | `azure_ai_search` | Protocol pre-indexed in Azure AI Search, retrieved per section, then LLM call | 1 |

### Running the Pipeline

```bash
# RLM backend (default) with OpenAI
python run_pipeline.py \
    --protocol data/Prot_000.pdf \
    --registry data/standard_ICF_template_breakdown.json \
    --verbose

# Naive backend with Azure OpenAI
python run_pipeline.py \
    --protocol data/Prot_000.pdf \
    --registry data/standard_ICF_template_breakdown.json \
    --extraction-backend naive \
    --backend azure_openai \
    --azure-endpoint https://your-resource.openai.azure.com/ \
    --azure-deployment gpt-5.1

# RAG backend (local embeddings + reranking)
python run_pipeline.py \
    --protocol data/Prot_000.pdf \
    --registry data/standard_ICF_template_breakdown.json \
    --extraction-backend rag \
    --backend azure_openai \
    --azure-endpoint https://your-resource.openai.azure.com/ \
    --azure-deployment gpt-5.1

# Azure AI Search backend (protocol must be pre-indexed)
python run_pipeline.py \
    --protocol data/Prot_000.pdf \
    --registry data/standard_ICF_template_breakdown.json \
    --extraction-backend azure_ai_search \
    --backend azure_openai \
    --azure-endpoint https://your-resource.openai.azure.com/ \
    --azure-deployment gpt-5.1

# Extract specific sections only
python run_pipeline.py \
    --protocol data/Prot_000.pdf \
    --registry data/standard_ICF_template_breakdown.json \
    --sections 2.1 3 6 8
```

### Azure AI Search Setup

To use the `azure_ai_search` backend:

1. **Index your protocol** in Azure AI Studio: upload the PDF, Azure handles chunking and embedding automatically.
2. **Note the index name** and set `AZURE_SEARCH_INDEX` in your `.env`.
3. You can index multiple protocols into separate indexes and switch between them with `--azure-search-index`.

## Evaluation Framework

The pipeline includes an evaluation framework that scores AI-generated ICF sections using LLM-as-a-judge, aligned with the UHN AI-Generated ICF Evaluation Outline (v3, March 2026).

Two evaluation modes are available:

| Mode | Flag | How it works | Cost |
|------|------|--------------|------|
| **Combined** (default) | `--eval-mode combined` | 1 LLM call per section scores all rubrics at once via direct Azure OpenAI call | ~$12-15/run |
| **Detailed** | `--eval-mode detailed` | 1 LLM call per rubric per section via [DeepEval](https://github.com/confident-ai/deepeval) GEval | ~$40-50/run |

Both modes produce the same JSON output format and comparison table. Combined mode does not require DeepEval installed.

### Evidence-Grounded Scoring

The judge does **not** receive the full protocol text. Instead, it receives per section:

- Extraction notes (what the backend found — placed first, used as primary context)
- Verbatim evidence quotes retrieved from the protocol
- Extraction confidence (`HIGH` / `MEDIUM` / `LOW`) and status (`FOUND` / `NOT_FOUND` / `PARTIAL` / `ERROR`)
- **UHN registry context** for this section: task instructions, required UHN guideline language, and conditional suggested text
- The REB-approved ground truth section (if `--ground-truth` is provided)
- The full REB-approved ICF concatenated once as document-level context (so the judge understands the complete consent story)

This keeps prompts focused and ensures the judge scores based on task completion and evidence grounding — not mechanical comparison against the ground truth.

#### Judge Scoring Rules

The judge applies 7 explicit rules to ensure accurate scoring:

1. **Task scope** — evaluate whether the AI completed its task per the section instructions, not whether it matches the GT word for word
2. **Required UHN language** — must be present and faithful to the guideline meaning; present = credit, missing or meaning-altered = penalize
3. **Suggested text** — only penalize if the protocol supports it but the AI omitted it
4. **GT extras** — do not penalize the AI for omitting GT content that goes beyond its task scope or what the protocol evidence supports
5. **Placeholders** — `[TO BE FILLED MANUALLY]`, `{{field}}`, `<<insert>>` where extraction notes confirm info was not found = correct abstention, not fabrication
6. **All-signals verification** — use notes + quotes + GT together; do not call fabrication from missing quotes alone if notes confirm the content exists in the protocol
7. **Genuine fabrication** — content not in evidence, notes, or required/suggested text that contradicts or exceeds the GT = penalized firmly

### Routing Policy

Before calling the judge, each section/rubric pair is routed through a configurable policy (`EvalPolicy` in `eval_rubrics.py`) that determines how to score it:

| Routing Mode | When it applies | Effect |
|---|---|---|
| `FULL` | High confidence, strong extraction (including partial protocol + HIGH confidence) | Judge scores normally |
| `SOFT` | Medium confidence, or partial protocol coverage with non-HIGH confidence | Judge is cautioned about extraction uncertainty |
| `HARD_PENALTY` | Backend confirmed NOT_FOUND but AI generated concrete content | Fixed low score (0.15) — hallucination flag |
| `SKIP` | Standard boilerplate, section not in protocol, extraction error, or correct abstention | Recorded as N/A — no judge call |

Fidelity and Honesty rubrics additionally return two judge-assessed fields per section:
- **Evidence Relevance** (`STRONG` / `PARTIAL` / `WEAK` / `IRRELEVANT`) — how well the retrieved quotes actually support this section
- **Support Level** (`WITHIN` / `EXCEEDS` / `NO_EVIDENCE`) — whether the AI text stays within what the evidence supports or makes claims beyond it

### Evaluation Dimensions (10 per-section rubrics + 1 document-level)

**Ground Truth Comparison:**
| Dimension | Method | Scope |
|-----------|--------|-------|
| Correctness vs approved ICF | LLM judge | All sections (requires `--ground-truth`) |

**Task Performance (UHN Rubric Table 6):**
| Dimension | Method | Scope |
|-----------|--------|-------|
| Fidelity to Protocol | LLM judge | All sections |
| Honesty | LLM judge | All sections — checks fabrication, contradiction, misrepresentation, unclarity, and missing acknowledgement |
| Over-inclusion | LLM judge | All sections |
| Inclusive Language | LLM judge | All sections |
| Reading Level (Flesch-Kincaid) | Deterministic (code) | Sections with 20+ words |
| Language Quality | LLM judge | Sections with 20+ words — combined reading level + plain language + comprehensibility |

**Effectiveness (UHN Rubric Table 7):**
| Dimension | Method | Scope |
|-----------|--------|-------|
| Misleading Language | LLM judge | All sections |
| Risks/Benefits/Voluntariness | LLM judge | Sections 7, 16, 18, 18.1, 18.2, 19, 20 only |
| Tone (neutral, non-coercive) | LLM judge | All sections |

**Document-Level (1 extra LLM call on full document):**
| Dimension | Method | Scope |
|-----------|--------|-------|
| Document Quality | LLM judge | Full concatenated ICF — abbreviation redundancy, repetition, terminology consistency, cross-section coherence |

Each dimension uses the exact **Excellent / Good / Borderline / Poor / Fail** scale from the UHN evaluation outline. Readability rubrics skip short fill-in fields (protocol number, sponsor name). Risks/Benefits/Voluntariness only runs on sections that discuss those topics. Standard boilerplate sections (e.g. signature blocks) are automatically skipped for grounding rubrics.

### Running Evaluation

The judge model is read automatically from `AZURE_OPENAI_DEPLOYMENT` in your `.env`. Output files are named to include the backend and protocol so runs never overwrite each other.

```bash
# Single backend, combined mode (default)
python run_eval.py --reports rlm=output/extraction_report_rlm_Prot_000.json --ground-truth data/ground_truth_icf.docx

# Compare multiple backends side by side
python run_eval.py \
    --reports \
        rlm=output/extraction_report_rlm_Prot_000.json \
        naive=output/extraction_report_naive_Prot_000.json \
        rag=output/extraction_report_rag_Prot_000.json \
        azure_ai_search=output/extraction_report_azure_ai_search_Prot_000.json \
    --ground-truth data/ground_truth_icf.docx \
    --verbose

# Detailed mode (DeepEval GEval, 1 call per rubric per section)
python run_eval.py \
    --reports rlm=output/extraction_report_rlm_Prot_000.json \
    --ground-truth data/ground_truth_icf.docx \
    --eval-mode detailed

# Evaluate specific sections only
python run_eval.py \
    --reports rlm=output/extraction_report_rlm_Prot_000.json \
    --ground-truth data/ground_truth_icf.docx \
    --sections 3 6 7 8
```

Output: a side-by-side comparison table printed to console and a JSON report saved to `output/eval_report_combined_<backends>_<protocol>.json`.

### Generating a Review Document

After running evaluation, generate a colour-coded Word document for human reviewers:

```bash
python run_eval_review.py \
    --eval-report output/eval_report_combined_rlm_Prot_000.json \
    --extraction-report output/extraction_report_rlm_Prot_000.json \
    --ground-truth data/ground_truth_icf.docx
```

The review document contains, per section:
- AI-generated text vs REB-approved ground truth side by side (blue / green headers)
- Protocol evidence quotes retrieved by the backend
- Per-rubric scores colour-coded by grade (green → red), with routing mode, evidence relevance, support level, and judge reasoning
- A blank reviewer comment box for written feedback

Output is saved to `output/review_<eval_report_stem>.docx`. `--ground-truth` is optional — omit it if you only want to review AI output and rubric scores without the ground truth column.

## Project Structure

```
.env.example                 # Unified env var template (copy to .env)
run_pipeline.py              # CLI entry point for ICF generation
run_eval.py                  # CLI entry point for evaluation
run_eval_review.py           # CLI entry point for review DOCX generation

icf/
  pipeline.py                # Main orchestrator (7 stages)
  ingest.py                  # Protocol PDF/DOCX parser
  registry.py                # ICF template registry loader (JSON/CSV)
  types.py                   # Data types (TemplateVariable, ExtractionResult, etc.)
  adapt.py                   # Dynamic section adaptation pass
  validate.py                # Quote verification + reading level check
  assemble.py                # Draft ICF DOCX + JSON report generator
  clean_icf.py               # Publication-quality ICF DOCX generator

  # Extraction backends
  extract.py                 # RLM extraction engine (default)
  prompts.py                 # RLM extraction prompts
  naive_extract.py           # Naive full-context extraction engine
  naive_prompts.py           # Naive extraction prompts
  rag_extract.py             # Local RAG extraction engine
  rag_index.py               # BM25 + dense embedding index
  rag_query.py               # Multi-query expansion
  rag_rerank.py              # Cross-encoder reranking
  rag_prompts.py             # RAG extraction prompts
  azure_search_extract.py    # Azure AI Search extraction engine
  azure_search_prompts.py    # Azure AI Search prompts

  # Evaluation
  eval_rubrics.py            # 10+1 rubric definitions + EvalPolicy routing (ScoringMode: FULL/SOFT/HARD_PENALTY/SKIP)
  eval_combined.py           # Combined evaluator (1 LLM call/section, all rubrics + evidence grounding)
  eval_ground_truth.py       # Ground truth DOCX parser
  eval_runner.py             # Evaluation engine (combined + detailed/DeepEval modes)
  eval_model.py              # Azure OpenAI judge wrapper for DeepEval
  eval_review.py             # Review DOCX generator (colour-coded per-section layout)

data/
  standard_ICF_template_breakdown.json   # ICF template registry
  UHN_logo.png                           # Logo for clean ICF header

EvalRubric/
  AI-Generated ICF Evaluation Outline - v3 - March2026.docx  # Evaluation criteria

output/                      # Generated at runtime
  draft_icf_*.docx           # Annotated draft ICF
  final_icf_*.docx           # Clean publication-quality ICF
  extraction_report_*.json   # Full structured extraction data
  adapted_registry.json      # Adaptation decisions audit trail
  eval_report_combined_<backends>_<protocol>.json   # Evaluation report (combined mode)
  eval_report_detailed_<backends>_<protocol>.json   # Evaluation report (detailed/DeepEval mode)
  review_<eval_report_stem>.docx                    # Colour-coded review document for human reviewers
```

## Based On

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

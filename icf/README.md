# UHN ICF Automation Pipeline

Automated generation of Informed Consent Forms (ICF) from study protocols using Recursive Language Models (RLMs).

## Overview

This pipeline automates the drafting of ICFs for clinical trials at UHN. It extracts information from clinical study protocols (50-200 pages of dense medical/legal text) and fills template variables with grounded, human-readable text at a Grade 6-8 reading level.

**Key Features:**
- **Template-driven extraction**: Fills specific ICF sections based on a structured JSON template registry
- **Grounded information**: Every extracted claim is backed by verbatim quotes with page numbers
- **Hallucination prevention**: Explicitly states when information is not found in the protocol
- **Semantic search**: Uses RLM-powered chunking and LLM queries to find information across varying protocol structures
- **Validation**: Automatic quote verification and reading level checks

## Architecture

The pipeline consists of 6 stages:

### 1. Ingest (`icf/ingest.py`)
Loads PDF or DOCX protocol files and indexes them with page markers for traceability.

### 2. Registry (`icf/registry.py`)
Loads the ICF template breakdown JSON to build a list of variables to extract, including:
- Section metadata (ID, heading, sub-section)
- Extraction instructions
- Complexity classification
- Availability flags (in protocol, partially in protocol, standard text)

### 3. Extract (`icf/extract.py`)
For each variable, spawns a fresh RLM instance that:
- Loads the full protocol as `context_0`
- Uses semantic chunking and `llm_query_batched()` to search for relevant information
- Extracts structured JSON with answer, filled template, evidence quotes, and confidence
- Routes based on complexity:
  - **Standard text**: Returns required text directly
  - **Not in protocol**: Returns `SKIPPED` status for manual entry
  - **Extractable**: Runs RLM extraction with iteration budget based on complexity

### 4. Validate (`icf/validate.py`)
Validates each extraction:
- **Quote verification**: Checks that cited quotes actually appear in the protocol
- **Reading level**: Calculates Flesch-Kincaid grade level (target: Grade 6-8)

### 5. Assemble (`icf/assemble.py`)
Generates output files:
- **Draft ICF (DOCX)**: Structured ICF document with extracted text
- **Extraction Report (JSON)**: Complete audit trail with evidence, confidence, and validation results

### 6. Orchestrate (`icf/pipeline.py`)
Main pipeline coordinator that manages the flow, progress reporting, and error handling.

## Installation

```bash
# Install the RLM framework in editable mode
cd /path/to/rlm
uv pip install -e .

# Install dependencies
uv pip install python-docx pypdf textstat

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

Extract all sections from a protocol:

```bash
uv run python run_pipeline.py \
    --protocol data/Prot_000.pdf \
    --registry data/standard_ICF_template_breakdown.json
```

### Extract Specific Sections

Extract only sections 6, 8, 9.1, 9.2:

```bash
uv run python run_pipeline.py \
    --protocol data/Prot_000.pdf \
    --registry data/standard_ICF_template_breakdown.json \
    --sections 6 8 9.1 9.2
```

### Verbose Mode

See detailed RLM iterations and REPL interactions:

```bash
uv run python run_pipeline.py \
    --protocol data/Prot_000.pdf \
    --registry data/standard_ICF_template_breakdown.json \
    --verbose
```

### Configuration Options

```
--protocol PATH          Path to clinical study protocol (PDF or DOCX)
--registry PATH          Path to ICF template registry (.json preferred, .csv legacy)
--output-dir PATH        Output directory (default: output)
--model NAME             LLM model name (default: gpt-5.1)
--backend NAME           RLM backend (default: openai)
--max-iterations N       Max RLM iterations per variable (default: 20)
--max-tokens N           Max output tokens per LLM call
--verbose                Enable verbose RLM output
--sections [IDs...]      Extract only these section IDs
--convert-registry       Convert --registry CSV to JSON and exit
```

## Output

The pipeline generates two files in the `output/` directory:

### 1. `draft_icf.docx`
Structured ICF document with:
- Extracted text for each section
- Placeholders (`[TO BE FILLED MANUALLY]`) for unavailable information
- Validation warnings for items requiring manual review

### 2. `extraction_report.json`
Complete extraction audit trail containing:

**Summary:**
- Total sections processed
- Status counts (FOUND, PARTIAL, NOT_FOUND, SKIPPED, STANDARD_TEXT, ERROR)
- Validation issue count
- Wall time

**Per-section extractions:**
- `section_id`, `heading`, `sub_section`
- `status`: Extraction status
- `answer`: Plain language answer (Grade 6 reading level)
- `filled_template`: Template with variables filled in
- `evidence`: List of verbatim quotes with page numbers
- `confidence`: HIGH, MEDIUM, or LOW
- `notes`: Caveats or manual review items
- `raw_response`: Raw RLM output for debugging
- `error`: Error message if extraction failed

**Validations:**
- `quotes_verified`: Boolean list indicating which quotes were verified
- `reading_grade_level`: Flesch-Kincaid grade level
- `issues`: List of validation warnings

## Template Registry Format

The pipeline reads `data/standard_ICF_template_breakdown.json`. The file has two top-level keys:

- **`schema`** — human-readable documentation of every field and text-markup symbol
- **`sections`** — array of section objects

Each section object contains:

| Field | Description |
|-------|-------------|
| `section_id` | Unique identifier (e.g., `"6"`, `"9.1.2"`) |
| `heading` | UPPERCASE section heading as it appears in the ICF |
| `sub_section` | Sub-section name, or `null` |
| `required` | `true` = must be included; `false` = include only if relevant |
| `complexity` | Tags driving iteration budget and availability flags |
| `is_in_protocol` | Whether the info is expected in the study protocol |
| `partially_in_protocol` | Whether only some fields come from the protocol |
| `is_standard_text` | Whether `required_text` is boilerplate needing no extraction |
| `instructions` | Author-facing extraction directions |
| `required_text` | Mandated ICF wording with `{{placeholders}}` |
| `suggested_text` | Sample language with `<conditions>`, `<<blocks>>`, and `OR` alternatives |
| `suggested_text_format` | `"text"` (default) or `"html"` for table content |
| `adaptation_notes` | Runtime field — set by the dynamic-adaptation pass; `null` in base registry |


## Iteration Budgets

The pipeline allocates RLM iterations based on extraction complexity:

| Complexity Label | Iterations | Description |
|------------------|------------|-------------|
| Easy | 10 | Simple, well-defined information |
| Moderate | 15 | Moderate mapping complexity |
| Complex | 20 | Complex mapping or multi-step extraction |
| Not in protocol | 8 | Brief search before returning `NOT_FOUND` |


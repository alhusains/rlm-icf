# UHN ICF Automation Pipeline

Automated generation of Informed Consent Forms (ICF) from study protocols using Recursive Language Models (RLMs).

## Overview

This pipeline automates the drafting of ICFs for clinical trials at UHN. It extracts information from clinical study protocols (50-200 pages of dense medical/legal text) and fills template variables with grounded, human-readable text at a Grade 6-8 reading level.

**Key Features:**
- **Template-driven extraction**: Fills specific ICF sections based on a structured CSV template
- **Grounded information**: Every extracted claim is backed by verbatim quotes with page numbers
- **Hallucination prevention**: Explicitly states when information is not found in the protocol
- **Semantic search**: Uses RLM-powered chunking and LLM queries to find information across varying protocol structures
- **Validation**: Automatic quote verification and reading level checks

## Architecture

The pipeline consists of 6 stages:

### 1. Ingest (`icf/ingest.py`)
Loads PDF or DOCX protocol files and indexes them with page markers for traceability.

### 2. Registry (`icf/registry.py`)
Parses the ICF template breakdown CSV to build a list of variables to extract, including:
- Section metadata (ID, heading, sub-section)
- Extraction instructions
- Protocol mapping hints
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
    --csv data/standard_ICF_template_breakdown.csv
```

### Extract Specific Sections

Extract only sections 6, 8, 9.1, 9.2:

```bash
uv run python run_pipeline.py \
    --protocol data/Prot_000.pdf \
    --csv data/standard_ICF_template_breakdown.csv \
    --sections 6 8 9.1 9.2
```

### Verbose Mode

See detailed RLM iterations and REPL interactions:

```bash
uv run python run_pipeline.py \
    --protocol data/Prot_000.pdf \
    --csv data/standard_ICF_template_breakdown.csv \
    --verbose
```

### Configuration Options

```
--protocol PATH          Path to clinical study protocol (PDF or DOCX)
--csv PATH              Path to ICF template breakdown CSV
--output-dir PATH       Output directory (default: output)
--model NAME            LLM model name (default: gpt-5.1)
--backend NAME          RLM backend (default: openai)
--max-iterations N      Max RLM iterations per variable (default: 20)
--verbose               Enable verbose RLM output
--sections [IDs...]     Extract only these section IDs
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

## Template CSV Format

The pipeline expects a CSV with the following columns:

| Column | Description |
|--------|-------------|
| `Section ID` | Unique identifier (e.g., "6", "9.1") |
| `Heading` | Section heading in ICF |
| `Sub-Section` | Sub-section name (or "N/A") |
| `Instructions for Filling` | Extraction instructions for the RLM |
| `Required Text in ICF Template` | ICF template text with `{{ variables }}` |
| `Suggested Text` | Suggested phrasing |
| `UHN Protocol Section` | Hints for where to find info in UHN protocols |
| `Sponsor Protocol Section` | Hints for sponsor protocol sections |
| `Complexity` | Extraction complexity (Easy, Moderate mapping complexity, Complex mapping, Potentially in protocol) |
| `Conventionally in protocol?` | "Yes", "No", or "Partially" |

## Iteration Budgets

The pipeline allocates RLM iterations based on extraction complexity:

| Complexity Label | Iterations | Description |
|------------------|------------|-------------|
| Easy | 10 | Simple, well-defined information |
| Moderate | 15 | Moderate mapping complexity |
| Complex | 20 | Complex mapping or multi-step extraction |
| Not in protocol | 8 | Brief search before returning `NOT_FOUND` |


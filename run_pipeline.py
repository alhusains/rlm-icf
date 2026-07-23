#!/usr/bin/env python3
"""
CLI entry point for the UHN ICF Automation Pipeline.

Example usage:

    # Full pipeline (JSON registry — preferred)
    python run_pipeline.py \\
        --protocol data/Prot_000.pdf \\
        --registry data/UHN_standard_ICF_template_breakdown_new.json

    # Legacy CSV registry still works
    python run_pipeline.py \\
        --protocol data/Prot_000.pdf \\
        --registry data/standard_ICF_template_breakdown.csv

    # One-time CSV -> JSON conversion
    python run_pipeline.py --convert-registry \\
        --registry data/standard_ICF_template_breakdown.csv

    # Extract specific sections only
    python run_pipeline.py \\
        --protocol data/Prot_000.pdf \\
        --registry data/UHN_standard_ICF_template_breakdown_new.json \\
        --sections 2.1 3 6 8

    # Verbose RLM output
    python run_pipeline.py \\
        --protocol data/Prot_000.pdf \\
        --registry data/UHN_standard_ICF_template_breakdown_new.json \\
        --verbose
"""

import argparse
import io
import os
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode characters in protocol text
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Load .env early so os.environ is populated before argparse default evaluation.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on shell env vars

from icf.pipeline import ICFPipeline
from icf.registry import convert_csv_to_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="UHN ICF Automation Pipeline - extract protocol data into "
        "Informed Consent Form sections using RLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--protocol",
        default=None,
        help="Path to the clinical study protocol (PDF or DOCX).",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help=(
            "Path to the ICF template registry — JSON (preferred) or CSV (legacy). "
            "If omitted, you will be prompted to choose Standard or Minimal Risk. "
            "Use --convert-registry to produce a JSON from a CSV once."
        ),
    )
    parser.add_argument(
        "--study-type",
        choices=["standard", "minimal_risk"],
        default=None,
        help=(
            "Study type used to select the default registry when --registry is not provided. "
            "  standard     — full ICF template for standard studies. "
            "  minimal_risk — simplified ICF template for minimal risk studies."
        ),
    )
    parser.add_argument(
        "--convert-registry",
        action="store_true",
        help=(
            "Convert --registry CSV to JSON and exit. "
            "Output file is the same path with .json extension."
        ),
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Path to ICF template DOCX (optional, for reference).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.6-sol"),
        help=(
            "LLM model name (default: AZURE_OPENAI_DEPLOYMENT env var, "
            "falling back to gpt-5.6-sol). Keeping this in sync with "
            "AZURE_OPENAI_DEPLOYMENT avoids mislabeled usage/logs when you "
            "switch models by only editing .env."
        ),
    )
    parser.add_argument(
        "--backend",
        default="azure_openai",
        help=(
            "LLM provider backend (default: openai). "
            "Choices: openai | azure_openai | vllm. "
            "Use 'azure_openai' for Azure AI Foundry deployments; "
            "use 'vllm' for local vLLM servers."
        ),
    )
    parser.add_argument(
        "--extraction-backend",
        default="rlm",
        choices=["rlm", "hybrid", "naive", "rag", "azure_ai_search"],
        help=(
            "Extraction strategy (default: rlm). "
            "  rlm    — iterative RLM with code execution and semantic chunking (default). "
            "  hybrid — RLM evidence gathering (narrow prompt) followed by a single "
            "structured drafting call; splits research from writing to reduce prompt "
            "overload. "
            "  naive  — full-context single LLM call per section (benchmarking baseline). "
            "  rag    — retrieval-augmented generation with hybrid search. "
            "  azure_ai_search — RAG via Azure AI Search (protocol must be pre-indexed). "
            "This flag is orthogonal to --backend: e.g., "
            "'--backend azure_openai --extraction-backend hybrid' is valid."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Base URL for the LLM API endpoint. Required when --backend vllm "
            "(e.g. http://localhost:8005/v1). Also works with any OpenAI-compatible server."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key for the LLM backend. For local vLLM servers use any non-empty "
            "string (e.g. 'EMPTY'). Defaults to the OPENAI_API_KEY env var for openai backend."
        ),
    )
    parser.add_argument(
        "--azure-endpoint",
        default=None,
        help=(
            "Azure OpenAI endpoint URL (e.g. https://rebicf.openai.azure.com/). "
            "Only used with --backend azure_openai. "
            "Defaults to the AZURE_OPENAI_ENDPOINT env var."
        ),
    )
    parser.add_argument(
        "--azure-deployment",
        default=None,
        help=(
            "Azure deployment name (e.g. gpt-5-chat). "
            "Only used with --backend azure_openai. "
            "Defaults to the AZURE_OPENAI_DEPLOYMENT env var."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Max RLM iterations per variable (default: 20).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max output tokens per LLM call (default: model default). Increase if responses are being truncated.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed passed to the LLM backend for reproducibility across runs (default: 42). "
            "Currently only honored by --backend azure_openai. Determinism is best-effort — "
            "the API may still vary, especially if Azure changes the serving backend "
            "(watch the printed system_fingerprint)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose RLM output (shows REPL interactions).",
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        default=None,
        help="Extract only these section IDs (e.g. 2.1 3 6 8).",
    )
    # ------------------------------------------------------------------
    # RAG backend options (only used when --extraction-backend rag)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--rag-embedding-deployment",
        default=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
        help=(
            "Embedding model / Azure deployment name for the RAG backend. "
            "Defaults to the AZURE_OPENAI_EMBEDDING_DEPLOYMENT env var, "
            "then 'text-embedding-3-large'. "
            "For Azure, this is your deployment name. "
            "For standard OpenAI, this is the model name."
        ),
    )
    parser.add_argument(
        "--rag-reranker",
        default="local",
        choices=["local", "none"],
        help=(
            "Reranker for the RAG backend (default: local). "
            "  local — cross-encoder/ms-marco-MiniLM-L-12-v2 via sentence-transformers. "
            "  none  — skip reranking (faster, useful for ablation studies)."
        ),
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=20,
        help="Number of candidate chunks retrieved before reranking (default: 20).",
    )
    parser.add_argument(
        "--rag-rerank-top-k",
        type=int,
        default=8,
        help="Number of chunks passed to the generator after reranking (default: 8).",
    )
    parser.add_argument(
        "--rag-num-queries",
        type=int,
        default=4,
        help="Number of search queries generated per ICF section (default: 4).",
    )
    parser.add_argument(
        "--rag-cache-dir",
        default=".rag_cache",
        help=(
            "Directory for caching protocol embeddings between runs (default: .rag_cache). "
            "Embeddings are keyed by protocol ID, model, and content fingerprint — "
            "so re-running the pipeline on the same protocol skips the embedding API calls. "
            "Pass an empty string to disable caching."
        ),
    )

    # ------------------------------------------------------------------
    # Azure AI Search backend options (only used when --extraction-backend azure_ai_search)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--azure-search-endpoint",
        default=os.environ.get("AZURE_SEARCH_ENDPOINT"),
        help=(
            "Azure AI Search service endpoint "
            "(e.g. https://my-search.search.windows.net). "
            "Defaults to AZURE_SEARCH_ENDPOINT env var. "
            "Required when --extraction-backend azure_ai_search."
        ),
    )
    parser.add_argument(
        "--azure-search-key",
        default=os.environ.get("AZURE_SEARCH_KEY"),
        help=(
            "API key for the Azure AI Search service. "
            "Defaults to AZURE_SEARCH_KEY env var. "
            "Required when --extraction-backend azure_ai_search."
        ),
    )
    parser.add_argument(
        "--azure-search-index",
        default=os.environ.get("AZURE_SEARCH_INDEX"),
        help=(
            "Name of the Azure AI Search index containing the protocol. "
            "Defaults to AZURE_SEARCH_INDEX env var. "
            "Required when --extraction-backend azure_ai_search."
        ),
    )
    parser.add_argument(
        "--azure-search-top-k",
        type=int,
        default=10,
        help="Number of documents to retrieve per query from Azure AI Search (default: 10).",
    )
    parser.add_argument(
        "--azure-search-num-queries",
        type=int,
        default=3,
        help="Number of search queries generated per ICF section (default: 3).",
    )
    parser.add_argument(
        "--azure-search-semantic",
        action="store_true",
        help="Enable semantic search (requires a semantic configuration on the index).",
    )
    parser.add_argument(
        "--azure-search-semantic-config",
        default=None,
        help="Name of the semantic configuration on the Azure AI Search index.",
    )

    parser.add_argument(
        "--skip-review",
        action="store_true",
        help=(
            "Skip the Stage 8 plain language review pass. "
            "The review reads the full assembled ICF and annotates terminology "
            "inconsistencies, passive voice, repetition, and other plain language issues. "
            "Useful for faster runs or when using --sections (partial ICF)."
        ),
    )
    parser.add_argument(
        "--skip-remediation",
        action="store_true",
        help=(
            "Skip the Stage 9 review-flag remediation pass. "
            "When enabled, review flags and cross-section terminology issues are "
            "annotated in the report but no automatic fixes are applied. "
            "Implies review still runs (unless --skip-review is also set)."
        ),
    )
    parser.add_argument(
        "--remediate-high-only",
        action="store_true",
        help=(
            "Stage 9 remediation: fix HIGH-severity flags only (default also fixes "
            "eligible MEDIUM flags). Eligible MEDIUM = PASSIVE_VOICE, "
            "SENTENCE_TOO_LONG, or PLAIN_LANGUAGE_VIOLATION, or any MEDIUM flag "
            "with a non-empty suggested_fix from review. MEDIUM REPETITION, "
            "UNCLEAR, and TONE are never auto-fixed."
        ),
    )
    parser.add_argument(
        "--skip-harmonize",
        action="store_true",
        help=(
            "Skip the Stage 5.5 section-group harmonization pass. "
            "Harmonization redistributes and de-duplicates content across related "
            "sub-sections (e.g., all 'WHAT ARE THE STUDY PROCEDURES?' sub-sections) "
            "that were extracted independently. A group is silently skipped when "
            "fewer than 2 of its sections have content, so using --sections with a "
            "full group (e.g. all 12.x) still triggers harmonization correctly."
        ),
    )
    parser.add_argument(
        "--us-funded",
        action="store_true",
        help=(
            "Study is funded or supported by a US federal agency (e.g. NIH, DHHS). "
            "Extracts Summary of Informed Consent Form sections (1.x) and inserts "
            "them at the beginning of the draft and marked-up ICF documents, "
            "with the study title from section 2.1 above the summary and a page break "
            "before the main ICF body."
        ),
    )
    parser.add_argument(
        "--sdm",
        action="store_true",
        help=(
            "ICF is intended for completion by a substitute decision maker (SDM). "
            "Prepends the SDM introduction paragraph to section 3 and uses SDM "
            "signature-page wording (no protocol search for SDM applicability)."
        ),
    )

    parser.add_argument(
        "--debug-log-dir",
        default=None,
        help=(
            "Directory to write a JSONL debug trace of every RLM iteration. "
            "Each line records the LLM response, code executed, REPL output "
            "(truncated), and final answer — without the full protocol text. "
            "Useful for auditing model behaviour and optimising prompts. "
            "Example: --debug-log-dir output/debug_logs"
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # --convert-registry mode: CSV -> JSON, then exit
    # ------------------------------------------------------------------
    if args.convert_registry:
        if args.registry is None:
            print("ERROR: --convert-registry requires --registry.", file=sys.stderr)
            return 1
        src = args.registry
        if not src.lower().endswith(".csv"):
            print(f"ERROR: --convert-registry expects a .csv file, got: {src}", file=sys.stderr)
            return 1
        dst = os.path.splitext(src)[0] + ".json"
        convert_csv_to_json(src, dst)
        return 0

    # ------------------------------------------------------------------
    # Normal pipeline run
    # ------------------------------------------------------------------
    if args.protocol is None:
        print("ERROR: --protocol is required when not using --convert-registry.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Resolve registry: explicit path > --study-type flag > interactive prompt
    # ------------------------------------------------------------------
    if args.registry is None:
        _repo_root = Path(__file__).resolve().parent
        _standard = _repo_root / "data" / "UHN_standard_ICF_template_breakdown_new.json"
        _minimal = _repo_root / "data" / "minimal_risk_ICF_template_breakdown.json"

        if args.study_type == "standard":
            args.registry = str(_standard)
            print("[REGISTRY] Standard ICF template selected.")
        elif args.study_type == "minimal_risk":
            args.registry = str(_minimal)
            print("[REGISTRY] Minimal Risk ICF template selected.")
        else:
            print("\nSelect study type:")
            print("  1. Standard study")
            print("  2. Minimal risk study")
            while True:
                try:
                    choice = input("Enter 1 or 2: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nAborted.", file=sys.stderr)
                    return 1
                if choice == "1":
                    args.registry = str(_standard)
                    print("[REGISTRY] Standard ICF template selected.")
                    break
                elif choice == "2":
                    args.registry = str(_minimal)
                    print("[REGISTRY] Minimal Risk ICF template selected.")
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")

    backend_kwargs: dict = {}
    if args.max_tokens is not None:
        backend_kwargs["max_tokens"] = args.max_tokens
    if args.seed is not None:
        backend_kwargs["seed"] = args.seed
    if args.base_url is not None:
        backend_kwargs["base_url"] = args.base_url
    if args.api_key is not None:
        backend_kwargs["api_key"] = args.api_key
    if args.azure_endpoint is not None:
        backend_kwargs["azure_endpoint"] = args.azure_endpoint
    if args.azure_deployment is not None:
        backend_kwargs["azure_deployment"] = args.azure_deployment

    pipeline = ICFPipeline(
        protocol_path=args.protocol,
        template_path=args.registry,
        template_docx_path=args.template,
        output_dir=args.output_dir,
        model_name=args.model,
        backend=args.backend,
        backend_kwargs=backend_kwargs,
        extraction_backend=args.extraction_backend,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        section_filter=args.sections,
        debug_log_dir=args.debug_log_dir,
        rag_embedding_deployment=args.rag_embedding_deployment,
        rag_reranker=args.rag_reranker,
        rag_top_k=args.rag_top_k,
        rag_rerank_top_k=args.rag_rerank_top_k,
        rag_num_queries=args.rag_num_queries,
        rag_cache_dir=args.rag_cache_dir,
        azure_search_endpoint=args.azure_search_endpoint,
        azure_search_key=args.azure_search_key,
        azure_search_index=args.azure_search_index,
        azure_search_top_k=args.azure_search_top_k,
        azure_search_num_queries=args.azure_search_num_queries,
        azure_search_semantic=args.azure_search_semantic,
        azure_search_semantic_config=args.azure_search_semantic_config,
        skip_review=args.skip_review,
        skip_remediation=args.skip_remediation,
        remediate_high_only=args.remediate_high_only,
        skip_harmonize=args.skip_harmonize,
        us_funded=args.us_funded,
        sdm=args.sdm,
    )

    result = pipeline.run()

    # Exit 1 if there were extraction errors so CI / scripts can detect issues
    if result.summary.get("errors", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
CLI entry point for the UHN ICF Automation Pipeline.

Example usage:

    # Full pipeline (JSON registry — preferred)
    python run_pipeline.py \\
        --protocol data/Prot_000.pdf \\
        --registry data/standard_ICF_template_breakdown.json

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
        --registry data/standard_ICF_template_breakdown.json \\
        --sections 2.1 3 6 8

    # Verbose RLM output
    python run_pipeline.py \\
        --protocol data/Prot_000.pdf \\
        --registry data/standard_ICF_template_breakdown.json \\
        --verbose
"""

import argparse
import io
import os
import sys

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
        required=True,
        help=(
            "Path to the ICF template registry — JSON (preferred) or CSV (legacy). "
            "Use --convert-registry to produce a JSON from a CSV once."
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
        default="gpt-5.1",
        help="LLM model name (default: gpt-5.1).",
    )
    parser.add_argument(
        "--backend",
        default="openai",
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
        choices=["rlm", "naive", "rag", "azure_ai_search"],
        help=(
            "Extraction strategy (default: rlm). "
            "  rlm   — iterative RLM with code execution and semantic chunking (default). "
            "  naive — full-context single LLM call per section (benchmarking baseline). "
            "  rag   — retrieval-augmented generation with hybrid search. "
            "  azure_ai_search — RAG via Azure AI Search (protocol must be pre-indexed). "
            "This flag is orthogonal to --backend: e.g., "
            "'--backend azure_openai --extraction-backend naive' is valid."
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

    backend_kwargs: dict = {}
    if args.max_tokens is not None:
        backend_kwargs["max_tokens"] = args.max_tokens
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
        azure_search_endpoint=args.azure_search_endpoint,
        azure_search_key=args.azure_search_key,
        azure_search_index=args.azure_search_index,
        azure_search_top_k=args.azure_search_top_k,
        azure_search_num_queries=args.azure_search_num_queries,
        azure_search_semantic=args.azure_search_semantic,
        azure_search_semantic_config=args.azure_search_semantic_config,
    )

    result = pipeline.run()

    # Exit 1 if there were extraction errors so CI / scripts can detect issues
    if result.summary.get("errors", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

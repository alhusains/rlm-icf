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
import os
import sys

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
            "RLM backend (default: openai). "
            "Choices: openai | azure_openai | vllm. "
            "Use 'azure_openai' for Azure AI Foundry deployments; "
            "use 'vllm' for local vLLM servers."
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
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        section_filter=args.sections,
    )

    result = pipeline.run()

    # Exit 1 if there were extraction errors so CI / scripts can detect issues
    if result.summary.get("errors", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

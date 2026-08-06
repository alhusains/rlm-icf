#!/usr/bin/env python3
"""
CLI entry point for RLM extraction consistency evaluation.

Runs the RLM extraction engine N times (default 3) on the SAME protocol
section and reports how consistent the outputs are: status/confidence
agreement, text and evidence overlap, plus an LLM-as-judge pass that reads
all N results side by side and flags missing facts or contradictions.

Example usage:

    python run_consistency_eval.py \\
        --protocol data/above_minimal/25-5177/25-5177_Protocol.docx \\
        --study-type standard \\
        --section 3

    # More runs, a different section, explicit registry, custom seed
    python run_consistency_eval.py \\
        --protocol data/Prot_000.pdf \\
        --registry data/UHN_standard_ICF_template_breakdown_new.json \\
        --section 8 \\
        --runs 5 \\
        --seed 42
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from icf.consistency_eval import print_report, run_consistency_eval, save_report


def _resolve_registry_path(registry: str | None, study_type: str | None) -> str:
    """Resolve the registry path from an explicit path or --study-type.

    Unlike run_pipeline.py, this does not fall back to an interactive prompt --
    consistency evaluation is meant to be scriptable/batch-friendly.
    """
    if registry:
        return registry
    repo_root = Path(__file__).resolve().parent
    if study_type == "standard":
        return str(repo_root / "data" / "UHN_standard_ICF_template_breakdown_new.json")
    if study_type == "minimal_risk":
        return str(repo_root / "data" / "minimal_risk_ICF_template_breakdown.json")
    raise ValueError("Either --registry or --study-type must be provided.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run RLM extraction N times on one ICF section and measure "
        "consistency across runs (status/confidence/text/evidence agreement + "
        "an LLM-as-judge pass).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--protocol", required=True, help="Path to the clinical study protocol.")
    parser.add_argument(
        "--registry",
        default=None,
        help="Path to the ICF template registry (JSON). Alternative to --study-type.",
    )
    parser.add_argument(
        "--study-type",
        choices=["standard", "minimal_risk"],
        default=None,
        help="Study type used to select the default registry when --registry is not provided.",
    )
    parser.add_argument(
        "--section",
        required=True,
        help="Single section ID to evaluate for consistency (e.g. 3, 5, 2.1, 8).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of fresh extraction runs to compare (default: 3).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4"),
        help="LLM model name (default: AZURE_OPENAI_DEPLOYMENT env var, falling back to gpt-5.4).",
    )
    parser.add_argument(
        "--backend",
        default="azure_openai",
        help="LLM provider backend (default: azure_openai). Same choices as run_pipeline.py.",
    )
    parser.add_argument(
        "--base-url", default=None, help="Base URL for the LLM API endpoint (e.g. vLLM)."
    )
    parser.add_argument("--api-key", default=None, help="API key for the LLM backend.")
    parser.add_argument(
        "--azure-endpoint", default=None, help="Azure OpenAI endpoint URL override."
    )
    parser.add_argument(
        "--azure-deployment", default=None, help="Azure OpenAI deployment name override."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed passed to the LLM backend on every run (default: 42). Currently only "
            "honored by --backend azure_openai. Kept fixed across runs by design -- this "
            "script measures how much variance remains even with the seed held constant."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Max RLM iterations per run (default: 20).",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("EVAL_JUDGE_MODEL"),
        help="Model used for the LLM-as-judge pass (default: same as --model).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the JSON consistency report (default: derived from protocol + section).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose RLM output for each run."
    )

    args = parser.parse_args()

    try:
        registry_path = _resolve_registry_path(args.registry, args.study_type)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    backend_kwargs: dict = {}
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

    print("=" * 64)
    print("RLM EXTRACTION CONSISTENCY EVALUATION")
    print("=" * 64)
    print(f"  Protocol:   {args.protocol}")
    print(f"  Registry:   {registry_path}")
    print(f"  Section:    {args.section}")
    print(f"  Runs:       {args.runs}")
    print(f"  Model:      {args.model} ({args.backend}), seed={args.seed}")
    print("=" * 64)

    report = run_consistency_eval(
        protocol_path=args.protocol,
        registry_path=registry_path,
        section_id=args.section,
        n_runs=args.runs,
        model_name=args.model,
        backend=args.backend,
        backend_kwargs=backend_kwargs,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        judge_model=args.judge_model,
    )

    print_report(report)

    if args.output:
        output_path = args.output
    else:
        protocol_stem = os.path.splitext(os.path.basename(args.protocol))[0]
        section_slug = args.section.replace(".", "_")
        output_dir = os.path.dirname(args.protocol) or "output"
        output_path = os.path.join(
            output_dir, f"consistency_eval_{protocol_stem}_sec{section_slug}.json"
        )
    save_report(report, output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
CLI entry point for whole-protocol RLM consistency evaluation.

Runs the FULL ICF pipeline (extraction -> harmonization -> plain-language
review -> remediation) N times end to end (default 3) on the same protocol,
then compares the FINAL extraction for every section across runs -- i.e.
exactly what would ship in the ICF. Produces a reviewer-friendly Word report
(one page per section, plus an executive-summary page flagging the sections
most likely to be inconsistent) alongside the raw JSON data.

This is a heavier, slower, more expensive evaluation than
``run_consistency_eval.py`` (which only tests raw extraction, one section at
a time). Intended to run unattended in the background -- see the nohup
example below.

Example usage:

    # All sections, 3 full pipeline runs
    python run_full_consistency_eval.py \\
        --protocol data/above_minimal/25-5953/25-5953_Protocol.docx \\
        --study-type standard

    # Just a few sections, in the background, with logs captured to a file
    nohup python run_full_consistency_eval.py \\
        --protocol data/above_minimal/25-5953/25-5953_Protocol.docx \\
        --study-type standard \\
        --sections 3 5 8 \\
        > consistency_25-5953.log 2>&1 &
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

from icf.consistency_eval import (
    generate_consistency_docx,
    print_multi_report_summary,
    run_full_pipeline_consistency_eval,
    save_reports_json,
)


def _resolve_registry_path(registry: str | None, study_type: str | None) -> str:
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
        description="Run the full ICF pipeline N times on a protocol and produce a "
        "reviewer-friendly Word report comparing the FINAL (post-harmonize/review/"
        "remediate) extraction for every section across runs.",
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
        "--template",
        default=None,
        help="Path to ICF template DOCX (optional, for reference; forwarded to ICFPipeline).",
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        default=None,
        help="Only evaluate these section IDs (e.g. 2.1 3 6 8). Default: every section.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of full end-to-end pipeline runs to compare (default: 3).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.6-sol"),
        help="LLM model name (default: AZURE_OPENAI_DEPLOYMENT env var, falling back to gpt-5.6-sol).",
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
        help="Seed passed to the LLM backend on every run (default: 42).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Max RLM iterations per section, per run (default: 20).",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("EVAL_JUDGE_MODEL"),
        help="Model used for the LLM-as-judge pass (default: same as --model).",
    )
    parser.add_argument(
        "--skip-harmonize",
        action="store_true",
        help="Skip Stage 5.5 harmonization in each pipeline run.",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Skip Stage 8 plain-language review (and therefore Stage 9 remediation) in each run.",
    )
    parser.add_argument(
        "--skip-remediation",
        action="store_true",
        help="Skip Stage 9 remediation in each pipeline run.",
    )
    parser.add_argument(
        "--remediate-high-only",
        action="store_true",
        help="Stage 9: fix HIGH-severity flags only in each run.",
    )
    parser.add_argument(
        "--us-funded", action="store_true", help="Forwarded to ICFPipeline (see run_pipeline.py)."
    )
    parser.add_argument("--sdm", action="store_true", help="Forwarded to ICFPipeline.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for the consistency report + per-run pipeline outputs "
            "(default: <protocol dir>/consistency_eval_<protocol_stem>/)."
        ),
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

    protocol_stem = os.path.splitext(os.path.basename(args.protocol))[0]
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.protocol) or "output", f"consistency_eval_{protocol_stem}"
    )
    os.makedirs(output_dir, exist_ok=True)
    pipeline_runs_dir = os.path.join(output_dir, "pipeline_runs")

    print("=" * 64)
    print("FULL-PIPELINE RLM CONSISTENCY EVALUATION")
    print("=" * 64)
    print(f"  Protocol:        {args.protocol}")
    print(f"  Registry:        {registry_path}")
    print(f"  Sections:        {args.sections or 'ALL'}")
    print(f"  Runs:            {args.runs}")
    print(f"  Model:           {args.model} ({args.backend}), seed={args.seed}")
    print(
        f"  Post-processing: harmonize={not args.skip_harmonize}, "
        f"review={not args.skip_review}, remediation={not args.skip_remediation}"
    )
    print(f"  Output dir:      {output_dir}")
    print("=" * 64)

    reports = run_full_pipeline_consistency_eval(
        protocol_path=args.protocol,
        registry_path=registry_path,
        n_runs=args.runs,
        model_name=args.model,
        backend=args.backend,
        backend_kwargs=backend_kwargs,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        section_filter=args.sections,
        judge_model=args.judge_model,
        skip_harmonize=args.skip_harmonize,
        skip_review=args.skip_review,
        skip_remediation=args.skip_remediation,
        remediate_high_only=args.remediate_high_only,
        us_funded=args.us_funded,
        sdm=args.sdm,
        template_docx_path=args.template,
        pipeline_runs_dir=pipeline_runs_dir,
    )

    if not reports:
        print("ERROR: No comparable sections produced content across the runs.", file=sys.stderr)
        return 1

    print_multi_report_summary(reports)

    docx_path = os.path.join(output_dir, f"consistency_report_{protocol_stem}.docx")
    json_path = os.path.join(output_dir, f"consistency_report_{protocol_stem}.json")

    generate_consistency_docx(
        reports,
        docx_path,
        protocol_path=args.protocol,
        n_runs=args.runs,
        model_name=args.model,
        seed=args.seed,
        post_processing={
            "harmonize": not args.skip_harmonize,
            "review": not args.skip_review,
            "remediation": not args.skip_remediation,
        },
    )
    save_reports_json(reports, json_path, args.protocol, args.runs, args.model)

    print("\nDone.")
    print(f"  Word report (share this with your intern): {docx_path}")
    print(f"  Raw JSON data:                              {json_path}")
    print(f"  Per-run pipeline outputs (draft/marked-up):  {pipeline_runs_dir}/run{{1..{args.runs}}}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())

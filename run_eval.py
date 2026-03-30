#!/usr/bin/env python3
"""
CLI entry point for ICF evaluation.

Evaluates AI-generated ICF extraction results against a ground truth ICF
and the UHN evaluation rubric, using DeepEval with LLM-as-a-judge.

Example usage:

    # Evaluate a single backend
    python run_eval.py \\
        --reports rlm=output/extraction_report_rlm_Prot_000.json \\
        --ground-truth data/ground_truth_icf.docx \\
        --registry data/standard_ICF_template_breakdown.json

    # Compare multiple backends side-by-side
    python run_eval.py \\
        --reports \\
            rlm=output/extraction_report_rlm_Prot_000.json \\
            naive=output/extraction_report_naive_Prot_000.json \\
            rag=output/extraction_report_rag_Prot_000.json \\
            azure_ai_search=output/extraction_report_azure_ai_search_Prot_000.json \\
        --ground-truth data/ground_truth_icf.docx \\
        --registry data/standard_ICF_template_breakdown.json \\
        --protocol data/Prot_000.pdf

    # Evaluate specific sections only
    python run_eval.py \\
        --reports rlm=output/extraction_report_rlm_Prot_000.json \\
        --ground-truth data/ground_truth_icf.docx \\
        --sections 3 6 7 8
"""

import argparse
import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from icf.eval_runner import ICFEvalRunner


def _parse_report_args(report_args: list[str]) -> dict[str, str]:
    """Parse 'name=path' report arguments into a dict."""
    result = {}
    for arg in report_args:
        if "=" not in arg:
            print(
                f"ERROR: --reports expects name=path pairs, got: {arg}",
                file=sys.stderr,
            )
            sys.exit(1)
        name, path = arg.split("=", 1)
        if not os.path.exists(path):
            print(f"ERROR: Report file not found: {path}", file=sys.stderr)
            sys.exit(1)
        result[name.strip()] = path.strip()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate AI-generated ICF extractions against ground truth "
        "and UHN evaluation rubrics using DeepEval (LLM-as-a-judge).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help=(
            "Extraction reports to evaluate, as name=path pairs. "
            "Example: rlm=output/extraction_report_rlm_Prot.json "
            "naive=output/extraction_report_naive_Prot.json"
        ),
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Path to the approved human-written ICF (DOCX) for correctness comparison.",
    )
    parser.add_argument(
        "--registry",
        default="data/standard_ICF_template_breakdown.json",
        help="Path to the ICF template registry (default: data/standard_ICF_template_breakdown.json).",
    )
    parser.add_argument(
        "--protocol",
        default=None,
        help=(
            "Path to the source protocol (PDF/DOCX). "
            "Used as context for fidelity and honesty checks. "
            "Optional but recommended for accurate scoring."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o"),
        help=(
            "LLM model for the judge (default: gpt-4o). "
            "Set EVAL_JUDGE_MODEL env var or pass directly. "
            "For Azure OpenAI, configure AZURE_OPENAI_* env vars "
            "and use the deployment name here."
        ),
    )
    parser.add_argument(
        "--sections",
        nargs="*",
        default=None,
        help="Evaluate only these section IDs (e.g. 3 6 7 8).",
    )
    parser.add_argument(
        "--eval-mode",
        default="combined",
        choices=["combined", "detailed"],
        help=(
            "Evaluation mode (default: combined). "
            "  combined  — 1 LLM call per section, all rubrics scored together (~90%% cheaper). "
            "  detailed  — 1 LLM call per rubric per section via DeepEval GEval (more granular)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path to save the JSON evaluation report. "
            "Default: output/eval_report_combined.json or output/eval_report_detailed.json "
            "based on --eval-mode."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-section scores and reasoning.",
    )

    args = parser.parse_args()

    # Parse report paths
    report_paths = _parse_report_args(args.reports)

    # Detect judge backend
    judge_backend = "Azure OpenAI" if os.environ.get("AZURE_OPENAI_ENDPOINT") else "OpenAI"

    # Default output path based on mode
    output_path = args.output or f"output/eval_report_{args.eval_mode}.json"

    mode_label = (
        "Combined (1 call/section)" if args.eval_mode == "combined"
        else "Detailed (DeepEval GEval)"
    )

    print("=" * 60)
    print(f"ICF EVALUATION - {mode_label}")
    print("=" * 60)
    print(f"  Eval mode:     {args.eval_mode}")
    print(f"  Backends:      {', '.join(report_paths.keys())}")
    print(f"  Ground truth:  {args.ground_truth or 'None'}")
    print(f"  Protocol:      {args.protocol or 'None'}")
    print(f"  Judge model:   {args.judge_model} ({judge_backend})")
    print(f"  Sections:      {args.sections or 'All'}")
    print(f"  Output:        {output_path}")
    print("=" * 60)

    runner = ICFEvalRunner(
        report_paths=report_paths,
        ground_truth_path=args.ground_truth,
        registry_path=args.registry,
        protocol_path=args.protocol,
        judge_model=args.judge_model,
        section_filter=args.sections,
        verbose=args.verbose,
    )

    if args.eval_mode == "combined":
        results = runner.run_combined()
    else:
        results = runner.run()

    # Print comparison table
    runner.print_comparison(results)

    # Save JSON report
    runner.save_report(results, output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

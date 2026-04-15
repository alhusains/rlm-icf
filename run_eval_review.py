#!/usr/bin/env python3
"""
CLI entry point for generating a human-readable review DOCX from ICF eval results.

Combines an eval report (JSON) and an extraction report (JSON) into a colour-coded
Word document where reviewers can read AI-generated text vs approved ground truth,
see per-rubric scores and evidence quotes, and write their own comments.

Example usage:

    python run_eval_review.py \\
        --eval-report output/eval_report_combined_rlm_24_5539_REBApprovedProtocol.json \\
        --extraction-report output/extraction_report_rlm_24_5539_REBApprovedProtocol.json \\
        --ground-truth data/24_5539_REBApprovedProtocol.docx

    # Custom output path
    python run_eval_review.py \\
        --eval-report output/eval_report_combined_rlm_24_5539_REBApprovedProtocol.json \\
        --extraction-report output/extraction_report_rlm_24_5539_REBApprovedProtocol.json \\
        --ground-truth data/24_5539_REBApprovedProtocol.docx \\
        --output output/review_rlm_24_5539.docx
"""

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a colour-coded DOCX review document from ICF eval results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval-report",
        required=True,
        help="Path to the eval report JSON (e.g. output/eval_report_combined_rlm_Prot.json).",
    )
    parser.add_argument(
        "--extraction-report",
        required=True,
        help="Path to the extraction report JSON (e.g. output/extraction_report_rlm_Prot.json).",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Path to the REB-approved ICF DOCX (optional but recommended).",
    )
    parser.add_argument(
        "--registry",
        default="data/standard_ICF_template_breakdown.json",
        help="Path to the ICF template registry (default: data/standard_ICF_template_breakdown.json).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output path for the review DOCX. "
            "Default: output/review_<eval_report_stem>.docx"
        ),
    )

    args = parser.parse_args()

    # Validate inputs
    for label, path in [("--eval-report", args.eval_report),
                         ("--extraction-report", args.extraction_report)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            return 1

    if args.ground_truth and not os.path.exists(args.ground_truth):
        print(f"ERROR: --ground-truth file not found: {args.ground_truth}", file=sys.stderr)
        return 1

    # Default output path derived from eval report filename
    if args.output:
        output_path = args.output
    else:
        stem = os.path.splitext(os.path.basename(args.eval_report))[0]
        output_path = f"output/review_{stem}.docx"

    print("=" * 60)
    print("ICF EVALUATION REVIEW DOCUMENT")
    print("=" * 60)
    print(f"  Eval report:       {args.eval_report}")
    print(f"  Extraction report: {args.extraction_report}")
    print(f"  Ground truth:      {args.ground_truth or 'Not provided'}")
    print(f"  Registry:          {args.registry}")
    print(f"  Output:            {output_path}")
    print("=" * 60)

    from icf.eval_review import generate_review_doc

    generate_review_doc(
        eval_report_path=args.eval_report,
        extraction_report_path=args.extraction_report,
        output_path=output_path,
        ground_truth_path=args.ground_truth,
        registry_path=args.registry,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

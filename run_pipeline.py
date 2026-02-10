#!/usr/bin/env python3
"""
CLI entry point for the UHN ICF Automation Pipeline.

Example usage:

    # Full pipeline
    python run_pipeline.py \\
        --protocol data/Prot_000.pdf \\
        --csv data/standard_ICF_template_breakdown.csv

    # Extract specific sections only
    python run_pipeline.py \\
        --protocol data/Prot_000.pdf \\
        --csv data/standard_ICF_template_breakdown.csv \\
        --sections 2.1 3 6 8

    # Verbose RLM output
    python run_pipeline.py \\
        --protocol data/Prot_000.pdf \\
        --csv data/standard_ICF_template_breakdown.csv \\
        --verbose
"""

import argparse
import sys

from icf.pipeline import ICFPipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="UHN ICF Automation Pipeline - extract protocol data into "
        "Informed Consent Form sections using RLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--protocol",
        required=True,
        help="Path to the clinical study protocol (PDF or DOCX).",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the ICF template breakdown CSV.",
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
        help="RLM backend (default: openai).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Max RLM iterations per variable (default: 20).",
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

    pipeline = ICFPipeline(
        protocol_path=args.protocol,
        template_csv_path=args.csv,
        template_docx_path=args.template,
        output_dir=args.output_dir,
        model_name=args.model,
        backend=args.backend,
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

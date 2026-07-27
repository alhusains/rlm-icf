#!/usr/bin/env python3
"""
Post-processing runner from an existing extraction report.

Loads extractions and validations from a prior pipeline run (no protocol
re-extraction) and runs the late-stage passes in order:

  Stage 5.5  Section-group harmonization (optional, default on)
  Stage 8    Plain-language review (optional, default on — always re-run)
  Stage 9    Review-flag remediation (optional, default on)

Writes updated outputs with a ``postprocessed_`` prefix:

  marked_up_icf_postprocessed_<stem>.docx
  draft_icf_postprocessed_<stem>.docx
  extraction_report_postprocessed_<stem>.json

Usage:
    python run_remediation_only.py \\
        --report data/above_minimal/24-5413/extraction_report_rlm_24_5413_Protocol.json \\
        --registry data/UHN_standard_ICF_template_breakdown_new.json \\
        --output-dir data/above_minimal/24-5413/ \\
        --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers to reconstruct typed objects from the report JSON
# ---------------------------------------------------------------------------


def _load_extractions(data: list[dict]):
    from icf.types import Evidence, ExtractionResult

    results = []
    for d in data:
        evidence = [
            Evidence(
                quote=e.get("quote", ""),
                page=str(e.get("page", "")),
                section=e.get("section", ""),
            )
            for e in d.get("evidence", [])
        ]
        results.append(
            ExtractionResult(
                section_id=d["section_id"],
                heading=d["heading"],
                sub_section=d.get("sub_section"),
                status=d["status"],
                answer=d.get("answer", ""),
                filled_template=d.get("filled_template", ""),
                evidence=evidence,
                confidence=d.get("confidence", "N/A"),
                notes=d.get("notes", ""),
                raw_response=d.get("raw_response", ""),
                error=d.get("error"),
            )
        )
    return results


def _load_validations(data: list[dict]):
    from icf.types import ValidationResult

    results = []
    for d in data:
        results.append(
            ValidationResult(
                section_id=d["section_id"],
                quotes_verified=d.get("quotes_verified", []),
                reading_grade_level=d.get("reading_grade_level"),
                issues=d.get("issues", []),
            )
        )
    return results


def _stem_from_report_path(report_path: str) -> str:
    report_basename = os.path.basename(report_path)
    for prefix in ("extraction_report_", "extraction_report"):
        if report_basename.startswith(prefix):
            return report_basename[len(prefix) :].replace(".json", "")
    return report_basename.replace(".json", "")


def _backend_kwargs_from_args(args: argparse.Namespace) -> dict:
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
    return backend_kwargs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run harmonization, plain-language review, and remediation on an "
            "existing extraction report (no protocol re-extraction)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Path to the extraction report JSON from a previous pipeline run.",
    )
    parser.add_argument(
        "--registry",
        default="data/UHN_standard_ICF_template_breakdown_new.json",
        help="Path to the ICF template registry JSON (default: data/UHN_standard_ICF_template_breakdown_new.json).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4"),
        help="LLM model name (default: AZURE_OPENAI_DEPLOYMENT env var, falling back to gpt-5.4).",
    )
    parser.add_argument(
        "--backend",
        default="azure_openai",
        help="LLM provider backend (default: azure_openai).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for the LLM API endpoint (e.g. for vLLM servers).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the LLM backend.",
    )
    parser.add_argument(
        "--azure-endpoint",
        default=None,
        help="Azure OpenAI endpoint URL.",
    )
    parser.add_argument(
        "--azure-deployment",
        default=None,
        help="Azure deployment name.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max output tokens per LLM call.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output for harmonize, review, and remediation.",
    )
    parser.add_argument(
        "--skip-harmonize",
        action="store_true",
        help="Skip Stage 5.5 section-group harmonization (see run_pipeline.py --skip-harmonize).",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Skip Stage 8 plain-language review (remediation requires review unless also skipped).",
    )
    parser.add_argument(
        "--skip-remediation",
        action="store_true",
        help="Skip Stage 9 remediation (review flags are still written to the report).",
    )
    parser.add_argument(
        "--remediate-high-only",
        action="store_true",
        help=(
            "Stage 9: fix HIGH-severity flags only (default also fixes eligible MEDIUM flags). "
            "See run_pipeline.py --remediate-high-only."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load report JSON
    # ------------------------------------------------------------------
    print(f"[LOAD] Reading report: {args.report}")
    with open(args.report, encoding="utf-8") as f:
        report = json.load(f)

    extractions = _load_extractions(report.get("extractions", []))
    validations = _load_validations(report.get("validations", []))
    summary = dict(report.get("summary", {}))

    if not extractions:
        print("ERROR: No extractions in report.", file=sys.stderr)
        return 1

    print(
        f"[LOAD] {len(extractions)} extractions, {len(validations)} validations "
        f"(prior review flags in file are ignored — review is re-run)"
    )

    # ------------------------------------------------------------------
    # Load template registry
    # ------------------------------------------------------------------
    print(f"[LOAD] Reading registry: {args.registry}")
    from icf.registry import load_template_registry

    variables = load_template_registry(args.registry)
    print(f"[LOAD] {len(variables)} template sections loaded.")

    logo_candidate = os.path.join(os.path.dirname(args.registry) or ".", "UHN_logo.png")
    logo_path = logo_candidate if os.path.isfile(logo_candidate) else None

    backend_kwargs = _backend_kwargs_from_args(args)

    # ------------------------------------------------------------------
    # Stage 5.5: Harmonization
    # ------------------------------------------------------------------
    if not args.skip_harmonize:
        from icf.harmonize import SectionGroupHarmonizer

        harmonizer = SectionGroupHarmonizer(
            model_name=args.model,
            backend=args.backend,
            backend_kwargs=backend_kwargs,
            verbose=args.verbose,
        )
        print("\n[HARMONIZE] Running section-group harmonization (Stage 5.5) ...")
        extractions, harmonize_audit = harmonizer.run_harmonization(extractions, variables)
        total_changed = sum(len(v) for v in harmonize_audit.values())
        print(
            f"[HARMONIZE] Done — {total_changed} sub-section(s) revised "
            f"across {len(harmonize_audit)} section(s)."
        )
    else:
        print("\n[HARMONIZE] Skipped (--skip-harmonize).")

    # ------------------------------------------------------------------
    # Stage 8: Review (always fresh)
    # ------------------------------------------------------------------
    review_result = None
    if not args.skip_review:
        from icf.review import ReviewEngine

        print("\n[REVIEW] Running plain language review (Stage 8) ...")
        reviewer = ReviewEngine(
            model_name=args.model,
            backend=args.backend,
            backend_kwargs=backend_kwargs,
            verbose=args.verbose,
        )
        review_result = reviewer.run_review(extractions, variables)
        n_flags = len(review_result.flags)
        high = sum(1 for f in review_result.flags if f.severity == "HIGH")
        medium = sum(1 for f in review_result.flags if f.severity == "MEDIUM")
        print(f"[REVIEW] {n_flags} flag(s): {high} HIGH, {medium} MEDIUM.")
        if review_result.cross_section_notes:
            preview = review_result.cross_section_notes[:200]
            print(f"[REVIEW] Cross-section notes: {preview}")
        summary["review_flags"] = n_flags
    else:
        print("\n[REVIEW] Skipped (--skip-review).")
        summary["review_flags"] = 0

    # ------------------------------------------------------------------
    # Stage 9: Remediation
    # ------------------------------------------------------------------
    remediation_result = None
    if args.skip_review:
        if not args.skip_remediation:
            print("\n[REMEDIATE] Skipped (requires review — use without --skip-review).")
    elif args.skip_remediation:
        print("\n[REMEDIATE] Skipped (--skip-remediation).")
    elif review_result is not None:
        from icf.remediate import RemediationEngine, _is_remediable_flag, _is_remediable_medium

        remediate_medium = not args.remediate_high_only
        high_count = sum(1 for f in review_result.flags if f.severity == "HIGH")
        medium_count = sum(
            1 for f in review_result.flags if remediate_medium and _is_remediable_medium(f)
        )
        remediable_count = sum(
            1 for f in review_result.flags if _is_remediable_flag(f, remediate_medium)
        )
        has_notes = bool(review_result.cross_section_notes.strip())
        # Abbreviation consistency is checked deterministically (see
        # icf/abbreviations.py) and must not depend on review having flagged
        # something else first -- check it directly for gating.
        from icf.abbreviations import find_abbreviation_fixes

        has_abbreviation_issues = bool(find_abbreviation_fixes(extractions, variables))

        if remediable_count == 0 and not has_notes and not has_abbreviation_issues:
            print(
                "\n[REMEDIATE] No remediable flags, cross-section notes, or abbreviation issues — skipping."
            )
        else:
            remediator = RemediationEngine(
                model_name=args.model,
                backend=args.backend,
                backend_kwargs=backend_kwargs,
                verbose=args.verbose,
                remediate_medium=remediate_medium,
            )
            print(
                f"\n[REMEDIATE] Running Stage 9 remediation "
                f"({high_count} HIGH, {medium_count} eligible MEDIUM flag(s), "
                f"cross-section notes: {bool(has_notes)}, "
                f"abbreviation fixes: {has_abbreviation_issues}) ..."
            )
            extractions, remediation_result = remediator.run_remediation(
                extractions, variables, review_result
            )
            n_patched = sum(1 for r in remediation_result.records if r.success)
            n_total = len(remediation_result.records)
            print(f"[REMEDIATE] {n_patched}/{n_total} section(s) patched successfully.")
            if remediation_result.unaddressed_notes:
                print(
                    f"[REMEDIATE] Unaddressed (human review): "
                    f"{remediation_result.unaddressed_notes[:300]}"
                )

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    stem = _stem_from_report_path(args.report)

    marked_up_path = os.path.join(args.output_dir, f"marked_up_icf_postprocessed_{stem}.docx")
    draft_path = os.path.join(args.output_dir, f"draft_icf_postprocessed_{stem}.docx")
    report_path = os.path.join(args.output_dir, f"extraction_report_postprocessed_{stem}.json")

    from icf.assemble import generate_marked_up_docx, generate_report_json
    from icf.clean_icf import generate_draft_docx

    print(f"\n[ASSEMBLE] Writing marked-up ICF -> {marked_up_path}")
    generate_marked_up_docx(extractions, validations, variables, marked_up_path, review_result)

    print(f"[ASSEMBLE] Writing report        -> {report_path}")
    generate_report_json(
        extractions,
        validations,
        summary,
        report_path,
        review_result,
        remediation_result,
    )

    print(f"[ASSEMBLE] Writing draft ICF     -> {draft_path}")
    generate_draft_docx(
        extractions=extractions,
        variables=variables,
        output_path=draft_path,
        logo_path=logo_path,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    sep = "=" * 60
    print(f"\n{sep}")
    print("POST-PROCESSING SUMMARY")
    print(sep)
    print(f"  Harmonize:     {'skipped' if args.skip_harmonize else 'done'}")
    if args.skip_review:
        review_summary = "skipped"
    else:
        review_summary = f"{summary.get('review_flags', 0)} flags"
    print(f"  Review:        {review_summary}")
    if remediation_result is not None:
        n_patched = sum(1 for r in remediation_result.records if r.success)
        n_total = len(remediation_result.records)
        print(f"  Remediation:   {n_patched}/{n_total} section(s) patched")
        failed = [r for r in remediation_result.records if not r.success]
        if failed:
            print(f"  Failed:        {', '.join(r.section_id for r in failed)}")
    else:
        print("  Remediation:   skipped")
    print(f"  Marked-up ICF: {marked_up_path}")
    print(f"  Draft ICF:     {draft_path}")
    print(f"  Report:        {report_path}")
    print(sep)

    return 0


if __name__ == "__main__":
    sys.exit(main())

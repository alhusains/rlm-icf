#!/usr/bin/env python3
"""
Standalone remediation runner.

Loads an existing extraction report JSON (produced by a previous pipeline run)
and runs only Stage 9 (HIGH flag remediation) on it, then writes the same three
output files that the full pipeline produces:

  draft_icf_remediated_<stem>.docx
  final_icf_remediated_<stem>.docx
  extraction_report_remediated_<stem>.json

Usage:
    python run_remediation_only.py \\
        --report output/extraction_report_rlm_23_5719_REBApprovedProtocol.json \\
        --registry data/standard_ICF_template_breakdown.json

Optional flags (same as run_pipeline.py):
    --output-dir output
    --model gpt-5.1
    --backend openai
    --verbose
"""

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


def _normalize_section_id(raw: str) -> str:
    """Strip accidental 'SECTION ' prefix from IDs stored in older report JSON files."""
    import re

    return re.sub(r"^(?:SECTION|Section|section)\s+", "", raw).strip()


def _load_review_result(data: dict | None):
    from icf.types import ReviewFlag, ReviewResult

    if data is None:
        return None
    flags = []
    for f in data.get("flags", []):
        flags.append(
            ReviewFlag(
                section_id=_normalize_section_id(f["section_id"]),
                flagged_text=f.get("flagged_text", ""),
                issue_type=f.get("issue_type", "UNCLEAR"),
                suggestion=f.get("suggestion", ""),
                severity=f.get("severity", "LOW"),
                suggested_fix=f.get("suggested_fix", ""),
            )
        )
    return ReviewResult(
        flags=flags,
        cross_section_notes=data.get("cross_section_notes", ""),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-run Stage 9 remediation on an existing extraction report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Path to the extraction report JSON from a previous pipeline run.",
    )
    parser.add_argument(
        "--registry",
        default="data/standard_ICF_template_breakdown.json",
        help="Path to the ICF template registry JSON (default: data/standard_ICF_template_breakdown.json).",
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
        help="LLM provider backend (default: openai).",
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
        help="Show verbose remediation output.",
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
    review_result = _load_review_result(report.get("review"))
    summary = report.get("summary", {})

    if review_result is None:
        print("ERROR: No 'review' section in report. Run the pipeline without --skip-review first.")
        return 1

    high_count = sum(1 for f in review_result.flags if f.severity == "HIGH")
    has_notes = bool(review_result.cross_section_notes.strip())
    print(
        f"[LOAD] {len(extractions)} extractions, {len(validations)} validations, "
        f"{len(review_result.flags)} review flag(s) ({high_count} HIGH), "
        f"cross-section notes: {bool(has_notes)}"
    )

    if high_count == 0 and not has_notes:
        print("[REMEDIATE] No HIGH flags or cross-section notes — nothing to remediate.")
        return 0

    # ------------------------------------------------------------------
    # Load template registry (needed for variables + output docs)
    # ------------------------------------------------------------------
    print(f"[LOAD] Reading registry: {args.registry}")
    from icf.registry import load_template_registry

    variables = load_template_registry(args.registry)
    print(f"[LOAD] {len(variables)} template sections loaded.")

    # Resolve logo path (same logic as pipeline.py)
    logo_candidate = os.path.join(os.path.dirname(args.registry) or ".", "UHN_logo.png")
    logo_path = logo_candidate if os.path.isfile(logo_candidate) else None

    # ------------------------------------------------------------------
    # Run Stage 9 remediation
    # ------------------------------------------------------------------
    from icf.remediate import RemediationEngine

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

    remediator = RemediationEngine(
        model_name=args.model,
        backend=args.backend,
        backend_kwargs=backend_kwargs,
        verbose=args.verbose,
    )

    print(f"\n[REMEDIATE] Running remediation ({high_count} HIGH flag(s)) ...")
    patched_extractions, remediation_result = remediator.run_remediation(
        extractions, variables, review_result
    )

    n_patched = sum(1 for r in remediation_result.records if r.success)
    n_total = len(remediation_result.records)
    print(f"[REMEDIATE] {n_patched}/{n_total} section(s) patched successfully.")
    if remediation_result.unaddressed_notes:
        print(
            f"[REMEDIATE] Unaddressed (human review): {remediation_result.unaddressed_notes[:300]}"
        )

    # ------------------------------------------------------------------
    # Build output paths
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # Derive stem from input report filename, replacing the prefix
    report_basename = os.path.basename(args.report)
    # Strip known prefix to get the original stem
    for prefix in ("extraction_report_", "extraction_report"):
        if report_basename.startswith(prefix):
            stem = report_basename[len(prefix) :].replace(".json", "")
            break
    else:
        stem = report_basename.replace(".json", "")

    draft_path = os.path.join(args.output_dir, f"draft_icf_remediated_{stem}.docx")
    final_path = os.path.join(args.output_dir, f"final_icf_remediated_{stem}.docx")
    report_path = os.path.join(args.output_dir, f"extraction_report_remediated_{stem}.json")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    from icf.assemble import generate_draft_docx, generate_report_json
    from icf.clean_icf import generate_clean_icf_docx

    print(f"\n[ASSEMBLE] Writing draft ICF  -> {draft_path}")
    generate_draft_docx(patched_extractions, validations, variables, draft_path, review_result)

    print(f"[ASSEMBLE] Writing report     -> {report_path}")
    generate_report_json(
        patched_extractions,
        validations,
        summary,
        report_path,
        review_result,
        remediation_result,
    )

    print(f"[ASSEMBLE] Writing final ICF  -> {final_path}")
    generate_clean_icf_docx(
        extractions=patched_extractions,
        variables=variables,
        output_path=final_path,
        logo_path=logo_path,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    sep = "=" * 60
    print(f"\n{sep}")
    print("REMEDIATION SUMMARY")
    print(sep)
    print(f"  Sections patched:    {n_patched}/{n_total}")
    failed = [r for r in remediation_result.records if not r.success]
    if failed:
        print(f"  Failed sections:     {', '.join(r.section_id for r in failed)}")
    print(f"  Global rules:        {len(remediation_result.global_rules)}")
    print(f"  Draft ICF:   {draft_path}")
    print(f"  Final ICF:   {final_path}")
    print(f"  Report:      {report_path}")
    print(sep)

    return 0


if __name__ == "__main__":
    sys.exit(main())

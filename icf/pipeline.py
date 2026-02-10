"""
ICF Pipeline orchestrator.

Coordinates all stages: ingest -> registry -> extract -> validate -> assemble.
"""

import os
import time

from icf.assemble import generate_draft_docx, generate_report_json
from icf.extract import ExtractionEngine
from icf.ingest import load_protocol
from icf.registry import load_template_registry
from icf.types import (
    ExtractionResult,
    PipelineResult,
    ValidationResult,
)
from icf.validate import validate_extractions


class ICFPipeline:
    """End-to-end Informed Consent Form extraction pipeline.

    Usage::

        pipeline = ICFPipeline(
            protocol_path="data/Prot_000.pdf",
            template_csv_path="data/standard_ICF_template_breakdown.csv",
        )
        result = pipeline.run()
        ICFPipeline.print_summary(result)
    """

    def __init__(
        self,
        protocol_path: str,
        template_csv_path: str,
        template_docx_path: str | None = None,
        output_dir: str = "output",
        model_name: str = "gpt-5.1",
        backend: str = "openai",
        backend_kwargs: dict | None = None,
        max_iterations: int = 20,
        verbose: bool = False,
        section_filter: list[str] | None = None,
    ):
        self.protocol_path = protocol_path
        self.template_csv_path = template_csv_path
        self.template_docx_path = template_docx_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.section_filter = section_filter

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> PipelineResult:
        wall_start = time.time()

        # -- Stage 1: Ingest ------------------------------------------------
        print(f"[INGEST] Loading protocol from {self.protocol_path} ...")
        protocol = load_protocol(self.protocol_path)
        print(f"[INGEST] Loaded: {protocol.total_pages} pages, {len(protocol.full_text):,} chars.")

        # -- Stage 2: Registry -----------------------------------------------
        print(f"[REGISTRY] Loading template from {self.template_csv_path} ...")
        all_variables = load_template_registry(self.template_csv_path)
        print(f"[REGISTRY] Loaded {len(all_variables)} template sections.")

        # Optional section filter
        if self.section_filter:
            variables = [v for v in all_variables if v.section_id in self.section_filter]
            print(f"[REGISTRY] Filtered to {len(variables)} sections: {self.section_filter}")
        else:
            variables = all_variables

        extractable = [v for v in variables if v.is_in_protocol or v.partially_in_protocol]
        standard = [v for v in variables if v.is_standard_text]
        skippable = [
            v
            for v in variables
            if not v.is_in_protocol and not v.partially_in_protocol and not v.is_standard_text
        ]
        print(
            f"[REGISTRY] {len(extractable)} extractable, "
            f"{len(standard)} standard text, "
            f"{len(skippable)} will be skipped (not in protocol)."
        )

        # -- Stages 3+4: Extract ---------------------------------------------
        print(
            f"\n[EXTRACT] Starting extraction  model={self.model_name}  "
            f"backend={self.backend}  max_iter={self.max_iterations}"
        )

        engine = ExtractionEngine(
            model_name=self.model_name,
            backend=self.backend,
            backend_kwargs=self.backend_kwargs,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
        )

        extractions: list[ExtractionResult] = []
        total = len(variables)

        try:
            for i, var in enumerate(variables, start=1):
                display = var.get_display_name()
                label = var.get_complexity_label()

                if var.is_standard_text:
                    print(f"[EXTRACT] [{i}/{total}] STD_TEXT: {display}")
                elif not var.is_in_protocol and not var.partially_in_protocol:
                    print(f"[EXTRACT] [{i}/{total}] SKIP: {display} (Not in protocol)")
                else:
                    print(f"[EXTRACT] [{i}/{total}] Extracting: {display} ({label}) ...")

                result = engine.extract_variable(protocol.full_text, var)
                extractions.append(result)

                self._print_extraction_status(i, total, result)

        except KeyboardInterrupt:
            print(
                f"\n[EXTRACT] Interrupted after {len(extractions)}/{total} "
                "sections. Saving partial results ..."
            )

        # -- Stage 5: Validate -----------------------------------------------
        print(f"\n[VALIDATE] Validating {len(extractions)} extractions ...")
        validations = validate_extractions(extractions, protocol.full_text)

        total_issues = sum(len(v.issues) for v in validations)
        fully_verified = sum(1 for v in validations if v.quotes_verified and all(v.quotes_verified))
        print(f"[VALIDATE] {fully_verified} fully verified, {total_issues} issues found.")

        # -- Stage 6: Assemble -----------------------------------------------
        os.makedirs(self.output_dir, exist_ok=True)

        elapsed = time.time() - wall_start
        summary = self._build_summary(extractions, validations, elapsed)

        docx_path = os.path.join(self.output_dir, "draft_icf.docx")
        json_path = os.path.join(self.output_dir, "extraction_report.json")

        print(f"\n[ASSEMBLE] Writing draft ICF -> {docx_path}")
        generate_draft_docx(extractions, validations, all_variables, docx_path)

        print(f"[ASSEMBLE] Writing report    -> {json_path}")
        generate_report_json(extractions, validations, summary, json_path)

        result = PipelineResult(
            extractions=extractions,
            validations=validations,
            output_docx_path=docx_path,
            report_path=json_path,
            summary=summary,
        )

        self.print_summary(result)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_extraction_status(idx: int, total: int, ext: ExtractionResult) -> None:
        if ext.status == "SKIPPED":
            print(f"[EXTRACT] [{idx}/{total}]  -> SKIPPED")
        elif ext.status == "STANDARD_TEXT":
            print(f"[EXTRACT] [{idx}/{total}]  -> STANDARD_TEXT")
        elif ext.status == "ERROR":
            print(f"[EXTRACT] [{idx}/{total}]  -> ERROR: {ext.error}")
        else:
            ev = len(ext.evidence)
            print(
                f"[EXTRACT] [{idx}/{total}]  -> {ext.status} | "
                f"Confidence: {ext.confidence} | Evidence: {ev} quote(s)"
            )

    @staticmethod
    def _build_summary(
        extractions: list[ExtractionResult],
        validations: list[ValidationResult],
        elapsed: float,
    ) -> dict:
        total = len(extractions)
        counts = {
            "FOUND": 0,
            "PARTIAL": 0,
            "NOT_FOUND": 0,
            "SKIPPED": 0,
            "STANDARD_TEXT": 0,
            "ERROR": 0,
        }
        for e in extractions:
            counts[e.status] = counts.get(e.status, 0) + 1

        return {
            "total_sections": total,
            "found": counts["FOUND"],
            "partial": counts["PARTIAL"],
            "not_found": counts["NOT_FOUND"],
            "skipped": counts["SKIPPED"],
            "standard_text": counts["STANDARD_TEXT"],
            "errors": counts["ERROR"],
            "validation_issues": sum(len(v.issues) for v in validations),
            "elapsed_seconds": round(elapsed, 1),
        }

    @staticmethod
    def print_summary(result: PipelineResult) -> None:
        s = result.summary
        sep = "=" * 60
        print(f"\n{sep}")
        print("ICF PIPELINE SUMMARY")
        print(sep)
        print(f"  Total sections:   {s['total_sections']}")
        print(f"  FOUND:            {s['found']}")
        print(f"  PARTIAL:          {s['partial']}")
        print(f"  NOT_FOUND:        {s['not_found']}")
        print(f"  SKIPPED:          {s['skipped']}")
        print(f"  STANDARD_TEXT:    {s['standard_text']}")
        print(f"  ERRORS:           {s['errors']}")
        print(f"  Validation issues:{s['validation_issues']}")
        print(f"  Wall time:        {s['elapsed_seconds']}s")
        print(sep)
        if result.output_docx_path:
            print(f"  Draft ICF:  {result.output_docx_path}")
        if result.report_path:
            print(f"  Report:     {result.report_path}")
        print(sep)

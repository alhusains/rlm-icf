"""
ICF Pipeline orchestrator.

Coordinates all stages: ingest -> registry -> extract -> adapt -> validate -> assemble.

Extraction runs in two phases:
  Phase A  Extract "trigger" sections (Introduction §3, Why Is This Study
           Being Done §6) first.
  Adapt    A lightweight LLM pass reviews the trigger-section content and
           marks irrelevant optional sections as adaptation_skipped so they
           are neither extracted nor written to the draft ICF.
  Phase B  Extract all remaining sections using the adapted registry.

The adapted registry is saved to <output_dir>/adapted_registry.json after
every run so the adaptation decisions are auditable.
"""

import json
import os
import time

from icf.adapt import ADAPTATION_TRIGGER_IDS, build_adapted_registry
from icf.assemble import generate_draft_docx, generate_report_json
from icf.clean_icf import generate_clean_icf_docx
from icf.debug_logger import ICFDebugLogger
from icf.extract import ExtractionEngine
from icf.ingest import load_protocol
from icf.registry import load_template_registry
from icf.types import (
    ExtractionResult,
    PipelineResult,
    ReviewResult,
    TemplateVariable,
    ValidationResult,
)
from icf.validate import validate_extractions

_VALID_EXTRACTION_BACKENDS = ("rlm", "naive", "rag", "azure_ai_search")


class ICFPipeline:
    """End-to-end Informed Consent Form extraction pipeline.

    Usage::

        pipeline = ICFPipeline(
            protocol_path="data/Prot_000.pdf",
            template_path="data/standard_ICF_template_breakdown.json",
        )
        result = pipeline.run()
        ICFPipeline.print_summary(result)
    """

    def __init__(
        self,
        protocol_path: str,
        template_path: str,
        template_docx_path: str | None = None,
        output_dir: str = "output",
        model_name: str = "gpt-5.1",
        backend: str = "openai",
        backend_kwargs: dict | None = None,
        extraction_backend: str = "rlm",
        max_iterations: int = 20,
        verbose: bool = False,
        section_filter: list[str] | None = None,
        debug_log_dir: str | None = None,
        # RAG-specific parameters (only used when extraction_backend="rag")
        rag_embedding_deployment: str = "text-embedding-3-large",
        rag_reranker: str = "local",
        rag_top_k: int = 20,
        rag_rerank_top_k: int = 8,
        rag_num_queries: int = 4,
        # Azure AI Search parameters (only used when extraction_backend="azure_ai_search")
        azure_search_endpoint: str | None = None,
        azure_search_key: str | None = None,
        azure_search_index: str | None = None,
        azure_search_top_k: int = 10,
        azure_search_num_queries: int = 3,
        azure_search_semantic: bool = False,
        azure_search_semantic_config: str | None = None,
        skip_review: bool = False,
    ):
        if extraction_backend not in _VALID_EXTRACTION_BACKENDS:
            raise ValueError(
                f"extraction_backend must be one of {_VALID_EXTRACTION_BACKENDS}, "
                f"got: {extraction_backend!r}"
            )
        self.protocol_path = protocol_path
        self.template_path = template_path
        self.template_docx_path = template_docx_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.extraction_backend = extraction_backend
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.section_filter = section_filter
        self.debug_log_dir = debug_log_dir
        self.rag_embedding_deployment = rag_embedding_deployment
        self.rag_reranker = rag_reranker
        self.rag_top_k = rag_top_k
        self.rag_rerank_top_k = rag_rerank_top_k
        self.rag_num_queries = rag_num_queries
        self.azure_search_endpoint = azure_search_endpoint
        self.azure_search_key = azure_search_key
        self.azure_search_index = azure_search_index
        self.azure_search_top_k = azure_search_top_k
        self.azure_search_num_queries = azure_search_num_queries
        self.azure_search_semantic = azure_search_semantic
        self.azure_search_semantic_config = azure_search_semantic_config
        self.skip_review = skip_review

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
        print(f"[REGISTRY] Loading template from {self.template_path} ...")
        all_variables = load_template_registry(self.template_path)
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

        # -- Stage 3+4+5: Two-phase extract with adaptation ------------------
        debug_logger: ICFDebugLogger | None = None
        if self.debug_log_dir:
            if self.extraction_backend != "rlm":
                print(
                    f"[DEBUG] --debug-log-dir is only supported for the 'rlm' backend "
                    f"(current: '{self.extraction_backend}'). Ignoring."
                )
            else:
                debug_logger = ICFDebugLogger(log_dir=self.debug_log_dir)
                print(f"[DEBUG] RLM trace will be saved -> {debug_logger.log_file_path}")

        print(f"[EXTRACT] Extraction backend: {self.extraction_backend.upper()}")
        engine = self._build_engine(debug_logger, protocol)

        # Split variables into trigger (adaptation seeds) and the rest.
        trigger_ids_in_run = ADAPTATION_TRIGGER_IDS & {v.section_id for v in variables}
        trigger_vars = [v for v in variables if v.section_id in trigger_ids_in_run]
        non_trigger_vars = [v for v in variables if v.section_id not in trigger_ids_in_run]

        total = len(variables)
        extractions: list[ExtractionResult] = []
        idx = 0  # running display counter

        try:
            # -- Phase A: trigger sections (Introduction + Why Is This Study Done)
            if trigger_vars:
                print(
                    f"\n[EXTRACT] Phase A: {len(trigger_vars)} trigger section(s) "
                    f"(adaptation seeds: {sorted(trigger_ids_in_run)})"
                )
            early_results: list[ExtractionResult] = []
            for var in trigger_vars:
                idx += 1
                self._print_pre_extraction(idx, total, var)
                result = engine.extract_variable(protocol.full_text, var)
                extractions.append(result)
                early_results.append(result)
                self._print_extraction_status(idx, total, result)

            # -- Adaptation pass
            if trigger_vars and early_results:
                n_optional = sum(1 for v in non_trigger_vars if not v.required)
                print(
                    f"\n[ADAPT] Running adaptation pass "
                    f"({n_optional} optional candidate section(s)) ..."
                )
                adapted_non_trigger = build_adapted_registry(
                    variables=non_trigger_vars,
                    early_results=early_results,
                    model_name=self.model_name,
                    backend=self.backend,
                    backend_kwargs=self.backend_kwargs,
                )
                n_skipped = sum(1 for v in adapted_non_trigger if v.adaptation_skipped)
                print(f"[ADAPT] {n_skipped} optional section(s) marked for skipping.")
            else:
                adapted_non_trigger = non_trigger_vars

            # Merge adaptations back so all_variables stays intact for the DOCX
            adapted_map: dict[str, TemplateVariable] = {v.section_id: v for v in trigger_vars}
            adapted_map.update({v.section_id: v for v in adapted_non_trigger})
            final_variables = [adapted_map.get(v.section_id, v) for v in all_variables]

            # Save the adapted registry for transparency / auditing
            os.makedirs(self.output_dir, exist_ok=True)
            self._save_adapted_registry(list(adapted_map.values()), self.output_dir)

            # -- Phase B: remaining sections (using adapted registry)
            if non_trigger_vars:
                print(f"\n[EXTRACT] Phase B: {len(adapted_non_trigger)} remaining section(s)")
            for var in adapted_non_trigger:
                idx += 1
                self._print_pre_extraction(idx, total, var)
                result = engine.extract_variable(protocol.full_text, var)
                extractions.append(result)
                self._print_extraction_status(idx, total, result)

        except KeyboardInterrupt:
            print(
                f"\n[EXTRACT] Interrupted after {len(extractions)}/{total} "
                "sections. Saving partial results ..."
            )
            # Ensure final_variables is defined even on early exit
            if "final_variables" not in dir():
                final_variables = all_variables

        # -- Stage 6: Validate -----------------------------------------------
        print(f"\n[VALIDATE] Validating {len(extractions)} extractions ...")
        validations = validate_extractions(extractions, protocol.full_text)

        total_issues = sum(len(v.issues) for v in validations)
        fully_verified = sum(1 for v in validations if v.quotes_verified and all(v.quotes_verified))
        print(f"[VALIDATE] {fully_verified} fully verified, {total_issues} issues found.")

        # -- Stage 7: Assemble (paths only — outputs written after review) ----
        os.makedirs(self.output_dir, exist_ok=True)

        elapsed = time.time() - wall_start
        summary = self._build_summary(extractions, validations, elapsed)

        stem = self._output_stem()
        docx_path = os.path.join(self.output_dir, f"draft_icf_{stem}.docx")
        json_path = os.path.join(self.output_dir, f"extraction_report_{stem}.json")
        clean_docx_path = os.path.join(self.output_dir, f"final_icf_{stem}.docx")

        # -- Stage 8: Review (optional plain-language annotation pass) --------
        review_result: ReviewResult | None = None
        if not self.skip_review:
            from icf.review import ReviewEngine

            print("\n[REVIEW] Running plain language review (Stage 8) ...")
            if self.section_filter:
                print(
                    "[REVIEW] NOTE: Section filter is active — cross-section analysis "
                    f"covers only: {self.section_filter}"
                )
            reviewer = ReviewEngine(
                model_name=self.model_name,
                backend=self.backend,
                backend_kwargs=self.backend_kwargs,
                verbose=self.verbose,
            )
            review_result = reviewer.run_review(extractions, final_variables)
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

        # -- Write outputs (now that review_result is available) --------------
        print(f"\n[ASSEMBLE] Writing draft ICF -> {docx_path}")
        generate_draft_docx(extractions, validations, final_variables, docx_path, review_result)

        print(f"[ASSEMBLE] Writing report    -> {json_path}")
        generate_report_json(extractions, validations, summary, json_path, review_result)

        # Resolve the UHN logo from the same directory as the template registry.
        _logo_candidate = os.path.join(
            os.path.dirname(self.template_path) or ".", "UHN_logo.png"
        )
        logo_path = _logo_candidate if os.path.isfile(_logo_candidate) else None

        print(f"[ASSEMBLE] Writing clean ICF -> {clean_docx_path}")
        generate_clean_icf_docx(
            extractions=extractions,
            variables=final_variables,
            output_path=clean_docx_path,
            logo_path=logo_path,
        )

        result = PipelineResult(
            extractions=extractions,
            validations=validations,
            output_docx_path=docx_path,
            clean_icf_path=clean_docx_path,
            report_path=json_path,
            summary=summary,
            review_result=review_result,
        )

        self.print_summary(result)
        return result

    # ------------------------------------------------------------------
    # Engine factory
    # ------------------------------------------------------------------

    def _build_engine(self, debug_logger: "ICFDebugLogger | None", protocol=None):
        """Instantiate the correct extraction engine for self.extraction_backend."""
        if self.extraction_backend == "rlm":
            return ExtractionEngine(
                model_name=self.model_name,
                backend=self.backend,
                backend_kwargs=self.backend_kwargs,
                max_iterations=self.max_iterations,
                verbose=self.verbose,
                debug_logger=debug_logger,
            )
        if self.extraction_backend == "naive":
            from icf.naive_extract import NaiveExtractionEngine

            return NaiveExtractionEngine(
                model_name=self.model_name,
                backend=self.backend,
                backend_kwargs=self.backend_kwargs,
                verbose=self.verbose,
            )
        if self.extraction_backend == "rag":
            from icf.rag_extract import RAGExtractionEngine
            from icf.rag_index import ProtocolIndex, RAGConfig, build_embedding_client
            from icf.rag_rerank import get_reranker

            assert protocol is not None, "protocol (IndexedProtocol) required for RAG backend"

            config = RAGConfig(
                embedding_model=self.rag_embedding_deployment,
                retrieval_top_k=self.rag_top_k,
                rerank_top_k=self.rag_rerank_top_k,
                num_queries=self.rag_num_queries,
                reranker=self.rag_reranker,
            )
            embedding_client = build_embedding_client(self.backend, self.backend_kwargs)
            index = ProtocolIndex(protocol, config, embedding_client)
            index.build()

            reranker = get_reranker(config)
            return RAGExtractionEngine(
                protocol_index=index,
                reranker=reranker,
                model_name=self.model_name,
                backend=self.backend,
                backend_kwargs=self.backend_kwargs,
                config=config,
                verbose=self.verbose,
            )

        if self.extraction_backend == "azure_ai_search":
            from icf.azure_search_extract import AzureSearchExtractionEngine

            if not self.azure_search_endpoint or not self.azure_search_key or not self.azure_search_index:
                raise ValueError(
                    "azure_ai_search backend requires --azure-search-endpoint, "
                    "--azure-search-key, and --azure-search-index."
                )

            return AzureSearchExtractionEngine(
                search_endpoint=self.azure_search_endpoint,
                search_key=self.azure_search_key,
                search_index=self.azure_search_index,
                model_name=self.model_name,
                backend=self.backend,
                backend_kwargs=self.backend_kwargs,
                search_top_k=self.azure_search_top_k,
                num_queries=self.azure_search_num_queries,
                use_semantic=self.azure_search_semantic,
                semantic_config=self.azure_search_semantic_config,
                verbose=self.verbose,
                debug_logger=debug_logger,
            )

        raise ValueError(f"Unknown extraction_backend: {self.extraction_backend!r}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_pre_extraction(idx: int, total: int, var: TemplateVariable) -> None:
        display = var.get_display_name()
        label = var.get_complexity_label()
        if var.adaptation_skipped:
            print(f"[EXTRACT] [{idx}/{total}] ADAPT_SKIP: {display}")
        elif var.is_standard_text:
            print(f"[EXTRACT] [{idx}/{total}] STD_TEXT: {display}")
        elif not var.is_in_protocol and not var.partially_in_protocol:
            print(f"[EXTRACT] [{idx}/{total}] SKIP: {display} (Not in protocol)")
        else:
            print(f"[EXTRACT] [{idx}/{total}] Extracting: {display} ({label}) ...")

    @staticmethod
    def _print_extraction_status(idx: int, total: int, ext: ExtractionResult) -> None:
        if ext.status in ("SKIPPED", "ADAPTATION_SKIPPED"):
            print(f"[EXTRACT] [{idx}/{total}]  -> {ext.status}")
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

    def _output_stem(self) -> str:
        """Return a short identifier used as a suffix in all output filenames.

        Format: <backend>_<protocol_stem>
        Example: rag_Prot_000   naive_StudyABC   rlm_Protocol_v2
        """
        protocol_stem = os.path.splitext(os.path.basename(self.protocol_path))[0]
        # Collapse any run of whitespace/special chars to underscores
        import re

        safe_stem = re.sub(r"[^\w]+", "_", protocol_stem).strip("_")
        return f"{self.extraction_backend}_{safe_stem}"

    @staticmethod
    def _save_adapted_registry(variables: list[TemplateVariable], output_dir: str) -> None:
        """Write a concise JSON summary of adaptation decisions to the output dir."""
        path = os.path.join(output_dir, "adapted_registry.json")
        data = [
            {
                "section_id": v.section_id,
                "heading": v.heading,
                "sub_section": v.sub_section,
                "required": v.required,
                "adaptation_skipped": v.adaptation_skipped,
                "adaptation_notes": v.adaptation_notes,
            }
            for v in sorted(variables, key=lambda v: v.section_id)
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[ADAPT] Adapted registry saved -> {path}")

    @staticmethod
    def _build_summary(
        extractions: list[ExtractionResult],
        validations: list[ValidationResult],
        elapsed: float,
    ) -> dict:
        total = len(extractions)
        counts: dict[str, int] = {
            "FOUND": 0,
            "PARTIAL": 0,
            "NOT_FOUND": 0,
            "SKIPPED": 0,
            "STANDARD_TEXT": 0,
            "ADAPTATION_SKIPPED": 0,
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
            "adaptation_skipped": counts["ADAPTATION_SKIPPED"],
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
        print(f"  Total sections:      {s['total_sections']}")
        print(f"  FOUND:               {s['found']}")
        print(f"  PARTIAL:             {s['partial']}")
        print(f"  NOT_FOUND:           {s['not_found']}")
        print(f"  SKIPPED:             {s['skipped']}")
        print(f"  STANDARD_TEXT:       {s['standard_text']}")
        print(f"  ADAPTATION_SKIPPED:  {s.get('adaptation_skipped', 0)}")
        print(f"  ERRORS:              {s['errors']}")
        print(f"  Validation issues:   {s['validation_issues']}")
        print(f"  Review flags:        {s.get('review_flags', 'N/A (skipped)')}")
        print(f"  Wall time:           {s['elapsed_seconds']}s")
        print(sep)
        if result.output_docx_path:
            print(f"  Draft ICF:   {result.output_docx_path}")
        if result.clean_icf_path:
            print(f"  Clean ICF:   {result.clean_icf_path}")
        if result.report_path:
            print(f"  Report:      {result.report_path}")
        print(sep)

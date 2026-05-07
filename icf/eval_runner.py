"""
ICF evaluation engine using DeepEval.

Evaluates AI-generated ICF extraction results against:
  1. Ground truth (approved human-written ICF) — correctness score
  2. Rubric dimensions (9 criteria from UHN evaluation outline) — GEval scores

Supports comparing multiple backends side-by-side by loading their
extraction_report_*.json files from the output directory.

Scores are aggregated into three category averages for publication reporting:
  task_performance  — Fidelity, Honesty, Over-inclusion
  effectiveness     — Misleading Language, RBVA, Tone
  accessibility     — Language Quality, Inclusive Language
  gt_alignment      — Ground Truth Correctness (separate, where GT available)

Usage::

    runner = ICFEvalRunner(
        report_paths={"rlm": "output/extraction_report_rlm_Prot.json", ...},
        ground_truth_path="data/ground_truth_icf.docx",
        registry_path="data/standard_ICF_template_breakdown.json",
        protocol_path="data/Prot_000.pdf",  # optional, for fidelity checks
    )
    results = runner.run()
    runner.print_comparison(results)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from icf.eval_ground_truth import parse_ground_truth_docx, print_ground_truth_summary
from icf.eval_rubrics import (
    ALL_RUBRICS,
    DEFAULT_POLICY,
    DOCUMENT_LEVEL_RUBRICS,
    GROUND_TRUTH_REQUIRED,
    GROUND_TRUTH_RUBRICS,
    GROUNDING_RUBRIC_NAMES,
    EvalPolicy,
    RubricDefinition,
    ScoringMode,
    has_placeholders,
    is_rubric_applicable,
    route_section,
)
from icf.registry import load_template_registry
from icf.types import TemplateVariable

# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------


@dataclass
class SectionScore:
    """Score for a single rubric dimension on a single section."""

    rubric_name: str
    score: float  # 0.0 - 1.0
    grade: str  # Excellent / Good / Borderline / Poor / Fail / N/A
    reason: str = ""
    routing_mode: str = ""  # SKIP / HARD_PENALTY / SOFT / FULL
    evidence_relevance: str = ""  # STRONG / PARTIAL / WEAK / IRRELEVANT
    support_level: str = ""  # WITHIN / EXCEEDS / NO_EVIDENCE
    confidence: str = ""  # HIGH / MEDIUM / LOW
    extraction_status: str = ""  # FOUND / PARTIAL / NOT_FOUND / ERROR


@dataclass
class SectionEvalResult:
    """All evaluation scores for a single ICF section."""

    section_id: str
    heading: str
    sub_section: str | None
    status: str  # FOUND, PARTIAL, etc. from extraction
    scores: list[SectionScore] = field(default_factory=list)


@dataclass
class CoverageAnalysis:
    """Coverage comparison between AI output and ground truth."""

    # Sections the AI generated content for but ground truth doesn't have
    ai_only: list[dict] = field(default_factory=list)
    # Sections in the ground truth but missing/skipped in AI output
    gt_only: list[dict] = field(default_factory=list)
    # Sections present in both
    matched: list[str] = field(default_factory=list)


@dataclass
class BackendEvalResult:
    """All evaluation results for a single backend."""

    backend_name: str
    sections: list[SectionEvalResult] = field(default_factory=list)
    coverage: CoverageAnalysis | None = None
    document_scores: list[dict] = field(default_factory=list)  # document-level rubric results

    def avg_score(self, rubric_name: str) -> float | None:
        """Average score for a rubric across all evaluated sections."""
        vals = [
            s.score
            for sec in self.sections
            for s in sec.scores
            if s.rubric_name == rubric_name and s.score >= 0
        ]
        return sum(vals) / len(vals) if vals else None


# ------------------------------------------------------------------
# DeepEval GEval metric builder
# ------------------------------------------------------------------


def _resolve_judge_model(model: str):
    """Resolve the judge model — use Azure OpenAI wrapper if Azure env vars are set.

    If AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are configured, returns
    an AzureOpenAIJudge instance so DeepEval uses Azure without needing
    ``deepeval set-azure-openai``.  Otherwise returns the model name string
    (DeepEval will use the OpenAI API with OPENAI_API_KEY).
    """
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")

    if azure_endpoint and azure_key:
        from icf.eval_model import AzureOpenAIJudge

        return AzureOpenAIJudge(model_name=model)

    # Fall back to plain model string (uses OpenAI API)
    return model


def _build_geval_metric(rubric: RubricDefinition, model):
    """Build a DeepEval GEval metric from a RubricDefinition."""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    param_map = {
        "actual_output": LLMTestCaseParams.ACTUAL_OUTPUT,
        "expected_output": LLMTestCaseParams.EXPECTED_OUTPUT,
        "context": LLMTestCaseParams.CONTEXT,
        "input": LLMTestCaseParams.INPUT,
    }

    eval_params = [param_map[p] for p in rubric.params if p in param_map]

    return GEval(
        name=rubric.name,
        criteria=rubric.criteria,
        evaluation_params=eval_params,
        model=model,
        threshold=0.5,
    )


def _build_geval_metric_without_expected(rubric: RubricDefinition, model):
    """Build a GEval metric variant that drops expected_output from params.

    Used when a rubric optionally accepts ground truth but none is available
    for the current section.
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    param_map = {
        "actual_output": LLMTestCaseParams.ACTUAL_OUTPUT,
        "context": LLMTestCaseParams.CONTEXT,
        "input": LLMTestCaseParams.INPUT,
    }

    eval_params = [param_map[p] for p in rubric.params if p in param_map]

    return GEval(
        name=rubric.name,
        criteria=rubric.criteria,
        evaluation_params=eval_params,
        model=model,
        threshold=0.5,
    )


def _build_test_case(
    section_id: str,
    actual_output: str,
    expected_output: str | None = None,
    context: list[str] | None = None,
    input_text: str = "",
):
    """Build a DeepEval LLMTestCase."""
    from deepeval.test_case import LLMTestCase

    kwargs = {
        "input": input_text or f"Extract ICF section {section_id}",
        "actual_output": actual_output,
    }
    if expected_output is not None:
        kwargs["expected_output"] = expected_output
    if context is not None:
        kwargs["context"] = context

    return LLMTestCase(**kwargs)


def _build_evidence_context(
    evidence: list[dict],
    confidence: str,
    mode: ScoringMode,
) -> list[str] | None:
    """Build a targeted evidence context string for the judge.

    Replaces the old protocol[:50000] dump with precise evidence quotes
    the extractor already found. Returns None if no evidence available.
    """
    if not evidence:
        return None

    lines = [f"Extraction confidence: {confidence}"]
    if mode == ScoringMode.SOFT:
        lines.append("Note: Extraction confidence is MEDIUM — evaluate cautiously.")
    lines.append("\nEvidence quotes retrieved from protocol:")
    for i, ev in enumerate(evidence[:10], 1):  # cap at 10 quotes
        quote = ev.get("quote", "").strip()
        page = ev.get("page", "")
        section = ev.get("section", "")
        loc = f"(Page {page}" + (f", {section}" if section else "") + ")"
        if quote:
            lines.append(f'  {i}. "{quote}" {loc}')

    return ["\n".join(lines)]


def _get_parent_gt(ground_truth: dict[str, str], sid: str) -> str | None:
    """Return GT text for the closest ancestor section when a sub-section has no GT.

    Enables sub-sections like '12.1' to inherit GT from '12' when the ground-truth
    DOCX uses top-level headings only. Tries progressively shorter dotted prefixes.
    """
    parts = sid.split(".")
    while len(parts) > 1:
        parts = parts[:-1]
        parent = ".".join(parts)
        if parent in ground_truth:
            return ground_truth[parent]
    return None


def _score_to_grade(score: float) -> str:
    """Convert a 0-1 score to Excellent/Good/Borderline/Poor/Fail.

    Boundaries match GRADE_SCALE in eval_combined.py so judge calibration
    and code-side grade labels are consistent.
    """
    if score >= 0.9:
        return "Excellent"
    if score >= 0.7:
        return "Good"
    if score >= 0.5:
        return "Borderline"
    if score >= 0.3:
        return "Poor"
    return "Fail"


# ------------------------------------------------------------------
# Main evaluation runner
# ------------------------------------------------------------------


class ICFEvalRunner:
    """Orchestrates evaluation of ICF extraction results.

    Args:
        report_paths:       Dict of backend_name -> extraction_report JSON path.
        ground_truth_path:  Path to the approved human-written ICF DOCX.
        registry_path:      Path to the ICF template registry JSON.
        protocol_path:      Optional path to the protocol (for fidelity context).
        judge_model:        Model name for the LLM judge (default: gpt-4o).
        rubrics:            List of rubrics to evaluate (default: all 10).
        section_filter:     Optional list of section IDs to evaluate (default: all).
        verbose:            Print detailed per-section results.
    """

    def __init__(
        self,
        report_paths: dict[str, str],
        ground_truth_path: str | None = None,
        registry_path: str = "data/standard_ICF_template_breakdown.json",
        protocol_path: str | None = None,
        judge_model: str = "gpt-4o",
        rubrics: list[RubricDefinition] | None = None,
        section_filter: list[str] | None = None,
        verbose: bool = False,
        policy: EvalPolicy | None = None,
    ):
        self.report_paths = report_paths
        self.ground_truth_path = ground_truth_path
        self.registry_path = registry_path
        self.protocol_path = protocol_path
        self.judge_model = judge_model
        self.rubrics = rubrics or ALL_RUBRICS
        self.section_filter = section_filter
        self.verbose = verbose
        self.policy = policy or DEFAULT_POLICY

    def run(self) -> dict[str, BackendEvalResult]:
        """Run evaluation across all backends. Returns {backend_name: BackendEvalResult}."""
        # Load registry
        variables = load_template_registry(self.registry_path)
        var_map: dict[str, TemplateVariable] = {v.section_id: v for v in variables}

        if self.section_filter:
            variables = [v for v in variables if v.section_id in self.section_filter]

        # Load ground truth
        ground_truth: dict[str, str] = {}
        if self.ground_truth_path and os.path.exists(self.ground_truth_path):
            print(f"[EVAL] Loading ground truth from {self.ground_truth_path} ...")
            ground_truth = parse_ground_truth_docx(self.ground_truth_path, variables)
            print_ground_truth_summary(ground_truth, variables)
        else:
            print("[EVAL] No ground truth provided. Skipping correctness comparison.")

        # Load protocol text (for fidelity/honesty context)
        protocol_text: str | None = None
        if self.protocol_path and os.path.exists(self.protocol_path):
            from icf.ingest import load_protocol

            print(f"[EVAL] Loading protocol from {self.protocol_path} ...")
            protocol = load_protocol(self.protocol_path)
            protocol_text = protocol.full_text

        # Build GEval metrics (once, reuse across backends)
        judge = _resolve_judge_model(self.judge_model)
        llm_rubrics = [r for r in self.rubrics if not r.deterministic]
        geval_metrics = {r.name: _build_geval_metric(r, judge) for r in llm_rubrics}

        # Evaluate each backend
        results: dict[str, BackendEvalResult] = {}
        for backend_name, report_path in self.report_paths.items():
            print(f"\n{'=' * 60}")
            print(f"[EVAL] Evaluating backend: {backend_name}")
            print(f"{'=' * 60}")
            result = self._evaluate_backend(
                backend_name=backend_name,
                report_path=report_path,
                var_map=var_map,
                ground_truth=ground_truth,
                protocol_text=protocol_text,
                geval_metrics=geval_metrics,
                judge=judge,
            )
            results[backend_name] = result

        return results

    def run_combined(self) -> dict[str, BackendEvalResult]:
        """Run evaluation using combined mode (1 LLM call per section).

        Same output format as run() but ~90% cheaper. Scores all rubrics
        in a single LLM call instead of one call per rubric.
        """

        # Load registry
        variables = load_template_registry(self.registry_path)
        var_map: dict[str, TemplateVariable] = {v.section_id: v for v in variables}

        if self.section_filter:
            variables = [v for v in variables if v.section_id in self.section_filter]

        # Load ground truth
        ground_truth: dict[str, str] = {}
        if self.ground_truth_path and os.path.exists(self.ground_truth_path):
            print(f"[EVAL] Loading ground truth from {self.ground_truth_path} ...")
            ground_truth = parse_ground_truth_docx(self.ground_truth_path, variables)
            print_ground_truth_summary(ground_truth, variables)
        else:
            print("[EVAL] No ground truth provided. Skipping correctness comparison.")

        # Load protocol text
        protocol_text: str | None = None
        if self.protocol_path and os.path.exists(self.protocol_path):
            from icf.ingest import load_protocol

            print(f"[EVAL] Loading protocol from {self.protocol_path} ...")
            protocol = load_protocol(self.protocol_path)
            protocol_text = protocol.full_text

        # Build Azure OpenAI client for combined calls
        import openai

        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        if azure_endpoint and azure_key:
            client = openai.AzureOpenAI(
                api_key=azure_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
            )
        else:
            client = openai.OpenAI()

        # Build compact study context summary once — used in every section's judge call
        # instead of dumping the full GT ICF, which would repeat ~4k tokens per section.
        from icf.eval_combined import build_study_context_summary

        study_context = ""
        if ground_truth:
            print("[EVAL-COMBINED] Building study context summary (1 LLM call) ...")
            study_context = build_study_context_summary(
                ground_truth=ground_truth,
                variables=variables,
                client=client,
                model_name=self.judge_model,
            )
            if study_context:
                word_count = len(study_context.split())
                print(f"[EVAL-COMBINED] Study context: {word_count} words")

        # Evaluate each backend
        results: dict[str, BackendEvalResult] = {}
        for backend_name, report_path in self.report_paths.items():
            print(f"\n{'=' * 60}")
            print(f"[EVAL-COMBINED] Evaluating backend: {backend_name}")
            print(f"{'=' * 60}")
            result = self._evaluate_backend_combined(
                backend_name=backend_name,
                report_path=report_path,
                var_map=var_map,
                ground_truth=ground_truth,
                protocol_text=protocol_text,
                client=client,
                study_context=study_context,
            )

            # Document-level pass — 1 LLM call on full concatenated output
            if DOCUMENT_LEVEL_RUBRICS:
                from icf.eval_combined import evaluate_document_level

                full_text = self._concatenate_backend_output_from_report(report_path)
                if full_text and len(full_text.split()) >= 50:
                    print(
                        f"\n[EVAL-COMBINED] Document-level quality pass ({len(full_text.split())} words)..."
                    )
                    for doc_rubric in DOCUMENT_LEVEL_RUBRICS:
                        doc_result = evaluate_document_level(
                            full_text=full_text,
                            rubric=doc_rubric,
                            client=client,
                            model_name=self.judge_model,
                            verbose=self.verbose,
                        )
                        result.document_scores.append(
                            {
                                "rubric": doc_rubric.name,
                                **doc_result,
                            }
                        )
                        grade = doc_result.get("grade", "ERROR")
                        score = doc_result.get("score", -1.0)
                        issues = doc_result.get("issues", [])
                        print(f"  {doc_rubric.name}: {grade} ({score:.2f})")
                        if issues and self.verbose:
                            for issue in issues[:5]:
                                print(f"    - {issue}")

            results[backend_name] = result

        return results

    def _evaluate_backend_combined(
        self,
        backend_name: str,
        report_path: str,
        var_map: dict[str, TemplateVariable],
        ground_truth: dict[str, str],
        protocol_text: str | None,
        client,
        study_context: str = "",
    ) -> BackendEvalResult:
        """Evaluate a single backend using combined mode (1 call per section)."""
        from icf.eval_combined import evaluate_section_combined
        from icf.eval_rubrics import GROUND_TRUTH_REQUIRED

        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        extractions = report.get("extractions", [])
        backend_result = BackendEvalResult(backend_name=backend_name)
        total = sum(
            1
            for e in extractions
            if e.get("status") not in ("SKIPPED", "ADAPTATION_SKIPPED", "STANDARD_TEXT")
            and e["section_id"] not in self.policy.skip_sections
            and len((e.get("filled_template") or e.get("answer") or "").split())
            >= self.policy.min_section_words
        )
        count = 0

        for ext in extractions:
            sid = ext["section_id"]
            if self.section_filter and sid not in self.section_filter:
                continue

            var = var_map.get(sid)
            if not var:
                continue

            status = ext.get("status", "")
            if status in ("SKIPPED", "ADAPTATION_SKIPPED", "STANDARD_TEXT"):
                continue

            # For NOT_FOUND sections the filled_template is intentionally empty —
            # the extraction correctly abstained. The `answer` field in these cases
            # holds an internal note/rationale, NOT patient-facing ICF content.
            # Falling back to `answer` would trick the routing logic into thinking
            # the AI hallucinated content (NOT_FOUND + concrete text = HARD_PENALTY),
            # when in reality the section was correctly omitted from the document.
            if status == "NOT_FOUND":
                actual_output = ext.get("filled_template") or ""
            else:
                actual_output = ext.get("filled_template") or ext.get("answer") or ""
            if not actual_output.strip():
                continue

            # Skip admin metadata sections (study title, PI name, site, etc.)
            if sid in self.policy.skip_sections:
                if self.verbose:
                    print(f"  [{sid}] SKIPPED (admin section in policy.skip_sections)")
                continue

            # Skip sections that are too short to evaluate meaningfully
            if len(actual_output.split()) < self.policy.min_section_words:
                if self.verbose:
                    print(
                        f"  [{sid}] SKIPPED ({len(actual_output.split())} words < min {self.policy.min_section_words})"
                    )
                continue

            # -- Grounding bundle from extraction report --
            evidence = ext.get("evidence", [])
            confidence = ext.get("confidence", "LOW")
            notes = ext.get("notes", "")
            has_ev = bool(evidence)

            count += 1
            display = var.get_display_name()
            print(
                f"\n[EVAL-COMBINED] [{count}/{total}] [{sid}] {display} (status: {status} | conf: {confidence})"
            )

            section_result = SectionEvalResult(
                section_id=sid,
                heading=var.heading,
                sub_section=var.sub_section,
                status=status,
            )

            # -- Combined LLM call for all rubrics --
            # Use exact GT text when available; fall back to parent-section GT for sub-sections
            # (e.g. section "12.1" inherits from "12" when the GT DOCX uses top-level headings).
            gt_text = ground_truth.get(sid) or _get_parent_gt(ground_truth, sid)

            # Filter rubrics: check applicability, GT requirements, deterministic, routing
            active_rubrics = []  # rubrics that go to the judge
            active_routing = []  # routing mode per active rubric
            skipped_rubrics = []  # (name, reason) skipped before judge
            hard_penalty_rubrics = []  # (name, reason) hard-penalized in code

            for r in self.rubrics:
                if r.deterministic:
                    continue

                # GT Correctness abstention-aware skip
                if r.name in GROUND_TRUTH_REQUIRED:
                    if not gt_text:
                        continue
                    # is_in_protocol=False + placeholders only → skip GT Correctness
                    if not var.is_in_protocol and not var.partially_in_protocol:
                        if (
                            has_placeholders(actual_output)
                            and self.policy.skip_gt_if_placeholders_only
                        ):
                            skipped_rubrics.append(
                                (
                                    r.name,
                                    "Non-protocol section with placeholders — correct abstention",
                                )
                            )
                            continue

                # Section-scope + min word check
                applicable, skip_reason = is_rubric_applicable(r, sid, actual_output)
                if not applicable:
                    skipped_rubrics.append((r.name, skip_reason))
                    continue

                # Routing decision (Honesty skip is handled by route_section
                # when section is not in protocol — no separate gate needed)
                mode, route_reason = route_section(
                    is_in_protocol=var.is_in_protocol,
                    partially_in_protocol=var.partially_in_protocol,
                    is_standard_text=var.is_standard_text,
                    status=status,
                    confidence=confidence,
                    has_evidence=has_ev,
                    text=actual_output,
                    rubric_name=r.name,
                    policy=self.policy,
                )

                if mode == ScoringMode.SKIP:
                    skipped_rubrics.append((r.name, route_reason))
                    continue
                elif mode == ScoringMode.HARD_PENALTY:
                    if r.name in GROUNDING_RUBRIC_NAMES:
                        # Grounding rubrics get the hard penalty — they measure factual
                        # accuracy relative to the protocol, which cannot be assessed
                        # when grounding is unconfirmed.
                        hard_penalty_rubrics.append((r, route_reason))
                    else:
                        # Content quality rubrics (tone, language, inclusion) are evaluated
                        # on the text itself regardless of grounding uncertainty. Route as
                        # SOFT so the judge is cautioned but still scores.
                        active_rubrics.append(r)
                        active_routing.append(ScoringMode.SOFT)
                    continue

                active_rubrics.append(r)
                active_routing.append(mode)

            # Record skipped rubrics as N/A — never silently drop
            for rname, skip_reason in skipped_rubrics:
                section_result.scores.append(
                    SectionScore(
                        rubric_name=rname,
                        score=-1.0,
                        grade="N/A",
                        reason=skip_reason,
                        routing_mode=ScoringMode.SKIP,
                        confidence=confidence,
                        extraction_status=status,
                    )
                )
                if self.verbose:
                    print(f"  {rname}: N/A - {skip_reason}")

            # Hard penalties — record without judge call
            for r, hp_reason in hard_penalty_rubrics:
                section_result.scores.append(
                    SectionScore(
                        rubric_name=r.name,
                        score=self.policy.hard_penalty_score,
                        grade="Poor",
                        reason=hp_reason,
                        routing_mode=ScoringMode.HARD_PENALTY,
                        confidence=confidence,
                        extraction_status=status,
                    )
                )
                if self.verbose:
                    print(f"  {r.name}: HARD_PENALTY ({hp_reason})")

            # Judge call for active rubrics
            if active_rubrics:
                scores = evaluate_section_combined(
                    section_id=sid,
                    section_heading=display,
                    actual_output=actual_output,
                    ground_truth=gt_text,
                    evidence=evidence,
                    confidence=confidence,
                    notes=notes,
                    routing_modes={
                        r.name: m for r, m in zip(active_rubrics, active_routing, strict=True)
                    },
                    rubrics=active_rubrics,
                    client=client,
                    model_name=self.judge_model,
                    verbose=self.verbose,
                    instructions=getattr(var, "instructions", "") or "",
                    required_text=getattr(var, "required_text", "") or "",
                    suggested_text=getattr(var, "suggested_text", "") or "",
                    study_context=study_context,
                )

                for r, mode in zip(active_rubrics, active_routing, strict=True):
                    result_data = scores.get(
                        r.name,
                        {
                            "score": -1.0,
                            "grade": "ERROR",
                            "reason": "Not returned by judge",
                            "evidence_relevance": "",
                            "support_level": "",
                        },
                    )
                    final_score = result_data["score"]
                    final_grade = result_data["grade"]
                    final_reason = result_data["reason"]
                    ev_relevance = result_data.get("evidence_relevance", "")
                    sup_level = result_data.get("support_level", "")

                    # Apply min_relevance_for_full_score policy post-scoring
                    # For Fidelity/Honesty: if evidence relevance is below threshold,
                    # cap score at 0.5 (Borderline) regardless of judge score
                    if ev_relevance and r.name in ("Fidelity to Protocol", "Honesty"):
                        relevance_rank = {"STRONG": 3, "PARTIAL": 2, "WEAK": 1, "IRRELEVANT": 0}
                        min_rank = relevance_rank.get(self.policy.min_relevance_for_full_score, 2)
                        actual_rank = relevance_rank.get(ev_relevance, 2)
                        if actual_rank < min_rank and final_score > 0.5:
                            final_score = 0.5
                            final_grade = "Borderline"
                            final_reason = (
                                f"[Score capped: evidence relevance={ev_relevance} below "
                                f"policy minimum={self.policy.min_relevance_for_full_score}] "
                                + final_reason
                            )

                    section_result.scores.append(
                        SectionScore(
                            rubric_name=r.name,
                            score=final_score,
                            grade=final_grade,
                            reason=final_reason,
                            routing_mode=mode,
                            evidence_relevance=ev_relevance,
                            support_level=sup_level,
                            confidence=confidence,
                            extraction_status=status,
                        )
                    )
                    if self.verbose:
                        print(
                            f"  {r.name}: {result_data['grade']} "
                            f"({result_data['score']:.2f}) "
                            f"[relevance={result_data.get('evidence_relevance', '')} "
                            f"support={result_data.get('support_level', '')}] "
                            f"- {result_data['reason'][:100]}"
                        )

            backend_result.sections.append(section_result)

        # Flag scoped rubrics whose sections were not generated
        self._flag_missing_scoped_sections(extractions, backend_name)

        # Coverage analysis
        if ground_truth:
            backend_result.coverage = self._compute_coverage(extractions, ground_truth, var_map)

        return backend_result

    def _evaluate_backend(
        self,
        backend_name: str,
        report_path: str,
        var_map: dict[str, TemplateVariable],
        ground_truth: dict[str, str],
        protocol_text: str | None,
        geval_metrics: dict,
        judge=None,
    ) -> BackendEvalResult:
        """Evaluate a single backend's extraction report."""
        # Load extraction report
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        extractions = report.get("extractions", [])
        backend_result = BackendEvalResult(backend_name=backend_name)

        for ext in extractions:
            sid = ext["section_id"]
            if self.section_filter and sid not in self.section_filter:
                continue

            var = var_map.get(sid)
            if not var:
                continue

            status = ext.get("status", "")
            if status in ("SKIPPED", "ADAPTATION_SKIPPED", "STANDARD_TEXT"):
                continue

            # For NOT_FOUND sections the filled_template is intentionally empty —
            # the extraction correctly abstained. Never fall back to `answer` here;
            # the `answer` field is an internal note, not evaluatable ICF content.
            if status == "NOT_FOUND":
                actual_output = ext.get("filled_template") or ""
            else:
                actual_output = ext.get("filled_template") or ext.get("answer") or ""
            if not actual_output.strip():
                continue

            # Skip admin metadata sections
            if sid in self.policy.skip_sections:
                if self.verbose:
                    print(f"  [{sid}] SKIPPED (admin section in policy.skip_sections)")
                continue

            # Skip sections that are too short to evaluate meaningfully
            if len(actual_output.split()) < self.policy.min_section_words:
                if self.verbose:
                    print(
                        f"  [{sid}] SKIPPED ({len(actual_output.split())} words < min {self.policy.min_section_words})"
                    )
                continue

            # -- Grounding bundle --
            evidence = ext.get("evidence", [])
            confidence = ext.get("confidence", "LOW")
            has_ev = bool(evidence)

            display = var.get_display_name()
            print(f"\n[EVAL] [{sid}] {display} (status: {status} | conf: {confidence})")

            section_result = SectionEvalResult(
                section_id=sid,
                heading=var.heading,
                sub_section=var.sub_section,
                status=status,
            )

            # -- GEval rubrics (LLM judge) --
            for rubric in self.rubrics:
                if rubric.deterministic:
                    continue

                metric = geval_metrics.get(rubric.name)
                if not metric:
                    continue

                # Section-scope + min word check
                applicable, skip_reason = is_rubric_applicable(rubric, sid, actual_output)
                if not applicable:
                    section_result.scores.append(
                        SectionScore(
                            rubric_name=rubric.name,
                            score=-1.0,
                            grade="N/A",
                            reason=skip_reason,
                            routing_mode=ScoringMode.SKIP,
                            confidence=confidence,
                            extraction_status=status,
                        )
                    )
                    if self.verbose:
                        print(f"  {rubric.name}: N/A - {skip_reason}")
                    continue

                # Honesty skip is handled by route_section when section
                # is not in protocol — no separate gate needed

                # GT Correctness abstention-aware skip.
                # Sub-sections inherit parent GT when direct match is absent.
                gt_text: str | None = None
                if rubric.name in GROUND_TRUTH_RUBRICS:
                    gt_text = ground_truth.get(sid) or _get_parent_gt(ground_truth, sid)
                if rubric.name in GROUND_TRUTH_REQUIRED:
                    if not gt_text:
                        section_result.scores.append(
                            SectionScore(
                                rubric_name=rubric.name,
                                score=-1.0,
                                grade="N/A",
                                reason="No ground truth available for this section",
                                routing_mode=ScoringMode.SKIP,
                                confidence=confidence,
                                extraction_status=status,
                            )
                        )
                        continue
                    if not var.is_in_protocol and not var.partially_in_protocol:
                        if (
                            has_placeholders(actual_output)
                            and self.policy.skip_gt_if_placeholders_only
                        ):
                            na_reason = (
                                "Non-protocol section with placeholders — correct abstention"
                            )
                            section_result.scores.append(
                                SectionScore(
                                    rubric_name=rubric.name,
                                    score=-1.0,
                                    grade="N/A",
                                    reason=na_reason,
                                    routing_mode=ScoringMode.SKIP,
                                    confidence=confidence,
                                    extraction_status=status,
                                )
                            )
                            if self.verbose:
                                print(f"  {rubric.name}: N/A - {na_reason}")
                            continue

                # Routing decision
                mode, route_reason = route_section(
                    is_in_protocol=var.is_in_protocol,
                    partially_in_protocol=var.partially_in_protocol,
                    is_standard_text=var.is_standard_text,
                    status=status,
                    confidence=confidence,
                    has_evidence=has_ev,
                    text=actual_output,
                    rubric_name=rubric.name,
                    policy=self.policy,
                )

                if mode == ScoringMode.SKIP:
                    section_result.scores.append(
                        SectionScore(
                            rubric_name=rubric.name,
                            score=-1.0,
                            grade="N/A",
                            reason=route_reason,
                            routing_mode=ScoringMode.SKIP,
                            confidence=confidence,
                            extraction_status=status,
                        )
                    )
                    if self.verbose:
                        print(f"  {rubric.name}: N/A - {route_reason}")
                    continue

                if mode == ScoringMode.HARD_PENALTY:
                    if rubric.name in GROUNDING_RUBRIC_NAMES:
                        section_result.scores.append(
                            SectionScore(
                                rubric_name=rubric.name,
                                score=self.policy.hard_penalty_score,
                                grade="Poor",
                                reason=route_reason,
                                routing_mode=ScoringMode.HARD_PENALTY,
                                confidence=confidence,
                                extraction_status=status,
                            )
                        )
                        if self.verbose:
                            print(f"  {rubric.name}: HARD_PENALTY ({route_reason})")
                        continue
                    else:
                        # Content quality rubrics still evaluated on text itself
                        mode = ScoringMode.SOFT

                # Build evidence context for the judge instead of protocol dump
                evidence_context = _build_evidence_context(evidence, confidence, mode)

                # Use GT as expected_output if available
                use_metric = metric
                if (
                    rubric.name in GROUND_TRUTH_RUBRICS
                    and not gt_text
                    and rubric.name not in GROUND_TRUTH_REQUIRED
                ):
                    use_metric = _build_geval_metric_without_expected(rubric, judge)

                test_case = _build_test_case(
                    section_id=sid,
                    actual_output=actual_output,
                    expected_output=gt_text,
                    context=evidence_context,
                    input_text=var.instructions,
                )

                try:
                    use_metric.measure(test_case)
                    score = use_metric.score
                    reason = use_metric.reason or ""
                    grade = _score_to_grade(score)
                except Exception as e:
                    score = -1.0
                    grade = "ERROR"
                    reason = f"{type(e).__name__}: {e}"

                # Note: evidence_relevance and support_level are only available in
                # combined mode. GEval (detailed mode) returns only score + reason.
                # Use combined mode for structured grounding analysis.
                section_result.scores.append(
                    SectionScore(
                        rubric_name=rubric.name,
                        score=score,
                        grade=grade,
                        reason=reason,
                        routing_mode=mode,
                        confidence=confidence,
                        extraction_status=status,
                    )
                )

                if self.verbose:
                    print(f"  {rubric.name}: {grade} ({score:.2f}) - {reason[:100]}")

            backend_result.sections.append(section_result)

        # Flag scoped rubrics whose sections were not generated
        self._flag_missing_scoped_sections(extractions, backend_name)

        # -- Coverage analysis: AI output vs ground truth --
        if ground_truth:
            backend_result.coverage = self._compute_coverage(extractions, ground_truth, var_map)

        return backend_result

    # ------------------------------------------------------------------
    # Scoped rubric flagging
    # ------------------------------------------------------------------

    def _flag_missing_scoped_sections(self, extractions: list[dict], backend_name: str) -> None:
        """Print warnings when a scoped rubric's target sections are missing."""
        # Build set of section IDs that were actually generated
        generated = set()
        for ext in extractions:
            status = ext.get("status", "")
            if status not in ("SKIPPED", "ADAPTATION_SKIPPED", "STANDARD_TEXT"):
                text = ext.get("filled_template") or ext.get("answer") or ""
                if text.strip():
                    generated.add(ext["section_id"])

        for rubric in self.rubrics:
            if not rubric.applicable_sections:
                continue
            missing = [s for s in rubric.applicable_sections if s not in generated]
            if missing:
                print(
                    f"  [FLAG] {rubric.name}: applicable sections not generated: "
                    f"{', '.join(missing)}"
                )

    # ------------------------------------------------------------------
    # Document-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _concatenate_backend_output_from_report(report_path: str) -> str:
        """Concatenate all section outputs from an extraction report for document-level eval."""
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        parts = []
        for ext in report.get("extractions", []):
            status = ext.get("status", "")
            if status in ("SKIPPED", "ADAPTATION_SKIPPED"):
                continue
            text = ext.get("filled_template") or ext.get("answer") or ""
            if not text.strip():
                continue
            sid = ext["section_id"]
            heading = ext.get("heading", "")
            parts.append(f"=== [{sid}] {heading} ===\n{text.strip()}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Coverage analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_coverage(
        extractions: list[dict],
        ground_truth: dict[str, str],
        var_map: dict[str, TemplateVariable],
    ) -> CoverageAnalysis:
        """Compare which sections appear in AI output vs ground truth.

        Sub-section awareness: if the AI generated sub-sections (e.g. 12.1, 12.2)
        but the GT only has the parent (12), those sub-sections are classified as
        "sub_of_matched" rather than "ai_only" to avoid inflating the over-inclusion
        count for legitimate granularity differences.
        """
        # AI sections that have actual content (FOUND or PARTIAL)
        ai_generated: dict[str, str] = {}
        for ext in extractions:
            sid = ext["section_id"]
            status = ext.get("status", "")
            if status in ("FOUND", "PARTIAL"):
                ai_generated[sid] = status

        gt_ids = set(ground_truth.keys())
        ai_ids = set(ai_generated.keys())

        coverage = CoverageAnalysis()

        # Direct matches
        coverage.matched = sorted(ai_ids & gt_ids)

        # AI generated but not a direct GT match
        for sid in sorted(ai_ids - gt_ids):
            var = var_map.get(sid)
            display = var.get_display_name() if var else sid
            # Check if this is a sub-section whose parent is in GT
            parent = _get_parent_gt(ground_truth, sid)
            if parent is not None:
                # Sub-section expansion of a matched parent — not a true over-inclusion
                coverage.ai_only.append(
                    {
                        "section_id": sid,
                        "heading": display,
                        "status": ai_generated[sid],
                        "flag": f"AI sub-section of matched GT section (parent matched: {sid.rsplit('.', 1)[0]})",
                    }
                )
            else:
                coverage.ai_only.append(
                    {
                        "section_id": sid,
                        "heading": display,
                        "status": ai_generated[sid],
                        "flag": "AI generated content for a section not in the approved ICF",
                    }
                )

        # In ground truth but AI didn't generate — potential gap
        for sid in sorted(gt_ids - ai_ids):
            var = var_map.get(sid)
            display = var.get_display_name() if var else sid
            ext_match = next((e for e in extractions if e["section_id"] == sid), None)
            ai_status = ext_match.get("status", "NOT_IN_REPORT") if ext_match else "NOT_IN_REPORT"
            coverage.gt_only.append(
                {
                    "section_id": sid,
                    "heading": display,
                    "ai_status": ai_status,
                    "flag": "Approved ICF has this section but AI did not generate it",
                }
            )

        return coverage

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_comparison(self, results: dict[str, BackendEvalResult]) -> None:
        """Print a side-by-side comparison table of all backends."""
        backends = list(results.keys())
        col_w = 18
        sep = "=" * (38 + col_w * len(backends))

        print(f"\n{sep}")
        print("ICF EVALUATION RESULTS")
        print(sep)

        # Header row
        header = f"{'Rubric':<38s}"
        for b in backends:
            header += f"  {b:>{col_w - 2}s}"
        print(header)
        print("-" * len(header))

        # Print rubrics grouped by category
        category_labels = {
            "ground_truth": "GT ALIGNMENT",
            "task_performance": "TASK PERFORMANCE",
            "effectiveness": "EFFECTIVENESS",
            "accessibility": "ACCESSIBILITY",
        }
        categories = ["ground_truth", "task_performance", "effectiveness", "accessibility"]

        for cat in categories:
            cat_rubrics = [r for r in self.rubrics if r.category == cat]
            if not cat_rubrics:
                continue
            print(f"\n  [{category_labels[cat]}]")
            for r in cat_rubrics:
                row = f"  {r.name:<36s}"
                for b in backends:
                    avg = results[b].avg_score(r.name)
                    if avg is not None:
                        grade = _score_to_grade(avg)
                        row += f"  {avg:>5.2f} {grade:>9s}"
                    else:
                        row += f"  {'N/A':>{col_w - 2}s}"
                print(row)

            # Category average
            row = f"  {'  → Category avg':<36s}"
            for b in backends:
                cat_scores = [
                    results[b].avg_score(r.name)
                    for r in cat_rubrics
                    if results[b].avg_score(r.name) is not None
                ]
                if cat_scores:
                    avg = sum(cat_scores) / len(cat_scores)
                    grade = _score_to_grade(avg)
                    row += f"  {avg:>5.2f} {grade:>9s}"
                else:
                    row += f"  {'N/A':>{col_w - 2}s}"
            print(row)

        print(f"\n{sep}")

        # Coverage analysis per backend
        for b in backends:
            cov = results[b].coverage
            if not cov:
                continue

            print(f"\n{'=' * 60}")
            print(f"COVERAGE ANALYSIS: {b}")
            print(f"{'=' * 60}")
            print(f"  Sections matched (AI + ground truth): {len(cov.matched)}")

            true_ai_only = [i for i in cov.ai_only if not i["flag"].startswith("AI sub-section")]
            sub_expanded = [i for i in cov.ai_only if i["flag"].startswith("AI sub-section")]

            if sub_expanded:
                print(
                    f"\n  AI sub-section expansions of matched GT sections ({len(sub_expanded)}):"
                )
                print("  (AI used finer-grained sections than GT — not an over-inclusion flag)")
                for item in sub_expanded:
                    print(
                        f"    [{item['section_id']}] {item['heading']} (status: {item['status']})"
                    )

            if true_ai_only:
                print(f"\n  AI generated but NOT in approved ICF ({len(true_ai_only)}):")
                print("  (Potential over-inclusion — REB experts did not include these)")
                for item in true_ai_only:
                    print(
                        f"    [{item['section_id']}] {item['heading']} (status: {item['status']})"
                    )

            if cov.gt_only:
                print(f"\n  In approved ICF but MISSING from AI output ({len(cov.gt_only)}):")
                print("  (Potential gaps — REB experts included these)")
                for item in cov.gt_only:
                    print(
                        f"    [{item['section_id']}] {item['heading']} (AI status: {item['ai_status']})"
                    )

            if not cov.ai_only and not cov.gt_only:
                print("  Perfect coverage — AI output matches ground truth sections exactly.")

            print(f"{'=' * 60}")

    def save_report(
        self,
        results: dict[str, BackendEvalResult],
        output_path: str,
    ) -> None:
        """Save the full evaluation results as a JSON report."""
        report = {}
        for backend_name, backend_result in results.items():
            sections = []
            for sec in backend_result.sections:
                scores = []
                for s in sec.scores:
                    scores.append(
                        {
                            "rubric": s.rubric_name,
                            "score": s.score,
                            "grade": s.grade,
                            "reason": s.reason,
                            "routing_mode": s.routing_mode,
                            "evidence_relevance": s.evidence_relevance,
                            "support_level": s.support_level,
                            "confidence": s.confidence,
                            "extraction_status": s.extraction_status,
                        }
                    )
                sections.append(
                    {
                        "section_id": sec.section_id,
                        "heading": sec.heading,
                        "sub_section": sec.sub_section,
                        "status": sec.status,
                        "scores": scores,
                    }
                )

            # Per-rubric averages
            averages = {}
            for rubric in self.rubrics:
                avg = backend_result.avg_score(rubric.name)
                if avg is not None:
                    averages[rubric.name] = {
                        "score": round(avg, 3),
                        "grade": _score_to_grade(avg),
                    }

            # Category averages for publication reporting.
            # Three primary categories + GT alignment reported separately.
            category_averages: dict[str, dict] = {}
            for cat in ("task_performance", "effectiveness", "accessibility"):
                cat_names = [r.name for r in self.rubrics if r.category == cat]
                cat_scores = [
                    backend_result.avg_score(n)
                    for n in cat_names
                    if backend_result.avg_score(n) is not None
                ]
                if cat_scores:
                    cat_avg = sum(cat_scores) / len(cat_scores)
                    category_averages[cat] = {
                        "score": round(cat_avg, 3),
                        "grade": _score_to_grade(cat_avg),
                        "rubrics": cat_names,
                    }
            # GT alignment — separate because it only applies where GT is available
            gt_names = [r.name for r in self.rubrics if r.category == "ground_truth"]
            gt_scores = [
                backend_result.avg_score(n)
                for n in gt_names
                if backend_result.avg_score(n) is not None
            ]
            if gt_scores:
                category_averages["gt_alignment"] = {
                    "score": round(sum(gt_scores) / len(gt_scores), 3),
                    "grade": _score_to_grade(sum(gt_scores) / len(gt_scores)),
                    "rubrics": gt_names,
                }

            # Coverage analysis
            coverage_data = None
            cov = backend_result.coverage
            if cov:
                # Split ai_only into genuine over-inclusions vs sub-section expansions
                true_ai_only = [
                    i for i in cov.ai_only if not i["flag"].startswith("AI sub-section")
                ]
                sub_expanded = [i for i in cov.ai_only if i["flag"].startswith("AI sub-section")]
                coverage_data = {
                    "matched_sections": cov.matched,
                    "matched_count": len(cov.matched),
                    "ai_only": true_ai_only,
                    "ai_only_count": len(true_ai_only),
                    "ai_subsections_of_matched": sub_expanded,
                    "ai_subsections_count": len(sub_expanded),
                    "gt_only": cov.gt_only,
                    "gt_only_count": len(cov.gt_only),
                }

            report[backend_name] = {
                "sections": sections,
                "averages": averages,
                "category_averages": category_averages,
                "coverage": coverage_data,
                "document_level": backend_result.document_scores
                if backend_result.document_scores
                else None,
            }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[EVAL] Report saved -> {output_path}")

"""
Hybrid extraction engine.

Splits each section's extraction into three focused stages instead of one
RLM run that has to research, draft, apply plain-language rules, and format
JSON all in a single overloaded prompt (that is what icf/extract.py does):

  Stage A -- Evidence gathering (RLM, narrow prompt, smaller iteration
             budget). Its only output is a small, verified evidence bundle:
             quotes, page numbers, a findings summary, and conditional-branch
             resolution. See icf/hybrid_prompts.py::build_evidence_gathering_prompt.
  Stage B -- Drafting (single non-agentic LLM call). Turns the evidence
             bundle into the final structured ICF output -- template symbol
             resolution, verbatim required/suggested text, plain-language
             guidelines, JSON schema. See icf/hybrid_prompts.py::build_draft_messages.
  Stage C -- Deterministic quality gate + one bounded repair retry of Stage B
             only (never re-runs Stage A) -- reuses the same regex/meta-
             commentary checks as the RLM backend (icf/validate.py).

Retries are split by stage so failures stay cheap: Stage A failures (RLM
errors) retry the whole research step; Stage B failures (JSON parse errors on
a single call) just retry that single call with the same evidence -- no need
to re-run the RLM. Stage C fires only when the parsed draft has a concrete,
fixable issue.

Interface
---------
HybridExtractionEngine implements the same extract_variable(protocol_text,
variable) -> ExtractionResult signature as ExtractionEngine, NaiveExtractionEngine,
and RAGExtractionEngine, so the pipeline -- and every downstream stage
(harmonize, validate, review, remediate, assemble) -- works unmodified.
"""

import os

from icf.debug_logger import ICFDebugLogger
from icf.extract import parse_extraction_json
from icf.hybrid_prompts import build_draft_messages, build_evidence_gathering_prompt
from icf.types import Evidence, ExtractionResult, TemplateVariable
from icf.validate import collect_quality_issues, is_garbage_result, quality_score
from rlm import RLM
from rlm.clients import get_client
from rlm.utils.prompts import RLM_SYSTEM_PROMPT


def build_evidence_system_prompt(protocol_length: int) -> str:
    """System prompt for the Stage A evidence-gathering RLM.

    Same execution-model corrections as icf/extract.py's system prompt addendum
    (models sometimes believe the REPL needs manual activation), adapted to the
    evidence_dict/evidence_json schema instead of result_dict/result_json since
    Stage A never produces filled_template.
    """
    addendum = (
        "\n\n=== ICF RESEARCH OPERATIONAL RULES (MANDATORY) ===\n"
        f"context_0 is LOADED with {protocol_length:,} characters of the clinical study "
        "protocol text. It is available RIGHT NOW — you do not need to load it, ask for it, "
        "or wait for any signal.\n\n"
        "EXECUTION MODEL:\n"
        "  • YOU write ```repl code blocks in YOUR response.\n"
        "  • The automation framework AUTOMATICALLY executes every ```repl block you write.\n"
        "  • You see the output in the NEXT message. There is NO human in the loop.\n"
        "  • Every 'Code executed:' block already in this conversation is a REAL Python\n"
        "    execution result — not simulated, not from a 'different environment'.\n\n"
        "PROHIBITED RESPONSES — never write any of the following:\n"
        "  ✗ 'The REPL is not active / not available in this interface'\n"
        "  ✗ 'Please send a ```repl block' / 'say continue' / 'BEGIN REPL'\n"
        "  ✗ 'I cannot continue' / 'I'm sorry, I cannot assist with that'\n"
        "  ✗ Any prose-only response when you still have iterations remaining\n"
        "  If you feel an urge to write any of the above, write a ```repl block instead.\n\n"
        "FINISHING — follow this two-step pattern exactly:\n"
        "  STEP A — Verify (run this block, NO FINAL_VAR inside it):\n"
        "  ```repl\n"
        "  import json\n"
        "  issues = []\n"
        "  _ev = evidence_dict.get('evidence', []) or []\n"
        "  for e in _ev:\n"
        "      q = e.get('quote', '')\n"
        "      if q and q not in context_0:\n"
        "          issues.append('Quote not verbatim in context_0: ' + q[:80])\n"
        "  if evidence_dict.get('status') in ('FOUND', 'PARTIAL') and not _ev:\n"
        "      issues.append('status is FOUND/PARTIAL but evidence is empty')\n"
        "  if issues:\n"
        "      for iss in issues: print('FIX: ' + iss)\n"
        "  else:\n"
        "      evidence_json = json.dumps(evidence_dict)\n"
        "      print('READY_TO_FINALIZE')\n"
        "  ```\n\n"
        "  STEP B — Finalize (only write this after you see READY_TO_FINALIZE in the output):\n"
        "  ```repl\n"
        "  FINAL_VAR(evidence_json)\n"
        "  ```\n\n"
        "  CRITICAL RULES:\n"
        "    ✗  Never write FINAL_VAR inside an if/else or conditional block\n"
        "    ✗  Never write FINAL_VAR in the same block as the verification check\n"
        "    ✗  Never write FINAL_VAR(json.dumps(evidence_dict)) — evidence_json must already exist\n"
        "    ✓  evidence_json is assigned in Step A's else branch — do NOT redefine it in Step B\n\n"
        "  RECOVERY RULE: If evidence_dict was already built in a prior iteration, skip straight\n"
        "  to Step A. If evidence_json was already assigned (Step A ran with no issues), write\n"
        "  Step B immediately.\n\n"
        "  FIRST-RESPONSE RULE: Your very first response MUST contain a ```repl block.\n"
    )
    return RLM_SYSTEM_PROMPT + addendum


def stage_a_iterations_for(variable: TemplateVariable, max_iterations: int) -> int:
    """Iteration budget for the Stage A research RLM.

    Deliberately smaller than the full RLM backend's budget (icf/extract.py):
    Stage A's job is narrow -- find and verify evidence, nothing else -- so it
    needs fewer turns to finish than a pass that also has to draft and format.
    """
    label = variable.get_complexity_label()
    budget_map = {
        "Easy": 5,
        "Moderate": 8,
        "Complex": max(10, max_iterations // 2),
        "Not in protocol": 4,
    }
    budget = budget_map.get(label, 6)
    return min(budget, max_iterations)


class HybridExtractionEngine:
    """Two-stage extraction: RLM evidence gathering -> single-call drafting.

    Implements the same extract_variable() interface as ExtractionEngine so
    the pipeline can swap backends without any other changes.
    """

    def __init__(
        self,
        model_name: str = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1"),
        backend: str = "openai",
        backend_kwargs: dict | None = None,
        max_iterations: int = 20,
        verbose: bool = False,
        debug_logger: ICFDebugLogger | None = None,
        max_research_retries: int = 3,
        max_draft_retries: int = 2,
    ):
        self.model_name = model_name
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.debug_logger = debug_logger
        self.max_research_retries = max_research_retries
        self.max_draft_retries = max_draft_retries

        # One client for Stage B (drafting) + Stage C (repair), reused across sections.
        kwargs = dict(self.backend_kwargs)
        kwargs["model_name"] = self.model_name
        self.draft_client = get_client(self.backend, kwargs)

    # ------------------------------------------------------------------
    # Public interface — matches ExtractionEngine.extract_variable()
    # ------------------------------------------------------------------

    def extract_variable(
        self,
        protocol_text: str,
        variable: TemplateVariable,
    ) -> ExtractionResult:
        """Extract a single variable via Stage A (research) -> Stage B (draft) -> Stage C (repair).

        Routing:
          is_standard_text   -> STANDARD_TEXT (no LLM call)
          not in protocol    -> SKIPPED (no LLM call)
          otherwise          -> Stage A, then Stage B, then a conditional Stage C repair
        """
        if variable.is_standard_text:
            return self.make_standard_result(variable)

        if not variable.is_in_protocol and not variable.partially_in_protocol:
            return self.make_skipped_result(variable)

        if self.debug_logger:
            self.debug_logger.set_section(
                variable.section_id, variable.heading, variable.sub_section or ""
            )

        # -- Stage A: research (retried on RLM error only) --
        evidence_bundle: dict | None = None
        research_error: ExtractionResult | None = None
        for attempt in range(1, self.max_research_retries + 1):
            evidence_bundle, research_error = self.run_stage_a(protocol_text, variable)
            if research_error is None:
                break
            if attempt < self.max_research_retries:
                print(
                    f"[HYBRID] Section {variable.section_id}: Stage A (research) attempt "
                    f"{attempt}/{self.max_research_retries} failed ({research_error.error}). "
                    "Retrying with a fresh research pass ..."
                )
            else:
                print(
                    f"[HYBRID] Section {variable.section_id}: Stage A (research) failed all "
                    f"{self.max_research_retries} attempts. Last error: {research_error.error}"
                )
        if evidence_bundle is None:
            assert research_error is not None
            return research_error

        # -- Stage B: draft (retried on parse error only — same evidence, no re-research) --
        result = self.run_stage_b(variable, evidence_bundle)
        for attempt in range(2, self.max_draft_retries + 1):
            if result.status != "ERROR" and not is_garbage_result(result):
                break
            print(
                f"[HYBRID] Section {variable.section_id}: Stage B (draft) attempt "
                f"{attempt - 1}/{self.max_draft_retries} produced non-JSON/garbage output. "
                "Retrying the draft call ..."
            )
            result = self.run_stage_b(variable, evidence_bundle)
        if result.status == "ERROR" or is_garbage_result(result):
            print(
                f"[HYBRID] Section {variable.section_id}: Stage B (draft) failed all "
                f"{self.max_draft_retries} attempts."
            )
            return result

        # -- Stage C: deterministic gate + one bounded repair of Stage B only --
        issues = collect_quality_issues(result)
        if issues:
            print(
                f"[HYBRID] Section {variable.section_id}: {len(issues)} quality issue(s) found "
                "— repairing draft ..."
            )
            for iss in issues[:4]:
                print(f"[HYBRID]   - {iss}")
            result = self.run_stage_c_repair(variable, evidence_bundle, result, issues)

        return result

    # ------------------------------------------------------------------
    # Stage A — evidence gathering (RLM)
    # ------------------------------------------------------------------

    def run_stage_a(
        self,
        protocol_text: str,
        variable: TemplateVariable,
    ) -> tuple[dict | None, ExtractionResult | None]:
        """Run the Stage A research RLM. Returns (evidence_bundle, error_result)."""
        max_iter = stage_a_iterations_for(variable, self.max_iterations)
        root_prompt = build_evidence_gathering_prompt(variable, protocol_length=len(protocol_text))

        kwargs = {"model_name": self.model_name}
        kwargs.update(self.backend_kwargs)

        try:
            rlm = RLM(
                backend=self.backend,
                backend_kwargs=kwargs,
                environment="local",
                verbose=self.verbose,
                max_iterations=max_iter,
                custom_system_prompt=build_evidence_system_prompt(len(protocol_text)),
                logger=self.debug_logger,
            )
            completion = rlm.completion(prompt=protocol_text, root_prompt=root_prompt)
        except Exception as e:
            return None, self.error_result(variable, f"{type(e).__name__}: {e}", stage="research")

        evidence_bundle = parse_extraction_json(completion.response)
        if evidence_bundle is None:
            return None, self.error_result(
                variable,
                "Failed to parse JSON from Stage A (research) response.",
                stage="research",
                raw=completion.response,
            )

        return evidence_bundle, None

    # ------------------------------------------------------------------
    # Stage B — drafting (single LLM call)
    # ------------------------------------------------------------------

    def run_stage_b(self, variable: TemplateVariable, evidence_bundle: dict) -> ExtractionResult:
        """Run the Stage B drafting call against the Stage A evidence bundle."""
        messages = build_draft_messages(variable, evidence_bundle)

        if self.verbose:
            print(f"[HYBRID] [{variable.section_id}] Stage B: calling drafting LLM ...")

        try:
            raw = self.draft_client.completion(messages)
        except Exception as e:
            return self.error_result(variable, f"{type(e).__name__}: {e}", stage="draft")

        if self.verbose:
            print(f"[HYBRID] [{variable.section_id}] Stage B raw response ({len(raw)} chars).")

        return self.parse_draft_response(raw, variable)

    # ------------------------------------------------------------------
    # Stage C — deterministic gate + bounded repair
    # ------------------------------------------------------------------

    def run_stage_c_repair(
        self,
        variable: TemplateVariable,
        evidence_bundle: dict,
        first_result: ExtractionResult,
        issues: list[str],
    ) -> ExtractionResult:
        """One bounded repair retry of Stage B only — never re-runs Stage A."""
        repair_feedback = "\n".join(f"- {iss}" for iss in issues)
        messages = build_draft_messages(variable, evidence_bundle, repair_feedback=repair_feedback)

        try:
            raw = self.draft_client.completion(messages)
        except Exception as e:
            print(
                f"[HYBRID] Section {variable.section_id}: repair call raised "
                f"{type(e).__name__} — keeping original draft."
            )
            return first_result

        repaired = self.parse_draft_response(raw, variable)

        if repaired.status == "ERROR":
            print(
                f"[HYBRID] Section {variable.section_id}: repair pass errored "
                f"({repaired.error}) — keeping original draft."
            )
            return first_result

        if is_garbage_result(repaired):
            print(
                f"[HYBRID] Section {variable.section_id}: repair returned "
                "non-JSON/prose output — keeping original draft."
            )
            return first_result

        if quality_score(repaired) < quality_score(first_result):
            print(
                f"[HYBRID] Section {variable.section_id}: repaired draft is lower quality "
                f"({repaired.status}/{repaired.confidence} vs original "
                f"{first_result.status}/{first_result.confidence}) — keeping original."
            )
            return first_result

        print(f"[HYBRID] Section {variable.section_id}: repair done.")
        return repaired

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def parse_draft_response(self, raw: str, variable: TemplateVariable) -> ExtractionResult:
        data = parse_extraction_json(raw)

        if data is None:
            return ExtractionResult(
                section_id=variable.section_id,
                heading=variable.heading,
                sub_section=variable.sub_section,
                status="ERROR",
                answer="",
                filled_template="",
                evidence=[],
                confidence="LOW",
                notes="Failed to parse JSON from Stage B (draft) response.",
                raw_response=raw,
                error="JSON parse failure",
            )

        evidence: list[Evidence] = []
        seen_quotes: set[str] = set()
        for e in data.get("evidence", []):
            if not isinstance(e, dict):
                continue
            quote = str(e.get("quote", ""))
            norm = " ".join(quote.lower().split())
            if not norm or norm in seen_quotes:
                continue
            seen_quotes.add(norm)
            evidence.append(
                Evidence(
                    quote=quote,
                    page=str(e.get("page", "")),
                    section=str(e.get("section", "")),
                )
            )

        return ExtractionResult(
            section_id=data.get("section_id", variable.section_id),
            heading=variable.heading,
            sub_section=variable.sub_section,
            status=data.get("status", "ERROR"),
            answer=data.get("answer", ""),
            filled_template=data.get("filled_template", ""),
            evidence=evidence,
            confidence=data.get("confidence", "LOW"),
            notes=data.get("notes", ""),
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    # Short-circuit / error helpers
    # ------------------------------------------------------------------

    @staticmethod
    def error_result(
        variable: TemplateVariable, error: str, stage: str, raw: str = ""
    ) -> ExtractionResult:
        return ExtractionResult(
            section_id=variable.section_id,
            heading=variable.heading,
            sub_section=variable.sub_section,
            status="ERROR",
            answer="",
            filled_template="",
            evidence=[],
            confidence="LOW",
            notes="",
            raw_response=raw,
            error=f"[{stage}] {error}",
        )

    @staticmethod
    def make_standard_result(variable: TemplateVariable) -> ExtractionResult:
        return ExtractionResult(
            section_id=variable.section_id,
            heading=variable.heading,
            sub_section=variable.sub_section,
            status="STANDARD_TEXT",
            answer=variable.required_text,
            filled_template=variable.required_text,
            evidence=[],
            confidence="HIGH",
            notes="Standard required text - no extraction needed.",
        )

    @staticmethod
    def make_skipped_result(variable: TemplateVariable) -> ExtractionResult:
        return ExtractionResult(
            section_id=variable.section_id,
            heading=variable.heading,
            sub_section=variable.sub_section,
            status="SKIPPED",
            answer="",
            filled_template="",
            evidence=[],
            confidence="N/A",
            notes=(
                "Section marked as 'Not in protocol - requires manual entry'. "
                "Use suggested text from template as a starting point."
            ),
        )

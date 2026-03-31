"""
RLM-based extraction engine.

For each template variable, spins up a fresh RLM instance, loads the full
protocol as context_0, and uses the extraction prompt to pull structured
JSON with evidence and grounding.
"""

import ast
import json
import re

from icf.debug_logger import ICFDebugLogger
from icf.prompts import build_extraction_prompt
from icf.refine_prompts import build_refinement_prompt, build_refinement_setup_code
from icf.types import Evidence, ExtractionResult, TemplateVariable
from icf.validate import check_meta_commentary
from rlm import RLM
from rlm.utils.prompts import RLM_SYSTEM_PROMPT


def _build_icf_system_prompt(protocol_length: int) -> str:
    """Return a system prompt that corrects the LLM's execution-model misconceptions.

    The base RLM_SYSTEM_PROMPT is good but newer models sometimes believe the
    REPL requires user-side activation ("please say 'continue'", "BEGIN REPL",
    etc.) or invoke imaginary policies to refuse writing code.  This addendum
    makes the following unambiguous:
      1. context_0 IS already loaded — here is its exact size.
      2. ```repl blocks the LLM writes are auto-executed; no user confirmation needed.
      3. Every 'Code executed' block in the message history is a REAL execution result.
      4. There is no policy against writing repl blocks or doing step-by-step reasoning.
      5. When the answer is ready, call FINAL_VAR immediately — no prose summaries.
    """
    addendum = (
        "\n\n=== ICF EXTRACTION OPERATIONAL RULES (MANDATORY) ===\n"
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
        "  ✗ 'OpenAI policy prohibits chain-of-thought / tool use'\n"
        "  ✗ 'I cannot continue' / 'I'm sorry, I cannot assist with that'\n"
        "  ✗ Any prose-only response when you still have iterations remaining\n"
        "  If you feel an urge to write any of the above, write a ```repl block instead.\n\n"
        "FINISHING — follow this two-step pattern exactly:\n"
        "  STEP A — Verify (run this block, NO FINAL_VAR inside it):\n"
        "  ```repl\n"
        "  import re, json\n"
        "  issues = []\n"
        "  ft = result_dict.get('filled_template', '')\n"
        "  for m in re.findall(r'{{[^}]+}}|<<[^>]+>>', ft):\n"
        "      issues.append('Unfilled: ' + m)\n"
        "  for b in ['not found in', 'study documents', 'cannot be found']:\n"
        "      if b in ft.lower(): issues.append('Meta-commentary: ' + b)\n"
        "  if issues:\n"
        "      for iss in issues: print('FIX: ' + iss)\n"
        "  else:\n"
        "      result_json = json.dumps(result_dict)\n"
        "      print('READY_TO_FINALIZE')\n"
        "  ```\n\n"
        "  STEP B — Finalize (only write this after you see READY_TO_FINALIZE in the output):\n"
        "  ```repl\n"
        "  FINAL_VAR(result_json)\n"
        "  ```\n\n"
        "  CRITICAL RULES:\n"
        "    ✗  Never write FINAL_VAR inside an if/else or conditional block\n"
        "    ✗  Never write FINAL_VAR in the same block as the verification check\n"
        "    ✗  Never write FINAL_VAR(json.dumps(result_dict)) — result_json must already exist\n"
        "    ✓  result_json is assigned in Step A's else branch — do NOT redefine it in Step B\n\n"
        "  RECOVERY RULE: If result_dict was already built in a prior iteration, skip straight\n"
        "  to Step A. If result_json was already assigned (Step A ran with no issues), write\n"
        "  Step B immediately.\n\n"
        "  FIRST-RESPONSE RULE: Your very first response MUST contain a ```repl block.\n"
    )
    return RLM_SYSTEM_PROMPT + addendum


def _quality_score(result: ExtractionResult) -> int:
    """Numeric quality score for comparing two results. Higher is better."""
    status_score = {"FOUND": 30, "PARTIAL": 20, "NOT_FOUND": 5, "ERROR": 0}.get(
        result.status, 0
    )
    confidence_score = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "N/A": 0}.get(
        result.confidence, 0
    )
    return status_score + confidence_score


def _is_garbage_result(result: ExtractionResult) -> bool:
    """Return True when the RLM produced non-JSON/policy-refusal output.

    Two main causes:
    1. Parser fallback: RLM returned prose, the fallback parser wrapped it as
       PARTIAL/LOW with empty filled_template and evidence.
    2. Policy-refusal hallucination: model said "I cannot continue" / "REPL not
       available" etc. These produce either garbage JSON or short prose answers.

    NOT_FOUND results with empty fields are explicitly excluded — that is the
    correct and expected output when the protocol contains no relevant information.
    Flagging them as garbage would cause pointless full-extraction retries.
    NOT_FOUND/LOW is handled instead by _collect_quality_issues → refinement pass.
    """
    refusal_signals = [
        "repl is not active",
        "repl is not available",
        "cannot run repl",
        "cannot execute repl",
        "i cannot continue",
        "this interface does not",
        "this chat interface",
        "this interface cannot",
        "i must stop here",
    ]
    raw = (result.raw_response or "").lower()
    if any(sig in raw for sig in refusal_signals):
        return True
    # Empty filled_template + empty evidence = fallback-wrapped prose, but ONLY
    # for FOUND or PARTIAL — NOT_FOUND legitimately has no template/evidence.
    if result.status != "NOT_FOUND" and not result.filled_template and not result.evidence:
        return True
    return False


def _collect_quality_issues(
    result: ExtractionResult,
    variable: TemplateVariable,
    protocol_text: str,
) -> list[str]:
    """Return a list of quality problems that warrant a refinement pass.

    Only triggers refinement for issues the RLM can concretely fix:
      1. Unfilled {{...}} or <<...>> markers left in filled_template.
      2. Meta-commentary leaking into patient-facing filled_template.
      3. LOW confidence (signals uncertain extraction; targeted search may help).

    Intentionally NOT triggering for:
    - PARTIAL status alone: means the protocol genuinely lacks the info.
      A second pass won't find what isn't there, and just wastes iterations.
    - Quote verification failures: Unicode chars, footnote numbers, and
      sub-LLM paraphrasing cause false failures the RLM cannot fix.
      Quote quality is surfaced in the validate_extractions step instead.

    Returns an empty list when the result is clean enough to keep as-is,
    or when the status is one that refinement cannot improve.
    """
    if result.status in (
        "SKIPPED",
        "ERROR",
        "STANDARD_TEXT",
        "ADAPTATION_SKIPPED",
    ):
        return []

    # NOT_FOUND with HIGH/MEDIUM confidence: the model searched thoroughly and is sure
    # the info isn't there. A second pass won't find what doesn't exist.
    # NOT_FOUND with LOW confidence: the model is uncertain — refinement may help.
    if result.status == "NOT_FOUND" and result.confidence != "LOW":
        return []

    # Garbage fallback results are handled by the fresh-RLM retry loop.
    if _is_garbage_result(result):
        return []

    issues: list[str] = []

    if result.confidence == "LOW":
        issues.append("confidence is LOW")

    unfilled = re.findall(r"\{\{[^}]+\}\}|<<[^>]+>>", result.filled_template)
    for m in unfilled[:3]:
        issues.append(f"unfilled marker in filled_template: {m}")

    issues.extend(check_meta_commentary(result.filled_template))

    return issues


class ExtractionEngine:
    """Drives per-variable extraction via fresh RLM calls."""

    def __init__(
        self,
        model_name: str = "gpt-5.1",
        backend: str = "openai",
        backend_kwargs: dict | None = None,
        max_iterations: int = 20,
        verbose: bool = False,
        debug_logger: ICFDebugLogger | None = None,
        max_retries: int = 5,
    ):
        self.model_name = model_name
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.debug_logger = debug_logger
        self.max_retries = max_retries

    def extract_variable(
        self,
        protocol_text: str,
        variable: TemplateVariable,
    ) -> ExtractionResult:
        """Extract a single variable from the protocol.

        Routing logic:
          - standard_text -> return required_text directly
          - not in protocol -> return SKIPPED
          - otherwise -> run RLM extraction with up to max_retries total attempts

        If the RLM returns an ERROR result (invalid/missing JSON, exception, etc.)
        the section is re-run with a fresh RLM up to max_retries total attempts.
        """
        if variable.adaptation_skipped:
            return self._make_adaptation_skipped_result(variable)

        if variable.is_standard_text:
            return self._make_standard_result(variable)

        if not variable.is_in_protocol and not variable.partially_in_protocol:
            return self._make_skipped_result(variable)

        if self.debug_logger:
            self.debug_logger.set_section(
                variable.section_id, variable.heading, variable.sub_section or ""
            )

        last_result: ExtractionResult | None = None
        for attempt in range(1, self.max_retries + 1):
            result = self._run_rlm_extraction(protocol_text, variable)

            if result.status != "ERROR" and _is_garbage_result(result):
                # Prose/policy-refusal wrapped as PARTIAL — treat as a retriable error.
                print(
                    f"[EXTRACT] Section {variable.section_id}: attempt {attempt}/{self.max_retries} "
                    f"produced non-JSON/garbage output. Retrying with fresh RLM ..."
                )
                result = ExtractionResult(
                    section_id=variable.section_id,
                    heading=variable.heading,
                    sub_section=variable.sub_section,
                    status="ERROR",
                    answer="",
                    filled_template="",
                    evidence=[],
                    confidence="LOW",
                    notes="RLM returned non-JSON or policy-refusal output.",
                    raw_response=result.raw_response,
                    error="non-JSON/garbage output",
                )

            if result.status != "ERROR":
                # Good structured result — run quality gate, optionally refine.
                issues = _collect_quality_issues(result, variable, protocol_text)
                if issues:
                    print(
                        f"[REFINE] Section {variable.section_id}: "
                        f"{len(issues)} quality issue(s) found — running refinement pass ..."
                    )
                    for iss in issues[:4]:
                        print(f"[REFINE]   - {iss}")
                    result = self._run_refinement_pass(protocol_text, variable, result, issues)
                return result

            last_result = result
            if attempt < self.max_retries:
                print(
                    f"[EXTRACT] Section {variable.section_id}: attempt {attempt}/{self.max_retries} "
                    f"produced an error ({result.error}). Retrying with fresh RLM ..."
                )
            else:
                print(
                    f"[EXTRACT] Section {variable.section_id}: all {self.max_retries} attempts "
                    f"failed. Last error: {result.error}"
                )

        assert last_result is not None
        return last_result

    # ------------------------------------------------------------------
    # RLM extraction
    # ------------------------------------------------------------------

    def _run_rlm_extraction(
        self,
        protocol_text: str,
        variable: TemplateVariable,
    ) -> ExtractionResult:
        max_iter = self._iterations_for(variable)
        root_prompt = build_extraction_prompt(variable, protocol_length=len(protocol_text))

        kwargs = {"model_name": self.model_name}
        kwargs.update(self.backend_kwargs)

        try:
            rlm = RLM(
                backend=self.backend,
                backend_kwargs=kwargs,
                environment="local",
                verbose=self.verbose,
                max_iterations=max_iter,
                custom_system_prompt=_build_icf_system_prompt(len(protocol_text)),
                logger=self.debug_logger,
            )

            completion = rlm.completion(
                prompt=protocol_text,
                root_prompt=root_prompt,
            )

            return self._parse_response(completion.response, variable)

        except Exception as e:
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
                raw_response="",
                error=f"{type(e).__name__}: {e}",
            )

    def _run_refinement_pass(
        self,
        protocol_text: str,
        variable: TemplateVariable,
        first_result: ExtractionResult,
        issues: list[str],
    ) -> ExtractionResult:
        """Run a focused second-pass RLM to fix specific quality issues.

        Uses a leaner prompt (no full template symbol guide or UHN guidelines)
        and a capped iteration budget so it doesn't burn as many tokens as
        the original extraction.  If the refinement itself errors or produces
        a worse result, the original first_result is returned unchanged.
        """
        # Minimum 10: 2 searches + update + verify + FINAL_VAR = ~5 steps,
        # plus buffer for the model to reason and retry if verification fails.
        max_iter = max(10, self._iterations_for(variable) // 2)
        refinement_prompt = build_refinement_prompt(variable, first_result, issues)
        setup_code = build_refinement_setup_code(first_result)

        kwargs = {"model_name": self.model_name}
        kwargs.update(self.backend_kwargs)

        try:
            rlm = RLM(
                backend=self.backend,
                backend_kwargs=kwargs,
                environment="local",
                environment_kwargs={"setup_code": setup_code},
                verbose=self.verbose,
                max_iterations=max_iter,
                custom_system_prompt=_build_icf_system_prompt(len(protocol_text)),
                logger=self.debug_logger,
            )
            completion = rlm.completion(
                prompt=protocol_text,
                root_prompt=refinement_prompt,
            )
            refined = self._parse_response(completion.response, variable)

            if refined.status == "ERROR":
                print(
                    f"[REFINE] Section {variable.section_id}: refinement pass errored "
                    f"({refined.error}) — keeping original result."
                )
                return first_result

            if _is_garbage_result(refined):
                print(
                    f"[REFINE] Section {variable.section_id}: refinement returned "
                    "non-JSON/prose output — keeping original result."
                )
                return first_result

            if _quality_score(refined) < _quality_score(first_result):
                print(
                    f"[REFINE] Section {variable.section_id}: refined result is lower "
                    f"quality ({refined.status}/{refined.confidence} vs original "
                    f"{first_result.status}/{first_result.confidence}) — keeping original."
                )
                return first_result

            print(
                f"[REFINE] Section {variable.section_id}: done. "
                f"Confidence {first_result.confidence} -> {refined.confidence}, "
                f"Status {first_result.status} -> {refined.status}."
            )
            return refined

        except Exception as e:
            print(
                f"[REFINE] Section {variable.section_id}: refinement pass raised "
                f"{type(e).__name__} — keeping original result."
            )
            return first_result

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        raw: str,
        variable: TemplateVariable,
    ) -> ExtractionResult:
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
                notes="Failed to parse JSON from RLM response.",
                raw_response=raw,
                error="JSON parse failure",
            )

        evidence: list[Evidence] = []
        for e in data.get("evidence", []):
            if isinstance(e, dict):
                evidence.append(
                    Evidence(
                        quote=str(e.get("quote", "")),
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
    # Short-circuit helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_standard_result(variable: TemplateVariable) -> ExtractionResult:
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
    def _make_adaptation_skipped_result(variable: TemplateVariable) -> ExtractionResult:
        reason = (
            variable.adaptation_notes
            or "Marked as not applicable for this study by adaptation pass."
        )
        return ExtractionResult(
            section_id=variable.section_id,
            heading=variable.heading,
            sub_section=variable.sub_section,
            status="ADAPTATION_SKIPPED",
            answer="",
            filled_template="",
            evidence=[],
            confidence="N/A",
            notes=reason,
        )

    @staticmethod
    def _make_skipped_result(variable: TemplateVariable) -> ExtractionResult:
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

    # ------------------------------------------------------------------
    # Iteration budget per complexity
    # ------------------------------------------------------------------

    def _iterations_for(self, variable: TemplateVariable) -> int:
        label = variable.get_complexity_label()
        budget_map = {
            "Easy": 10,
            "Moderate": 15,
            "Complex": self.max_iterations,
            "Not in protocol": 8,
        }
        budget = budget_map.get(label, 12)
        return min(budget, self.max_iterations)


# ======================================================================
# Robust JSON extraction from free-form RLM output
# ======================================================================


def parse_extraction_json(raw: str) -> dict | None:
    """Try every reasonable strategy to recover a JSON dict from *raw*.

    Strategies (in order):
      1. Direct json.loads
      2. Python literal_eval (handles single-quoted dicts)
      3. JSON inside a markdown code fence
      4. Outermost { ... } containing a "status" key
      5. Any outermost { ... } that parses as valid JSON dict
      6. Regex field extraction from truncated / malformed JSON
      7. Fallback: wrap the raw text as a best-effort PARTIAL result
    """
    if not raw:
        return None

    stripped = raw.strip()

    # 1. Direct parse
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Python literal
    try:
        obj = ast.literal_eval(stripped)
        if isinstance(obj, dict):
            return obj
    except (ValueError, SyntaxError):
        pass

    # 3. Markdown code fence
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if md_match:
        try:
            data = json.loads(md_match.group(1).strip())
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # 4+5. Balanced-brace extraction: prefer one with "status", accept any
    candidates = _extract_brace_candidates(raw)
    any_valid: dict | None = None
    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                if "status" in data:
                    return data
                if any_valid is None:
                    any_valid = data
        except json.JSONDecodeError:
            continue

    if any_valid is not None:
        return any_valid

    # 6. Truncated / malformed JSON: pull individual fields with regex.
    #    Handles the common case where the LLM output is cut off mid-string
    #    so balanced braces never close, but the content is still there.
    partial = _parse_partial_json_fields(raw)
    if partial is not None:
        return partial

    # 7. Fallback: the RLM returned free-form text instead of JSON.
    #    Wrap it so the pipeline can still use the content.
    if len(stripped) > 20:
        return {
            "status": "PARTIAL",
            "answer": stripped,
            "filled_template": "",
            "evidence": [],
            "confidence": "LOW",
            "notes": "RLM returned free-form text instead of JSON. Content may need manual review.",
        }

    return None


def _extract_brace_candidates(text: str) -> list[str]:
    """Return all top-level { … } substrings in *text*."""
    depth = 0
    start: int | None = None
    results: list[str] = []
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                results.append(text[start : i + 1])
                start = None
    return results


def _unescape_json_str(s: str) -> str:
    """Decode common JSON string escape sequences."""
    return s.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').replace("\\\\", "\\")


def _parse_partial_json_fields(raw: str) -> dict | None:
    """Extract key fields via regex from a truncated or malformed JSON string.

    Called when all standard parse strategies fail (e.g. the LLM output was
    cut off mid-string so braces never balance).  Recovers as many fields as
    possible and returns a PARTIAL-status dict.
    """
    # Only attempt if the text looks like it contains JSON-style key-value pairs
    if not re.search(r'"(?:status|answer|filled_template)"', raw):
        return None

    result: dict = {}

    # Simple enum / id fields — values never contain backslash escapes
    for key in ("section_id", "status", "confidence"):
        m = re.search(rf'"{key}"\s*:\s*"([^"\\]*)"', raw)
        if m:
            result[key] = m.group(1)

    # Potentially long string fields — must handle JSON escape sequences
    for key in ("answer", "filled_template", "notes"):
        # Try a properly closed JSON string first
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
        if m:
            result[key] = _unescape_json_str(m.group(1))
        else:
            # Truncated string: grab whatever is there (still useful)
            m = re.search(rf'"{key}"\s*:\s*"(.*)', raw, re.DOTALL)
            if m:
                result[key] = _unescape_json_str(m.group(1).rstrip())

    # Evidence array — try to parse just that slice
    ev_m = re.search(r'"evidence"\s*:\s*(\[.*?\])', raw, re.DOTALL)
    if ev_m:
        try:
            result["evidence"] = json.loads(ev_m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    if not result:
        return None

    return result

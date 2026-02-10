"""
RLM-based extraction engine.

For each template variable, spins up a fresh RLM instance, loads the full
protocol as context_0, and uses the extraction prompt to pull structured
JSON with evidence and grounding.
"""

import ast
import json
import re

from icf.prompts import build_extraction_prompt
from icf.types import Evidence, ExtractionResult, TemplateVariable
from rlm import RLM


class ExtractionEngine:
    """Drives per-variable extraction via fresh RLM calls."""

    def __init__(
        self,
        model_name: str = "gpt-5.1",
        backend: str = "openai",
        backend_kwargs: dict | None = None,
        max_iterations: int = 20,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.max_iterations = max_iterations
        self.verbose = verbose

    def extract_variable(
        self,
        protocol_text: str,
        variable: TemplateVariable,
    ) -> ExtractionResult:
        """Extract a single variable from the protocol.

        Routing logic:
          - standard_text -> return required_text directly
          - not in protocol -> return SKIPPED
          - otherwise -> run RLM extraction
        """
        if variable.is_standard_text:
            return self._make_standard_result(variable)

        if not variable.is_in_protocol and not variable.partially_in_protocol:
            return self._make_skipped_result(variable)

        return self._run_rlm_extraction(protocol_text, variable)

    # ------------------------------------------------------------------
    # RLM extraction
    # ------------------------------------------------------------------

    def _run_rlm_extraction(
        self,
        protocol_text: str,
        variable: TemplateVariable,
    ) -> ExtractionResult:
        max_iter = self._iterations_for(variable)
        root_prompt = build_extraction_prompt(variable)

        kwargs = {"model_name": self.model_name}
        kwargs.update(self.backend_kwargs)

        try:
            rlm = RLM(
                backend=self.backend,
                backend_kwargs=kwargs,
                environment="local",
                verbose=self.verbose,
                max_iterations=max_iter,
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
      6. Fallback: wrap the raw text as a best-effort PARTIAL result
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

    # 6. Fallback: the RLM returned free-form text instead of JSON.
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

"""
Naive (full-context) extraction engine.

Sends the entire protocol text + extraction task to the LLM in a single call.
No REPL, no iteration, no chunking — pure single-shot structured generation.

This backend exists as a benchmarking baseline.  It answers the question:
"How much does the RLM's iterative retrieval actually help over just giving
the model the whole document at once?"

Key design choices:
  - One LM client instance is created and reused across all sections (efficient).
  - The client's completion() method is called with a messages list (system +
    user), rather than a raw string, so the protocol is cleanly separated from
    the extraction task in the context window.
  - The 'reasoning' field in the JSON schema provides chain-of-thought before
    extraction fields, improving quality without extra round-trips.
  - parse_extraction_json() from extract.py is reused directly — the output
    format is identical to the RLM backend.
  - max_retries is intentionally small (2): unlike RLM, failures here are
    almost always JSON parse errors on a well-formed single-call output, not
    search strategy failures, so a single retry is usually sufficient.
"""

from icf.debug_logger import ICFDebugLogger
from icf.extract import parse_extraction_json
from icf.naive_prompts import build_naive_messages
from icf.types import Evidence, ExtractionResult, TemplateVariable
from rlm.clients import get_client


class NaiveExtractionEngine:
    """Single-pass extraction: full protocol → one LLM call → ExtractionResult.

    Implements the same extract_variable() interface as ExtractionEngine so
    the pipeline can swap backends without any other changes.
    """

    def __init__(
        self,
        model_name: str = "gpt-5.1",
        backend: str = "openai",
        backend_kwargs: dict | None = None,
        max_retries: int = 2,
        verbose: bool = False,
        debug_logger: ICFDebugLogger | None = None,
    ):
        self.model_name = model_name
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.max_retries = max_retries
        self.verbose = verbose
        self.debug_logger = debug_logger

        # Build one client and reuse across all sections.
        kwargs = dict(self.backend_kwargs)
        kwargs["model_name"] = self.model_name
        self.client = get_client(self.backend, kwargs)

    # ------------------------------------------------------------------
    # Public interface — matches ExtractionEngine.extract_variable()
    # ------------------------------------------------------------------

    def extract_variable(
        self,
        protocol_text: str,
        variable: TemplateVariable,
    ) -> ExtractionResult:
        """Extract a single ICF section using a full-context single LLM call.

        Routing:
          adaptation_skipped → ADAPTATION_SKIPPED (no LLM call)
          is_standard_text   → STANDARD_TEXT (no LLM call)
          not in protocol    → SKIPPED (no LLM call)
          otherwise          → single LLM call with up to max_retries attempts
        """
        if variable.adaptation_skipped:
            return self._make_adaptation_skipped_result(variable)

        if variable.is_standard_text:
            return self._make_standard_result(variable)

        if not variable.is_in_protocol and not variable.partially_in_protocol:
            return self._make_skipped_result(variable)

        last_result: ExtractionResult | None = None
        for attempt in range(1, self.max_retries + 1):
            result = self._run_extraction(protocol_text, variable)
            if result.status != "ERROR":
                return result
            last_result = result
            if attempt < self.max_retries:
                print(
                    f"[NAIVE] Section {variable.section_id}: attempt {attempt}/{self.max_retries} "
                    f"produced an error ({result.error}). Retrying ..."
                )
            else:
                print(
                    f"[NAIVE] Section {variable.section_id}: all {self.max_retries} attempts "
                    f"failed. Last error: {result.error}"
                )

        assert last_result is not None
        return last_result

    # ------------------------------------------------------------------
    # Internal extraction call
    # ------------------------------------------------------------------

    def _run_extraction(
        self,
        protocol_text: str,
        variable: TemplateVariable,
    ) -> ExtractionResult:
        messages = build_naive_messages(variable, protocol_text)

        if self.verbose:
            task_preview = messages[-1]["content"][-300:]
            print(f"[NAIVE] [{variable.section_id}] Calling LLM ... (task tail: ...{task_preview!r})")

        try:
            raw = self.client.completion(messages)
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

        if self.verbose:
            print(f"[NAIVE] [{variable.section_id}] Raw response ({len(raw)} chars):\n{raw[:500]}")

        return self._parse_response(raw, variable)

    # ------------------------------------------------------------------
    # Response parsing — identical logic to ExtractionEngine._parse_response
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
                notes="Failed to parse JSON from naive LLM response.",
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
    # Short-circuit helpers (same behaviour as ExtractionEngine)
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
            notes="Standard required text — no extraction needed.",
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
                "Section marked as 'Not in protocol — requires manual entry'. "
                "Use suggested text from template as a starting point."
            ),
        )

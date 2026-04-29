"""
Stage 8 — ICF Plain Language Review.

After all sections are assembled, ReviewEngine reads the full generated ICF
and returns a structured list of ReviewFlags pointing to specific text.
Flags are annotations only — the engine never modifies ICF content.

Standard-text sections (is_standard_text=True) are completely protected:
  1. They are shown in the assembled document with a [STANDARD TEXT - DO NOT FLAG]
     header so the LLM can see the protection clearly.
  2. Their section IDs are listed explicitly in the prompt header.
  3. Any flags referencing protected section IDs are silently dropped in
     _parse_review_response() as a final backstop.

Design mirrors adapt.py: get_client() once, single direct LLM call (no REPL),
graceful failure returns an empty ReviewResult rather than raising.
"""

from __future__ import annotations

import json
import re

from icf.review_prompts import build_icf_document_for_review, build_review_messages
from icf.types import ExtractionResult, ReviewFlag, ReviewResult, TemplateVariable
from rlm.clients import get_client

# Strip accidental "SECTION " prefix from section IDs returned by the review LLM.
_SECTION_PREFIX_RE = re.compile(r"^(?:SECTION|Section|section)\s+", re.IGNORECASE)


def _normalize_review_section_id(raw: str) -> str:
    return _SECTION_PREFIX_RE.sub("", raw).strip()


class ReviewEngine:
    """Run the Stage 8 plain-language review over the assembled ICF.

    Reuses the same LLM backend configured for the pipeline (model_name,
    backend, backend_kwargs) so no additional credentials are needed.
    """

    def __init__(
        self,
        model_name: str,
        backend: str,
        backend_kwargs: dict | None = None,
        max_retries: int = 2,
        verbose: bool = False,
    ):
        self.max_retries = max_retries
        self.verbose = verbose

        kwargs = dict(backend_kwargs or {})
        kwargs["model_name"] = model_name
        self.client = get_client(backend, kwargs)

    def run_review(
        self,
        extractions: list[ExtractionResult],
        variables: list[TemplateVariable],
    ) -> ReviewResult:
        """Assemble the ICF and run the review LLM call.

        Returns a ReviewResult with flags and cross_section_notes.
        On failure returns an empty ReviewResult rather than raising.
        """
        icf_document, standard_text_ids = build_icf_document_for_review(extractions, variables)

        if not icf_document.strip():
            return ReviewResult(flags=[], cross_section_notes="No extractable content to review.")

        messages = build_review_messages(icf_document, standard_text_ids)

        for attempt in range(1, self.max_retries + 1):
            result = self._call_llm(messages, standard_text_ids)
            if result is not None:
                return result
            if attempt < self.max_retries:
                print(f"[REVIEW] Attempt {attempt}/{self.max_retries} failed. Retrying ...")

        return ReviewResult(flags=[], cross_section_notes="Review LLM call failed after retries.")

    def _call_llm(
        self,
        messages: list[dict],
        standard_text_ids: set[str],
    ) -> ReviewResult | None:
        """Issue the LLM call and parse the JSON response."""
        try:
            raw = self.client.completion(messages)
        except Exception as e:
            print(f"[REVIEW] LLM call error: {type(e).__name__}: {e}")
            return None

        if self.verbose:
            preview = raw[:600] if raw else "(empty)"
            print(f"[REVIEW] Raw response ({len(raw) if raw else 0} chars):\n{preview}")

        return _parse_review_response(raw, standard_text_ids)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _parse_review_response(raw: str, standard_text_ids: set[str]) -> ReviewResult | None:
    """Extract ReviewResult from the LLM response.

    Tries three strategies: direct parse, markdown fence, outermost { ... }.
    Filters out any flags targeting protected (standard-text) sections.
    Returns None only if JSON cannot be extracted at all.
    """
    if not raw:
        return None

    data = _extract_json_object(raw)
    if data is None or not isinstance(data, dict):
        return None

    flags: list[ReviewFlag] = []
    for f in data.get("flags", []):
        if not isinstance(f, dict):
            continue
        section_id = _normalize_review_section_id(str(f.get("section_id", "")))
        # Safety backstop: drop any flag targeting a protected section.
        if section_id in standard_text_ids:
            continue
        flags.append(
            ReviewFlag(
                section_id=section_id,
                flagged_text=str(f.get("flagged_text", "")),
                issue_type=str(f.get("issue_type", "UNCLEAR")),
                suggestion=str(f.get("suggestion", "")),
                severity=str(f.get("severity", "LOW")),
                suggested_fix=str(f.get("suggested_fix", "")),
            )
        )

    return ReviewResult(
        flags=flags,
        cross_section_notes=str(data.get("cross_section_notes", "")),
    )


def _extract_json_object(raw: str) -> dict | None:
    """Extract the first JSON object from an LLM response string.

    Three strategies (same pattern used in adapt.py and extract.py):
      1. Direct json.loads on the stripped string.
      2. Content of the first ```json ... ``` or ``` ... ``` fence.
      3. Outermost { ... } with balanced-brace extraction.
    """
    # Strategy 1: direct parse
    try:
        data = json.loads(raw.strip())
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: markdown code fence
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1).strip())
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: outermost { ... } with balanced-brace extraction
    start = raw.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(raw[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(raw[start : i + 1])
                        if isinstance(data, dict):
                            return data
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break

    return None

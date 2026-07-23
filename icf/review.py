"""
Stage 8 — ICF Plain Language Review.

After all sections are assembled, ReviewEngine reads the full generated ICF
and returns a structured list of ReviewFlags pointing to specific text.
Flags are annotations only — the engine never modifies ICF content.

Standard-text sections (is_standard_text=True) AND section group 2.x
(cover-page fields: title, protocol #, study doctor, sponsor, emergency
contact) are completely protected:
  1. They are shown in the assembled document with a [STANDARD TEXT - DO NOT
     FLAG] or [PROTECTED FIELD - DO NOT FLAG] header so the LLM can see the
     protection clearly.
  2. Their section IDs are listed explicitly in the prompt header.
  3. Any flags referencing protected section IDs are silently dropped in
     _parse_review_response() as a final backstop.
  4. Flags whose flagged_text overlaps locked literal phrases from
     required_text or suggested_text are also dropped (same
     extract_locked_phrases logic as remediation).

Design mirrors adapt.py: get_client() once, single direct LLM call (no REPL),
graceful failure returns an empty ReviewResult rather than raising.
"""

from __future__ import annotations

import json
import re

from icf.remediate_prompts import extract_locked_phrases
from icf.review_prompts import build_icf_document_for_review, build_review_messages
from icf.types import (
    ExtractionResult,
    ReviewFlag,
    ReviewResult,
    TemplateVariable,
    normalize_section_id,
)
from rlm.clients import get_client


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
        icf_document, protected_section_ids = build_icf_document_for_review(extractions, variables)

        if not icf_document.strip():
            return ReviewResult(flags=[], cross_section_notes="No extractable content to review.")

        messages = build_review_messages(icf_document, protected_section_ids)

        for attempt in range(1, self.max_retries + 1):
            result = self._call_llm(messages, protected_section_ids, variables, extractions)
            if result is not None:
                return result
            if attempt < self.max_retries:
                print(f"[REVIEW] Attempt {attempt}/{self.max_retries} failed. Retrying ...")

        return ReviewResult(flags=[], cross_section_notes="Review LLM call failed after retries.")

    def _call_llm(
        self,
        messages: list[dict],
        protected_section_ids: set[str],
        variables: list[TemplateVariable],
        extractions: list[ExtractionResult],
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

        return _parse_review_response(raw, protected_section_ids, variables, extractions)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _build_locked_phrases_by_section(
    variables: list[TemplateVariable],
    extractions: list[ExtractionResult],
) -> dict[str, list[str]]:
    ext_map = {e.section_id: e for e in extractions}
    locked: dict[str, list[str]] = {}
    for var in variables:
        if (not var.required_text or not var.required_text.strip()) and (
            not var.suggested_text or not var.suggested_text.strip()
        ):
            continue
        ext = ext_map.get(var.section_id)
        filled = (ext.filled_template or ext.answer or "") if ext else ""
        phrases = extract_locked_phrases(var.required_text, filled)
        for p in extract_locked_phrases(var.suggested_text, filled):
            if p not in phrases:
                phrases.append(p)
        if phrases:
            locked[var.section_id] = phrases
    return locked


# A flagged excerpt sharing this many consecutive words with the start/end of a
# locked phrase is treated as touching locked text -- see _shares_locked_boundary.
_MIN_BOUNDARY_OVERLAP_WORDS = 2
_MIN_BOUNDARY_OVERLAP_CHARS = 12


def _run_of_shared_words(tail_words: list[str], head_words: list[str]) -> int:
    """Longest n such that tail_words[-n:] == head_words[:n]."""
    for n in range(min(len(tail_words), len(head_words)), 0, -1):
        if tail_words[-n:] == head_words[:n]:
            return n
    return 0


def _shares_locked_boundary(excerpt_lower: str, phrase_lower: str) -> bool:
    """True if excerpt overlaps a locked phrase's boundary by >= 2 substantive words.

    The reviewer only sees the rendered document, not where locked template text
    ends and placeholder-filled content begins. A flagged span can therefore grab
    the tail of a locked phrase plus adjacent filled text (e.g. locked "...on
    individuals with" + filled "kidney transplantation" flagged together as
    "individuals with kidney transplantation"). Full-string containment doesn't
    catch this; check for a shared run of words straddling either boundary instead.
    """
    excerpt_words = excerpt_lower.split()
    phrase_words = phrase_lower.split()

    # Locked phrase's tail == excerpt's head (excerpt starts mid-phrase and runs on).
    n1 = _run_of_shared_words(phrase_words, excerpt_words)
    # Excerpt's tail == locked phrase's head (excerpt ends where the phrase begins).
    n2 = _run_of_shared_words(excerpt_words, phrase_words)

    for n, overlap_words in ((n1, excerpt_words[:n1]), (n2, excerpt_words[-n2:] if n2 else [])):
        if n >= _MIN_BOUNDARY_OVERLAP_WORDS and len(" ".join(overlap_words)) >= (
            _MIN_BOUNDARY_OVERLAP_CHARS
        ):
            return True
    return False


def _flag_targets_locked_text(flagged_text: str, locked_phrases: list[str]) -> bool:
    excerpt = flagged_text.strip()
    if not excerpt or not locked_phrases:
        return False
    excerpt_lower = excerpt.lower()
    for phrase in locked_phrases:
        p = phrase.strip()
        if not p:
            continue
        p_lower = p.lower()
        if excerpt_lower in p_lower or p_lower in excerpt_lower:
            return True
        if _shares_locked_boundary(excerpt_lower, p_lower):
            return True
    return False


def _parse_review_response(
    raw: str,
    protected_section_ids: set[str],
    variables: list[TemplateVariable],
    extractions: list[ExtractionResult],
) -> ReviewResult | None:
    """Extract ReviewResult from the LLM response.

    Tries three strategies: direct parse, markdown fence, outermost { ... }.
    Filters out any flags targeting protected (standard-text or 2.x cover-page)
    sections, or locked literal phrases from required_text/suggested_text.
    Returns None only if JSON cannot be extracted at all.
    """
    if not raw:
        return None

    data = _extract_json_object(raw)
    if data is None or not isinstance(data, dict):
        return None

    locked_by_section = _build_locked_phrases_by_section(variables, extractions)

    flags: list[ReviewFlag] = []
    for f in data.get("flags", []):
        if not isinstance(f, dict):
            continue
        section_id = normalize_section_id(str(f.get("section_id", "")))
        # Safety backstop: drop any flag targeting a protected section.
        if section_id in protected_section_ids:
            continue
        flagged_text = str(f.get("flagged_text", ""))
        section_locked = locked_by_section.get(section_id, [])
        if _flag_targets_locked_text(flagged_text, section_locked):
            continue
        flags.append(
            ReviewFlag(
                section_id=section_id,
                flagged_text=flagged_text,
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

"""
Dynamic registry adaptation pass.

After extracting the "trigger" sections (Introduction and Why Is This Study
Being Done — IDs "3" and "6"), runs a single lightweight LLM call to decide
which remaining OPTIONAL sections are clearly irrelevant for this specific
study, and marks them adaptation_skipped=True.

Design principles:
  - The original variables list is NEVER mutated — a deep copy is returned.
  - Only sections with required=False are candidates for skipping.
  - The LLM is called directly (no RLM REPL loop) since this is a simple
    classification task that needs no protocol search.
  - When in doubt the LLM is instructed to keep sections, not skip them.
  - The decision is transparent: adaptation_notes records the reason.
"""

from __future__ import annotations

import copy
import json
import re

from icf.types import ExtractionResult, TemplateVariable
from rlm.clients import get_client

# Section IDs that trigger the adaptation pass once extracted.
ADAPTATION_TRIGGER_IDS: set[str] = {"3", "6"}


def build_adapted_registry(
    variables: list[TemplateVariable],
    early_results: list[ExtractionResult],
    model_name: str,
    backend: str,
    backend_kwargs: dict,
) -> list[TemplateVariable]:
    """Return a deep copy of *variables* with irrelevant optional sections
    marked adaptation_skipped=True and adaptation_notes set to the reason.

    Only sections where required=False and not yet extracted are candidates.
    If the early results contain no usable text, returns the copy unmodified.
    """
    adapted = copy.deepcopy(variables)

    early_context = _format_early_results(early_results)
    if not early_context:
        print("[ADAPT] No usable content from trigger sections — skipping adaptation.")
        return adapted

    already_extracted_ids = {r.section_id for r in early_results}
    candidates = [
        v for v in adapted if not v.required and v.section_id not in already_extracted_ids
    ]

    if not candidates:
        return adapted

    skip_decisions = _run_adaptation_llm(
        early_context=early_context,
        candidates=candidates,
        model_name=model_name,
        backend=backend,
        backend_kwargs=backend_kwargs,
    )

    skip_map = {d["section_id"]: d.get("reason", "") for d in skip_decisions}
    for var in adapted:
        if var.section_id in skip_map:
            var.adaptation_skipped = True
            var.adaptation_notes = skip_map[var.section_id]

    return adapted


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _format_early_results(results: list[ExtractionResult]) -> str:
    """Summarise early extraction results as context for the adaptation prompt."""
    parts = []
    for r in results:
        if r.status in ("FOUND", "PARTIAL", "STANDARD_TEXT"):
            text = r.answer or r.filled_template or ""
            if text.strip():
                parts.append(f"Section {r.section_id} — {r.heading}:\n{text[:2000]}")
    return "\n\n---\n\n".join(parts)


def _build_adaptation_prompt(
    early_context: str,
    candidates: list[TemplateVariable],
) -> str:
    section_list = "\n".join(
        f"  {v.section_id}: {v.heading}" + (f" > {v.sub_section}" if v.sub_section else "")
        for v in candidates
    )
    return (
        "You are a clinical research expert reviewing an Informed Consent Form (ICF) template.\n"
        "Based on early information extracted from a study protocol, decide which OPTIONAL ICF\n"
        "sections are CLEARLY and DEFINITELY not applicable to this specific study.\n\n"
        "EARLY STUDY INFORMATION (Introduction + Purpose):\n"
        f"{early_context}\n\n"
        "OPTIONAL CANDIDATE SECTIONS (only these may be skipped):\n"
        f"{section_list}\n\n"
        "RULES:\n"
        "- Only skip a section if you are CERTAIN it is irrelevant.\n"
        "  Example: skip 'SAMPLE COLLECTION' only if the study clearly collects no samples.\n"
        "- When in doubt, DO NOT skip — including an unnecessary section is safer than\n"
        "  omitting a needed one.\n"
        "- Only reference section IDs from the list above.\n\n"
        "Respond with a JSON array of sections to skip (use [] if nothing should be skipped):\n"
        '[\n  {"section_id": "...", "reason": "One-sentence explanation"}\n]\n\n'
        "Return ONLY the JSON array, nothing else."
    )


def _run_adaptation_llm(
    early_context: str,
    candidates: list[TemplateVariable],
    model_name: str,
    backend: str,
    backend_kwargs: dict,
) -> list[dict]:
    """Call the LM directly and return a list of {section_id, reason} dicts."""
    prompt = _build_adaptation_prompt(early_context, candidates)

    # Build kwargs with model_name merged in (get_client unpacks these).
    kwargs = dict(backend_kwargs)
    kwargs["model_name"] = model_name

    try:
        client = get_client(backend, kwargs)
        raw = client.completion(prompt)
        return _parse_adaptation_response(raw)
    except Exception as e:
        print(f"[ADAPT] LLM call failed ({type(e).__name__}: {e}). No sections will be skipped.")
        return []


def _parse_adaptation_response(raw: str) -> list[dict]:
    """Extract the JSON array from the LLM response."""
    if not raw:
        return []

    def _valid(lst: list) -> list[dict]:
        return [d for d in lst if isinstance(d, dict) and "section_id" in d]

    # Direct parse
    try:
        data = json.loads(raw.strip())
        if isinstance(data, list):
            return _valid(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Markdown code fence
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1).strip())
            if isinstance(data, list):
                return _valid(data)
        except (json.JSONDecodeError, ValueError):
            pass

    # Bare [...] anywhere in the response
    m = re.search(r"\[.*?\]", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return _valid(data)
        except (json.JSONDecodeError, ValueError):
            pass

    return []

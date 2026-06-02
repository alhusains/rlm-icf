"""
Section Group Harmonization pass (Stage 5.5).

After all sections are extracted independently, groups of related sub-sections
(e.g., all "WHAT ARE THE STUDY PROCEDURES?" sub-sections) may contain repeated
or misplaced content because each RLM call had no visibility into the others.

SectionGroupHarmonizer runs a single LLM call per group to:
  1. Redistribute content so each piece of information appears in the correct
     sub-section only.
  2. De-duplicate information that appears in multiple sub-sections.
  3. Mark sub-sections with no applicable content as empty (NOT_FOUND) so they
     are omitted from the assembled document.

Design mirrors adapt.py / remediate.py:
  - The input extraction list is NEVER mutated — a deep copy is returned.
  - Direct LLM call (no RLM REPL loop) — purely a redistribution task.
  - Locked-phrase safety check before applying any revision (same as Stage 9).
  - Graceful failure: if a call fails or produces invalid JSON the original
    text is kept; the run never degrades.
  - ADAPTATION_SKIPPED sections are excluded from each group automatically.
"""

from __future__ import annotations

import copy
import json
import re

from icf.harmonize_prompts import build_harmonization_prompt
from icf.remediate_prompts import extract_locked_phrases
from icf.types import ExtractionResult, TemplateVariable
from rlm.clients import get_client

# ---------------------------------------------------------------------------
# Section groups
#
# Maps a short group key to an ordered list of section IDs.
# All IDs in a group share the same top-level heading and may overlap.
# Groups are processed only when at least _MIN_ACTIVE_SECTIONS of their
# sections have usable extracted content (FOUND or PARTIAL with non-empty text).
# ---------------------------------------------------------------------------

HARMONIZATION_GROUPS: dict[str, list[str]] = {
    # "What are the study procedures?" parent + all sub-sections
    "12": ["12", "12.1", "12.2", "12.3", "12.4", "12.5.0", "12.6", "12.7", "12.8"],
    # "Sample collection" parent + all sub-sections
    "13": ["13", "13.1", "13.2", "13.3", "13.4", "13.5", "13.6"],
}

# Minimum number of sections with non-empty content needed to bother harmonizing.
_MIN_ACTIVE_SECTIONS = 2

_CONTENT_STATUSES = frozenset({"FOUND", "PARTIAL"})


# ---------------------------------------------------------------------------
# Public engine
# ---------------------------------------------------------------------------


class SectionGroupHarmonizer:
    """Redistribute and de-duplicate content across related ICF sub-sections."""

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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_harmonization(
        self,
        extractions: list[ExtractionResult],
        variables: list[TemplateVariable],
    ) -> tuple[list[ExtractionResult], dict[str, list[str]]]:
        """Run harmonization over all configured section groups.

        Returns:
            patched_extractions: deep copy with revised ``filled_template`` values
                                 where the LLM redistributed content.
            audit_notes:         dict mapping section_id → list of change notes.
        """
        ext_map: dict[str, ExtractionResult] = {e.section_id: e for e in extractions}
        var_map: dict[str, TemplateVariable] = {v.section_id: v for v in variables}

        patched = copy.deepcopy(extractions)
        patched_map: dict[str, ExtractionResult] = {e.section_id: e for e in patched}
        audit: dict[str, list[str]] = {}

        for group_key, ids in HARMONIZATION_GROUPS.items():
            # Build ordered (var, ext) pairs for sections present in this run.
            pairs: list[tuple[TemplateVariable, ExtractionResult | None]] = []
            for sid in ids:
                var = var_map.get(sid)
                if var is None:
                    continue
                ext = ext_map.get(sid)
                # Never include adaptation-skipped sections.
                if ext is not None and ext.status == "ADAPTATION_SKIPPED":
                    continue
                pairs.append((var, ext))

            if not pairs:
                continue

            # Skip groups with too little content to be worth harmonizing.
            active = sum(
                1
                for _, ext in pairs
                if ext is not None
                and ext.status in _CONTENT_STATUSES
                and (ext.filled_template or ext.answer or "").strip()
            )
            if active < _MIN_ACTIVE_SECTIONS:
                if self.verbose:
                    print(
                        f"[HARMONIZE] Group {group_key!r}: {active} active section(s) "
                        "— skipping (nothing to harmonize)."
                    )
                continue

            group_label = pairs[0][0].heading
            section_ids = [v.section_id for v, _ in pairs]
            print(
                f"[HARMONIZE] Group {group_key!r}: {len(pairs)} section(s) "
                f"({active} with content): {section_ids}"
            )

            revisions = self._call_llm(group_label, pairs)
            if revisions is None:
                print(
                    f"[HARMONIZE] Group {group_key!r}: LLM call failed after "
                    f"{self.max_retries} attempt(s). Original text kept."
                )
                continue

            if not revisions:
                print(f"[HARMONIZE] Group {group_key!r}: no changes needed.")
                continue

            n_applied = 0
            for rev in revisions:
                section_id = str(rev.get("section_id", "")).strip()
                revised_text = str(rev.get("revised_text", "")).strip()
                notes = str(rev.get("notes", "")).strip()

                var = var_map.get(section_id)
                target = patched_map.get(section_id)

                if var is None or target is None:
                    if self.verbose:
                        print(
                            f"[HARMONIZE] Unknown section_id {section_id!r} "
                            "in LLM response — ignoring."
                        )
                    continue

                # Safety: locked phrases from required_text must survive.
                current_text = target.filled_template or target.answer or ""
                locked = extract_locked_phrases(var.required_text, current_text)
                if locked and not _all_phrases_present(revised_text, locked):
                    missing = [p for p in locked if p not in revised_text]
                    preview = "; ".join(f'"{p[:55]}"' for p in missing[:2])
                    print(
                        f"[HARMONIZE] Section {section_id}: revision rejected — "
                        f"required text would be altered. Missing: {preview}"
                    )
                    continue

                # Empty revised_text → mark section NOT_FOUND so it is omitted.
                if not revised_text:
                    if target.status in _CONTENT_STATUSES:
                        target.filled_template = ""
                        target.answer = ""
                        target.status = "NOT_FOUND"
                        target.confidence = "N/A"
                        target.notes = _append_note(
                            target.notes,
                            "Harmonization: no applicable content for this study.",
                        )
                    if self.verbose:
                        print(
                            f"[HARMONIZE] Section {section_id}: "
                            "cleared (no applicable content after redistribution)."
                        )
                    continue

                # If the LLM moved content INTO a previously empty section,
                # promote its status so it appears in the assembled document.
                if target.status not in _CONTENT_STATUSES:
                    target.status = "FOUND"
                    target.confidence = "MEDIUM"

                target.filled_template = revised_text
                if notes:
                    target.notes = _append_note(target.notes, f"Harmonized: {notes}")
                audit.setdefault(section_id, []).append(
                    notes or "Content redistributed by harmonizer."
                )
                n_applied += 1
                if self.verbose:
                    print(f"[HARMONIZE] Section {section_id}: revision applied.")

            print(
                f"[HARMONIZE] Group {group_key!r}: "
                f"{n_applied}/{len(revisions)} revision(s) applied."
            )

        return patched, audit

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        group_label: str,
        pairs: list[tuple[TemplateVariable, ExtractionResult | None]],
    ) -> list[dict] | None:
        """Issue the harmonization LLM call with retries. Returns revisions or None."""
        messages = build_harmonization_prompt(group_label, pairs)

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.client.completion(messages)
            except Exception as e:
                print(
                    f"[HARMONIZE] LLM error (attempt {attempt}/{self.max_retries}): "
                    f"{type(e).__name__}: {e}"
                )
                raw = None

            if raw:
                if self.verbose:
                    preview = raw[:500]
                    print(f"[HARMONIZE] Raw response ({len(raw)} chars): {preview}")
                result = _parse_response(raw)
                if result is not None:
                    return result
                print(
                    f"[HARMONIZE] Could not parse LLM response "
                    f"(attempt {attempt}/{self.max_retries})."
                )

            if attempt < self.max_retries:
                print(f"[HARMONIZE] Attempt {attempt}/{self.max_retries} failed. Retrying...")

        return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _all_phrases_present(text: str, phrases: list[str]) -> bool:
    return all(phrase in text for phrase in phrases)


def _append_note(existing: str, new_note: str) -> str:
    if existing and existing.strip():
        return f"{existing.strip()} | {new_note}"
    return new_note


def _parse_response(raw: str) -> list[dict] | None:
    """Extract a list[dict] from the LLM response.

    Three strategies, same pattern used throughout the codebase:
      1. Direct json.loads on the stripped string.
      2. Content of the first ```json ... ``` or ``` ... ``` fence.
      3. Outermost [ ... ] with balanced-bracket extraction.
    """
    if not raw:
        return None

    def _valid(data) -> list[dict] | None:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict) and "section_id" in item]
        return None

    # Strategy 1
    try:
        data = json.loads(raw.strip())
        result = _valid(data)
        if result is not None:
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: markdown code fence
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1).strip())
            result = _valid(data)
            if result is not None:
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: outermost [ ... ]
    start = raw.find("[")
    if start != -1:
        depth = 0
        for i, ch in enumerate(raw[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(raw[start : i + 1])
                        result = _valid(data)
                        if result is not None:
                            return result
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break

    return None

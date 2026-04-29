"""
Stage 9 — HIGH Flag Remediation.

After Stage 8 review produces ReviewFlags, RemediationEngine runs two passes:

  Pass A  One LLM call on cross_section_notes + flags to extract a list of
          GlobalFixRules (document-wide terminology / abbreviation fixes).
          Structural repetition is acknowledged as note_only and never auto-fixed.

  Pass B  One LLM call per affected section to patch the filled_template:
            - addresses all HIGH flags for that section
            - applies any applicable GlobalFixRules
          Sections reach the patch step via two routes:
            1. They contain at least one HIGH-severity ReviewFlag.
            2. They appear in the affected_section_ids of a non-note_only rule.

  After patching, a programmatic safety check verifies that all literal phrases
  from required_text survive verbatim.  If the check fails the patch is rejected
  and the original text is kept (success=False in the audit log).

Design mirrors adapt.py: direct LLM calls (no RLM REPL loop), deep-copy safety,
graceful failure never degrades an extraction.
"""

from __future__ import annotations

import copy
import json
import re

from icf.remediate_prompts import (
    build_global_rules_prompt,
    build_patch_prompt,
    extract_locked_phrases,
)
from icf.types import (
    ExtractionResult,
    GlobalFixRule,
    RemediationRecord,
    RemediationResult,
    ReviewFlag,
    ReviewResult,
    TemplateVariable,
)
from rlm.clients import get_client


class RemediationEngine:
    """Run Stage 9 HIGH-flag remediation over the assembled ICF.

    Reuses the same LLM backend configured for the pipeline.
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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_remediation(
        self,
        extractions: list[ExtractionResult],
        variables: list[TemplateVariable],
        review_result: ReviewResult,
    ) -> tuple[list[ExtractionResult], RemediationResult]:
        """Run Pass A then Pass B and return patched extractions + audit log.

        The returned extractions list is a deep copy with filled_template
        updated for successfully patched sections.  The input is never mutated.
        """
        var_map: dict[str, TemplateVariable] = {v.section_id: v for v in variables}
        ext_map: dict[str, ExtractionResult] = {e.section_id: e for e in extractions}

        # -- Pass A: extract document-wide fix rules ----------------------
        global_rules = self._extract_global_rules(review_result, variables)

        actionable_rules = [r for r in global_rules if r.rule_type != "note_only"]
        note_only_rules = [r for r in global_rules if r.rule_type == "note_only"]
        unaddressed_notes = (
            "; ".join(r.description for r in note_only_rules) if note_only_rules else ""
        )

        # -- Compute remediation scope -----------------------------------
        # HIGH-flagged section IDs (excluding standard_text sections).
        # Normalize here as a backstop for review flags loaded from older reports
        # where the Stage 8 LLM may have stored IDs with a "SECTION " prefix.
        standard_ids = {v.section_id for v in variables if v.is_standard_text}

        high_flagged_ids: set[str] = {
            _normalize_section_id(f.section_id)
            for f in review_result.flags
            if f.severity == "HIGH" and _normalize_section_id(f.section_id) not in standard_ids
        }

        # Sections pulled in by actionable global rules
        rule_section_ids: set[str] = {
            sid
            for rule in actionable_rules
            for sid in rule.affected_section_ids
            if sid not in standard_ids
        }

        scope = high_flagged_ids | rule_section_ids

        if self.verbose:
            print(f"[REMEDIATE] Scope: {sorted(scope)} ({len(scope)} section(s))")

        # -- Pass B: patch each section in scope -------------------------
        patched_extractions = copy.deepcopy(extractions)
        patched_ext_map: dict[str, ExtractionResult] = {
            e.section_id: e for e in patched_extractions
        }

        records: list[RemediationRecord] = []

        for section_id in sorted(scope, key=lambda s: (len(s), s)):
            ext = patched_ext_map.get(section_id) or ext_map.get(section_id)
            var = var_map.get(section_id)

            if ext is None or var is None:
                records.append(
                    RemediationRecord(
                        section_id=section_id,
                        high_flag_count=0,
                        global_rules_applied=[],
                        original_text="",
                        patched_text="",
                        success=False,
                        notes="Section not found in extractions or variables.",
                    )
                )
                continue

            # Only patch sections with actual generated content.
            if ext.status not in ("FOUND", "PARTIAL"):
                if self.verbose:
                    print(f"[REMEDIATE] Skip {section_id} (status={ext.status})")
                continue

            current_text = ext.filled_template or ext.answer or ""
            if not current_text.strip():
                continue

            locked_phrases = extract_locked_phrases(var.required_text)

            section_high_flags = [
                f
                for f in review_result.flags
                if _normalize_section_id(f.section_id) == section_id and f.severity == "HIGH"
            ]
            applicable_rules = [r for r in actionable_rules if section_id in r.affected_section_ids]

            original_text = current_text
            patched_text = self._patch_section(
                section_id=section_id,
                heading=var.get_display_name(),
                filled_template=current_text,
                locked_phrases=locked_phrases,
                high_flags=section_high_flags,
                applicable_rules=applicable_rules,
            )

            if patched_text is None:
                records.append(
                    RemediationRecord(
                        section_id=section_id,
                        high_flag_count=len(section_high_flags),
                        global_rules_applied=[r.description for r in applicable_rules],
                        original_text=original_text,
                        patched_text=original_text,
                        success=False,
                        notes="LLM patch call failed after retries.",
                    )
                )
                continue

            if not _validate_patch(patched_text, locked_phrases):
                # Find which phrases were dropped so the note is actionable.
                missing = [p for p in locked_phrases if p not in patched_text]
                missing_preview = "; ".join(f'"{p[:60]}"' for p in missing[:3])
                records.append(
                    RemediationRecord(
                        section_id=section_id,
                        high_flag_count=len(section_high_flags),
                        global_rules_applied=[r.description for r in applicable_rules],
                        original_text=original_text,
                        patched_text=original_text,
                        success=False,
                        notes=(
                            "Patch rejected: the fix would alter required/locked text. "
                            f"Missing phrase(s): {missing_preview}. "
                            "Human review required."
                        ),
                    )
                )
                continue

            # Apply patch to the deep-copied extraction.
            target = patched_ext_map.get(section_id)
            if target is not None:
                target.filled_template = patched_text

            records.append(
                RemediationRecord(
                    section_id=section_id,
                    high_flag_count=len(section_high_flags),
                    global_rules_applied=[r.description for r in applicable_rules],
                    original_text=original_text,
                    patched_text=patched_text,
                    success=True,
                )
            )

            if self.verbose:
                print(f"[REMEDIATE] Patched section {section_id} OK.")

        return patched_extractions, RemediationResult(
            records=records,
            global_rules=global_rules,
            unaddressed_notes=unaddressed_notes,
        )

    # ------------------------------------------------------------------
    # Pass A helpers
    # ------------------------------------------------------------------

    def _extract_global_rules(
        self,
        review_result: ReviewResult,
        variables: list[TemplateVariable],
    ) -> list[GlobalFixRule]:
        """Call the LLM to parse cross_section_notes into GlobalFixRules."""
        if not review_result.cross_section_notes.strip():
            return []

        messages = build_global_rules_prompt(
            cross_section_notes=review_result.cross_section_notes,
            flags=review_result.flags,
            variables=variables,
        )

        for attempt in range(1, self.max_retries + 1):
            rules = self._call_global_rules_llm(messages)
            if rules is not None:
                if self.verbose:
                    print(f"[REMEDIATE] Pass A: {len(rules)} global rule(s) extracted.")
                return rules
            if attempt < self.max_retries:
                print(
                    f"[REMEDIATE] Pass A attempt {attempt}/{self.max_retries} failed. Retrying..."
                )

        print("[REMEDIATE] Pass A failed after retries. Proceeding with HIGH flags only.")
        return []

    def _call_global_rules_llm(self, messages: list[dict]) -> list[GlobalFixRule] | None:
        try:
            raw = self.client.completion(messages)
        except Exception as e:
            print(f"[REMEDIATE] Pass A LLM error: {type(e).__name__}: {e}")
            return None

        if self.verbose:
            preview = raw[:400] if raw else "(empty)"
            print(f"[REMEDIATE] Pass A raw ({len(raw) if raw else 0} chars): {preview}")

        return _parse_global_rules_response(raw)

    # ------------------------------------------------------------------
    # Pass B helpers
    # ------------------------------------------------------------------

    def _patch_section(
        self,
        section_id: str,
        heading: str,
        filled_template: str,
        locked_phrases: list[str],
        high_flags: list[ReviewFlag],
        applicable_rules: list[GlobalFixRule],
    ) -> str | None:
        """Make the patch LLM call for one section. Returns patched text or None."""
        messages = build_patch_prompt(
            section_id=section_id,
            heading=heading,
            filled_template=filled_template,
            locked_phrases=locked_phrases,
            high_flags=high_flags,
            applicable_rules=applicable_rules,
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.client.completion(messages)
            except Exception as e:
                print(f"[REMEDIATE] Patch {section_id} LLM error: {type(e).__name__}: {e}")
                raw = None

            if raw and raw.strip():
                return raw.strip()

            if attempt < self.max_retries:
                print(
                    f"[REMEDIATE] Patch {section_id} attempt {attempt}/{self.max_retries} "
                    "returned empty. Retrying..."
                )

        return None


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

# Prefixes the LLM sometimes adds before a bare section ID.
_SECTION_PREFIX_RE = re.compile(r"^(?:SECTION|Section|section)\s+", re.IGNORECASE)


def _normalize_section_id(raw_id: str) -> str:
    """Strip accidental 'SECTION ' prefixes the LLM may add to section IDs.

    E.g. 'SECTION 3' -> '3', 'Section 9.2' -> '9.2', '21.1' -> '21.1'.
    """
    return _SECTION_PREFIX_RE.sub("", str(raw_id)).strip()


def _parse_global_rules_response(raw: str) -> list[GlobalFixRule] | None:
    """Extract a list[GlobalFixRule] from the LLM Pass A response."""
    if not raw:
        return None

    data = _extract_json_array(raw)
    if data is None:
        return None

    rules: list[GlobalFixRule] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        rule_type = str(item.get("rule_type", "note_only"))
        description = str(item.get("description", ""))
        affected = item.get("affected_section_ids", [])
        if not isinstance(affected, list):
            affected = []
        rules.append(
            GlobalFixRule(
                rule_type=rule_type,
                description=description,
                affected_section_ids=[_normalize_section_id(s) for s in affected],
            )
        )
    return rules


def _extract_json_array(raw: str) -> list | None:
    """Extract the first JSON array from an LLM response.

    Three strategies (same pattern as adapt.py):
      1. Direct json.loads on the stripped string.
      2. Content of the first ```json ... ``` or ``` ... ``` fence.
      3. Outermost [ ... ] with balanced-bracket extraction.
    """
    try:
        data = json.loads(raw.strip())
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1).strip())
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

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
                        if isinstance(data, list):
                            return data
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break

    return None


# ---------------------------------------------------------------------------
# Patch validation
# ---------------------------------------------------------------------------


def _validate_patch(patched_text: str, locked_phrases: list[str]) -> bool:
    """Return True if every locked phrase still appears verbatim in patched_text."""
    for phrase in locked_phrases:
        if phrase not in patched_text:
            return False
    return True

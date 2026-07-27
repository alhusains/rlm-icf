"""
Stage 9 — Review Flag Remediation.

After Stage 8 review produces ReviewFlags, RemediationEngine runs two passes:

  Pass A  One LLM call on cross_section_notes + flags to extract a list of
          GlobalFixRules for document-wide terminology fixes (standardize_term,
          fix_inconsistency; structural repetition is acknowledged as note_only
          and never auto-fixed). It deliberately does NOT generate abbreviation
          placement rules -- see icf/abbreviations.py's docstring for why an
          LLM guess about WHERE to define an abbreviation can contradict the
          deterministic scan below and leave it undefined everywhere.

          A deterministic, regex-based scan (icf/abbreviations.py) separately
          adds GlobalFixRules for redundant, out-of-order, or orphaned/garbled
          abbreviation definitions -- this is the ONLY source of
          define_abbreviation rules, since it is the only pass that sees the
          whole final document at once and can decide placement consistently.

  Pass B  One LLM call per affected section to patch the filled_template:
            - addresses all HIGH flags for that section
            - addresses eligible MEDIUM flags when remediate_medium is enabled
            - applies any applicable GlobalFixRules
          Sections reach the patch step via two routes:
            1. They contain at least one remediable ReviewFlag (HIGH always;
               MEDIUM only when issue_type is whitelisted or suggested_fix is set).
            2. They appear in the affected_section_ids of a non-note_only rule.

  After each patch attempt, a programmatic safety check verifies that all
  literal phrases from required_text AND suggested_text survive verbatim.  If
  the check fails, the model gets one corrective retry naming exactly which
  phrase(s) it dropped/altered, so a single overreaching edit doesn't discard
  every other valid fix in the section.  If it still fails after retries, the
  patch is rejected and the original text is kept (success=False in the audit
  log).

Section group 2.x (cover page fields: title, protocol #, study doctor,
sponsor, emergency contact) is excluded from remediation scope entirely --
these are short factual identifiers, never eligible for automatic patching.

Design mirrors adapt.py: direct LLM calls (no RLM REPL loop), deep-copy safety,
graceful failure never degrades an extraction.
"""

from __future__ import annotations

import copy
import json
import re

from icf.abbreviations import find_abbreviation_fixes
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
    normalize_section_id,
)
from rlm.clients import get_client

# MEDIUM flags auto-fixed only when issue_type is in this set OR suggested_fix is set.
# REPETITION, UNCLEAR, TONE at MEDIUM are never auto-fixed (too subjective / risky).
_MEDIUM_AUTO_FIX_ISSUE_TYPES = frozenset(
    {
        "PASSIVE_VOICE",
        "SENTENCE_TOO_LONG",
        "PLAIN_LANGUAGE_VIOLATION",
    }
)

# Top-level section group always excluded from remediation. 2.x holds cover-page
# fields (title, protocol #, study doctor, sponsor, emergency contact) -- short
# factual identifiers that must never be auto-patched.
_REMEDIATION_LOCKED_TOPS = frozenset({"2"})


def _is_remediation_locked(section_id: str) -> bool:
    top = (section_id or "").strip().split(".", 1)[0]
    return top in _REMEDIATION_LOCKED_TOPS


def _locked_phrases_for(var: TemplateVariable, current_text: str) -> list[str]:
    """Literal phrases from required_text AND suggested_text that must survive
    verbatim in the patched output (order-preserving union)."""
    phrases = extract_locked_phrases(var.required_text, current_text)
    for p in extract_locked_phrases(var.suggested_text, current_text):
        if p not in phrases:
            phrases.append(p)
    return phrases


def _is_remediable_medium(flag: ReviewFlag) -> bool:
    if flag.severity != "MEDIUM":
        return False
    if flag.suggested_fix.strip():
        return True
    return flag.issue_type in _MEDIUM_AUTO_FIX_ISSUE_TYPES


def _is_remediable_flag(flag: ReviewFlag, remediate_medium: bool) -> bool:
    if flag.severity == "HIGH":
        return True
    return remediate_medium and _is_remediable_medium(flag)


class RemediationEngine:
    """Run Stage 9 review-flag remediation over the assembled ICF.

    Reuses the same LLM backend configured for the pipeline.
    """

    def __init__(
        self,
        model_name: str,
        backend: str,
        backend_kwargs: dict | None = None,
        max_retries: int = 2,
        verbose: bool = False,
        remediate_medium: bool = True,
    ):
        self.max_retries = max_retries
        self.verbose = verbose
        self.remediate_medium = remediate_medium

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

        # Deterministic document-wide abbreviation check -- catches redundant
        # or out-of-order abbreviation definitions across sections regardless
        # of whether Stage 8 review happened to flag them (see abbreviations.py
        # for why this needs to be mechanical rather than LLM-detected).
        abbreviation_fixes = find_abbreviation_fixes(extractions, variables)
        if abbreviation_fixes and self.verbose:
            print(f"[REMEDIATE] Abbreviation check: {len(abbreviation_fixes)} fix(es) needed.")
        global_rules.extend(
            GlobalFixRule(
                rule_type="define_abbreviation",
                description=fix.instruction,
                affected_section_ids=[fix.section_id],
            )
            for fix in abbreviation_fixes
        )

        actionable_rules = [r for r in global_rules if r.rule_type != "note_only"]
        note_only_rules = [r for r in global_rules if r.rule_type == "note_only"]
        unaddressed_notes = (
            "; ".join(r.description for r in note_only_rules) if note_only_rules else ""
        )

        # -- Compute remediation scope -----------------------------------
        # HIGH-flagged section IDs (excluding standard_text and 2.x cover-page sections).
        # Normalize here as a backstop -- review.py already normalizes section_id
        # when parsing Stage 8 flags, but older cached ReviewResults (or a future
        # caller that constructs ReviewFlags directly) may not have gone through it.
        protected_ids = {
            v.section_id
            for v in variables
            if v.is_standard_text or _is_remediation_locked(v.section_id)
        }

        remediable_flagged_ids: set[str] = {
            normalize_section_id(f.section_id)
            for f in review_result.flags
            if _is_remediable_flag(f, self.remediate_medium)
            and normalize_section_id(f.section_id) not in protected_ids
        }

        # Sections pulled in by actionable global rules
        rule_section_ids: set[str] = {
            sid
            for rule in actionable_rules
            for sid in rule.affected_section_ids
            if sid not in protected_ids
        }

        scope = remediable_flagged_ids | rule_section_ids

        if self.verbose:
            mode = "HIGH + eligible MEDIUM" if self.remediate_medium else "HIGH only"
            print(f"[REMEDIATE] Scope ({mode}): {sorted(scope)} ({len(scope)} section(s))")

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

            locked_phrases = _locked_phrases_for(var, current_text)

            section_flags = [
                f
                for f in review_result.flags
                if normalize_section_id(f.section_id) == section_id
                and _is_remediable_flag(f, self.remediate_medium)
            ]
            applicable_rules = [r for r in actionable_rules if section_id in r.affected_section_ids]

            original_text = current_text
            patched_text, failure_notes = self._patch_section(
                section_id=section_id,
                heading=var.get_display_name(),
                filled_template=current_text,
                locked_phrases=locked_phrases,
                flags=section_flags,
                applicable_rules=applicable_rules,
            )

            if patched_text is None:
                records.append(
                    RemediationRecord(
                        section_id=section_id,
                        high_flag_count=len(section_flags),
                        global_rules_applied=[r.description for r in applicable_rules],
                        original_text=original_text,
                        patched_text=original_text,
                        success=False,
                        notes=failure_notes,
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
                    high_flag_count=len(section_flags),
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
        flags: list[ReviewFlag],
        applicable_rules: list[GlobalFixRule],
    ) -> tuple[str | None, str]:
        """Make the patch LLM call for one section, retrying on failure.

        Two distinct failure modes are retried differently:
          - Empty response: just re-issue the same prompt.
          - Locked-phrase violation: rather than discarding every fix in the
            section (throwing away good fixes because of one bad one), retry
            with the offending draft plus the specific missing phrase(s) so the
            model can redo it -- keeping required wording intact while still
            applying whichever fixes don't touch it.

        Returns (patched_text, failure_notes). failure_notes is empty on success.
        """
        messages = build_patch_prompt(
            section_id=section_id,
            heading=heading,
            filled_template=filled_template,
            locked_phrases=locked_phrases,
            flags=flags,
            applicable_rules=applicable_rules,
        )

        missing: list[str] = []
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self.client.completion(messages)
            except Exception as e:
                print(f"[REMEDIATE] Patch {section_id} LLM error: {type(e).__name__}: {e}")
                raw = None

            if not raw or not raw.strip():
                if attempt < self.max_retries:
                    print(
                        f"[REMEDIATE] Patch {section_id} attempt {attempt}/{self.max_retries} "
                        "returned empty. Retrying..."
                    )
                continue

            raw = raw.strip()
            missing = [p for p in locked_phrases if p not in raw]
            if not missing:
                return raw, ""

            if attempt < self.max_retries:
                print(
                    f"[REMEDIATE] Patch {section_id} attempt {attempt}/{self.max_retries} "
                    "dropped required wording. Retrying with corrective feedback..."
                )
                missing_block = "\n".join(f"  - {p}" for p in missing)
                messages = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "That revision dropped or altered required wording that must "
                            "survive unchanged. Missing/altered phrase(s):\n"
                            f"{missing_block}\n\n"
                            "Redo the revision: keep every phrase above exactly as written, "
                            "word-for-word, and still apply the requested fixes to any other "
                            "part of the text. If a specific fix cannot be made without "
                            "touching one of these phrases, skip that fix rather than "
                            "altering the required wording."
                        ),
                    },
                ]

        if missing:
            missing_preview = "; ".join(f'"{p[:60]}"' for p in missing[:3])
            return None, (
                "Patch rejected after retries: the fix would alter required/locked text. "
                f"Missing phrase(s): {missing_preview}. Human review required."
            )
        return None, "LLM patch call failed after retries."


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


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
                affected_section_ids=[normalize_section_id(s) for s in affected],
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

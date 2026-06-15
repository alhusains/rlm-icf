"""
Prompt building for Stage 8 — ICF Plain Language Review.

Two public functions:
  build_icf_document_for_review  — assembles the full ICF into flat text for
                                    the LLM to read, protecting standard-text
                                    sections from being flagged.
  build_review_messages          — returns a [system, user] messages list ready
                                    to pass to client.completion().
"""

from __future__ import annotations

from icf.plain_language import UHN_PLAIN_LANGUAGE_GUIDELINES
from icf.runtime_injections import SIGNATURE_CONSENT_SECTION_ID
from icf.types import ExtractionResult, TemplateVariable

# Statuses whose content should be included in the review document.
_REVIEWABLE_STATUSES = ("FOUND", "PARTIAL", "STANDARD_TEXT")

# Rough chars-per-token estimate; used for soft token-budget enforcement.
_CHARS_PER_TOKEN = 4
# Default budget: ~100 000 tokens worth of assembled ICF text.
_DEFAULT_TOKEN_BUDGET = 100_000

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

REVIEW_SYSTEM_PROMPT = (
    "You are a plain-language reviewer for Informed Consent Forms (ICFs) at UHN "
    "(University Health Network).\n\n"
    "You will receive the assembled text of a draft ICF, followed by the UHN Plain "
    "Language Guidelines. Your job is to annotate problems — NOT to rewrite anything.\n\n"
    "Rules:\n"
    "  1. Return ONLY a JSON object in the exact schema requested. No prose outside the JSON.\n"
    "  2. Never suggest edits that constitute a rewrite. The 'suggestion' field must be brief "
    "guidance (e.g. 'Consider active voice: You will receive …'), not replacement text.\n"
    "  3. Sections marked [STANDARD TEXT - DO NOT FLAG] are legally mandated verbatim wording. "
    "Do NOT generate any flags for those sections regardless of reading level or style.\n"
    "  4. The 'flagged_text' must be a short verbatim excerpt (≤ 30 words) copied exactly from "
    "the section content shown to you.\n"
    "  5. Focus on issues that the participant reading this form would actually notice: unclear "
    "language, unexplained jargon, passive voice, very long sentences, repeated information "
    "across sections, inconsistent terminology, or an unwelcoming tone. Note: the participant "
    "may be a patient, a clinician, a healthy volunteer, a caregiver, or another research subject "
    "— calibrate your expectations for technical language accordingly.\n"
    "  6. Severity guide: HIGH = likely confuses or misleads the participant; "
    "MEDIUM = noticeable problem but meaning is still clear; LOW = minor style issue.\n"
    "  7. suggested_fix: Provide a ready-to-copy plain-language replacement whenever "
    "the fix is local to the flagged excerpt. This field powers automated remediation — "
    "empty suggested_fix means the issue will likely stay in the draft.\n"
    "     • REQUIRED (non-empty) for HIGH and MEDIUM flags of types PLAIN_LANGUAGE_VIOLATION, "
    "PASSIVE_VOICE, SENTENCE_TOO_LONG, UNCLEAR, and TONE when a single excerpt can be "
    "rewritten without changing facts or crossing sections.\n"
    "     • REQUIRED for unexplained medical jargon, technical procedure names, dosing "
    "units, and formal words (e.g. 'inform', 'utilize') — give the simplified phrase "
    "participants should read.\n"
    "     • Leave empty ONLY when: (a) the issue is REPETITION or TERMINOLOGY_INCONSISTENCY "
    "that must be fixed consistently across multiple sections, (b) the span is legally "
    "mandated verbatim text you were told not to flag, or (c) no single excerpt replacement "
    "can fix the problem without omitting required facts.\n"
    "     • suggested_fix must replace only the flagged_text span (same facts, plainer words); "
    "the suggestion field stays brief guidance, not the replacement itself.\n"
)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def build_icf_document_for_review(
    extractions: list[ExtractionResult],
    variables: list[TemplateVariable],
) -> tuple[str, set[str]]:
    """Assemble the full ICF into flat text for the review LLM.

    Returns:
        (assembled_document_text, standard_text_section_ids)

    The assembled text has the form::

        === SECTION 3: INTRODUCTION ===
        You are being invited to participate in a research study ...

        === SECTION 2.1: STUDY CONTACTS ===
        [STANDARD TEXT - DO NOT FLAG]
        If you have questions about this study ...

    Only sections with status FOUND, PARTIAL, or STANDARD_TEXT are included.
    Sections with status SKIPPED, NOT_FOUND, or ERROR are omitted from the
    review document (there is no generated text to review).
    """
    ext_map: dict[str, ExtractionResult] = {e.section_id: e for e in extractions}

    standard_text_ids: set[str] = set()
    parts: list[str] = []

    # Iterate in registry order for natural document flow.
    for var in variables:
        if var.section_id == SIGNATURE_CONSENT_SECTION_ID:
            continue
        ext = ext_map.get(var.section_id)
        if ext is None or ext.status not in _REVIEWABLE_STATUSES:
            continue

        heading = var.heading
        if var.sub_section:
            heading += f" — {var.sub_section}"

        header = f"=== SECTION {var.section_id}: {heading.upper()} ==="

        if var.is_standard_text:
            standard_text_ids.add(var.section_id)
            text = ext.filled_template or ext.answer or ""
            parts.append(f"{header}\n[STANDARD TEXT - DO NOT FLAG]\n{text.strip()}")
        else:
            text = ext.filled_template or ext.answer or ""
            if text.strip():
                parts.append(f"{header}\n{text.strip()}")

    assembled = "\n\n".join(parts)
    return assembled, standard_text_ids


def build_review_messages(
    icf_document: str,
    standard_text_ids: set[str],
    token_budget: int = _DEFAULT_TOKEN_BUDGET,
) -> list[dict]:
    """Build the [system, user] messages list for the review LLM call.

    If the assembled ICF text exceeds the token budget, it is truncated and a
    visible warning banner is prepended so the LLM knows the review is partial.
    """
    char_budget = token_budget * _CHARS_PER_TOKEN

    if len(icf_document) > char_budget:
        icf_document = (
            "[WARNING: ICF document was truncated to fit the token budget. "
            "The review below covers only the first portion of the document.]\n\n"
            + icf_document[:char_budget]
        )

    protected_list = (
        ", ".join(sorted(standard_text_ids)) if standard_text_ids else "(none)"
    )

    user_content = (
        f"PROTECTED SECTIONS — DO NOT FLAG THESE: {protected_list}\n\n"
        f"{UHN_PLAIN_LANGUAGE_GUIDELINES}\n\n"
        "=== ICF DOCUMENT TO REVIEW ===\n"
        f"{icf_document}\n"
        "=== END OF DOCUMENT ===\n\n"
        "OUTPUT — respond with ONLY this JSON object, nothing else:\n"
        "{\n"
        '  "flags": [\n'
        "    {\n"
        '      "section_id": "...",\n'
        '      "flagged_text": "short verbatim excerpt (≤ 30 words) from the section",\n'
        '      "issue_type": "REPETITION | PASSIVE_VOICE | SENTENCE_TOO_LONG | '
        'TERMINOLOGY_INCONSISTENCY | UNCLEAR | TONE | PLAIN_LANGUAGE_VIOLATION",\n'
        '      "suggestion": "brief explanation of the issue",\n'
        '      "severity": "HIGH | MEDIUM | LOW",\n'
        '      "suggested_fix": "Ready-to-copy plain-language replacement for flagged_text '
        'only (Grade 6–8, active voice, short sentences). REQUIRED for HIGH/MEDIUM '
        'PLAIN_LANGUAGE_VIOLATION, PASSIVE_VOICE, SENTENCE_TOO_LONG, UNCLEAR, and TONE '
        'when the fix fits in one excerpt. REQUIRED for jargon and unexplained technical '
        'terms — define or simplify in place. Empty string ONLY for cross-section '
        'REPETITION/TERMINOLOGY_INCONSISTENCY or when no local rewrite is possible."\n'
        "    }\n"
        "  ],\n"
        '  "cross_section_notes": "Overall observations about terminology consistency, '
        'repeated information, or structural issues spanning multiple sections. '
        'Write an empty string if there are no cross-section issues."\n'
        "}"
    )

    return [
        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

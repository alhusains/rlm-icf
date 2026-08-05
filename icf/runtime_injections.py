"""
Runtime study-context injections for ICF extraction and document assembly.

User-selected flags (CLI or web UI) append ``adaptation_notes`` on specific registry
sections and adjust document output (e.g. signature pages). Add new injections here
rather than one-off pipeline branches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from icf.types import ExtractionResult, TemplateVariable

# ---------------------------------------------------------------------------
# US federal funding (sections 21.2 minimal / 21.3 standard)
# ---------------------------------------------------------------------------

US_FUNDING_PARAGRAPH_SNIPPET = (
    "Your study data {{and/or samples}} will not be used or shared with other "
    "researchers for future studies, even if the researchers remove any information "
    "that could directly identify you."
)

US_FUNDING_RUNTIME_NOTE = (
    "This study IS funded or supported by a US federal funding agency (e.g., NIH, "
    "DHHS). Include the US federal funding paragraph from suggested_text (the block "
    f'beginning with "{US_FUNDING_PARAGRAPH_SNIPPET[:40]}...") in this section\'s '
    "filled_template. Place it as the last paragraph of this section when other "
    "suggested_text blocks also apply. Resolve {{and/or samples}} from the protocol "
    "(use biological-sample wording only if the study collects samples). Omit that "
    "paragraph only if the protocol explicitly states specimens or study information "
    "WILL be used or shared for future research beyond this study."
)

# ---------------------------------------------------------------------------
# Substitute decision maker — section 3 (INTRODUCTION)
# ---------------------------------------------------------------------------

SDM_INTRO_SECTION_ID = "3"

SDM_OPENING_PARAGRAPH = (
    "As a Substitute Decision Maker, you are being asked to provide informed consent "
    "on behalf of a person who is unable to provide consent for themselves.  If the "
    "participant gains the capacity to consent for themselves, your consent for them "
    'will end.  Throughout this form, "you" means the person you are representing.'
)

SDM_INTRO_RUNTIME_NOTE = (
    "This ICF is for completion by a Substitute Decision Maker (SDM) — confirmed by "
    "the study team, not inferred from the protocol.\n\n"
    "You MUST prepend the following paragraph verbatim as the first paragraph of "
    "filled_template (then a blank line, then the main introduction from required_text):\n\n"
    f"{SDM_OPENING_PARAGRAPH}\n\n"
    "Do NOT search the protocol to decide whether SDM applies.\n"
    "Do NOT leave any <<...>> conditional markers in filled_template.\n"
    "Still extract from the protocol: population description, study vs clinical trial "
    "wording (if applicable), decision timeframe, voluntary participation, and other "
    "required_text placeholders."
)

# ---------------------------------------------------------------------------
# Substitute decision maker — section 32 (signature page consent attestation)
# ---------------------------------------------------------------------------

SIGNATURE_CONSENT_SECTION_ID = "32"

PARTICIPANT_SIGNATURE_FINAL_BULLET = "I agree to take part in this study."

SDM_SIGNATURE_FINAL_BULLET = (
    "I agree, or agree to allow the person I am responsible for, to take part in "
    "this study."
)

SDM_SIGNATURE_RUNTIME_NOTE = (
    "This ICF is for completion by a Substitute Decision Maker (SDM) — confirmed by "
    "the study team, not inferred from the protocol.\n\n"
    "The final consent attestation bullet MUST read exactly:\n"
    f"• {SDM_SIGNATURE_FINAL_BULLET}\n\n"
    "Do NOT search the protocol to decide whether SDM applies.\n"
    "Do NOT use the default participant-only agreement wording."
)

_CONSENT_BULLET_LINE_RE = re.compile(r"^[•\-–\*·]\s+")


@dataclass(frozen=True)
class StudyRuntimeFlags:
    """User-selected study context from CLI flags or the web UI."""

    us_funded: bool = False
    sdm: bool = False


def _append_runtime_note(var: TemplateVariable, note: str) -> None:
    if var.adaptation_notes:
        var.adaptation_notes = f"{var.adaptation_notes.strip()}\n\n{note}"
    else:
        var.adaptation_notes = note


def is_us_funding_future_research_section(var: TemplateVariable) -> bool:
    """True when this registry section carries the US future-research paragraph."""
    blob = f"{var.suggested_text}\n{var.instructions}"
    return US_FUNDING_PARAGRAPH_SNIPPET[:50] in blob or (
        "US federal funding agency" in blob
        and "will not be used or shared with other researchers for future studies" in blob
    )


def is_sdm_intro_section(var: TemplateVariable) -> bool:
    return var.section_id == SDM_INTRO_SECTION_ID


def runtime_locked_phrases(var: TemplateVariable, filled_template: str) -> list[str]:
    """Verbatim runtime-injected paragraphs that must survive review/remediation.

    These are not stored in registry required_text/suggested_text — they come
    only from runtime injections (e.g. the SDM opening paragraph in section 3).
    Include a phrase only when it is already present in the draft, so unused
    conditionals never block editing.
    """
    if not filled_template or not filled_template.strip():
        return []
    phrases: list[str] = []
    if is_sdm_intro_section(var):
        normalized_draft = " ".join(filled_template.split())
        normalized_sdm = " ".join(SDM_OPENING_PARAGRAPH.split())
        if normalized_sdm in normalized_draft:
            phrases.append(SDM_OPENING_PARAGRAPH)
    return phrases


def is_sdm_signature_consent_section(var: TemplateVariable) -> bool:
    return var.section_id == SIGNATURE_CONSENT_SECTION_ID and (
        var.sub_section or ""
    ).lower() == "consent"


def apply_sdm_injections(variables: list[TemplateVariable]) -> list[str]:
    """Inject SDM context into section 3 and signature consent (section 32)."""
    logs: list[str] = []
    for var in variables:
        if is_sdm_intro_section(var):
            _append_runtime_note(var, SDM_INTRO_RUNTIME_NOTE)
            logs.append(f"section {var.section_id} (INTRODUCTION)")
        if is_sdm_signature_consent_section(var):
            _append_runtime_note(var, SDM_SIGNATURE_RUNTIME_NOTE)
            logs.append(f"section {var.section_id} (signature consent)")
    return logs


def apply_us_funding_injections(variables: list[TemplateVariable]) -> list[str]:
    """Inject the US-funding future-research note into the relevant section(s)."""
    logs: list[str] = []
    for var in variables:
        if not is_us_funding_future_research_section(var):
            continue
        _append_runtime_note(var, US_FUNDING_RUNTIME_NOTE)
        logs.append(f"section {var.section_id}")
    return logs


def apply_runtime_injections(
    variables: list[TemplateVariable], flags: StudyRuntimeFlags
) -> list[str]:
    """Apply all user-flag-driven section injections before extraction.

    This is the single extension point for adapting specific sections based on
    options the user selects in the UI/CLI (currently SDM and US federal
    funding). Add new flag-driven injections here.
    """
    messages: list[str] = []
    if flags.sdm:
        sections = apply_sdm_injections(variables)
        if sections:
            messages.append(
                "SDM form: context wired to " + ", ".join(sections)
            )
    if flags.us_funded:
        sections = apply_us_funding_injections(variables)
        if sections:
            messages.append(
                "US-funded future-research paragraph wired to " + ", ".join(sections)
            )
    return messages


def prompt_runtime_context(var: TemplateVariable) -> str:
    """Optional prompt block from adaptation_notes (runtime injections)."""
    if not var.adaptation_notes:
        return ""
    return (
        "RUNTIME STUDY CONTEXT (confirmed by the study team — apply in addition to "
        "protocol search; overrides inferring these flags from the protocol alone):\n"
        f"{var.adaptation_notes.strip()}\n\n"
    )


def parse_consent_attestation_bullets(text: str) -> list[str]:
    """Parse bullet lines from a signature consent attestation block."""
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _CONSENT_BULLET_LINE_RE.match(stripped)
        if match:
            items.append(stripped[match.end() :].strip())
    return items


def apply_signature_final_bullet(bullets: list[str], *, sdm: bool) -> list[str]:
    """Replace the last consent bullet with the participant or SDM agreement line."""
    if not bullets:
        return bullets
    final = SDM_SIGNATURE_FINAL_BULLET if sdm else PARTICIPANT_SIGNATURE_FINAL_BULLET
    return bullets[:-1] + [final]


def default_signature_consent_bullets(*, sdm: bool) -> list[str]:
    """Fallback consent attestation bullets when section 32 extraction is unavailable."""
    return apply_signature_final_bullet(
        [
            "All of my questions have been answered",
            "I allow access to medical records and related personal health information "
            "as explained in this consent form",
            "I do not give up any legal rights by signing this consent form,",
            PARTICIPANT_SIGNATURE_FINAL_BULLET,
        ],
        sdm=sdm,
    )


def resolve_signature_consent_bullets(
    ext_map: dict[str, ExtractionResult],
    *,
    sdm: bool,
) -> list[str]:
    """Resolve signature-page consent bullets from section 32, with SDM override."""
    ext = ext_map.get(SIGNATURE_CONSENT_SECTION_ID)
    if ext is not None and ext.status in {"FOUND", "PARTIAL", "STANDARD_TEXT"}:
        text = (ext.filled_template or ext.answer or "").strip()
        if text:
            bullets = parse_consent_attestation_bullets(text)
            if bullets:
                return apply_signature_final_bullet(bullets, sdm=sdm)
    return default_signature_consent_bullets(sdm=sdm)

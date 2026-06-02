"""
Runtime study-context injections for ICF extraction and document assembly.

User-selected flags (CLI or web UI) append ``adaptation_notes`` on specific registry
sections and adjust document output (e.g. signature pages). Add new injections here
rather than one-off pipeline branches.
"""

from __future__ import annotations

from dataclasses import dataclass

from icf.types import TemplateVariable

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
    "required_text placeholders — but ignore any SDM conditional in suggested_text."
)


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
    return var.section_id == SDM_INTRO_SECTION_ID and "Substitute Decision Maker" in (
        f"{var.suggested_text}\n{var.instructions}"
    )


def apply_sdm_injections(variables: list[TemplateVariable]) -> list[str]:
    """Inject SDM context into section 3 before extraction (incl. Phase A triggers)."""
    logs: list[str] = []
    for var in variables:
        if not is_sdm_intro_section(var):
            continue
        _append_runtime_note(var, SDM_INTRO_RUNTIME_NOTE)
        logs.append(f"section {var.section_id} (INTRODUCTION)")
    return logs


def apply_us_funding_injections(variables: list[TemplateVariable]) -> list[str]:
    """Inject US-funding context; call after adaptation so optional 21.3 is not skipped."""
    logs: list[str] = []
    for var in variables:
        if not is_us_funding_future_research_section(var):
            continue
        var.adaptation_skipped = False
        _append_runtime_note(var, US_FUNDING_RUNTIME_NOTE)
        logs.append(f"section {var.section_id}")
    return logs


def apply_pre_extraction_injections(
    variables: list[TemplateVariable], flags: StudyRuntimeFlags
) -> list[str]:
    """Injections that must be visible before any extraction (e.g. SDM on section 3)."""
    messages: list[str] = []
    if flags.sdm:
        sections = apply_sdm_injections(variables)
        if sections:
            messages.append(
                "SDM form: introduction paragraph wired to " + ", ".join(sections)
            )
    return messages


def apply_post_adaptation_injections(
    variables: list[TemplateVariable], flags: StudyRuntimeFlags
) -> list[str]:
    """Injections applied after the adaptation pass (e.g. US-funded 21.2/21.3)."""
    messages: list[str] = []
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

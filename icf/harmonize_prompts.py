"""
Prompt building for the Section Group Harmonization pass.

One public function:
  build_harmonization_prompt  — builds the [system, user] messages list for a
                                single section-group harmonization LLM call.

The LLM receives the fully extracted text for every sub-section in the group
and returns a JSON array describing which sub-sections need revised content.
"""

from __future__ import annotations

from icf.plain_language import PLAIN_LANGUAGE_SCOPE, UHN_PLAIN_LANGUAGE_GUIDELINES
from icf.types import ExtractionResult, TemplateVariable

# Truncation limits applied to template fields in the prompt to keep token
# usage bounded.  Instructions are kept full (they define what belongs where).
# Suggested text is truncated since it is only a format reference.
_SUGGESTED_TEXT_PREVIEW = 350

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_HARMONIZE_SYSTEM = (
    "You are a clinical document editor harmonizing a group of related sub-sections "
    "in a draft Informed Consent Form (ICF) at UHN (University Health Network).\n\n"
    "Each sub-section was extracted independently from the same study protocol, so "
    "the same information may appear in multiple sub-sections, or content may have "
    "landed in the wrong sub-section.\n\n"
    "Your task:\n"
    "  1. Read the current extracted text for every sub-section in the group.\n"
    "  2. Redistribute content so each piece of information appears in exactly the\n"
    "     right sub-section (as defined by its Instructions) and nowhere else.\n"
    "  3. Remove duplicated content that appears in more than one sub-section.\n"
    "  4. Return revised text ONLY for sub-sections that need changes.\n\n"
    "Hard constraints:\n"
    "  • DO NOT add any clinical information not already present in the extracted drafts.\n"
    "    You may only move, trim, or lightly rephrase existing content.\n"
    "  • DO NOT drop any clinical fact entirely — move it to the correct sub-section.\n"
    "  • REQUIRED TEXT: if a sub-section shows 'REQUIRED TEXT', those exact words must\n"
    "    appear verbatim in your revised output for that sub-section.\n"
    "  • SUGGESTED TEXT is the UHN template format — use it as a structural guide for\n"
    "    phrasing and layout when rewriting content for a sub-section.\n"
    "  • If a sub-section has no applicable content for this study after harmonization,\n"
    "    return an empty string for revised_text (it will be marked NOT_FOUND).\n"
    "  • Follow the UHN Plain Language Guidelines for all text you write or revise.\n"
    "  • Return ONLY a JSON array — no prose, no preamble, no explanation.\n\n"
    + PLAIN_LANGUAGE_SCOPE
    + "\n"
    + UHN_PLAIN_LANGUAGE_GUIDELINES
)


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def build_harmonization_prompt(
    group_label: str,
    subsections: list[tuple[TemplateVariable, ExtractionResult | None]],
) -> list[dict]:
    """Build the [system, user] messages list for a harmonization LLM call.

    Args:
        group_label: Human-readable group heading (e.g., "WHAT ARE THE STUDY PROCEDURES?").
        subsections: Ordered list of (variable, extraction_result) pairs for the group.
                     extraction_result is None if the section was not in the run.
    """
    section_blocks: list[str] = []

    for var, ext in subsections:
        sub_label = f"Section {var.section_id}"
        if var.sub_section:
            sub_label += f" — {var.sub_section}"
        sub_label += f"  ({'Required' if var.required else 'Optional'})"

        # Instructions — kept full: they define what belongs in this sub-section.
        instructions_block = f"Instructions:\n{var.instructions.strip()}"

        # Required text — must be preserved verbatim.
        if var.required_text.strip():
            required_block = (
                "REQUIRED TEXT (copy verbatim into your revised output):\n"
                + var.required_text.strip()
            )
        else:
            required_block = "REQUIRED TEXT: (none)"

        # Suggested text — format/phrasing reference, truncated.
        if var.suggested_text.strip():
            st = var.suggested_text.strip()
            truncated = len(st) > _SUGGESTED_TEXT_PREVIEW
            st_preview = st[:_SUGGESTED_TEXT_PREVIEW] + (" ... [truncated]" if truncated else "")
            suggested_block = f"Suggested text format (follow this structure):\n{st_preview}"
        else:
            suggested_block = "Suggested text format: (none)"

        # Current extracted content.
        if ext is not None and ext.status in ("FOUND", "PARTIAL"):
            current = (ext.filled_template or ext.answer or "").strip()
            if current:
                content_block = (
                    f"Current extracted text (status={ext.status}):\n{current}"
                )
            else:
                content_block = f"Current extracted text: (empty — status={ext.status})"
        elif ext is not None:
            content_block = f"Current extracted text: (none — status={ext.status})"
        else:
            content_block = "Current extracted text: (none — section not in this run)"

        section_blocks.append(
            "\n".join([
                f"=== {sub_label} ===",
                instructions_block,
                required_block,
                suggested_block,
                content_block,
            ])
        )

    all_ids = [var.section_id for var, _ in subsections]
    valid_ids_line = (
        "VALID SECTION IDs (use these exact strings in your JSON): "
        + ", ".join(all_ids)
    )

    user_content = (
        f"SECTION GROUP: {group_label.upper()}\n\n"
        f"{valid_ids_line}\n\n"
        + "\n\n".join(section_blocks)
        + "\n\n"
        "TASK:\n"
        "Redistribute and de-duplicate the content above so that:\n"
        "  • Each procedure, test, or piece of information appears in exactly one sub-section.\n"
        "  • Each sub-section contains only what its Instructions say it should contain.\n"
        "  • No clinical fact is lost — move it to the correct sub-section if misplaced.\n"
        "  • Required text wording is preserved verbatim in the sub-section it belongs to.\n"
        "  • Suggested text format is followed closely for structure and phrasing.\n"
        "  • Plain language guidelines are applied throughout.\n"
        "  • Sub-sections with no applicable content have empty revised_text.\n"
        "\n"
        "Return ONLY a JSON array. Include an entry ONLY for sub-sections whose text\n"
        "needs to change. Return [] if no redistribution is needed.\n"
        "\n"
        "[\n"
        "  {\n"
        '    "section_id": "12.1",\n'
        '    "revised_text": "The complete revised patient-facing text for this sub-section.",\n'
        '    "notes": "One sentence describing what was changed (e.g., moved blood draw details here from 12.2)."\n'
        "  }\n"
        "]"
    )

    return [
        {"role": "system", "content": _HARMONIZE_SYSTEM},
        {"role": "user", "content": user_content},
    ]

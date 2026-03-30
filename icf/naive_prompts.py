"""
Prompt templates for the naive (full-context) extraction backend.

The entire protocol text is embedded directly in the user message so the LLM
can read it in a single pass.  No REPL, no code execution, no iteration — just
one structured LLM call per ICF section.

A 'reasoning' field is placed first in the JSON schema so the model can think
through the relevant protocol passages before committing to extraction values
(chain-of-thought inside structured output).
"""

from icf.plain_language import PLAIN_LANGUAGE_SCOPE, UHN_PLAIN_LANGUAGE_GUIDELINES
from icf.types import TemplateVariable

# ---------------------------------------------------------------------------
# Shared symbol guide (same rules as the RLM prompt)
# ---------------------------------------------------------------------------

_SYMBOL_GUIDE = (
    "TEMPLATE SYMBOL GUIDE — read carefully before processing the template text below:\n"
    "  {{placeholder}}         → REQUIRED fill-in. Replace the entire {{...}} token with\n"
    "                            study-specific text from the protocol. The {{...}} markers\n"
    "                            must NOT appear in your output.\n"
    "  {{option1/option2}}     → CHOOSE ONE. Pick the applicable option from the slash-separated\n"
    "                            list (e.g., {{will/may}} → 'will' or 'may'). The {{...}}\n"
    "                            markers must NOT appear in your output.\n"
    "  <<Condition block>>     → CONDITIONAL SECTION (double angle brackets). Include the text\n"
    "                            that follows ONLY if the stated condition applies to this study.\n"
    "                            Remove the <<...>> marker itself — it must NEVER appear in the\n"
    "                            final ICF text.\n"
    "  <Condition label>       → CONDITIONAL SENTENCE/PARAGRAPH (single angle brackets). Same\n"
    "                            rule: include only if the condition applies; strip the <...>\n"
    "                            marker from the output.\n"
    "  OR (standalone line)    → ALTERNATIVE. Choose exactly ONE of the blocks immediately above\n"
    "                            or below this marker. Do not include both, and do not include\n"
    "                            the word 'OR' itself in the final text.\n"
    "  • or -                  → BULLET POINT. Both are used interchangeably as list items.\n\n"
    "OUTPUT RULE: The filled_template field must contain clean ICF prose — no <<...>>, <...>,\n"
    "{{...}}, or standalone OR lines remaining.\n"
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

NAIVE_SYSTEM_PROMPT = (
    "You are a Clinical Data Extraction Specialist producing Informed Consent Form (ICF) "
    "content for a clinical study at UHN (University Health Network).\n\n"
    "You will receive the full text of a clinical study protocol, followed by a specific ICF "
    "section to extract. Your job is to:\n"
    "  1. Read the protocol carefully.\n"
    "  2. Find all relevant passages for the requested ICF section.\n"
    "  3. Return a JSON object with your reasoning and the extracted content.\n\n"
    "Core rules:\n"
    "  • 'filled_template' is READ BY PATIENTS. It must contain ONLY: required ICF wording "
    "(with placeholders filled), protocol information, and [TO BE FILLED MANUALLY] for "
    "missing fields. NEVER include sentences about what was or wasn't found, references to "
    "'the protocol', 'study documents', or any internal process. Put internal notes in 'notes'.\n"
    "  • Do NOT fabricate information. If information is not in the protocol, say NOT_FOUND.\n"
    "  • Every evidence quote must be a verbatim substring from the protocol text.\n"
    "  • The 'filled_template' must be clean ICF prose — no template markers remaining.\n"
    "  • If only partial information is found, use status='PARTIAL' and note what is missing.\n"
    "  • For unfillable placeholders, write [TO BE FILLED MANUALLY] — never explain why.\n\n"
    "UHN PLAIN LANGUAGE GUIDELINES — apply these when generating any text:\n"
    + PLAIN_LANGUAGE_SCOPE
    + UHN_PLAIN_LANGUAGE_GUIDELINES
    + "\n"
)

# ---------------------------------------------------------------------------
# Availability note (same logic as RLM prompts)
# ---------------------------------------------------------------------------


def _availability_note(var: TemplateVariable) -> str:
    if not var.is_in_protocol:
        return (
            "IMPORTANT: This information is typically NOT found in clinical protocols and "
            "requires manual entry by the study team. Search the protocol briefly — if you "
            "cannot find explicit evidence, return status='NOT_FOUND' immediately. "
            "Do NOT fabricate information."
        )
    if var.partially_in_protocol:
        return (
            "NOTE: Some fields in this section may not be in the protocol and require manual "
            "entry. Extract what you can find, mark unfound fields as [TO BE FILLED MANUALLY], "
            "and use status='PARTIAL' if only some information is found."
        )
    return (
        "This information should be findable in the protocol. "
        "Search thoroughly before concluding NOT_FOUND."
    )


# ---------------------------------------------------------------------------
# JSON output schema
# ---------------------------------------------------------------------------

_JSON_SCHEMA = """{
    "reasoning": "Your step-by-step thought process: which protocol sections/pages you looked at, what you found, what decisions you made for conditional template blocks (<<...>>, <...>, OR alternatives), and why you chose the status and confidence level you did.",
    "section_id": "{section_id}",
    "status": "FOUND" | "PARTIAL" | "NOT_FOUND",
    "answer": "Extracted information following UHN Plain Language Guidelines (patient-facing).",
    "filled_template": "PATIENT-FACING OUTPUT. Required ICF wording with all {{placeholders}} filled from the protocol, <<conditions>> resolved, OR alternatives chosen. Contains ONLY protocol information and [TO BE FILLED MANUALLY] for genuinely missing fields — never sentences about the extraction process or references to the protocol/study documents.",
    "evidence": [
        {{"quote": "Exact verbatim quote from the protocol text", "page": "Page number (from --- PAGE X --- markers)", "section": "Protocol section heading if identifiable"}}
    ],
    "confidence": "HIGH" | "MEDIUM" | "LOW",
    "notes": "Any caveats, items needing manual review, or decisions made during extraction."
}"""


# ---------------------------------------------------------------------------
# Main prompt builder
# ---------------------------------------------------------------------------


def build_naive_extraction_task(var: TemplateVariable) -> str:
    """Build the extraction task portion of the user message (without protocol text).

    The protocol text is prepended separately in build_naive_messages() so this
    function can be used independently for logging / inspection.
    """
    sub = f" > {var.sub_section}" if var.sub_section else ""
    availability = _availability_note(var)
    importance = (
        "REQUIRED — this section must appear in every ICF."
        if var.required
        else "OPTIONAL — include only if directly relevant to this specific study."
    )
    schema = _JSON_SCHEMA.replace("{section_id}", var.section_id)

    lines: list[str] = [
        f"=== EXTRACTION TASK: ICF Section [{var.section_id}] ===\n",
        f"TARGET: {var.heading}{sub}",
        f"WHAT TO EXTRACT: {var.instructions}\n",
        f"AVAILABILITY: {availability}",
        f"IMPORTANCE: {importance}\n",
        _SYMBOL_GUIDE,
    ]

    if var.required_text:
        lines.append(
            "REQUIRED ICF TEXT (mandatory wording — resolve all template markers and "
            "fill in all {{placeholders}} with study-specific content from the protocol):\n"
            f"{var.required_text}\n"
        )

    if var.suggested_text:
        lines.append(
            "SUGGESTED ICF TEXT (use as a starting point — adapt to this study by "
            "resolving all conditional blocks and filling placeholders):\n"
            f"{var.suggested_text}\n"
        )

    lines.append(
        "CHAIN-OF-THOUGHT REQUIREMENT: Fill in the 'reasoning' field FIRST with your "
        "step-by-step thought process before filling in any other field. "
        "Reference specific page numbers and protocol sections in your reasoning.\n"
    )
    lines.append(f"OUTPUT — respond with ONLY this JSON object, nothing else:\n{schema}")

    return "\n".join(lines)


def build_naive_messages(var: TemplateVariable, protocol_text: str) -> list[dict]:
    """Build the full messages list for a single naive extraction call.

    Structure:
      system  — role + core rules
      user    — protocol text (labelled) + extraction task
    """
    task = build_naive_extraction_task(var)

    user_content = (
        "=== FULL CLINICAL STUDY PROTOCOL ===\n"
        "(Pages are delimited by --- PAGE X --- markers. "
        "Use these markers to record page numbers in your evidence quotes.)\n\n"
        f"{protocol_text}\n\n"
        "=== END OF PROTOCOL ===\n\n"
        f"{task}"
    )

    return [
        {"role": "system", "content": NAIVE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

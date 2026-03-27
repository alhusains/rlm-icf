"""
Prompt templates for the RAG extraction backend.

Instead of the full protocol text (naive backend) or an active REPL session
(RLM backend), the RAG backend passes a compact set of pre-retrieved, pre-ranked
protocol passages to the generator LLM.

Design choices
--------------
- A 'reasoning' field is placed first in the JSON schema so the model can
  think through what each retrieved passage says before populating the
  extraction fields (chain-of-thought inside structured generation).
- Retrieved passages are labelled with [Pages X-Y, Section: Z] so the model
  can attribute quotes to specific page numbers without re-reading the whole
  protocol.
- The system prompt is concise: no REPL execution model to explain, no
  FINAL_VAR mechanics — just role + rules.
- The symbol guide ({{placeholders}}, <<conditions>>, OR alternatives) is
  identical across all three backends for consistency.
"""

from __future__ import annotations

from icf.plain_language import PLAIN_LANGUAGE_SCOPE, UHN_PLAIN_LANGUAGE_GUIDELINES
from icf.rag_index import Chunk
from icf.types import TemplateVariable

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = (
    "You are a Clinical Data Extraction Specialist producing Informed Consent Form (ICF) "
    "content for a clinical study at UHN (University Health Network).\n\n"
    "You will receive a set of passages retrieved from a clinical study protocol, "
    "followed by a specific ICF section to fill in. Your job is to:\n"
    "  1. Carefully read ALL retrieved passages — the answer may span multiple passages.\n"
    "  2. Identify relevant information and note its source (page number).\n"
    "  3. Return a JSON object: start with your reasoning, then the extraction fields.\n\n"
    "Core rules:\n"
    "  • 'filled_template' is READ BY PATIENTS. It must contain ONLY: required ICF wording "
    "(with placeholders filled), protocol information, and [TO BE FILLED MANUALLY] for "
    "missing fields. NEVER include sentences about what was or wasn't found, references to "
    "'the protocol', 'study documents', or any internal process. Put internal notes in 'notes'.\n"
    "  • Do NOT fabricate information. If a field cannot be answered from the passages,\n"
    "    write [TO BE FILLED MANUALLY] or return status='NOT_FOUND'.\n"
    "  • Every evidence quote must be a verbatim substring from one of the retrieved passages.\n"
    "  • The 'filled_template' must be clean ICF prose — no template markers remaining.\n"
    "  • Tables in retrieved passages contain important procedural data; read them carefully.\n\n"
    "PLAIN LANGUAGE GUIDELINES — apply these when generating any text:\n"
    + PLAIN_LANGUAGE_SCOPE
    + UHN_PLAIN_LANGUAGE_GUIDELINES
    + "\n"
)

# ---------------------------------------------------------------------------
# Shared symbol guide (identical to naive_prompts.py)
# ---------------------------------------------------------------------------

_SYMBOL_GUIDE = (
    "TEMPLATE SYMBOL GUIDE — read carefully before processing the template text below:\n"
    "  {{placeholder}}         → REQUIRED fill-in. Replace the entire {{...}} token with\n"
    "                            study-specific text from the retrieved passages.\n"
    "                            The {{...}} markers must NOT appear in your output.\n"
    "  {{option1/option2}}     → CHOOSE ONE option from the slash-separated list.\n"
    "                            The {{...}} markers must NOT appear in your output.\n"
    "  <<Condition block>>     → CONDITIONAL SECTION. Include the text ONLY if the stated\n"
    "                            condition applies to this study. Strip the <<...>> marker.\n"
    "  <Condition label>       → CONDITIONAL SENTENCE/PARAGRAPH. Same rule: include only\n"
    "                            if condition applies; strip the <...> marker.\n"
    "  OR (standalone line)    → ALTERNATIVE. Choose exactly ONE of the adjacent blocks.\n"
    "                            Do not include both, and do not include 'OR' in output.\n"
    "  • or -                  → BULLET POINT (used interchangeably).\n\n"
    "OUTPUT RULE: filled_template must contain clean ICF prose — no <<...>>, <...>,\n"
    "{{...}}, or standalone OR lines remaining.\n"
)

# ---------------------------------------------------------------------------
# JSON output schema
# ---------------------------------------------------------------------------

_JSON_SCHEMA_TEMPLATE = """{
    "reasoning": "Step-by-step analysis: reference specific retrieved passages by their [Page X] label, note what information each passage provides, explain how you resolved conditional template blocks (<<...>>, <...>, OR alternatives), and justify your status and confidence choices.",
    "section_id": "{section_id}",
    "status": "FOUND" | "PARTIAL" | "NOT_FOUND",
    "answer": "Extracted information following UHN Plain Language Guidelines (patient-facing).",
    "filled_template": "PATIENT-FACING OUTPUT. Required ICF wording with all {{placeholders}} filled from the retrieved passages, <<conditions>> resolved, OR alternatives chosen. Contains ONLY protocol information and [TO BE FILLED MANUALLY] for genuinely missing fields — never sentences about the extraction process or references to the protocol/study documents.",
    "evidence": [
        {"quote": "Exact verbatim quote from one of the retrieved passages", "page": "Page number from [Pages X-Y] label", "section": "Protocol section if identifiable"}
    ],
    "confidence": "HIGH" | "MEDIUM" | "LOW",
    "notes": "Caveats, items needing manual review, or decisions made during extraction. Note if key information was absent from retrieved passages."
}"""

# ---------------------------------------------------------------------------
# Availability note
# ---------------------------------------------------------------------------


def _availability_note(var: TemplateVariable) -> str:
    if not var.is_in_protocol:
        return (
            "IMPORTANT: This information is typically NOT found in clinical protocols. "
            "If the retrieved passages do not contain explicit evidence, "
            "return status='NOT_FOUND'. Do NOT fabricate information."
        )
    if var.partially_in_protocol:
        return (
            "NOTE: Only some fields may be in the protocol. Extract what the retrieved "
            "passages contain, mark unfound fields as [TO BE FILLED MANUALLY], "
            "and use status='PARTIAL' if only partial information is found."
        )
    return (
        "This information should be findable in the protocol. "
        "If the retrieved passages do not contain it, use status='PARTIAL' and note "
        "that additional protocol passages may hold the answer."
    )


# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------


def format_retrieved_context(parent_chunks: list[Chunk]) -> str:
    """Format a list of parent chunks as labelled passages for the prompt.

    Passages are sorted by page_start so the generator reads them in document
    order.  Each passage is labelled with its page range and section header
    to help the model attribute quotes accurately.
    """
    sorted_chunks = sorted(parent_chunks, key=lambda c: c.page_start)
    parts: list[str] = []

    for i, chunk in enumerate(sorted_chunks, start=1):
        label_parts = [f"Pages {chunk.page_start}–{chunk.page_end}"]
        if chunk.section_header:
            label_parts.append(f"Section: {chunk.section_header}")
        if chunk.is_table:
            label_parts.append("TABLE")
        label = " | ".join(label_parts)
        parts.append(f"[Passage {i}: {label}]\n{chunk.text}")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main prompt builder
# ---------------------------------------------------------------------------


def build_rag_extraction_task(var: TemplateVariable) -> str:
    """Build the extraction task portion of the message (without retrieved passages).

    This is split out so it can be logged/inspected independently of the
    context block, which can be large.
    """
    sub = f" > {var.sub_section}" if var.sub_section else ""
    availability = _availability_note(var)
    importance = (
        "REQUIRED — this section must appear in every ICF."
        if var.required
        else "OPTIONAL — include only if directly relevant to this specific study."
    )
    schema = _JSON_SCHEMA_TEMPLATE.replace("{section_id}", var.section_id)

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
            "REQUIRED ICF TEXT (mandatory wording — fill all {{placeholders}} from the "
            "retrieved passages; resolve all conditional blocks):\n"
            f"{var.required_text}\n"
        )

    if var.suggested_text:
        lines.append(
            "SUGGESTED ICF TEXT (adapt to this study using the retrieved passages):\n"
            f"{var.suggested_text}\n"
        )

    lines.append(
        "CHAIN-OF-THOUGHT REQUIREMENT: Fill in the 'reasoning' field FIRST. "
        "Reference specific passages by their [Passage N: Pages X-Y] label. "
        "Explain which passages were relevant, what you found, and how you resolved "
        "conditional blocks and template choices.\n"
    )
    lines.append(f"OUTPUT — respond with ONLY this JSON object:\n{schema}")

    return "\n".join(lines)


def build_rag_messages(
    var: TemplateVariable,
    parent_chunks: list[Chunk],
) -> list[dict]:
    """Build the full messages list for a single RAG extraction call.

    Structure:
      system  — role + rules (no REPL instructions)
      user    — retrieved passages (labelled) + extraction task
    """
    context_block = format_retrieved_context(parent_chunks)
    task = build_rag_extraction_task(var)

    user_content = (
        "=== RETRIEVED PASSAGES FROM THE CLINICAL STUDY PROTOCOL ===\n"
        "(These passages were retrieved as most relevant to the ICF section below. "
        "Page numbers in [Pages X-Y] labels come from the original protocol.)\n\n"
        f"{context_block}\n\n"
        "=== END OF RETRIEVED PASSAGES ===\n\n"
        f"{task}"
    )

    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

"""
Prompt templates for the Azure AI Search extraction backend.

Uses the same extraction schema and symbol guide as the other RAG backend,
but the retrieved context comes from Azure AI Search instead of local
BM25 + dense embeddings.
"""

from __future__ import annotations

from icf.types import TemplateVariable

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

AZURE_SEARCH_SYSTEM_PROMPT = (
    "You are a Clinical Data Extraction Specialist producing Informed Consent Form (ICF) "
    "content for a clinical study at UHN (University Health Network).\n\n"
    "You will receive a set of passages retrieved from a clinical study protocol via "
    "Azure AI Search, followed by a specific ICF section to fill in. Your job is to:\n"
    "  1. Carefully read ALL retrieved passages — the answer may span multiple passages.\n"
    "  2. Identify relevant information and note its source (document/page if available).\n"
    "  3. Return a JSON object: start with your reasoning, then the extraction fields.\n\n"
    "Core rules:\n"
    "  - Do NOT fabricate information. If a field cannot be answered from the passages,\n"
    "    write [TO BE FILLED MANUALLY] or return status='NOT_FOUND'.\n"
    "  - Every evidence quote must be a verbatim substring from one of the retrieved passages.\n"
    "  - Write the 'answer' field at a Grade 6 reading level (plain, patient-friendly language).\n"
    "  - The 'filled_template' must be clean ICF prose — no template markers remaining.\n"
    "  - Tables in retrieved passages contain important procedural data; read them carefully.\n"
)

# ---------------------------------------------------------------------------
# Shared symbol guide (identical to other backends)
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
    "reasoning": "Step-by-step analysis: reference specific retrieved documents, note what information each provides, explain how you resolved conditional template blocks (<<...>>, <...>, OR alternatives), and justify your status and confidence choices.",
    "section_id": "{section_id}",
    "status": "FOUND" | "PARTIAL" | "NOT_FOUND",
    "answer": "Extracted information in plain language at Grade 6 reading level.",
    "filled_template": "The required/suggested template text with all {{placeholders}} filled, <<conditions>> resolved, and OR alternatives chosen — clean ICF prose with no template markers remaining.",
    "evidence": [
        {"quote": "Exact verbatim quote from one of the retrieved documents", "page": "Page number or document index if identifiable", "section": "Protocol section if identifiable"}
    ],
    "confidence": "HIGH" | "MEDIUM" | "LOW",
    "notes": "Caveats, items needing manual review, or decisions made during extraction."
}"""

# ---------------------------------------------------------------------------
# Availability note
# ---------------------------------------------------------------------------


def _availability_note(var: TemplateVariable) -> str:
    if not var.is_in_protocol:
        return (
            "IMPORTANT: This information is typically NOT found in clinical protocols. "
            "If the retrieved documents do not contain explicit evidence, "
            "return status='NOT_FOUND'. Do NOT fabricate information."
        )
    if var.partially_in_protocol:
        return (
            "NOTE: Only some fields may be in the protocol. Extract what the retrieved "
            "documents contain, mark unfound fields as [TO BE FILLED MANUALLY], "
            "and use status='PARTIAL' if only partial information is found."
        )
    return (
        "This information should be findable in the protocol. "
        "If the retrieved documents do not contain it, use status='PARTIAL' and note "
        "that additional protocol sections may hold the answer."
    )


# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------


def format_search_results(documents: list[dict]) -> str:
    """Format Azure AI Search results as labelled passages for the prompt."""
    parts: list[str] = []
    for i, doc in enumerate(documents, start=1):
        # Try common content field names from Azure AI Search indices
        content = ""
        for field in ("content", "chunk", "text", "body", "description", "merged_content"):
            if field in doc and doc[field]:
                content = str(doc[field])
                break
        if not content:
            # Fallback: stringify the whole document
            content = str(doc)

        # Build a label from available metadata
        label_parts: list[str] = [f"Document {i}"]
        for title_field in ("title", "name", "filename", "metadata_storage_name"):
            if title_field in doc and doc[title_field]:
                label_parts.append(f"Source: {doc[title_field]}")
                break
        label = " | ".join(label_parts)

        parts.append(f"[{label}]\n{content}")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main prompt builder
# ---------------------------------------------------------------------------


def build_azure_search_messages(
    var: TemplateVariable,
    documents: list[dict],
) -> list[dict]:
    """Build the full messages list for a single Azure AI Search extraction call."""
    sub = f" > {var.sub_section}" if var.sub_section else ""
    availability = _availability_note(var)
    importance = (
        "REQUIRED — this section must appear in every ICF."
        if var.required
        else "OPTIONAL — include only if directly relevant to this specific study."
    )
    schema = _JSON_SCHEMA_TEMPLATE.replace("{section_id}", var.section_id)
    context_block = format_search_results(documents)

    task_lines: list[str] = [
        f"=== EXTRACTION TASK: ICF Section [{var.section_id}] ===\n",
        f"TARGET: {var.heading}{sub}",
        f"WHAT TO EXTRACT: {var.instructions}\n",
        f"AVAILABILITY: {availability}",
        f"IMPORTANCE: {importance}\n",
        _SYMBOL_GUIDE,
    ]

    if var.required_text:
        task_lines.append(
            "REQUIRED ICF TEXT (mandatory wording — fill all {{placeholders}} from the "
            "retrieved documents; resolve all conditional blocks):\n"
            f"{var.required_text}\n"
        )

    if var.suggested_text:
        task_lines.append(
            "SUGGESTED ICF TEXT (adapt to this study using the retrieved documents):\n"
            f"{var.suggested_text}\n"
        )

    task_lines.append(
        "CHAIN-OF-THOUGHT REQUIREMENT: Fill in the 'reasoning' field FIRST. "
        "Reference specific documents by their [Document N] label. "
        "Explain which documents were relevant, what you found, and how you resolved "
        "conditional blocks and template choices.\n"
    )
    task_lines.append(f"OUTPUT — respond with ONLY this JSON object:\n{schema}")

    task = "\n".join(task_lines)

    user_content = (
        "=== RETRIEVED DOCUMENTS FROM AZURE AI SEARCH ===\n"
        "(These documents were retrieved from the study protocol index as most relevant "
        "to the ICF section below.)\n\n"
        f"{context_block}\n\n"
        "=== END OF RETRIEVED DOCUMENTS ===\n\n"
        f"{task}"
    )

    return [
        {"role": "system", "content": AZURE_SEARCH_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

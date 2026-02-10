"""
Prompt templates for RLM-based extraction.

The root_prompt is shown to the RLM orchestrator as its task. The protocol
text is loaded separately as context_0 in the REPL environment.
"""

from icf.types import TemplateVariable


def _availability_note(var: TemplateVariable) -> str:
    """Generate the availability warning for the extraction prompt."""
    if not var.is_in_protocol:
        return (
            "IMPORTANT: This information is typically NOT found in clinical "
            "protocols and requires manual entry by the study team. Search "
            "the protocol briefly, but if you cannot find explicit evidence, "
            'return status="NOT_FOUND" immediately. Do NOT spend many '
            "iterations searching. Do NOT fabricate information."
        )
    if var.partially_in_protocol:
        return (
            "NOTE: Some fields in this section may not be found in the "
            "protocol and require manual entry. Extract what you can find, "
            "mark unfound fields as [TO BE FILLED MANUALLY], and use "
            'status="PARTIAL" if only some information is found.'
        )
    return (
        "This information should be findable in the protocol. "
        "Search thoroughly before concluding NOT_FOUND."
    )


def build_extraction_prompt(var: TemplateVariable) -> str:
    """Build the root_prompt for extracting a single template variable."""
    sub = f" > {var.sub_section}" if var.sub_section else ""
    availability = _availability_note(var)

    json_schema = (
        "{\n"
        f'    "section_id": "{var.section_id}",\n'
        '    "status": "FOUND" | "NOT_FOUND" | "PARTIAL",\n'
        '    "answer": "Extracted text in plain language (Grade 6 reading level)",\n'
        '    "filled_template": "The required text with curly-brace variables filled in",\n'
        '    "evidence": [\n'
        '        {"quote": "Exact verbatim quote from protocol", "page": "Page number"}\n'
        "    ],\n"
        '    "confidence": "HIGH" | "MEDIUM" | "LOW",\n'
        '    "notes": "Any caveats or items needing manual review"\n'
        "}"
    )

    prompt = (
        "You are a Clinical Data Extraction Specialist.\n"
        f"Extract information from a clinical study protocol for ICF section [{var.section_id}].\n\n"
        f"TARGET: {var.heading}{sub}\n"
        f"WHAT TO EXTRACT: {var.instructions}\n\n"
    )

    if var.protocol_mapping or var.sponsor_mapping:
        prompt += (
            "WHERE TO LOOK (hints - actual section names vary between protocols):\n"
            f"  Protocol sections: {var.protocol_mapping}\n"
            f"  Sponsor sections: {var.sponsor_mapping}\n\n"
        )

    prompt += f"{availability}\n\n"

    if var.required_text:
        prompt += f"ICF TEMPLATE TEXT (fill the {{{{ ... }}}} variables):\n{var.required_text}\n\n"

    if var.suggested_text:
        prompt += f"SUGGESTED TEXT:\n{var.suggested_text}\n\n"

    prompt += (
        "ENVIRONMENT NOTES:\n"
        "- `context` and `context_0` are the SAME variable: a plain STRING (not a list).\n"
        "  Use `context_0` directly. Do NOT index it like context[0] (that returns one character).\n"
        "- `globals()` is blocked. Access variables by name directly.\n"
        "- Pages are delimited by `--- PAGE X ---` markers in the text.\n\n"
        "APPROACH:\n"
        "1. First, check the size of context_0:\n"
        "   ```repl\n"
        "   print(f'Protocol length: {len(context_0)} chars')\n"
        "   ```\n"
        "2. Chunk context_0 and use llm_query_batched() to semantically search for the target info.\n"
        "   This is CRITICAL because section names vary greatly between protocols -- regex alone will miss them.\n"
        "   ```repl\n"
        "   chunk_size = 100000\n"
        "   chunks = [context_0[i:i+chunk_size] for i in range(0, len(context_0), chunk_size)]\n"
        "   prompts = [f'Find information about TARGET_INFO in this clinical protocol excerpt. "
        "Return relevant quotes with page numbers (look for --- PAGE X --- markers). "
        'If not found, say NOT FOUND.\\n\\n{chunk}" for chunk in chunks]\n'
        "   results = llm_query_batched(prompts)\n"
        "   for i, r in enumerate(results):\n"
        "       print(f'Chunk {i}: {r[:500]}')\n"
        "   ```\n"
        "3. Once you find relevant passages, verify quotes exist in context_0 with `quote in context_0`.\n"
        "4. Build a JSON result dict and return it:\n"
        "   ```repl\n"
        "   import json\n"
        "   result_json = json.dumps(result_dict)\n"
        "   ```\n"
        "   Then call FINAL_VAR(result_json)\n\n"
        f"RESULT JSON SCHEMA:\n{json_schema}\n\n"
        "RULES:\n"
        '1. DO NOT fabricate information. If not found, set status="NOT_FOUND".\n'
        "2. Every claim must be backed by a verbatim quote from the protocol.\n"
        "3. Simplify medical language to Grade 6 reading level in the 'answer' field.\n"
        '4. If only partial info is found, set status="PARTIAL" and note what is missing.\n'
        "5. For unfillable template placeholders, write [TO BE FILLED MANUALLY].\n"
    )

    return prompt

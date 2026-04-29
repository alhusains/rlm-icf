"""
Prompt templates for RLM-based extraction.

The root_prompt is shown to the RLM orchestrator as its task. The protocol
text is loaded separately as context_0 in the REPL environment.
"""

from icf.plain_language import PLAIN_LANGUAGE_SCOPE, UHN_PLAIN_LANGUAGE_GUIDELINES
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


def build_extraction_prompt(var: TemplateVariable, protocol_length: int = 0) -> str:
    """Build the root_prompt for extracting a single template variable.

    Args:
        var: The template variable to extract.
        protocol_length: Actual character count of the loaded protocol.  When
            provided (> 0) it is included in the ENVIRONMENT NOTES so the LLM
            has concrete proof that context_0 is loaded.
    """
    sub = f" > {var.sub_section}" if var.sub_section else ""
    availability = _availability_note(var)

    json_schema = (
        "{\n"
        f'    "section_id": "{var.section_id}",\n'
        '    "status": "FOUND" | "NOT_FOUND" | "PARTIAL",\n'
        '    "filled_template": "PATIENT-FACING OUTPUT. Required ICF wording with all {{placeholders}} filled from the protocol, <<conditions>> resolved, OR alternatives chosen. Contains ONLY protocol information and [TO BE FILLED MANUALLY] for genuinely missing fields — never sentences about the extraction process or references to the protocol/study documents.",\n'
        '    "evidence": [\n'
        '        {"quote": "Exact verbatim quote from protocol", "page": "Page number"}\n'
        "    ],\n"
        '    "confidence": "HIGH" | "MEDIUM" | "LOW",\n'
        '    "answer": "Plain-language summary of what was found by the extraction process (not patient-facing)",\n'
        '    "notes": "Any caveats or items needing manual review"\n'
        "}"
    )

    prompt = (
        "You are a Clinical Data Extraction Specialist.\n"
        f"Extract information from a clinical study protocol for ICF section [{var.section_id}].\n\n"
        f"TARGET: {var.heading}{sub}\n"
        f"WHAT TO EXTRACT: {var.instructions}\n\n"
    )

    if var.required:
        importance = "REQUIRED — this section must appear in every ICF."
    else:
        importance = "OPTIONAL — include only if directly relevant to this specific study."

    prompt += f"{availability}\n\nIMPORTANCE: {importance}\n\n"

    prompt += (
        "TEMPLATE SYMBOL GUIDE — read this carefully before processing the template text below:\n"
        "  {{placeholder}}         → REQUIRED fill-in. Replace the entire {{...}} token with\n"
        "                            study-specific text extracted from the protocol. The\n"
        "                            {{...}} markers must NOT appear in your output.\n"
        "  {{option1/option2}}     → CHOOSE ONE. Pick the applicable option from the\n"
        "                            slash-separated list (e.g. {{will/may}} → 'will' or 'may').\n"
        "                            The {{...}} markers must NOT appear in your output.\n"
        "  <<Condition block>>     → CONDITIONAL SECTION (double angle brackets). Include the\n"
        "                            text that follows ONLY if the stated condition applies to\n"
        "                            this study. Remove the <<...>> marker itself entirely —\n"
        "                            it must NEVER appear in the final ICF text.\n"
        "  <Condition label>       → CONDITIONAL SENTENCE/PARAGRAPH (single angle brackets).\n"
        "                            Same rule: include the text only if the condition applies;\n"
        "                            strip the <...> marker from the output.\n"
        "  OR (standalone line)    → ALTERNATIVE. Choose exactly ONE of the blocks immediately\n"
        "                            above or below this marker. Do not include both, and do\n"
        "                            not include the word 'OR' itself in the final text.\n"
        "  • or -                  → BULLET POINT. Both are used interchangeably as list items.\n\n"
        "OUTPUT RULE: The filled_template field in your JSON result must contain clean ICF\n"
        "prose — no <<...>>, <...>, {{...}}, or standalone OR lines remaining.\n\n"
    )

    if var.required_text:
        prompt += f"ICF TEMPLATE TEXT:\n{var.required_text}\n\n"

    if var.suggested_text:
        prompt += f"SUGGESTED TEXT (adapt to this study; apply the symbol rules above):\n{var.suggested_text}\n\n"

    prompt += (
        "ENVIRONMENT NOTES:\n"
        + (
            f"- context_0 is PRE-LOADED with {protocol_length:,} characters of protocol text.\n"
            if protocol_length
            else ""
        )
        + "- `context` and `context_0` are the SAME variable: a plain STRING (not a list).\n"
        "  Use `context_0` directly. Do NOT index it like context[0] (that returns one character).\n"
        "- `globals()` is blocked. Access variables by name directly.\n"
        "- Pages are delimited by `--- PAGE X ---` markers in the text.\n"
        "- REPL blocks you write are AUTOMATICALLY executed. Never ask for user permission.\n"
        "- NEVER wrap a ```repl block inside another fence (e.g. ````repl). Write code blocks\n"
        "  DIRECTLY as ```repl ... ``` with no outer wrapper. Nested fences cause SyntaxError.\n\n"
        "APPROACH:\n"
        "1. Chunk context_0 and use llm_query_batched() to semantically search for the target info.\n"
        "   CRITICAL: The sub-LLM called by llm_query/llm_query_batched only receives the prompt\n"
        "   string you write — it cannot see context_0 or your REPL session. YOU (the orchestrator)\n"
        "   always have full REPL access to context_0 and must embed the chunk text in each prompt.\n"
        "   ```repl\n"
        "   chunk_size = 50000\n"
        "   chunks = [context_0[i:i+chunk_size] for i in range(0, len(context_0), chunk_size)]\n"
        "   # Embed chunk text directly in each prompt so the sub-LLM can read it\n"
        "   prompts = [\n"
        "       f'Find information about TARGET_INFO. Return quotes with page numbers '\n"
        "       f'(--- PAGE X --- markers). If not found, say NOT FOUND.\\n\\nExcerpt:\\n{chunk}'\n"
        "       for chunk in chunks\n"
        "   ]\n"
        "   results = llm_query_batched(prompts)\n"
        "   for i, r in enumerate(results):\n"
        "       print(f'Chunk {i}: {r[:500]}')\n"
        "   ```\n"
        "2. Once you find relevant passages, verify quotes exist in context_0 with `quote in context_0`.\n"
        "3. Build result_dict, then run this verification block (NO FINAL_VAR inside it):\n"
        "   ```repl\n"
        "   import re, json\n"
        "   result_dict = {\"section_id\": ..., \"status\": ..., \"filled_template\": ...,\n"
        "                  \"evidence\": [{\"quote\": ..., \"page\": ...}],\n"
        "                  \"confidence\": ..., \"answer\": ..., \"notes\": ...}\n"
        "   issues = []\n"
        "   ft = result_dict.get('filled_template', '')\n"
        "   for m in re.findall(r'{{[^}]+}}|<<[^>]+>>', ft):\n"
        "       issues.append('Unfilled marker: ' + m)\n"
        "   for b in ['not found in', 'study documents', 'cannot be found', 'in these passages']:\n"
        "       if b in ft.lower():\n"
        "           issues.append('Meta-commentary: ' + b)\n"
        "   if issues:\n"
        "       for iss in issues: print('FIX: ' + iss)\n"
        "   else:\n"
        "       result_json = json.dumps(result_dict)\n"
        "       print('READY_TO_FINALIZE')\n"
        "   ```\n"
        "   - FIX lines printed? Fix result_dict and re-run this block.\n"
        "   - Output is READY_TO_FINALIZE? Write this as your next block (and nothing else):\n"
        "   ```repl\n"
        "   FINAL_VAR(result_json)\n"
        "   ```\n"
        "   IMPORTANT: result_json was assigned in the verification block above — do NOT redefine it.\n"
        "   IMPORTANT: Never write FINAL_VAR inside an if/else block.\n\n"
        f"RESULT JSON SCHEMA:\n{json_schema}\n\n"
        "RULES:\n"
        "1. 'filled_template' is READ BY PATIENTS. It must contain ONLY: required ICF wording "
        "(with placeholders filled), protocol information, and [TO BE FILLED MANUALLY] for "
        "missing fields. NEVER include sentences about what was or wasn't found, references to "
        "'the protocol', 'study documents', or any internal process. Put internal notes in 'notes'.\n"
        '2. DO NOT fabricate information. If not found, set status="NOT_FOUND".\n'
        "3. Every claim must be backed by a verbatim quote from the protocol.\n"
        '4. If only partial info is found, set status="PARTIAL" and note what is missing.\n'
        "5. For unfillable template placeholders, write [TO BE FILLED MANUALLY] — never explain why.\n\n"
        "UHN PLAIN LANGUAGE GUIDELINES — apply these when generating any text:\n"
        + PLAIN_LANGUAGE_SCOPE
        + UHN_PLAIN_LANGUAGE_GUIDELINES
        + "\n"
    )

    return prompt

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
        "APPROACH (follow this funnel — each step is cheaper than the next):\n\n"
        "STEP 1 — Keyword filter (free, no LLM tokens):\n"
        "   Use regex to find the most relevant passages before calling any LLM.\n"
        "   Extract a window of text around each match so you get full context.\n"
        "   ```repl\n"
        "   import re\n"
        "\n"
        "   def find_snippets(text, keywords, window=2000, max_hits=10):\n"
        "       seen = set()\n"
        "       hits = []\n"
        "       for kw in keywords:\n"
        "           for m in re.finditer(re.escape(kw), text, re.IGNORECASE):\n"
        "               s = max(0, m.start() - window)\n"
        "               e = min(len(text), m.end() + window)\n"
        "               snippet = text[s:e]\n"
        "               key = (s // window)  # deduplicate overlapping hits\n"
        "               if key not in seen:\n"
        "                   seen.add(key)\n"
        "                   hits.append(snippet)\n"
        "               if len(hits) >= max_hits:\n"
        "                   return hits\n"
        "       return hits\n"
        "\n"
        "   # Choose keywords relevant to TARGET -- use multiple synonyms/variants\n"
        "   keywords = ['keyword1', 'keyword2', 'keyword3']\n"
        "   snippets = find_snippets(context_0, keywords)\n"
        "   combined = '\\n\\n--- SNIPPET BREAK ---\\n\\n'.join(snippets)\n"
        "   print(f'Found {len(snippets)} snippets, {len(combined)} chars total')\n"
        "   print(combined[:2000])  # preview\n"
        "   ```\n\n"
        "STEP 2 — Sub-LM on snippets only (cheap, focused):\n"
        "   If Step 1 found relevant snippets, pass ONLY those to a sub-LM.\n"
        "   Do NOT pass the entire protocol when snippets are available.\n"
        "   ```repl\n"
        "   result = llm_query(\n"
        "       f'TARGET_INSTRUCTION\\n\\n'\n"
        "       f'Relevant protocol excerpts (with --- PAGE X --- markers):\\n\\n{combined}'\n"
        "   )\n"
        "   print(result)\n"
        "   ```\n\n"
        "STEP 3 — Full-protocol scan (fallback only, use if Step 1 found nothing):\n"
        "   Only use this if keyword search returned no useful snippets.\n"
        "   ```repl\n"
        "   chunk_size = 100000\n"
        "   chunks = [context_0[i:i+chunk_size] for i in range(0, len(context_0), chunk_size)]\n"
        "   prompts = [\n"
        "       f'TARGET_INSTRUCTION\\n\\nProtocol excerpt:\\n\\n{chunk}'\n"
        "       for chunk in chunks\n"
        "   ]\n"
        "   results = llm_query_batched(prompts)\n"
        "   for i, r in enumerate(results):\n"
        "       print(f'Chunk {i}: {r[:800]}')\n"
        "   ```\n\n"
        "STEP 4 — Verify and return:\n"
        "   For each quote in your evidence, verify it exists: `assert quote in context_0`.\n"
        "   Build the result dict and call FINAL_VAR(result_json).\n\n"
        f"RESULT JSON SCHEMA:\n{json_schema}\n\n"
        "RULES:\n"
        '1. DO NOT fabricate information. If not found, set status="NOT_FOUND".\n'
        "2. Every claim must be backed by a verbatim quote from the protocol.\n"
        "3. Simplify medical language to Grade 6 reading level in the 'answer' field.\n"
        '4. If only partial info is found, set status="PARTIAL" and note what is missing.\n'
        "5. For unfillable template placeholders, write [TO BE FILLED MANUALLY].\n"
    )

    return prompt

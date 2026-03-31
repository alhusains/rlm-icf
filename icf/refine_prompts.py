"""
Prompt for the Tier 2 targeted refinement pass.

Called when the first-pass extraction has quality issues: LOW confidence,
PARTIAL status on a required section, unverified quotes, or meta-commentary
leaking into patient-facing text.

Intentionally leaner than the full extraction prompt — no template symbol
guide, no UHN plain language guidelines block (the model already knows them
from the extraction pass). This pass only needs to fix specific problems.

The caller (_run_refinement_pass in extract.py) pre-seeds result_dict into
the REPL via setup_code before the RLM starts, so the prompt can truthfully
say "result_dict is already in scope — modify it to fix the issues below."
"""

import json

from icf.types import ExtractionResult, TemplateVariable


def build_refinement_setup_code(first_result: ExtractionResult) -> str:
    """Return Python setup code that pre-loads result_dict into the REPL.

    This is injected via environment_kwargs['setup_code'] so that
    result_dict is available in the REPL from the very first iteration,
    making the "do not re-extract from scratch" instruction coherent.
    """
    data = {
        "section_id": first_result.section_id,
        "status": first_result.status,
        "confidence": first_result.confidence,
        "answer": first_result.answer,
        "filled_template": first_result.filled_template,
        "evidence": [
            {"quote": e.quote, "page": e.page, "section": e.section}
            for e in first_result.evidence
        ],
        "notes": first_result.notes,
    }
    return f"import json\nresult_dict = json.loads({repr(json.dumps(data))})"


def build_refinement_prompt(
    var: TemplateVariable,
    first_result: ExtractionResult,
    issues: list[str],
) -> str:
    """Build a focused repair prompt for the refinement RLM pass.

    Args:
        var: The template variable being extracted.
        first_result: The first-pass extraction result to improve.
        issues: Specific quality problems identified post-extraction.
    """
    sub = f" > {var.sub_section}" if var.sub_section else ""
    issues_block = "\n".join(f"  - {iss}" for iss in issues)

    json_schema = (
        "{\n"
        f'    "section_id": "{var.section_id}",\n'
        '    "status": "FOUND" | "NOT_FOUND" | "PARTIAL",\n'
        '    "filled_template": "Clean patient-facing ICF text — no {{...}}, <<...>>, or internal notes.",\n'
        '    "evidence": [{"quote": "Verbatim quote from protocol", "page": "X"}],\n'
        '    "confidence": "HIGH" | "MEDIUM" | "LOW",\n'
        '    "answer": "Patient-facing plain language summary (Grade 6-8)",\n'
        '    "notes": "Caveats or items needing manual review"\n'
        "}"
    )

    # Extract any [TO BE FILLED MANUALLY] gaps from filled_template so we can
    # give the model concrete search targets rather than vague issue descriptions.
    import re as _re

    manual_gaps = _re.findall(
        r"[^\n]*\[TO BE FILLED MANUALLY[^\]]*\][^\n]*", first_result.filled_template
    )
    gaps_block = ""
    if manual_gaps:
        gap_list = "\n".join(f"  • {g.strip()}" for g in manual_gaps[:5])
        gaps_block = (
            "\nSPECIFIC GAPS to resolve (lines from the previous extraction that "
            "still need information from the protocol):\n"
            f"{gap_list}\n"
        )

    prompt = (
        f"You are fixing a previous extraction for ICF section [{var.section_id}] — "
        f"{var.heading}{sub}.\n\n"
        f"WHAT TO EXTRACT:\n{var.instructions}\n\n"
    )

    if var.required_text:
        rt = var.required_text[:1500]
        suffix = "... [truncated]" if len(var.required_text) > 1500 else ""
        prompt += f"ICF TEMPLATE TEXT:\n{rt}{suffix}\n\n"

    prompt += (
        "ISSUES TO FIX:\n"
        f"{issues_block}\n"
        f"{gaps_block}\n"
        "CONTEXT:\n"
        "result_dict is ALREADY LOADED in the REPL from the previous extraction.\n"
        "context_0 holds the full protocol text — search it to fix the issues above.\n\n"
        "WORKFLOW — follow these exact steps:\n\n"
        "STEP 1 — Search context_0 (MAXIMUM 2 searches):\n"
        "```repl\n"
        "prompt = 'Find [the specific missing item] in the protocol. Return verbatim quotes with page numbers.'\n"
        "found = llm_query(prompt + '\\n\\n' + context_0[:40000])\n"
        "print(found[:2000])\n"
        "```\n\n"
        "STEP 2 — After searching, update result_dict in a code block:\n"
        "  a) If you FOUND the information:\n"
        "```repl\n"
        "result_dict['status'] = 'FOUND'\n"
        "result_dict['confidence'] = 'HIGH'\n"
        "result_dict['filled_template'] = '...'  # patient-facing text\n"
        "result_dict['evidence'] = [{'quote': '...verbatim quote...', 'page': 'X'}]\n"
        "result_dict['answer'] = '...plain language summary...'\n"
        "```\n\n"
        "  b) If searches confirm the information is NOT in the protocol — stop searching\n"
        "     and confirm result_dict is already correct:\n"
        "```repl\n"
        "# Info confirmed absent — result_dict is already correct as-is\n"
        "print('confirmed not in protocol')\n"
        "```\n\n"
        "NEVER run more than 2 searches. If 2 searches both return NOT FOUND, "
        "move immediately to STEP 3.\n\n"
        "RULES:\n"
        "1. Modify result_dict in place — do NOT redefine it from scratch.\n"
        "2. For meta-commentary in filled_template: move it to result_dict['notes'].\n"
        "3. For unfilled {{...}} or <<...>> markers: resolve them from the protocol.\n"
        "4. Every claim needs a verbatim protocol quote in result_dict['evidence'].\n"
        "5. Use [TO BE FILLED MANUALLY] only for genuinely absent information.\n"
        "6. Plain language: Grade 6-8, active voice, short sentences.\n\n"
        "FINALIZE — run this verification block (NO FINAL_VAR inside it):\n"
        "```repl\n"
        "import re, json\n"
        "issues = []\n"
        "ft = result_dict.get('filled_template', '')\n"
        "for m in re.findall(r'{{[^}]+}}|<<[^>]+>>', ft):\n"
        "    issues.append('Unfilled: ' + m)\n"
        "for b in ['not found in', 'study documents', 'cannot be found']:\n"
        "    if b in ft.lower():\n"
        "        issues.append('Meta-commentary: ' + b)\n"
        "if issues:\n"
        "    for iss in issues: print('STILL BROKEN: ' + iss)\n"
        "else:\n"
        "    result_json = json.dumps(result_dict)\n"
        "    print('READY_TO_FINALIZE')\n"
        "```\n"
        "If output is READY_TO_FINALIZE:\n"
        "  → STOP. Do NOT run any more searches.\n"
        "  → Your IMMEDIATE next block must be ONLY this — no other code:\n"
        "```repl\n"
        "FINAL_VAR(result_json)\n"
        "```\n"
        "Never write FINAL_VAR inside an if/else block.\n"
        "Never run another search after seeing READY_TO_FINALIZE.\n\n"
        f"RESULT JSON SCHEMA:\n{json_schema}\n"
    )

    return prompt

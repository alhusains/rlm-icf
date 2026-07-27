"""
Prompt templates for the hybrid extraction backend.

The hybrid backend splits the RLM's job into two focused stages instead of
one prompt that mixes research, template legalese, plain-language rules,
and JSON formatting all at once (see icf/prompts.py for the original,
monolithic version):

  Stage A (build_evidence_gathering_prompt) -- an RLM whose ONLY job is to
  search the protocol and return a small, verified evidence bundle
  (verbatim quotes + a findings summary). It never sees the JSON output
  schema, the template symbol guide, or the plain-language guidelines --
  none of that helps it search better, and today it competes for
  attention with the search task itself.

  Stage B (build_draft_messages) -- a single non-agentic LLM call that
  turns the Stage A evidence bundle into the final structured ICF output.
  It owns everything Stage A does not: template symbol resolution,
  verbatim required/suggested text handling, plain-language guidelines,
  and the JSON schema. It is not searching a 100-page protocol -- it only
  sees the curated evidence bundle plus the section template -- so it can
  focus entirely on writing well.

icf/hybrid_extract.py orchestrates both stages plus a bounded Stage C
repair retry; see that module for the deterministic quality gate.
"""

from icf.plain_language import (
    PLAIN_LANGUAGE_SCOPE,
    STUDY_TEAM_NOTES_GUIDANCE,
    UHN_PLAIN_LANGUAGE_GUIDELINES,
    is_cover_page_section,
)
from icf.runtime_injections import prompt_runtime_context
from icf.types import TemplateVariable

# ===========================================================================
# Stage A -- evidence gathering (RLM root_prompt)
# ===========================================================================

RESEARCH_PHILOSOPHY = (
    "RESEARCH PHILOSOPHY (read first):\n"
    "- Report facts precisely and technically: exact numbers, names, procedures, "
    "timeframes, and eligibility criteria. Do NOT simplify, summarize loosely, or soften "
    "anything for a lay reader -- a separate writing step turns your findings into plain, "
    "participant-facing language, and it can only be as accurate as the facts you hand it.\n"
    "- Only report information the protocol explicitly states, and back every fact with "
    "a verbatim quote you have confirmed exists in context_0. Because this research feeds "
    "an informed-consent decision, precision and traceability matter more than completeness "
    "-- never guess or approximate a fact you cannot quote.\n"
    "- It is correct and expected that some information is not in the protocol. Say so "
    "plainly in search_notes -- do not stretch adjacent or loosely related content to "
    "cover a gap.\n"
    "- A partially-answerable section is normal: report what is supported and clearly "
    'flag what is missing, using status="PARTIAL".\n\n'
)

EVIDENCE_SUFFICIENCY = (
    "EVIDENCE SUFFICIENCY:\n"
    "- A quote can be one sentence or a short passage of consecutive sentences -- whichever "
    "is needed to preserve the complete meaning of a fact. Do not truncate a passage that "
    "cuts off mid-explanation, and do not fragment one coherent statement (e.g. a risk "
    "described together with its severity and management) into disconnected pieces.\n"
    "- Give the drafter enough material to write a complete, accurate section. If a topic has "
    "several distinct facts (e.g. a procedure's purpose, steps, duration, and who performs "
    "it), capture each with its own quote instead of reporting only the single most obvious "
    "sentence and leaving the rest for the drafter to guess at.\n"
    "- Still be selective: do not quote unrelated surrounding text just because it is nearby "
    "-- every quote must directly support a fact you state in findings_summary.\n\n"
)

CONTRADICTION_AND_RELEVANCE = (
    "HANDLING CONTRADICTIONS AND RELEVANCE:\n"
    "- If two passages state different things about the same fact (e.g. different numbers, "
    "differing eligibility criteria, or a protocol amendment that changes an earlier "
    "section), report BOTH quotes with page numbers, note the discrepancy explicitly in "
    "search_notes, and set confidence to MEDIUM or LOW so the drafter and study team know "
    "this is not settled. As a rule of thumb a later amendment or the main protocol body "
    "supersedes an earlier version or the synopsis -- say so if it applies, but always flag "
    "the conflict either way rather than silently picking one value.\n"
    "- Not everything near this topic belongs in an ICF. Only report facts directly relevant "
    "to WHAT TO LOOK FOR above. Leave out internal administrative detail, statistical/"
    "analysis-plan minutiae, and sponsor-only or operational information that a participant "
    "does not need to make an informed decision, even if it appears in the same passage as "
    "something relevant.\n\n"
)


def research_availability_note(var: TemplateVariable) -> str:
    """Availability heuristic for the Stage A research prompt.

    Same underlying logic as the full RLM prompt (icf/prompts.py), reframed
    around researching rather than drafting.
    """
    if not var.is_in_protocol:
        return (
            "IMPORTANT: This information is typically NOT found in clinical protocols "
            "and requires manual entry by the study team. Search briefly, but if you "
            'cannot find explicit evidence, return status="NOT_FOUND" immediately. '
            "Do NOT spend many iterations searching."
        )
    if var.partially_in_protocol:
        return (
            "NOTE: Some facts for this section may not be in the protocol and require "
            'manual entry. Report what you can find and use status="PARTIAL" if only '
            "some information is found."
        )
    return (
        "This information should be findable in the protocol. Search thoroughly before "
        "concluding NOT_FOUND."
    )


def build_evidence_gathering_prompt(var: TemplateVariable, protocol_length: int = 0) -> str:
    """Build the Stage A root_prompt: research only, no drafting concerns.

    Args:
        var: The template variable to research.
        protocol_length: Character count of the loaded protocol, surfaced in
            ENVIRONMENT NOTES as concrete proof that context_0 is loaded.
    """
    sub = f" > {var.sub_section}" if var.sub_section else ""
    availability = research_availability_note(var)

    json_schema = (
        "{\n"
        f'    "section_id": "{var.section_id}",\n'
        '    "status": "FOUND" | "NOT_FOUND" | "PARTIAL",\n'
        '    "evidence": [\n'
        '        {"quote": "Verbatim quote -- a COMPLETE sentence or clause copied exactly from the protocol, not a fragment", "page": "Page number"}\n'
        "    ],\n"
        '    "findings_summary": "Precise, technical notes for the person who will draft the ICF text: the exact facts, numbers, names, procedures, and timeframes you found. Write for a clinical reader, not the study participant -- do not simplify or soften.",\n'
        '    "conditional_resolution": "If the template below has <<...>>/<...>/OR alternatives, state which branch applies to this study and why. Empty string if not applicable.",\n'
        '    "confidence": "HIGH" | "MEDIUM" | "LOW",\n'
        '    "search_notes": "What you searched, what you could not find, and any ambiguity or contradictory statements the drafter should know about (e.g. conflicting numbers between sections/amendments)."\n'
        "}"
    )

    prompt = (
        "You are a Clinical Research Analyst supporting the creation of an Informed Consent "
        "Form (ICF) for a clinical study. An ICF is the document a potential study "
        "participant reads to decide whether to join the study, so every fact in it must be "
        "traceable to an exact statement in the protocol.\n\n"
        f"Your job is to research the protocol and report verified evidence for ICF section "
        f"[{var.section_id}] -- a separate writing step will turn your findings into the "
        "plain-language text the participant actually reads. You do NOT write that final "
        "text yourself.\n\n"
        f"TARGET: {var.heading}{sub}\n"
        f"WHAT TO LOOK FOR: {var.instructions}\n\n"
    )
    prompt += prompt_runtime_context(var)
    prompt += RESEARCH_PHILOSOPHY
    prompt += EVIDENCE_SUFFICIENCY
    prompt += CONTRADICTION_AND_RELEVANCE

    if var.required:
        importance = "REQUIRED -- this section must appear in every ICF."
    else:
        importance = "OPTIONAL -- include only if directly relevant to this specific study."
    prompt += f"{availability}\n\nIMPORTANCE: {importance}\n\n"

    # The drafter needs the template text to resolve conditionals/alternatives against
    # what you find -- but you report FACTS, not wording. Do not reproduce or rewrite it.
    if var.required_text:
        prompt += (
            "ICF TEMPLATE TEXT (context only -- identify which facts below fill its "
            "{{...}} placeholders and which <<...>>/OR branches apply; do not copy or "
            "rewrite this text yourself):\n"
            f"{var.required_text}\n\n"
        )
    if var.suggested_text:
        prompt += (
            "SUGGESTED TEXT (context only, same rule as above):\n" f"{var.suggested_text}\n\n"
        )

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
        "   string you write -- it cannot see context_0 or your REPL session. YOU (the orchestrator)\n"
        "   always have full REPL access to context_0 and must embed the chunk text in each prompt.\n"
        "   ```repl\n"
        "   chunk_size = 50000\n"
        "   chunks = [context_0[i:i+chunk_size] for i in range(0, len(context_0), chunk_size)]\n"
        "   # Embed chunk text directly in each prompt so the sub-LLM can read it\n"
        "   prompts = [\n"
        "       f'Find information about TARGET_INFO. Return the relevant passage(s) verbatim '\n"
        "       f'-- a full sentence, or several consecutive sentences together if that is '\n"
        "       f'what it takes to keep one fact complete (e.g. a risk with its severity and '\n"
        "       f'management). Not isolated fragments, and not more than the fact needs. '\n"
        "       f'Include page numbers (--- PAGE X --- markers). If not found, say NOT FOUND.'\n"
        "       f'\\n\\nExcerpt:\\n{chunk}'\n"
        "       for chunk in chunks\n"
        "   ]\n"
        "   results = llm_query_batched(prompts)\n"
        "   for i, r in enumerate(results):\n"
        "       print(f'Chunk {i}: {r[:500]}')\n"
        "   ```\n"
        "2. Once you find relevant passages, verify quotes exist in context_0 with `quote in context_0`.\n"
        "3. Build evidence_dict, then run this verification block (NO FINAL_VAR inside it):\n"
        "   ```repl\n"
        "   import json\n"
        "   evidence_dict = {\"section_id\": ..., \"status\": ..., \"evidence\": [{\"quote\": ..., \"page\": ...}],\n"
        "                     \"findings_summary\": ..., \"conditional_resolution\": ...,\n"
        "                     \"confidence\": ..., \"search_notes\": ...}\n"
        "   issues = []\n"
        "   _ev = evidence_dict.get('evidence', []) or []\n"
        "   for e in _ev:\n"
        "       q = e.get('quote', '')\n"
        "       if q and q not in context_0:\n"
        "           issues.append('Quote not verbatim in context_0: ' + q[:80])\n"
        "   if evidence_dict.get('status') in ('FOUND', 'PARTIAL') and not _ev:\n"
        "       issues.append('status is FOUND/PARTIAL but evidence list is empty')\n"
        "   if issues:\n"
        "       for iss in issues: print('FIX: ' + iss)\n"
        "   else:\n"
        "       evidence_json = json.dumps(evidence_dict)\n"
        "       print('READY_TO_FINALIZE')\n"
        "   ```\n"
        "   - FIX lines printed? Fix evidence_dict and re-run this block.\n"
        "   - Output is READY_TO_FINALIZE? Write this as your next block (and nothing else):\n"
        "   ```repl\n"
        "   FINAL_VAR(evidence_json)\n"
        "   ```\n"
        "   IMPORTANT: evidence_json was assigned in the verification block above -- do NOT redefine it.\n"
        "   IMPORTANT: Never write FINAL_VAR inside an if/else block.\n\n"
        f"RESULT JSON SCHEMA:\n{json_schema}\n\n"
        "RULES:\n"
        '1. DO NOT fabricate information. If not found, set status="NOT_FOUND".\n'
        "2. Every fact in findings_summary must be backed by its OWN verbatim quote in "
        "evidence -- one quote per distinct fact, no duplicates -- and together the quotes "
        "must cover every fact you report.\n"
        '3. If only partial info is found, set status="PARTIAL" and say what is missing in search_notes.\n'
        + (
            "4. This section is OPTIONAL. After a brief search, if the protocol contains no direct "
            'evidence that this topic applies to this study, return status="NOT_FOUND" immediately. '
            "A related or adjacent procedure does NOT count -- for example, blood draws are not "
            "tissue collection.\n"
            if not var.required
            else ""
        )
    )

    return prompt


# ===========================================================================
# Stage B -- drafting (single non-agentic LLM call)
# ===========================================================================

SYMBOL_GUIDE = (
    "TEMPLATE SYMBOL GUIDE -- read carefully before processing the template text below:\n"
    "  {{placeholder}}         -> REQUIRED fill-in. Replace the entire {{...}} token with\n"
    "                            study-specific text from the research findings. The {{...}}\n"
    "                            markers must NOT appear in your output.\n"
    "  {{option1/option2}}     -> CHOOSE ONE. Pick the applicable option from the slash-separated\n"
    "                            list (e.g. {{will/may}} -> 'will' or 'may'). The {{...}}\n"
    "                            markers must NOT appear in your output.\n"
    "  <<Condition block>>     -> CONDITIONAL SECTION (double angle brackets). Include the text\n"
    "                            that follows ONLY if the condition applies to this study --\n"
    "                            use the conditional/alternative resolution in the research\n"
    "                            findings below. Remove the <<...>> marker itself entirely --\n"
    "                            it must NEVER appear in the final ICF text.\n"
    "  <Condition label>       -> CONDITIONAL SENTENCE/PARAGRAPH (single angle brackets). Same\n"
    "                            rule: include only if the condition applies; strip the <...>\n"
    "                            marker from the output.\n"
    "  OR (standalone line)    -> ALTERNATIVE. Choose exactly ONE of the blocks immediately\n"
    "                            above or below this marker. Do not include both, and do not\n"
    "                            include the word 'OR' itself in the final text.\n"
    "  \u2022 or -                  -> BULLET POINT. Both are used interchangeably as list items.\n\n"
    "OUTPUT RULE: The filled_template field must contain clean ICF prose -- no <<...>>, <...>,\n"
    "{{...}}, or standalone OR lines remaining.\n"
)

_HYBRID_DRAFT_SYSTEM_CORE = (
    "You are a Clinical Consent Form Writer producing Informed Consent Form (ICF) content "
    "for a clinical study at UHN (University Health Network).\n\n"
    "A separate research step has already searched the protocol for you. You will receive "
    "its verified findings -- verbatim quotes, page numbers, and a findings summary -- for "
    "one ICF section, plus that section's template text. Your ONLY job is to turn those "
    "findings into the final structured ICF output. Do NOT invent facts beyond what the "
    "findings state, and do NOT search for more information -- if the findings don't cover "
    "something the template needs, mark it [PLEASE COMPLETE].\n\n"
    "Core rules:\n"
    "  - 'filled_template' is READ BY THE STUDY PARTICIPANT. The participant may be a patient, "
    "a clinician, a healthy volunteer, a caregiver, or any other person the study is enrolling "
    "-- the protocol defines who. Write for whoever the protocol says is being recruited. It "
    "must contain ONLY: required ICF wording (with placeholders filled), findings content, "
    "and [PLEASE COMPLETE] for missing fields. NEVER include sentences about what was or "
    "wasn't found, references to 'the protocol', 'the research step', 'study documents', or "
    "any internal process. Put internal notes in 'notes'.\n"
    "  - Do NOT fabricate information beyond what the findings state.\n"
    "  - Every evidence quote you cite must be copied EXACTLY from the findings you were "
    "given -- never paraphrase a quote, never invent a new one.\n"
    "  - The 'filled_template' must be clean ICF prose -- no template markers remaining.\n"
    "  - If only partial information is found, use status='PARTIAL' and note what is missing.\n"
    "  - For unfillable placeholders, write [PLEASE COMPLETE] -- never explain why.\n"
    "  - The findings may include multi-sentence passages, not just single facts -- use the "
    "full context of each finding to write a complete, coherent section instead of a terse "
    "one-liner when the material supports more detail.\n"
    "  - If search_notes flags a contradiction between passages (e.g. differing numbers "
    "between an amendment and an earlier section), do NOT silently pick one value. Use the "
    "value the findings indicate is more authoritative (e.g. the later amendment); if the "
    "findings don't say which one to trust, use the one that best matches the required/"
    "suggested text below. Either way, add a brief note in 'notes' flagging the discrepancy "
    "for the study team to confirm. Never present both conflicting values as if both are true.\n"
    "  - Not everything in the findings belongs in this section. Include only what is "
    "directly relevant to WHAT TO WRITE below and appropriate for participant-facing consent "
    "material -- omit internal administrative, statistical, or sponsor-only details from the "
    "findings even if present, rather than writing around them.\n\n"
)


def _hybrid_draft_system_prompt(section_id: str) -> str:
    """Stage B system prompt; omits plain-language guidelines for cover-page 2.x."""
    plain_language = (
        ""
        if is_cover_page_section(section_id)
        else (
            "UHN PLAIN LANGUAGE GUIDELINES -- apply these when generating any text:\n"
            + PLAIN_LANGUAGE_SCOPE
            + UHN_PLAIN_LANGUAGE_GUIDELINES
            + "\n\n"
        )
    )
    return _HYBRID_DRAFT_SYSTEM_CORE + plain_language + STUDY_TEAM_NOTES_GUIDANCE + "\n"


def draft_availability_note(evidence_bundle: dict) -> str:
    """Availability note derived from what Stage A actually found (not the registry flag)."""
    stage_a_status = evidence_bundle.get("status", "")
    if stage_a_status == "NOT_FOUND":
        return (
            "The research step did NOT find this information in the protocol. Use "
            'status="NOT_FOUND" unless the required/suggested text below is pure standard '
            "wording that needs no study-specific facts."
        )
    if stage_a_status == "PARTIAL":
        return (
            "The research step found some but not all of the information for this section. "
            "Fill in what the findings support, mark the rest [PLEASE COMPLETE], and use "
            'status="PARTIAL".'
        )
    return (
        "The research step found information relevant to this section -- use it to fill "
        "the template below."
    )


def format_evidence_bundle(evidence_bundle: dict) -> str:
    """Render the Stage A evidence bundle as readable text for the Stage B prompt."""
    status = evidence_bundle.get("status", "UNKNOWN")
    confidence = evidence_bundle.get("confidence", "LOW")
    findings_summary = evidence_bundle.get("findings_summary", "") or "(none)"
    conditional_resolution = evidence_bundle.get("conditional_resolution", "") or "(not applicable)"
    search_notes = evidence_bundle.get("search_notes", "") or "(none)"
    evidence_list = evidence_bundle.get("evidence", []) or []

    quote_lines = [
        f'  [{i}] "{e.get("quote", "")}" (page {e.get("page", "?")})'
        for i, e in enumerate(evidence_list, start=1)
        if isinstance(e, dict) and e.get("quote")
    ]
    quotes_block = "\n".join(quote_lines) if quote_lines else "  (no verbatim quotes found)"

    return (
        f"Research status: {status} (confidence: {confidence})\n\n"
        "Findings summary (precise/technical research notes -- turn these into the plain-"
        f"language, participant-facing text below):\n{findings_summary}\n\n"
        f"Conditional/alternative resolution:\n{conditional_resolution}\n\n"
        "Verbatim quotes found (copy these EXACTLY if you cite them -- do not paraphrase or "
        f"invent new ones):\n{quotes_block}\n\n"
        f"Research notes/caveats:\n{search_notes}"
    )


def build_draft_messages(
    var: TemplateVariable,
    evidence_bundle: dict,
    repair_feedback: str | None = None,
) -> list[dict]:
    """Build the Stage B [system, user] messages for the drafting LLM call.

    Args:
        var: The template variable being drafted.
        evidence_bundle: The parsed Stage A evidence dict.
        repair_feedback: When set (Stage C bounded repair), a description of
            issues found in a previous draft that this call must fix.
    """
    sub = f" > {var.sub_section}" if var.sub_section else ""
    availability = draft_availability_note(evidence_bundle)
    importance = (
        "REQUIRED -- this section must appear in every ICF."
        if var.required
        else "OPTIONAL -- include only if directly relevant to this specific study."
    )

    json_schema = (
        "{\n"
        '    "reasoning": "Step-by-step: which findings map to which placeholders/conditionals, '
        'which OR alternative applies and why, and what (if anything) is unresolved.",\n'
        f'    "section_id": "{var.section_id}",\n'
        '    "status": "FOUND" | "NOT_FOUND" | "PARTIAL",\n'
        '    "filled_template": "PARTICIPANT-FACING OUTPUT. Required ICF wording with all {{placeholders}} filled from the research findings, <<conditions>> resolved, OR alternatives chosen. Contains ONLY findings content and [PLEASE COMPLETE] for genuinely missing fields -- never sentences about the research process or references to the protocol/findings.",\n'
        '    "evidence": [\n'
        '        {"quote": "Verbatim quote copied EXACTLY from the research findings above", "page": "Page number from the findings"}\n'
        "    ],\n"
        '    "confidence": "HIGH" | "MEDIUM" | "LOW",\n'
        '    "answer": "Plain-language summary of what was found (not patient-facing).",\n'
        '    "notes": "See STUDY TEAM NOTES GUIDANCE above for what belongs here."\n'
        "}"
    )

    lines: list[str] = [
        f"=== DRAFTING TASK: ICF Section [{var.section_id}] ===\n",
        f"TARGET: {var.heading}{sub}",
        f"WHAT TO WRITE: {var.instructions}\n",
    ]
    runtime_ctx = prompt_runtime_context(var)
    if runtime_ctx:
        lines.append(runtime_ctx.rstrip())
    lines.extend([f"AVAILABILITY: {availability}", f"IMPORTANCE: {importance}\n", SYMBOL_GUIDE])

    if var.required_text:
        lines.append(
            "REQUIRED ICF TEXT (mandatory wording -- resolve all template markers and fill in "
            "all {{placeholders}} using the research findings below):\n"
            f"{var.required_text}\n"
        )
    if var.suggested_text:
        lines.append(
            "SUGGESTED ICF TEXT (follow verbatim; apply the symbol rules above. Only modify "
            "when absolutely necessary to keep sentence structure and meaning accurate after "
            "resolving placeholders/conditionals -- do NOT paraphrase or rewrite for style):\n"
            f"{var.suggested_text}\n"
        )

    lines.append("=== RESEARCH FINDINGS (verified by a separate research step) ===")
    lines.append(format_evidence_bundle(evidence_bundle))
    lines.append("=== END OF RESEARCH FINDINGS ===\n")

    if repair_feedback:
        lines.append(
            "YOUR PREVIOUS DRAFT HAD THE FOLLOWING ISSUES -- fix them without changing anything "
            "else, and without inventing new facts beyond the research findings above:\n"
            f"{repair_feedback}\n"
        )

    lines.append(
        "CHAIN-OF-THOUGHT REQUIREMENT: Fill in the 'reasoning' field FIRST with which findings "
        "map to which placeholders, which conditional branches apply, and what (if anything) "
        "remains unresolved.\n"
    )
    lines.append(f"OUTPUT -- respond with ONLY this JSON object, nothing else:\n{json_schema}")

    return [
        {"role": "system", "content": _hybrid_draft_system_prompt(var.section_id)},
        {"role": "user", "content": "\n".join(lines)},
    ]

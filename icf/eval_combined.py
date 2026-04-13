"""
Combined single-call-per-section ICF evaluator.

Instead of separate LLM calls per rubric via DeepEval GEval,
this module scores ALL rubric dimensions in a single LLM call per section.
This reduces cost by ~90% while preserving per-rubric scores and reasoning.

The LLM judge receives:
  - The AI-generated ICF section text
  - Evidence quotes the extractor retrieved (targeted, not a protocol dump)
  - Extraction confidence + status
  - The REB-approved ground truth section (if available)
  - All applicable rubric criteria in one prompt
  - Routing mode (SOFT = caution warning, FULL = normal)

And returns structured JSON with score + grade + reason + evidence_relevance
+ support_level for each rubric.
"""

from __future__ import annotations

import json

from icf.eval_rubrics import RubricDefinition, ScoringMode

# Grade scale matching the detailed mode
GRADE_SCALE = """
Score each rubric on this scale:
  1.0  Excellent  - Fully meets all criteria
  0.8  Good       - Minor issues only
  0.5  Borderline - Noticeable issues requiring revision
  0.3  Poor       - Major issues, significant revision needed
  0.0  Fail       - Does not meet criteria at all

Use intermediate values (e.g. 0.9, 0.7, 0.4, 0.2, 0.1) when appropriate.
"""


def _build_evidence_block(
    evidence: list[dict],
    confidence: str,
    notes: str,
) -> str:
    """Format evidence quotes into a readable block for the judge prompt."""
    lines = [f"Extraction confidence: {confidence}"]
    if notes:
        lines.append(f"Extraction notes: {notes}")
    if evidence:
        lines.append("\nEvidence quotes retrieved from source protocol:")
        for i, ev in enumerate(evidence[:10], 1):
            quote = ev.get("quote", "").strip()
            page = ev.get("page", "")
            section = ev.get("section", "")
            loc = f"(Page {page}" + (f", {section}" if section else "") + ")"
            if quote:
                lines.append(f'  {i}. "{quote}" {loc}')
    else:
        lines.append("\nNo evidence quotes were retrieved from the protocol.")
    return "\n".join(lines)


def _build_combined_prompt(
    section_id: str,
    section_heading: str,
    actual_output: str,
    ground_truth: str | None,
    evidence: list[dict],
    confidence: str,
    notes: str,
    routing_modes: dict[str, ScoringMode],
    rubrics: list[RubricDefinition],
) -> list[dict]:
    """Build a single prompt that asks the judge to score all rubrics at once."""

    # Determine if any rubric is in SOFT mode
    has_soft = any(m == ScoringMode.SOFT for m in routing_modes.values())
    caution_block = ""
    if has_soft:
        caution_block = (
            "\n⚠️  CAUTION: Extraction confidence is MEDIUM or section is partially "
            "in protocol. Evaluate conservatively. Do not penalize for gaps that may "
            "be due to extraction uncertainty rather than AI content errors.\n"
        )

    # Build rubric descriptions block
    rubric_block = ""
    for i, r in enumerate(rubrics, 1):
        rubric_block += f"\n--- Rubric {i}: {r.name} ---\n"
        rubric_block += f"Category: {r.category}\n"
        rubric_block += f"Description: {r.description}\n"
        rubric_block += f"Criteria:\n{r.criteria}\n"

    system_prompt = (
        "You are an expert evaluator for Informed Consent Forms (ICFs) in clinical "
        "research. You are reviewing an AI-generated ICF section for a Canadian "
        "research ethics board (REB) at the University Health Network (UHN).\n\n"
        "You will evaluate the AI-generated text against multiple rubric dimensions "
        "in a SINGLE evaluation.\n\n"
        "For each rubric, provide:\n"
        "  - score (0.0-1.0)\n"
        "  - grade (Excellent/Good/Borderline/Poor/Fail)\n"
        "  - reason (specific explanation)\n"
        "  - evidence_relevance: STRONG / PARTIAL / WEAK / IRRELEVANT "
        "(only for Fidelity and Honesty rubrics — how relevant are the evidence "
        "quotes to this section?)\n"
        "  - support_level: WITHIN / EXCEEDS / NO_EVIDENCE "
        "(only for Fidelity and Honesty rubrics — does the AI text stay within "
        "what the evidence supports, or make claims beyond it?)\n\n"
        f"{GRADE_SCALE}\n"
        "IMPORTANT RULES:\n"
        "- Evidence quotes are verbatim excerpts the extraction engine retrieved "
        "from the protocol. Use them as the grounding reference, not the full protocol.\n"
        "- Unfilled placeholders like {{PI Name}}, <<insert date>> are EXPECTED "
        "for admin/user-fill fields. Do NOT penalize these.\n"
        "- If confidence is LOW or MEDIUM, note uncertainty but do not automatically "
        "penalize — distinguish extraction issues from content quality issues.\n"
        "- Be specific in your reasoning — cite what the AI text does well or misses.\n"
        "- Each rubric is scored independently.\n\n"
        "Respond ONLY with valid JSON in this exact format (no markdown, no extra text):\n"
        "{\n"
        '  "Rubric Name 1": {\n'
        '    "score": 0.8, "grade": "Good", "reason": "...",\n'
        '    "evidence_relevance": "STRONG", "support_level": "WITHIN"\n'
        '  },\n'
        '  "Rubric Name 2": {\n'
        '    "score": 0.5, "grade": "Borderline", "reason": "...",\n'
        '    "evidence_relevance": "", "support_level": ""\n'
        '  }\n'
        "}\n\n"
        "Use the exact rubric names as JSON keys. "
        "For rubrics other than Fidelity/Honesty, leave evidence_relevance and "
        "support_level as empty strings."
    )

    evidence_block = _build_evidence_block(evidence, confidence, notes)

    user_parts = [
        "=== ICF SECTION BEING EVALUATED ===",
        f"Section: [{section_id}] {section_heading}",
        f"\nAI-Generated Text:\n{actual_output}",
        f"\n=== EXTRACTION GROUNDING ===\n{evidence_block}",
    ]

    if caution_block:
        user_parts.append(caution_block)

    if ground_truth:
        user_parts.append(
            f"\n=== REB-APPROVED GROUND TRUTH (reference) ===\n{ground_truth}"
        )

    user_parts.append(f"\n=== RUBRICS TO EVALUATE ===\n{rubric_block}")
    user_parts.append(
        "\nNow score each rubric. Return ONLY valid JSON with the exact rubric names as keys."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def _parse_combined_response(
    raw: str,
    rubrics: list[RubricDefinition],
) -> dict[str, dict]:
    """Parse the JSON response from the combined evaluation call.

    Returns {rubric_name: {score, grade, reason, evidence_relevance, support_level}}.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    results = {}
    rubric_names = {r.name for r in rubrics}

    for key, val in data.items():
        matched_name = None
        if key in rubric_names:
            matched_name = key
        else:
            for rn in rubric_names:
                if key.lower().replace("_", " ") == rn.lower().replace("_", " "):
                    matched_name = rn
                    break
            if not matched_name:
                for rn in rubric_names:
                    if key.lower() in rn.lower() or rn.lower() in key.lower():
                        matched_name = rn
                        break

        if matched_name and isinstance(val, dict):
            score = float(val.get("score", 0))
            grade = str(val.get("grade", ""))
            reason = str(val.get("reason", ""))
            evidence_relevance = str(val.get("evidence_relevance", ""))
            support_level = str(val.get("support_level", ""))

            valid_grades = {"Excellent", "Good", "Borderline", "Poor", "Fail"}
            if grade not in valid_grades:
                if score >= 0.9:
                    grade = "Excellent"
                elif score >= 0.7:
                    grade = "Good"
                elif score >= 0.5:
                    grade = "Borderline"
                elif score >= 0.25:
                    grade = "Poor"
                else:
                    grade = "Fail"

            valid_relevance = {"STRONG", "PARTIAL", "WEAK", "IRRELEVANT", ""}
            if evidence_relevance not in valid_relevance:
                evidence_relevance = ""

            valid_support = {"WITHIN", "EXCEEDS", "NO_EVIDENCE", ""}
            if support_level not in valid_support:
                support_level = ""

            results[matched_name] = {
                "score": max(0.0, min(1.0, score)),
                "grade": grade,
                "reason": reason,
                "evidence_relevance": evidence_relevance,
                "support_level": support_level,
            }

    return results


def evaluate_section_combined(
    section_id: str,
    section_heading: str,
    actual_output: str,
    ground_truth: str | None,
    evidence: list[dict],
    confidence: str,
    notes: str,
    routing_modes: dict[str, ScoringMode],
    rubrics: list[RubricDefinition],
    client,
    model_name: str,
    verbose: bool = False,
    # Legacy parameter kept for backwards compatibility — no longer used
    protocol_context: str | None = None,
) -> dict[str, dict]:
    """Evaluate a single section across all rubrics in one LLM call.

    Uses evidence quotes from the extraction report instead of a protocol dump.

    Returns
    -------
    dict mapping rubric_name -> {score, grade, reason, evidence_relevance, support_level}
    """
    llm_rubrics = [r for r in rubrics if not r.deterministic]
    if not llm_rubrics:
        return {}

    messages = _build_combined_prompt(
        section_id=section_id,
        section_heading=section_heading,
        actual_output=actual_output,
        ground_truth=ground_truth,
        evidence=evidence,
        confidence=confidence,
        notes=notes,
        routing_modes=routing_modes,
        rubrics=llm_rubrics,
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        raw = response.choices[0].message.content
    except Exception as e:
        if verbose:
            print(f"  [COMBINED] LLM call error: {e}")
        return {
            r.name: {
                "score": -1.0, "grade": "ERROR",
                "reason": f"{type(e).__name__}: {e}",
                "evidence_relevance": "", "support_level": "",
            }
            for r in llm_rubrics
        }

    results = _parse_combined_response(raw, llm_rubrics)

    for r in llm_rubrics:
        if r.name not in results:
            results[r.name] = {
                "score": -1.0, "grade": "ERROR",
                "reason": "Rubric not found in LLM response",
                "evidence_relevance": "", "support_level": "",
            }

    return results

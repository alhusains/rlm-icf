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

    # Notes first — describe what the backend found, even when few quotes stored
    if notes:
        lines.append(f"\nWhat the backend found in the protocol:\n{notes}")

    if evidence:
        lines.append(f"\nVerbatim evidence quotes ({len(evidence)}):")
        for i, ev in enumerate(evidence[:10], 1):
            quote = ev.get("quote", "").strip()
            page = ev.get("page", "")
            section = ev.get("section", "")
            loc = f"(Page {page}" + (f", {section}" if section else "") + ")"
            if quote:
                lines.append(f'  {i}. "{quote}" {loc}')
    else:
        lines.append("\nNo verbatim evidence quotes were stored for this section.")
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
    instructions: str = "",
    required_text: str = "",
    suggested_text: str = "",
    full_gt_icf: str = "",
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
        "IMPORTANT RULES:\n\n"
        "1. TASK SCOPE — The AI was given specific instructions for this section "
        "(shown below). Evaluate whether the AI completed its task correctly. "
        "Do not penalize the AI for content it was not asked to generate.\n\n"
        "2. REQUIRED UHN LANGUAGE — Each section has required UHN guideline language "
        "(shown below). The AI must reflect this language faithfully — same meaning "
        "and same consent obligation as defined by UHN guidelines. If it is present "
        "and faithful, give full credit for that component. If it is missing or its "
        "consent meaning is altered, penalize.\n\n"
        "3. SUGGESTED TEXT — Each section has optional conditional template content "
        "(shown below). The AI should include relevant suggested content only when "
        "the protocol evidence supports it. Do NOT penalize the AI for omitting "
        "suggested content that the protocol does not support.\n\n"
        "4. GROUND TRUTH EXTRAS — The REB-approved ground truth may contain "
        "additional sentences or elaborations beyond the AI's task scope or beyond "
        "what the protocol evidence supports. Do NOT penalize the AI for omitting "
        "GT content that goes beyond its task instructions or its evidence. The AI "
        "is evaluated on whether it correctly completed its task — not on whether "
        "it matches the GT word for word.\n\n"
        "5. PLACEHOLDERS — [TO BE FILLED MANUALLY], {{field name}}, <<insert here>> "
        "in the AI output mean the AI correctly acknowledged that information was "
        "not available. When the extraction notes confirm the information was not "
        "found in the protocol, treat placeholders as correct abstention — do NOT "
        "score as fabrication or failure for Honesty or Fidelity.\n\n"
        "6. VERIFY USING ALL SIGNALS — Use extraction notes + verbatim quotes + "
        "ground truth together to verify content. The notes describe what the "
        "backend actually found in the protocol even when few verbatim quotes are "
        "stored. Do not conclude fabrication based on missing quotes alone if the "
        "notes confirm the content exists in the protocol.\n\n"
        "7. GENUINE FABRICATION — Content that is not in the evidence, not in the "
        "notes, not in the required/suggested text, and contradicts or goes beyond "
        "what the GT shows = fabrication. Penalize this firmly.\n\n"
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

    user_parts = []

    # Full GT ICF — document-level reference so judge understands the full consent story
    if full_gt_icf:
        user_parts.append(
            "=== FULL REB-APPROVED ICF (document-level reference) ===\n"
            "This is the complete approved ICF for this study. Use it to understand "
            "the full consent story and context — NOT as a section-by-section answer key.\n\n"
            + full_gt_icf
        )

    user_parts += [
        "=== ICF SECTION BEING EVALUATED ===",
        f"Section: [{section_id}] {section_heading}",
        f"\nAI-Generated Text:\n{actual_output}",
        f"\n=== EXTRACTION GROUNDING ===\n{evidence_block}",
    ]

    if caution_block:
        user_parts.append(caution_block)

    # Task context from registry
    task_block = "\n=== TASK CONTEXT (UHN ICF Guidelines for this section) ==="
    if instructions:
        task_block += f"\n\nSection Instructions (what the AI was asked to do):\n{instructions}"
    if required_text:
        task_block += f"\n\nRequired UHN Language (must be present and faithful):\n{required_text}"
    if suggested_text:
        task_block += (
            f"\n\nSuggested Template Content (conditional — include only when "
            f"protocol supports it):\n{suggested_text}"
        )
    if instructions or required_text or suggested_text:
        user_parts.append(task_block)

    if ground_truth:
        user_parts.append(
            f"\n=== REB-APPROVED GROUND TRUTH FOR THIS SECTION (reference) ===\n{ground_truth}"
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
    instructions: str = "",
    required_text: str = "",
    suggested_text: str = "",
    full_gt_icf: str = "",
    # Legacy parameter kept for backwards compatibility — no longer used
    protocol_context: str | None = None,
) -> dict[str, dict]:
    """Evaluate a single section across all rubrics in one LLM call.

    Uses evidence quotes from the extraction report instead of a protocol dump.
    Passes UHN registry context (instructions, required_text, suggested_text)
    so the judge understands the task and can distinguish required from optional content.

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
        instructions=instructions,
        required_text=required_text,
        suggested_text=suggested_text,
        full_gt_icf=full_gt_icf,
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


# ------------------------------------------------------------------
# Document-level evaluation (runs once on full concatenated output)
# ------------------------------------------------------------------


def evaluate_document_level(
    full_text: str,
    rubric,
    client,
    model_name: str,
    verbose: bool = False,
) -> dict:
    """Score a single document-level rubric on the full concatenated ICF output.

    Returns {score, grade, reason, issues}.
    """
    system_msg = (
        "You are an expert clinical research ethics reviewer evaluating an "
        "AI-generated Informed Consent Form (ICF). You are reviewing the "
        "ENTIRE document for cross-section quality issues.\n\n"
        + GRADE_SCALE
    )

    user_msg = (
        f"## Rubric: {rubric.name}\n\n"
        f"{rubric.criteria}\n\n"
        "---\n\n"
        "## Full AI-Generated ICF Document\n\n"
        f"{full_text}\n\n"
        "---\n\n"
        "Respond with ONLY valid JSON:\n"
        "```json\n"
        "{\n"
        f'  "{rubric.name}": {{\n'
        '    "score": <float 0.0-1.0>,\n'
        '    "grade": "<Excellent|Good|Borderline|Poor|Fail>",\n'
        '    "reason": "<specific issues found or why the document scores well>",\n'
        '    "issues": ["<issue 1>", "<issue 2>", "..."]\n'
        "  }\n"
        "}\n"
        "```\n"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        raw = response.choices[0].message.content
    except Exception as e:
        if verbose:
            print(f"  [DOC-LEVEL] LLM call error: {e}")
        return {
            "score": -1.0, "grade": "ERROR",
            "reason": f"{type(e).__name__}: {e}",
            "issues": [],
        }

    # Parse response
    try:
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1]
        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0]
        parsed = json.loads(cleaned.strip())

        if rubric.name in parsed:
            data = parsed[rubric.name]
        else:
            data = parsed

        score = float(data.get("score", -1.0))
        grade = data.get("grade", "ERROR")
        reason = data.get("reason", "")
        issues = data.get("issues", [])

        valid_grades = {"Excellent", "Good", "Borderline", "Poor", "Fail"}
        if grade not in valid_grades:
            grade = "ERROR"

        return {"score": score, "grade": grade, "reason": reason, "issues": issues}
    except Exception as e:
        if verbose:
            print(f"  [DOC-LEVEL] Parse error: {e}\nRaw: {raw[:300]}")
        return {
            "score": -1.0, "grade": "ERROR",
            "reason": f"Failed to parse judge response: {e}",
            "issues": [],
        }

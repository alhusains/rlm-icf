"""
Combined single-call-per-section ICF evaluator.

Instead of 9 separate LLM calls per section (one per rubric via DeepEval GEval),
this module scores ALL rubric dimensions in a single LLM call per section.
This reduces cost by ~90% while preserving per-rubric scores and reasoning.

The LLM judge receives:
  - The AI-generated ICF section text
  - The REB-approved ground truth section (if available)
  - The source protocol context (if available)
  - All rubric criteria in one prompt

And returns a structured JSON with score + grade + reason for each rubric.
"""

from __future__ import annotations

import json
import os

from icf.eval_rubrics import ALL_RUBRICS, READING_LEVEL, RubricDefinition

# Grade scale matching the detailed mode
GRADE_SCALE = """
Score each rubric on this scale:
  1.0  Excellent - Fully meets all criteria
  0.8  Good      - Minor issues only
  0.5  Borderline - Noticeable issues requiring revision
  0.3  Poor      - Major issues, significant revision needed
  0.0  Fail      - Does not meet criteria at all

Use intermediate values (e.g. 0.9, 0.7, 0.4, 0.2, 0.1) when appropriate.
"""

def _build_combined_prompt(
    section_id: str,
    section_heading: str,
    actual_output: str,
    ground_truth: str | None,
    protocol_context: str | None,
    rubrics: list[RubricDefinition],
) -> list[dict]:
    """Build a single prompt that asks the judge to score all rubrics at once."""

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
        "in a SINGLE evaluation. For each rubric, provide a score (0.0-1.0), a grade, "
        "and a brief but specific reason explaining why you gave that score.\n\n"
        f"{GRADE_SCALE}\n"
        "IMPORTANT:\n"
        "- Be specific in your reasoning — cite what the AI text does well or misses.\n"
        "- If ground truth (approved ICF) is provided, compare against it.\n"
        "- If protocol context is provided, verify fidelity and honesty against it.\n"
        "- Each rubric should be scored independently.\n\n"
        "Respond ONLY with valid JSON in this exact format (no markdown, no extra text):\n"
        "{\n"
        '  "rubric_name_1": {"score": 0.8, "grade": "Good", "reason": "..."},\n'
        '  "rubric_name_2": {"score": 0.5, "grade": "Borderline", "reason": "..."},\n'
        "  ...\n"
        "}\n\n"
        "Use the exact rubric names as JSON keys."
    )

    # Build user message with all context
    user_parts = [
        f"=== ICF SECTION BEING EVALUATED ===",
        f"Section: [{section_id}] {section_heading}",
        f"\nAI-Generated Text:\n{actual_output}",
    ]

    if ground_truth:
        user_parts.append(
            f"\n=== REB-APPROVED GROUND TRUTH (reference) ===\n{ground_truth}"
        )

    if protocol_context:
        # Trim protocol to avoid token limits
        trimmed = protocol_context[:40000]
        user_parts.append(
            f"\n=== SOURCE PROTOCOL (context) ===\n{trimmed}"
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

    Returns {rubric_name: {"score": float, "grade": str, "reason": str}}.
    """
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON block in the response
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
        # Match rubric name (exact or fuzzy)
        matched_name = None
        if key in rubric_names:
            matched_name = key
        else:
            # Try case-insensitive match
            for rn in rubric_names:
                if key.lower().replace("_", " ") == rn.lower().replace("_", " "):
                    matched_name = rn
                    break
            # Try substring match
            if not matched_name:
                for rn in rubric_names:
                    if key.lower() in rn.lower() or rn.lower() in key.lower():
                        matched_name = rn
                        break

        if matched_name and isinstance(val, dict):
            score = float(val.get("score", 0))
            grade = str(val.get("grade", ""))
            reason = str(val.get("reason", ""))

            # Validate grade
            valid_grades = {"Excellent", "Good", "Borderline", "Poor", "Fail"}
            if grade not in valid_grades:
                # Infer from score
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

            results[matched_name] = {
                "score": max(0.0, min(1.0, score)),
                "grade": grade,
                "reason": reason,
            }

    return results


def evaluate_section_combined(
    section_id: str,
    section_heading: str,
    actual_output: str,
    ground_truth: str | None,
    protocol_context: str | None,
    rubrics: list[RubricDefinition],
    client,
    model_name: str,
    verbose: bool = False,
) -> dict[str, dict]:
    """Evaluate a single section across all rubrics in one LLM call.

    Parameters
    ----------
    client : openai.AzureOpenAI or similar
        The LLM client to call.
    model_name : str
        The deployment/model name.

    Returns
    -------
    dict mapping rubric_name -> {"score": float, "grade": str, "reason": str}
    """
    # Filter out deterministic rubrics (handled separately)
    llm_rubrics = [r for r in rubrics if not r.deterministic]

    if not llm_rubrics:
        return {}

    messages = _build_combined_prompt(
        section_id=section_id,
        section_heading=section_heading,
        actual_output=actual_output,
        ground_truth=ground_truth,
        protocol_context=protocol_context,
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
        # Return error scores for all rubrics
        return {
            r.name: {"score": -1.0, "grade": "ERROR", "reason": f"{type(e).__name__}: {e}"}
            for r in llm_rubrics
        }

    results = _parse_combined_response(raw, llm_rubrics)

    # Fill in any missing rubrics with error
    for r in llm_rubrics:
        if r.name not in results:
            results[r.name] = {
                "score": -1.0,
                "grade": "ERROR",
                "reason": "Rubric not found in LLM response",
            }

    return results

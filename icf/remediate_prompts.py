"""
Prompt building for Stage 9 — HIGH Flag Remediation.

Three public functions:
  extract_locked_phrases      — extract literal mandatory phrases from required_text
                                so the validation step can check them after patching.
  build_global_rules_prompt   — Pass A prompt: parse cross_section_notes + flag list
                                into a structured list[GlobalFixRule].
  build_patch_prompt          — Pass B prompt: rewrite one section to fix HIGH flags
                                and apply applicable global rules, while preserving
                                all required text and every clinical fact.
"""

from __future__ import annotations

import re

from icf.types import GlobalFixRule, ReviewFlag, TemplateVariable

# ---------------------------------------------------------------------------
# Protected text extraction
# ---------------------------------------------------------------------------

# Template markers used as split-points when extracting locked literal text.
# The literal segments BETWEEN these markers are the phrases we must preserve.
_TEMPLATE_MARKER_RE = re.compile(
    r"\{\{[^}]+\}\}"  # {{placeholder}} or {{option1/option2}}
    r"|<<[^>]+>>"  # <<conditional block>>
    r"|<[^>]+>"  # <conditional marker>
    r"|\bOR\b"  # standalone OR keyword
    r"|^\s*[•\-]\s*",  # leading bullet markers
    re.MULTILINE,
)

_MIN_PHRASE_LEN = 10  # ignore fragments shorter than this


def extract_locked_phrases(required_text: str) -> list[str]:
    """Return literal text fragments that must survive verbatim after patching.

    Splits required_text on template markers (placeholders, conditionals,
    OR alternatives, bullet markers) and returns the non-trivial literal
    segments between them.  Each returned fragment is a piece of real text
    that the validation step checks is still present in the patched output.
    """
    if not required_text or not required_text.strip():
        return []

    # Split on every template marker; the parts between them are literal text.
    segments = _TEMPLATE_MARKER_RE.split(required_text)

    phrases: list[str] = []
    for seg in segments:
        for line in re.split(r"\n+", seg):
            fragment = line.strip()
            if len(fragment) >= _MIN_PHRASE_LEN:
                phrases.append(fragment)

    return phrases


# ---------------------------------------------------------------------------
# Pass A — Cross-Section Global Rules prompt
# ---------------------------------------------------------------------------

_GLOBAL_RULES_SYSTEM = (
    "You are a clinical document editor reviewing the quality notes for a draft "
    "Informed Consent Form (ICF) intended to be read by patients.\n\n"
    "You will be given:\n"
    "  1. Cross-section notes written by a plain-language reviewer.\n"
    "  2. A list of individual section flags (issue type, severity, suggestion).\n\n"
    "Your job is to produce a structured list of DOCUMENT-WIDE fix rules that should "
    "be applied consistently across all affected sections.\n\n"
    "Rules:\n"
    "  1. Return ONLY a JSON array. No prose outside the JSON.\n"
    "  2. Each item must have exactly these keys:\n"
    '     "rule_type": one of "define_abbreviation" | "standardize_term" | '
    '"fix_inconsistency" | "note_only"\n'
    '     "description": one clear sentence describing the fix.\n'
    '     "affected_section_ids": list of section ID strings where this rule applies.\n'
    "  3. Use rule_type 'note_only' for structural repetition — do NOT recommend "
    "automated removal of repeated content. ICF repetition is often intentional for "
    "patient comprehension.\n"
    "  4. Keep rules targeted: only include a rule if it is clearly warranted by the "
    "notes or flags. Do not invent rules.\n"
    "  5. Affected_section_ids must only contain IDs actually mentioned in the input.\n"
    "  6. If there are no actionable cross-section rules, return an empty array [].\n"
)


def build_global_rules_prompt(
    cross_section_notes: str,
    flags: list[ReviewFlag],
    variables: list[TemplateVariable],
) -> list[dict]:
    """Build the [system, user] messages list for the Pass A global-rules LLM call."""
    var_index = {v.section_id: v.heading for v in variables}

    # List the valid bare section IDs so the LLM copies them exactly.
    all_ids = sorted({f.section_id for f in flags} | set(var_index.keys()))
    valid_ids_line = (
        "VALID SECTION IDs (use these exact strings in affected_section_ids): " + ", ".join(all_ids)
    )

    flag_lines = []
    for f in flags:
        heading = var_index.get(f.section_id, "")
        # Format as "id (heading)" so the bare ID is clearly separated from the label.
        flag_lines.append(
            f"  id={f.section_id!r} ({heading}) "
            f"severity={f.severity} type={f.issue_type}: {f.suggestion}"
        )
    flags_text = "\n".join(flag_lines) if flag_lines else "  (none)"

    user_content = (
        f"{valid_ids_line}\n\n"
        "CROSS-SECTION REVIEWER NOTES:\n"
        f"{cross_section_notes.strip()}\n\n"
        "INDIVIDUAL SECTION FLAGS (all severities, for context):\n"
        f"{flags_text}\n\n"
        "OUTPUT — respond with ONLY a JSON array, nothing else.\n"
        'Use the bare section IDs from the VALID SECTION IDs line above (e.g. "3", "9.2", "21.1").\n'
        "Do NOT prefix them with 'SECTION' or any other word.\n"
        "[\n"
        "  {\n"
        '    "rule_type": "define_abbreviation | standardize_term | fix_inconsistency | note_only",\n'
        '    "description": "One clear sentence describing the fix.",\n'
        '    "affected_section_ids": ["3", "9.2"]\n'
        "  }\n"
        "]"
    )

    return [
        {"role": "system", "content": _GLOBAL_RULES_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Pass B — Per-Section Patch prompt
# ---------------------------------------------------------------------------

_PATCH_SYSTEM = (
    "You are a plain-language editor for Informed Consent Forms (ICFs) at UHN "
    "(University Health Network).\n\n"
    "You will receive the current text of one ICF section, a list of HIGH-severity "
    "issues to fix, and any document-wide terminology rules to apply.\n\n"
    "Rules:\n"
    "  1. Return ONLY the revised section text. No JSON wrapper, no preamble, no "
    "explanation — just the corrected text.\n"
    "  2. THIS IS A PATIENT-FACING DOCUMENT. Preserve every clinical fact. Do not "
    "remove any information the patient needs to make an informed decision.\n"
    "  3. Make the minimum change necessary to fix each issue. Do not rewrite "
    "sentences that are not flagged.\n"
    "  4. If LOCKED PHRASES are provided, those exact strings must appear verbatim "
    "in your output. Do not alter, paraphrase, or omit them.\n"
    "  5. When applying terminology rules, replace like-for-like. Do not change "
    "meaning or omit surrounding context.\n"
    "  6. When adding an abbreviation definition, insert the expansion in "
    "parentheses immediately after the first occurrence of the abbreviation in "
    "this section, e.g. 'alloHCT (allogeneic stem cell transplant)'.\n"
)


def build_patch_prompt(
    section_id: str,
    heading: str,
    filled_template: str,
    locked_phrases: list[str],
    high_flags: list[ReviewFlag],
    applicable_rules: list[GlobalFixRule],
) -> list[dict]:
    """Build the [system, user] messages list for a single Pass B patch call.

    high_flags may be empty if the section is only in scope due to global rules.
    """
    # Locked phrases block
    if locked_phrases:
        locked_block = (
            "LOCKED PHRASES — copy these verbatim into your output, do not alter:\n"
            + "\n".join(f"  - {p}" for p in locked_phrases)
        )
    else:
        locked_block = "LOCKED PHRASES: (none — this section has no mandated wording)"

    # HIGH flags block
    if high_flags:
        flag_lines = []
        for i, f in enumerate(high_flags, 1):
            flag_lines.append(
                f'  {i}. [{f.issue_type}] Flagged text: "{f.flagged_text}"\n'
                f"     Suggestion: {f.suggestion}"
            )
        flags_block = "HIGH-SEVERITY ISSUES TO FIX:\n" + "\n".join(flag_lines)
    else:
        flags_block = "HIGH-SEVERITY ISSUES TO FIX: (none — apply global rules only)"

    # Global rules block
    if applicable_rules:
        rule_lines = [
            f"  {i}. [{r.rule_type}] {r.description}" for i, r in enumerate(applicable_rules, 1)
        ]
        rules_block = "DOCUMENT-WIDE RULES TO APPLY IN THIS SECTION:\n" + "\n".join(rule_lines)
    else:
        rules_block = "DOCUMENT-WIDE RULES: (none)"

    user_content = (
        f"SECTION {section_id}: {heading.upper()}\n\n"
        f"{locked_block}\n\n"
        f"{flags_block}\n\n"
        f"{rules_block}\n\n"
        "CURRENT SECTION TEXT:\n"
        f"{filled_template.strip()}\n\n"
        "OUTPUT — the revised section text only, nothing else:"
    )

    return [
        {"role": "system", "content": _PATCH_SYSTEM},
        {"role": "user", "content": user_content},
    ]

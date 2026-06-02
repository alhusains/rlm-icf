"""
Prompt building for Stage 9 — Review Flag Remediation.

Three public functions:
  extract_locked_phrases      — extract literal mandatory phrases from required_text
                                so the validation step can check them after patching.
  build_global_rules_prompt   — Pass A prompt: parse cross_section_notes + flag list
                                into a structured list[GlobalFixRule].
  build_patch_prompt          — Pass B prompt: rewrite one section to fix remediable
                                review flags and apply global rules, while preserving
                                all required text and every clinical fact.
"""

from __future__ import annotations

import re

from icf.plain_language import PLAIN_LANGUAGE_SCOPE, UHN_PLAIN_LANGUAGE_GUIDELINES
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

# Mutually exclusive template branches (e.g. <<Option 1: ...>> / <<Option 2: ...>>).
_OPTION_HEADER_RE = re.compile(
    r"<<Option\s+\d+[^>]*>>|<<OPTION\s+[A-Z][^>]*>>",
    re.IGNORECASE,
)

# Standalone <<instruction>> lines (not option headers) — omitted from locking.
_DIRECTIVE_LINE_RE = re.compile(r"^\s*<<[^>]+>>\s*$", re.MULTILINE)


def _partition_exclusive_option_bodies(required_text: str) -> list[str]:
    """Split required_text into bodies following <<Option N>> / <<OPTION X>> headers."""
    headers = list(_OPTION_HEADER_RE.finditer(required_text))
    if not headers:
        return []

    bodies: list[str] = []
    for i, match in enumerate(headers):
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(required_text)
        bodies.append(required_text[start:end])
    return bodies


def _strip_directive_lines(text: str) -> str:
    """Remove instruction-only <<...>> lines; keep literal text and option bodies."""
    return _DIRECTIVE_LINE_RE.sub("", text)


def _literal_phrases_from_text(text: str) -> list[str]:
    """Extract non-trivial literal fragments between template markers."""
    segments = _TEMPLATE_MARKER_RE.split(_strip_directive_lines(text))
    phrases: list[str] = []
    for seg in segments:
        for line in re.split(r"\n+", seg):
            fragment = line.strip()
            if len(fragment) >= _MIN_PHRASE_LEN:
                phrases.append(fragment)
    return phrases


def _select_option_body(required_text: str, filled_template: str) -> str | None:
    """Pick the mutually exclusive option branch that best matches the draft."""
    bodies = _partition_exclusive_option_bodies(required_text)
    if not bodies:
        return None

    scored = [
        (body, sum(1 for p in _literal_phrases_from_text(body) if p in filled_template))
        for body in bodies
    ]
    best_score = max(score for _, score in scored)
    if best_score == 0:
        return None

    # First branch wins ties (stable, deterministic).
    for body, score in scored:
        if score == best_score:
            return body
    return None


def extract_locked_phrases(
    required_text: str,
    filled_template: str | None = None,
) -> list[str]:
    """Return literal text fragments that must survive verbatim after patching.

    Splits required_text on template markers (placeholders, conditionals,
    OR alternatives, bullet markers) and returns the non-trivial literal
    segments between them.

    When ``filled_template`` is provided:

      1. Mutually exclusive ``<<Option N: ...>>`` / ``<<OPTION X: ...>>`` blocks
         are resolved to the branch whose literals appear in the draft (fixes
         sections like Health Canada Option 1 vs Option 2).
      2. Any phrase not present as a substring of the draft is dropped, so locks
         from unused template branches never block remediation.
    """
    if not required_text or not required_text.strip():
        return []

    scope = required_text
    if filled_template and filled_template.strip():
        selected = _select_option_body(required_text, filled_template)
        if selected is not None:
            scope = selected
        else:
            scope = _strip_directive_lines(required_text)

    phrases = _literal_phrases_from_text(scope)

    if filled_template and filled_template.strip():
        phrases = [p for p in phrases if p in filled_template]

    # Preserve order, drop duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for phrase in phrases:
        if phrase not in seen:
            seen.add(phrase)
            unique.append(phrase)
    return unique


# ---------------------------------------------------------------------------
# Pass A — Cross-Section Global Rules prompt
# ---------------------------------------------------------------------------

_GLOBAL_RULES_SYSTEM = (
    "You are a clinical document editor reviewing the quality notes for a draft "
    "Informed Consent Form (ICF) intended to be read by study participants "
    "(who may be patients, clinicians, healthy volunteers, caregivers, or other research subjects).\n\n"
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
    "participant comprehension.\n"
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
    "You will receive draft section text to revise, a list of review issues to fix "
    "(HIGH and eligible MEDIUM), and any document-wide terminology rules to apply.\n\n"
    "UHN PLAIN LANGUAGE GUIDELINES — apply when revising any text you change:\n"
    + PLAIN_LANGUAGE_SCOPE
    + UHN_PLAIN_LANGUAGE_GUIDELINES
    + "\n"
    "Rules:\n"
    "  1. Respond with the revised section text as plain text (no JSON, no preamble, "
    "no commentary).\n"
    "  2. This is participant-facing consent language. Preserve every clinical fact. "
    "Do not remove information the participant needs to make an informed decision.\n"
    "  3. Make the minimum change necessary to fix each issue. Do not rewrite "
    "sentences that are not flagged.\n"
    "  4. When a suggested replacement is provided, use that wording for the flagged "
    "span when possible.\n"
    "  5. When required wording is listed, keep each listed phrase unchanged "
    "word-for-word in the revised section (including punctuation).\n"
    "  6. When applying terminology rules, replace like-for-like. Do not change "
    "meaning or omit surrounding context.\n"
    "  7. When adding an abbreviation definition, insert the expansion in "
    "parentheses immediately after the first occurrence of the abbreviation in "
    "this section, e.g. 'alloHCT (allogeneic stem cell transplant)'.\n"
    "  8. For PLAIN_LANGUAGE_VIOLATION flags without a suggested replacement, "
    "simplify jargon in the flagged span using the guidelines above while keeping "
    "required wording fragments intact.\n"
)


def build_patch_prompt(
    section_id: str,
    heading: str,
    filled_template: str,
    locked_phrases: list[str],
    flags: list[ReviewFlag],
    applicable_rules: list[GlobalFixRule],
) -> list[dict]:
    """Build the [system, user] messages list for a single Pass B patch call.

    flags may be empty if the section is only in scope due to global rules.
    """
    # Required wording block (avoid "LOCKED"/"copy verbatim into output" — Azure
    # content filters often classify that phrasing as jailbreak/prompt injection).
    if locked_phrases:
        locked_block = (
            "Required wording (preserve each phrase unchanged in the revised section):\n"
            + "\n".join(f"  - {p}" for p in locked_phrases)
        )
    else:
        locked_block = "Required wording: (none — no mandated phrases for this section)"

    # Review flags block
    if flags:
        flag_lines = []
        for i, f in enumerate(flags, 1):
            line = (
                f'  {i}. [{f.severity}] [{f.issue_type}] Flagged text: "{f.flagged_text}"\n'
                f"     Suggestion: {f.suggestion}"
            )
            if f.suggested_fix.strip():
                line += f"\n     Suggested replacement: {f.suggested_fix}"
            flag_lines.append(line)
        flags_block = "ISSUES TO FIX:\n" + "\n".join(flag_lines)
    else:
        flags_block = "ISSUES TO FIX: (none — apply global rules only)"

    # Global rules block
    if applicable_rules:
        rule_lines = [
            f"  {i}. [{r.rule_type}] {r.description}" for i, r in enumerate(applicable_rules, 1)
        ]
        rules_block = "DOCUMENT-WIDE RULES TO APPLY IN THIS SECTION:\n" + "\n".join(rule_lines)
    else:
        rules_block = "DOCUMENT-WIDE RULES: (none)"

    user_content = (
        f"Section {section_id}: {heading}\n\n"
        f"{locked_block}\n\n"
        f"{flags_block}\n\n"
        f"{rules_block}\n\n"
        "Draft section text to revise:\n"
        "---\n"
        f"{filled_template.strip()}\n"
        "---\n\n"
        "Provide the revised section text:"
    )

    return [
        {"role": "system", "content": _PATCH_SYSTEM},
        {"role": "user", "content": user_content},
    ]

"""
Prompt building for Stage 9 — Review Flag Remediation.

Three public functions:
  extract_locked_phrases      — extract literal mandatory phrases from a template
                                text (required_text or suggested_text) so the
                                validation step can check them after patching.
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


# Placeholders like {{will/may}} or {{study/clinical trial (a type of study
# that involves research)}} present a small closed set of literal alternative
# WORDINGS the drafter must pick between and copy verbatim -- unlike free-text
# writing instructions such as {{insert name(s) of product/agent/device}} or
# {{specify condition}}. Once the section is drafted, the alternative actually
# used is a factual/classification choice (e.g. "clinical trial" vs "study"),
# not a stylistic word the plain-language passes should be free to swap out --
# lock it the same way <<Option N>> branches already are.
_INSTRUCTIONAL_PLACEHOLDER_RE = re.compile(
    r"\b(insert|specify|describe|explain|list|choose|include)\b|e\.g\.?",
    re.IGNORECASE,
)
_MAX_LITERAL_ALTERNATIVE_CHARS = 70
_PLACEHOLDER_RE = re.compile(r"\{\{([^}]+)\}\}")


def _literal_choice_alternatives(placeholder_body: str) -> list[str]:
    """Return the literal alternative wordings for a closed-choice placeholder.

    Returns [] if the placeholder reads as a free-text writing instruction
    instead (contains a verb like "insert"/"specify"/"describe", or an
    alternative too long to be a bare word/phrase choice).
    """
    if "/" not in placeholder_body:
        return []
    if _INSTRUCTIONAL_PLACEHOLDER_RE.search(placeholder_body):
        return []
    alternatives = [a.strip() for a in placeholder_body.split("/")]
    if any(not a or len(a) > _MAX_LITERAL_ALTERNATIVE_CHARS for a in alternatives):
        return []
    return alternatives


def _literal_choice_phrases_from_text(text: str) -> list[str]:
    """Collect literal alternative wordings from every closed-choice placeholder in text."""
    phrases: list[str] = []
    for m in _PLACEHOLDER_RE.finditer(text):
        phrases.extend(_literal_choice_alternatives(m.group(1)))
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
    segments between them. Also pulls the chosen alternative out of any
    closed-choice placeholder (e.g. {{will/may}}, {{study/clinical trial
    (a type of study that involves research)}}) -- see
    _literal_choice_alternatives -- since those are factual/classification
    picks the drafter must copy verbatim, not free text.

    When ``filled_template`` is provided:

      1. Mutually exclusive ``<<Option N: ...>>`` / ``<<OPTION X: ...>>`` blocks
         are resolved to the branch whose literals appear in the draft (fixes
         sections like Health Canada Option 1 vs Option 2).
      2. Any phrase not present as a substring of the draft is dropped, so locks
         from unused template branches (or unused closed-choice alternatives)
         never block remediation.
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
    phrases.extend(_literal_choice_phrases_from_text(scope))

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
    '     "rule_type": one of "standardize_term" | "fix_inconsistency" | "note_only"\n'
    '     "description": one clear sentence describing the fix.\n'
    '     "affected_section_ids": list of section ID strings where this rule applies.\n'
    "  3. Use rule_type 'note_only' for structural repetition — do NOT recommend "
    "automated removal of repeated content. ICF repetition is often intentional for "
    "participant comprehension.\n"
    "  4. Keep rules targeted: only include a rule if it is clearly warranted by the "
    "notes or flags. Do not invent rules.\n"
    "  5. Affected_section_ids must only contain IDs actually mentioned in the input.\n"
    "  6. If there are no actionable cross-section rules, return an empty array [].\n"
    "  7. Do NOT produce rules about WHERE an abbreviation should be defined vs. used "
    "bare (e.g. 'define X in section Y, use alone elsewhere') even if the notes mention "
    "it -- that is handled separately by a deterministic document scan that always sees "
    "the full final text, which you do not. An LLM guess about placement here can "
    "contradict that scan and cause the abbreviation to end up undefined everywhere. "
    "If the notes raise an abbreviation issue, leave it out entirely rather than guessing.\n"
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
        '    "rule_type": "standardize_term | fix_inconsistency | note_only",\n'
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
    "sentences that are not flagged, EXCEPT when a fix would otherwise leave the section "
    "incoherent -- e.g. a nearby sentence that only made sense because of a term/phrase "
    "you just replaced elsewhere, or a defining sentence ('X means ...') that is no longer "
    "needed because the term it defined was simplified away or removed. In those cases, "
    "make the smallest possible adjustment to the affected nearby text (never to required "
    "wording, never adding or removing facts) so the section reads as one coherent passage. "
    "This also applies when several issues are flagged inside the SAME sentence: if applying "
    "every suggested replacement in place would stack them into one long run-on (this is "
    "common when a sentence lists two or more alternative options, e.g. 'treatment is X, or "
    "Y'), split that sentence into shorter ones instead of inserting every fix into the same "
    "clause -- e.g. one short sentence naming the options, then one sentence per option for "
    "its detail. Never split or reorder text that is Required wording (see below).\n"
    "  4. When a suggested replacement is provided, use that wording for the flagged "
    "span when possible. If the exact same replacement wording would then appear twice in "
    "nearby sentences, vary the second occurrence (e.g. a short callback like 'these tests' "
    "or 'this') instead of repeating the full phrase verbatim -- do not trade jargon for "
    "robotic repetition.\n"
    "  5. When required wording is listed, keep each listed phrase unchanged "
    "word-for-word in the revised section (including punctuation).\n"
    "  6. When applying terminology rules, replace like-for-like. Do not change "
    "meaning or omit surrounding context.\n"
    "  7. Abbreviation rules (define_abbreviation) are DOCUMENT-WIDE. Always use "
    "the form Full Term (ABB) — full term first, abbreviation in parentheses — "
    "e.g. 'Magnetic Resonance Imaging (MRI)'. NEVER write 'MRI (Magnetic Resonance "
    "Imaging)'. If this section is the first document occurrence named in the rule, "
    "introduce the term once as Full Term (ABB) and use the abbreviation alone for "
    "any later mentions in the same section. If this section is a later occurrence, "
    "use the abbreviation alone and remove any redundant full-term expansions.\n"
    "  8. For PLAIN_LANGUAGE_VIOLATION flags without a suggested replacement, "
    "simplify jargon in the flagged span using the guidelines above while keeping "
    "required wording fragments intact.\n"
    "  8b. Whenever you simplify or rewrite a phrase, check whether it contains a "
    "parenthetical abbreviation, e.g. 'a pre-emptive therapy strategy (PET)'. Never leave "
    "the abbreviation behind attached to unrelated words after paraphrasing away the term "
    "it stood for (e.g. rewriting to 'early treatment if the virus appears (PET)' is WRONG "
    "-- nothing left in that sentence means 'PET' anymore). Per the plain-language "
    "guidelines' abbreviation section, either keep the term+abbreviation together (named "
    "tests/scans/drugs/devices the participant must recognize later) or remove the term AND "
    "its abbreviation together (internal study-only labels) -- never one without the other.\n"
    "  9. Before finalizing, re-read the full section once as connected prose (not "
    "flag-by-flag): fix any redundancy, broken transitions, or leftover sentences your "
    "edits created, while preserving every fact and every phrase listed in 'Required "
    "wording' unchanged.\n"
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

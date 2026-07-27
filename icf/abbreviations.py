"""
Stage 9.5 -- Document-Wide Abbreviation Consistency.

The plain-language guidelines require every abbreviation to be spelled out in
full only ONCE, at its first use in reading order, with the bare abbreviation
used everywhere after (see UHN_PLAIN_LANGUAGE_GUIDELINES in plain_language.py).
Each section is drafted independently (no cross-section context), so a
drafting LLM call that dutifully follows "define on first use" only ever sees
"first use within THIS section" -- the same abbreviation routinely gets
spelled out again in every section that happens to use it.

Stage 8 review is a single LLM call over the whole assembled document, so in
principle it could catch this, but each repeated definition looks completely
correct in isolation ("Magnetic Resonance Imaging (MRI)" is properly formatted
plain language) -- the violation only exists when comparing across sections,
which is exactly the kind of holistic, easy-to-miss bookkeeping an LLM's
best-effort attention over a long document is unreliable at. This is also a
fully mechanical rule (first occurrence in reading order wins), so rather than
depending on the review LLM to notice every repeat, detect it programmatically
and feed the results into Stage 9 remediation as ordinary GlobalFixRules --
reusing the same per-section patch + locked-phrase safety net as every other
remediation rule.

A related but distinct failure mode is an ORPHANED abbreviation: review's
suggested_fix or remediation's own patch simplifies away the term an
abbreviation stood for but leaves the abbreviation itself behind, e.g.
"a pre-emptive therapy strategy (PET)" rewritten to "early treatment if the
virus appears (PET)" -- nothing in the rewrite means "PET" anymore. Prompt
guidance (see UHN_PLAIN_LANGUAGE_GUIDELINES and the review/patch system
prompts) tells the model not to do this, but since it is a subtle, easy edit
to miss, this module also acts as a deterministic backstop: any "(ABBR)" whose
preceding words have no plausible relationship to it is flagged for repair
regardless of prompt compliance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from icf.types import ExtractionResult, TemplateVariable

# Statuses whose content participates in the document (matches review_prompts.py).
_REVIEWABLE_STATUSES = ("FOUND", "PARTIAL", "STANDARD_TEXT")

# Top-level section group excluded from the document scan, matching the same
# exclusion in review_prompts.py (_REVIEW_PROTECTED_TOPS) and remediate.py
# (_REMEDIATION_LOCKED_TOPS). 2.x holds cover-page fields (title, protocol #,
# study doctor, sponsor) -- short factual identifiers, not prose. An
# abbreviation used incidentally in the study's short title (e.g. "Short
# Title: Paralytic in SNM") is not a genuine "first use" that later prose
# should be measured against, and this section can't be patched anyway (see
# remediate.py's own protected_ids) -- including it here only produces
# incorrect "define it here" instructions for an untouchable section.
_PROTECTED_TOPS = frozenset({"2"})


def _is_protected(section_id: str) -> bool:
    top = (section_id or "").strip().split(".", 1)[0]
    return top in _PROTECTED_TOPS


# Common function words excluded from the captured term so the match doesn't
# run backwards across a clause boundary, e.g. "...blood tests to check for
# Cytomegalovirus (CMV)" should capture "Cytomegalovirus", not the whole
# clause. This only affects the wording used in the generated instruction --
# it has no effect on which abbreviation/section pairs are detected.
_STOPWORDS = (
    "a|an|the|of|for|to|with|and|or|in|on|at|by|from|is|are|was|were|will|"
    "may|can|this|that|these|those|as|if|than|then|your|you|their|its|it|"
    "be|been|has|have|had|check|checking|tests|test|testing|about|into|"
    "where|which|who|whom|whose|when|what|why|how|o"
)
_TERM_WORD = rf"\b(?!(?i:{_STOPWORDS})\b)[A-Za-z][A-Za-z\-]*"

# "Full Term (ABBR)" -- a short run of words immediately followed by a
# parenthetical. The abbreviation must contain >= 2 uppercase letters (see
# _is_real_abbreviation) so this doesn't fire on ordinary capitalized words
# that happen to precede a parenthetical aside, e.g. "Canada (the country)".
_DEFINITION_RE = re.compile(
    rf"(?P<term>{_TERM_WORD}(?:\s+{_TERM_WORD}){{0,4}})"
    r"\s*\(\s*(?P<abbr>[A-Z][A-Za-z0-9\-]{1,9})\s*\)"
)


def _is_real_abbreviation(token: str) -> bool:
    return sum(1 for c in token if c.isupper()) >= 2


def _term_plausibly_matches_abbr(term: str, abbr: str) -> bool:
    """Reject "definitions" where the term has no real relationship to the
    abbreviation -- e.g. a botched plain-language edit can leave a dangling
    "(PET)" stuck onto unrelated leftover words like "the virus appears
    (PET)". Require at least one word in the term to start with the
    abbreviation's first letter; genuine expansions always satisfy this
    (multi-word acronyms, or a single technical noun sharing the initial),
    while orphaned/garbled parentheticals typically don't.
    """
    first_letter = abbr[0].lower()
    return any(word[0].lower() == first_letter for word in term.split())


@dataclass
class _Occurrence:
    section_id: str
    term: str
    abbr: str


@dataclass
class AbbreviationFix:
    """One document-wide abbreviation consistency correction to apply."""

    section_id: str
    instruction: str


def _document_text_by_section(
    extractions: list[ExtractionResult],
    variables: list[TemplateVariable],
) -> list[tuple[str, str]]:
    """Return [(section_id, text)] in document reading order.

    ``variables`` is assumed to already be in reading order -- the same
    assumption build_icf_document_for_review() makes in review_prompts.py.
    """
    ext_map = {e.section_id: e for e in extractions}
    ordered: list[tuple[str, str]] = []
    for var in variables:
        if _is_protected(var.section_id):
            continue
        ext = ext_map.get(var.section_id)
        if ext is None or ext.status not in _REVIEWABLE_STATUSES:
            continue
        text = ext.filled_template or ext.answer or ""
        if text.strip():
            ordered.append((var.section_id, text))
    return ordered


def _all_candidates(text: str) -> list[tuple[str, str, bool]]:
    """Return [(term, abbr, plausible), ...] for every real-abbreviation-shaped
    parenthetical in text, in order -- including ones whose term doesn't
    plausibly match (see _term_plausibly_matches_abbr), which callers use to
    detect orphaned abbreviations rather than just discarding them.
    """
    results = []
    for m in _DEFINITION_RE.finditer(text):
        term, abbr = m.group("term").strip(), m.group("abbr")
        if _is_real_abbreviation(abbr):
            results.append((term, abbr, _term_plausibly_matches_abbr(term, abbr)))
    return results


def _find_definitions(text: str) -> list[tuple[str, str]]:
    """Return [(term, abbr), ...] for every genuine definition-style match, in order."""
    return [(term, abbr) for term, abbr, plausible in _all_candidates(text) if plausible]


def _bare_use_pattern(abbr: str) -> re.Pattern:
    """Match ABBR as a standalone token, excluding the "(ABBR)" of a definition."""
    escaped = re.escape(abbr)
    return re.compile(rf"(?<!\()\b{escaped}\b(?!\s*\))")


def find_abbreviation_fixes(
    extractions: list[ExtractionResult],
    variables: list[TemplateVariable],
) -> list[AbbreviationFix]:
    """Detect abbreviation-consistency violations across the assembled document.

    Three violation types, all mechanical (no LLM judgment needed to detect,
    though fixing #3 well does need judgment -- see its instruction text):

      1. Redundant redefinition: the same abbreviation is spelled out with its
         full term in more than one section. Only the earliest section (by
         document reading order) should keep the definition; later sections
         should use the bare abbreviation.

      2. Used before defined: a section uses the bare abbreviation before any
         section actually defines it (each section is drafted independently,
         so this ordering mismatch is easy to introduce). The earliest section
         using it should gain the definition; whichever section originally
         carried the definition should drop back to the bare form once it is
         no longer genuinely first.

      3. Orphaned/garbled abbreviation: a "(ABBR)" whose preceding term has no
         plausible relationship to it -- typically left behind when a plain-
         language edit simplified away the term but not the abbreviation
         itself (see module docstring). The instruction hands the patcher the
         same keep-vs-drop framework from UHN_PLAIN_LANGUAGE_GUIDELINES so it
         can decide whether to restore a correct definition or drop the
         abbreviation entirely.

    Returns one AbbreviationFix per (section, violation) -- callers typically
    wrap each into a GlobalFixRule and let Stage 9's existing per-section
    patch step apply it.
    """
    doc = _document_text_by_section(extractions, variables)
    section_order = [sid for sid, _ in doc]

    first_definition: dict[str, tuple[str, str]] = {}  # abbr -> (section_id, term)
    all_definitions: list[_Occurrence] = []
    for section_id, text in doc:
        for term, abbr in _find_definitions(text):
            all_definitions.append(_Occurrence(section_id, term, abbr))
            first_definition.setdefault(abbr, (section_id, term))

    first_bare_use: dict[str, str] = {}  # abbr -> earliest section_id using it bare
    for section_id, text in doc:
        for abbr in first_definition:
            if abbr not in first_bare_use and _bare_use_pattern(abbr).search(text):
                first_bare_use[abbr] = section_id

    fixes: list[AbbreviationFix] = []
    emitted: set[tuple[str, str]] = set()  # (section_id, abbr) -- avoid duplicate rules

    # Violation 1: redundant redefinition.
    for occ in all_definitions:
        canonical_section, canonical_term = first_definition[occ.abbr]
        if occ.section_id == canonical_section:
            continue
        key = (occ.section_id, occ.abbr)
        if key in emitted:
            continue
        emitted.add(key)
        fixes.append(
            AbbreviationFix(
                section_id=occ.section_id,
                instruction=(
                    f"'{occ.term}' is already introduced as '{canonical_term} ({occ.abbr})' "
                    f"in section {canonical_section} (earlier in the document). Do not "
                    f"redefine it again here -- use '{occ.abbr}' alone."
                ),
            )
        )

    # Violation 2: used before defined.
    for abbr, (def_section, term) in first_definition.items():
        bare_first = first_bare_use.get(abbr)
        if bare_first is None or bare_first == def_section:
            continue
        if section_order.index(bare_first) >= section_order.index(def_section):
            continue

        bare_key = (bare_first, abbr)
        if bare_key not in emitted:
            emitted.add(bare_key)
            fixes.append(
                AbbreviationFix(
                    section_id=bare_first,
                    instruction=(
                        f"'{abbr}' is used here before it is defined anywhere in the "
                        f"document -- this is its earliest use. Introduce it in full the "
                        f"first time it appears here: '{term} ({abbr})', then use '{abbr}' "
                        f"alone for any later mention in this section."
                    ),
                )
            )

        def_key = (def_section, abbr)
        if def_key not in emitted:
            emitted.add(def_key)
            fixes.append(
                AbbreviationFix(
                    section_id=def_section,
                    instruction=(
                        f"'{term} ({abbr})' here is not actually the first use of {abbr} in "
                        f"the document -- it is used earlier (without being spelled out) in "
                        f"section {bare_first}. Once that section defines it, use '{abbr}' "
                        f"alone here instead of spelling it out again."
                    ),
                )
            )

    # Violation 3: orphaned/garbled abbreviation.
    for section_id, text in doc:
        for term, abbr, plausible in _all_candidates(text):
            if plausible:
                continue
            key = (section_id, abbr)
            if key in emitted:
                continue
            emitted.add(key)

            canonical = first_definition.get(abbr)
            if canonical is not None:
                canonical_section, canonical_term = canonical
                guidance = (
                    f"'{abbr}' is already properly defined elsewhere as "
                    f"'{canonical_term} ({abbr})' in section {canonical_section} -- if this "
                    f"is the same thing, just use '{abbr}' alone here instead of this broken "
                    f"parenthetical."
                )
            else:
                guidance = (
                    f"'{abbr}' is not defined anywhere else in the document either. If it "
                    "names something the participant needs to recognize on its own later "
                    "(e.g. a scan, test, drug, or device name), restore the correct full term "
                    "here so it reads 'Full Term (" + abbr + ")'. If it is only an internal "
                    "study/strategy label with no meaning outside this document, drop the "
                    "abbreviation entirely and keep just the plain-language description."
                )
            fixes.append(
                AbbreviationFix(
                    section_id=section_id,
                    instruction=(
                        f"'{term} ({abbr})' looks like a leftover error: '{term}' has no real "
                        f"relationship to '{abbr}', which likely happened when a nearby phrase "
                        f"was simplified but the abbreviation was accidentally left attached. "
                        f"{guidance}"
                    ),
                )
            )

    return fixes

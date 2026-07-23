"""
Validation pipeline for ICF extractions.

Checks:
  1. Quote verification   - does the cited quote actually appear in the protocol?
  2. Meta-commentary      - does filled_template contain internal process notes?
  3. Quality gate         - is a result garbage / does it have fixable issues?
     (shared by the RLM and hybrid extraction backends to decide when a
     refinement/repair pass is worth running)
  4. Issue aggregation    - collect all problems for the report.
"""

import re

from icf.types import ExtractionResult, ValidationResult

# ------------------------------------------------------------------
# 1. Quote verification
# ------------------------------------------------------------------


def verify_quote(
    quote: str,
    protocol_text: str,
    threshold: float = 0.80,
) -> bool:
    """Check whether *quote* appears (exactly or fuzzily) in *protocol_text*.

    Strategy:
      a) Exact substring match (after whitespace normalisation).
      b) Match the first 120 chars of the quote (handles trailing OCR noise).
      c) Phrase-level match: split on commas/periods and require >=50 %% of
         phrases to appear in the protocol.
    """
    if not quote or not protocol_text:
        return False

    norm_q = _normalise(quote)
    norm_p = _normalise(protocol_text)

    # (a) Exact containment
    if norm_q in norm_p:
        return True

    # (b) Prefix match (first 120 chars)
    prefix = norm_q[:120]
    if len(prefix) > 30 and prefix in norm_p:
        return True

    # (c) Phrase-level match
    phrases = [p.strip() for p in re.split(r"[,.]", norm_q) if len(p.strip()) > 15]
    if phrases:
        found = sum(1 for p in phrases if p in norm_p)
        if found / len(phrases) >= 0.5:
            return True

    return False


def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace."""
    return " ".join(text.lower().split())


# ------------------------------------------------------------------
# 2. Meta-commentary detection
# ------------------------------------------------------------------

# Patterns that indicate the LLM leaked internal extraction notes into
# patient-facing text. Each tuple is (label, compiled regex).
_META_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "references to source documents",
        re.compile(
            # "study documents" or "study document" are never appropriate in patient text.
            # Do NOT flag bare "protocol" — it legitimately appears in ICF text
            # (e.g. "The study protocol was approved by the ethics board").
            r"\bstudy\s+documents?\b|\bclinical\s+trial\s+documents?\b"
            r"|\bin\s+(the\s+)?(retrieved\s+)?passages?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "extraction-process commentary",
        re.compile(
            # "not found / not described / not specified ... in the protocol/study/passages"
            r"\b(not\s+)?(clearly\s+)?(found|described|specified|mentioned|stated|provided|available|documented|included)\s+"
            r"(in\s+the\s+(protocol|study|documents?|passages?)|in\s+these?\s+(passages?|documents?|sources?))\b"
            r"|not\s+enough\s+information"
            r"|will\s+need\s+(more\s+)?details?\s+later"
            r"|more\s+information\s+(is\s+)?(needed|required)"
            r"|additional\s+information\s+(will\s+be\s+)?needed"
            r"|cannot\s+be\s+found|could\s+not\s+(be\s+)?found"
            r"|\bwill\s+need\s+to\s+be\s+(filled|completed|provided)",
            re.IGNORECASE,
        ),
    ),
    (
        "protocol-as-source commentary",
        re.compile(
            # Catches sentences where the LLM references the protocol as the
            # subject of a negative verb — e.g. "The protocol does not state
            # the visit length" or "The protocol does not clearly describe
            # hospitalisation". Patient-facing text must never mention the
            # protocol as an information source.
            r"\bthe\s+(?:study\s+)?protocol\s+"
            r"(?:does\s+not|doesn'?t|did\s+not|didn'?t|does\s+not\s+clearly|doesn'?t\s+clearly)\s+"
            r"(?:state|describe|specify|mention|include|provide|contain|detail|indicate|"
            r"address|cover|discuss|explain|list|note|outline|report|document|define)",
            re.IGNORECASE,
        ),
    ),
]


def check_meta_commentary(text: str) -> list[str]:
    """Return a list of issues if *text* contains internal extraction commentary.

    Checks sentence-by-sentence so the issue message can quote the offending
    sentence rather than just flagging the whole field.
    """
    if not text:
        return []

    issues: list[str] = []
    # Split into rough sentences for targeted reporting.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        for label, pattern in _META_PATTERNS:
            if pattern.search(sentence):
                short = sentence[:120].replace("\n", " ")
                issues.append(
                    f"[META-COMMENTARY] Patient-facing text contains {label}: \"{short}\""
                )
                break  # one issue per sentence is enough

    return issues


# ------------------------------------------------------------------
# 3. Quality gate — shared by the RLM and hybrid extraction backends
# ------------------------------------------------------------------


def quality_score(result: ExtractionResult) -> int:
    """Numeric quality score for comparing two results. Higher is better."""
    status_score = {"FOUND": 30, "PARTIAL": 20, "NOT_FOUND": 5, "ERROR": 0}.get(
        result.status, 0
    )
    confidence_score = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "N/A": 0}.get(
        result.confidence, 0
    )
    return status_score + confidence_score


def is_garbage_result(result: ExtractionResult) -> bool:
    """Return True when the extraction produced non-JSON/policy-refusal output.

    Two main causes:
    1. Parser fallback: the LLM returned prose, the fallback parser wrapped it as
       PARTIAL/LOW with empty filled_template and evidence.
    2. Policy-refusal hallucination: model said "I cannot continue" / "REPL not
       available" etc. These produce either garbage JSON or short prose answers.

    NOT_FOUND results with empty fields are explicitly excluded — that is the
    correct and expected output when the protocol contains no relevant information.
    Flagging them as garbage would cause pointless full-extraction retries.
    """
    refusal_signals = [
        "repl is not active",
        "repl is not available",
        "cannot run repl",
        "cannot execute repl",
        "i cannot continue",
        "this interface does not",
        "this chat interface",
        "this interface cannot",
        "i must stop here",
    ]
    raw = (result.raw_response or "").lower()
    if any(sig in raw for sig in refusal_signals):
        return True
    # Empty filled_template + empty evidence = fallback-wrapped prose, but ONLY
    # for FOUND or PARTIAL — NOT_FOUND legitimately has no template/evidence.
    if result.status != "NOT_FOUND" and not result.filled_template and not result.evidence:
        return True
    return False


def collect_quality_issues(result: ExtractionResult) -> list[str]:
    """Return a list of quality problems that warrant a refinement/repair pass.

    Only flags issues a follow-up LLM call can concretely fix:
      1. Unfilled {{...}} or <<...>> markers left in filled_template.
      2. Meta-commentary leaking into patient-facing filled_template.

    Intentionally NOT flagging:
    - LOW confidence alone: second passes do not reliably improve quality and
      can produce worse output.  Low-confidence results are accepted as-is and
      surfaced to the reviewer via the confidence annotations.
    - PARTIAL status alone: means the protocol genuinely lacks the info.
      A second pass won't find what isn't there, and just wastes iterations.
    - Quote verification failures: Unicode chars, footnote numbers, and
      sub-LLM paraphrasing cause false failures a follow-up call cannot fix.
      Quote quality is surfaced in validate_extractions() instead.

    Returns an empty list when the result is clean enough to keep as-is, or
    when the status is one that a refinement/repair pass cannot improve.
    """
    if result.status in ("SKIPPED", "ERROR", "STANDARD_TEXT"):
        return []

    # NOT_FOUND: the model searched thoroughly and found nothing.
    # A second pass won't find what doesn't exist.
    if result.status == "NOT_FOUND":
        return []

    # Garbage fallback results are handled by the caller's fresh-attempt retry loop.
    if is_garbage_result(result):
        return []

    issues: list[str] = []

    unfilled = re.findall(r"\{\{[^}]+\}\}|<<[^>]+>>", result.filled_template)
    for m in unfilled[:3]:
        issues.append(f"unfilled marker in filled_template: {m}")

    issues.extend(check_meta_commentary(result.filled_template))

    return issues


# ------------------------------------------------------------------
# 4. Aggregate validation
# ------------------------------------------------------------------


def validate_extractions(
    extractions: list[ExtractionResult],
    protocol_text: str,
) -> list[ValidationResult]:
    """Run all validation checks on a list of extractions."""
    results: list[ValidationResult] = []

    for ext in extractions:
        # Skip non-extractable statuses
        if ext.status in ("SKIPPED", "ERROR", "NOT_FOUND", "STANDARD_TEXT"):
            results.append(
                ValidationResult(
                    section_id=ext.section_id,
                    quotes_verified=[],
                    reading_grade_level=None,
                    issues=[],
                )
            )
            continue

        issues: list[str] = []

        # Quote verification
        quotes_ok: list[bool] = []
        for ev in ext.evidence:
            ok = verify_quote(ev.quote, protocol_text)
            quotes_ok.append(ok)
            if not ok:
                short = ev.quote[:80].replace("\n", " ")
                issues.append(f'Quote not verified in protocol: "{short}..."')

        # Meta-commentary in patient-facing text
        issues.extend(check_meta_commentary(ext.filled_template))

        results.append(
            ValidationResult(
                section_id=ext.section_id,
                quotes_verified=quotes_ok,
                reading_grade_level=None,
                issues=issues,
            )
        )

    return results

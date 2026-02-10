"""
Validation pipeline for ICF extractions.

Three checks:
  1. Quote verification  - does the cited quote actually appear in the protocol?
  2. Reading level        - is the answer written at Grade 6-8?
  3. Issue aggregation    - collect all problems for the report.
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
# 2. Reading level
# ------------------------------------------------------------------


def check_reading_level(text: str) -> float | None:
    """Return Flesch-Kincaid grade level, or None if unavailable."""
    if not text or len(text.split()) < 10:
        return None
    try:
        import textstat

        return textstat.flesch_kincaid_grade(text)
    except ImportError:
        return None


# ------------------------------------------------------------------
# 3. Aggregate validation
# ------------------------------------------------------------------

READING_LEVEL_WARN = 8.0  # flag answers above Grade 8


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

        # Reading level
        grade = check_reading_level(ext.answer)
        if grade is not None and grade > READING_LEVEL_WARN:
            issues.append(
                f"Reading level ({grade:.1f}) exceeds Grade {READING_LEVEL_WARN:.0f} target."
            )

        results.append(
            ValidationResult(
                section_id=ext.section_id,
                quotes_verified=quotes_ok,
                reading_grade_level=grade,
                issues=issues,
            )
        )

    return results

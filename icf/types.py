from __future__ import annotations

import re
from dataclasses import dataclass

# Section IDs in this codebase are always a dotted-numeric token (e.g. "3", "9.2",
# "21.1"). LLM-produced section_id values sometimes get contaminated with the
# surrounding "=== SECTION N: HEADING ===" document header the model was shown --
# either a leading "SECTION " prefix ("SECTION 3") or a trailing ": HEADING" suffix
# ("3: INTRODUCTION"), or both. Strip either/both so lookups against the bare ID
# (used everywhere else: TemplateVariable.section_id, ExtractionResult.section_id)
# don't silently fail.
_SECTION_PREFIX_RE = re.compile(r"^(?:SECTION|Section|section)\s+")
_SECTION_ID_RE = re.compile(r"^\d+(?:\.\d+)*")


def normalize_section_id(raw_id: str) -> str:
    """Return the bare dotted-numeric section ID from a possibly-contaminated string.

    Examples: "SECTION 3" -> "3", "3: INTRODUCTION" -> "3", "9.2" -> "9.2".
    Falls back to the (prefix-stripped) input unchanged if it doesn't start with
    a recognizable numeric ID, rather than silently discarding it.
    """
    cleaned = _SECTION_PREFIX_RE.sub("", str(raw_id)).strip()
    match = _SECTION_ID_RE.match(cleaned)
    return match.group(0) if match else cleaned


@dataclass
class ProtocolPage:
    """A single page of the protocol with its text and page number."""

    page_number: int
    text: str


@dataclass
class IndexedProtocol:
    """A parsed protocol with page-level text indexing."""

    pages: list[ProtocolPage]
    full_text: str
    total_pages: int
    source_path: str

    def get_page_text(self) -> str:
        """Return the full text with --- PAGE X --- markers."""
        parts = []
        for page in self.pages:
            parts.append(f"--- PAGE {page.page_number} ---\n{page.text}")
        return "\n".join(parts)

    def get_pages_range(self, start: int, end: int) -> str:
        """Get text for a range of pages (inclusive)."""
        parts = []
        for page in self.pages:
            if start <= page.page_number <= end:
                parts.append(f"--- PAGE {page.page_number} ---\n{page.text}")
        return "\n".join(parts)


@dataclass
class TemplateVariable:
    """A single ICF template section/variable to extract."""

    section_id: str
    heading: str
    sub_section: str | None
    required: bool
    instructions: str
    required_text: str
    # Plain text or HTML string; check suggested_text_format to know which.
    suggested_text: str
    complexity: list[str]
    is_in_protocol: bool
    partially_in_protocol: bool
    is_standard_text: bool
    # "text" (default) or "html" — allows rich table content in JSON registry.
    suggested_text_format: str = "text"
    # Runtime study-context note injected by user-selected flags (e.g. US-funding,
    # SDM). Surfaced to the extraction prompt via prompt_runtime_context.
    # None = no override.
    adaptation_notes: str | None = None

    def get_display_name(self) -> str:
        name = f"[{self.section_id}] {self.heading}"
        if self.sub_section:
            name += f" > {self.sub_section}"
        return name

    def get_complexity_label(self) -> str:
        for c in self.complexity:
            cl = c.lower()
            if "easy" in cl:
                return "Easy"
            if "moderate" in cl:
                return "Moderate"
            if "complex mapping" in cl:
                return "Complex"
            if "potentially in protocol" in cl:
                return "Moderate"
        if not self.is_in_protocol:
            return "Not in protocol"
        return "Moderate"


@dataclass
class Evidence:
    """A piece of evidence supporting an extraction."""

    quote: str
    page: str
    section: str = ""

    def to_dict(self) -> dict:
        return {"quote": self.quote, "page": self.page, "section": self.section}


@dataclass
class ExtractionResult:
    """The result of extracting a single template variable."""

    section_id: str
    heading: str
    sub_section: str | None
    status: str  # FOUND, NOT_FOUND, PARTIAL, SKIPPED, STANDARD_TEXT, ERROR
    answer: str
    filled_template: str
    evidence: list[Evidence]
    confidence: str  # HIGH, MEDIUM, LOW, N/A
    notes: str
    raw_response: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        d = {
            "section_id": self.section_id,
            "heading": self.heading,
            "sub_section": self.sub_section,
            "status": self.status,
            "answer": self.answer,
            "filled_template": self.filled_template,
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": self.confidence,
            "notes": self.notes,
            "error": self.error,
        }
        if self.raw_response:
            d["raw_response"] = self.raw_response
        return d


@dataclass
class ValidationResult:
    """Validation results for a single extraction."""

    section_id: str
    quotes_verified: list[bool]
    reading_grade_level: float | None
    issues: list[str]

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "quotes_verified": self.quotes_verified,
            "reading_grade_level": self.reading_grade_level,
            "issues": self.issues,
        }


@dataclass
class ReviewFlag:
    """A single plain-language issue flagged by the Stage 8 review pass."""

    section_id: str
    flagged_text: str  # verbatim snippet from the ICF section text
    issue_type: str  # REPETITION | PASSIVE_VOICE | SENTENCE_TOO_LONG |
    # TERMINOLOGY_INCONSISTENCY | UNCLEAR | TONE |
    # PLAIN_LANGUAGE_VIOLATION
    suggestion: str  # brief guidance explaining the issue
    severity: str  # HIGH | MEDIUM | LOW
    suggested_fix: str = ""  # ready-to-copy replacement text; empty if not applicable

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "flagged_text": self.flagged_text,
            "issue_type": self.issue_type,
            "suggestion": self.suggestion,
            "severity": self.severity,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class ReviewResult:
    """The output of the Stage 8 plain-language review pass."""

    flags: list[ReviewFlag]
    cross_section_notes: str  # overall observations spanning multiple sections

    def to_dict(self) -> dict:
        return {
            "flags": [f.to_dict() for f in self.flags],
            "cross_section_notes": self.cross_section_notes,
        }


@dataclass
class GlobalFixRule:
    """A document-wide fix rule extracted from Stage 8 cross-section notes.

    rule_type values:
      define_abbreviation  -- first document use: Full Term (ABB); later uses: ABB only
      standardize_term     -- replace an inconsistent term across sections
      fix_inconsistency    -- correct a factual/structural inconsistency
      note_only            -- acknowledged but not auto-applied (e.g. repetition)
    """

    rule_type: str
    description: str
    affected_section_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "rule_type": self.rule_type,
            "description": self.description,
            "affected_section_ids": self.affected_section_ids,
        }


@dataclass
class RemediationRecord:
    """Audit record for one section patched during Stage 9 remediation."""

    section_id: str
    high_flag_count: int
    global_rules_applied: list[str]  # descriptions of GlobalFixRules applied
    original_text: str
    patched_text: str  # equals original_text when success=False
    success: bool
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "high_flag_count": self.high_flag_count,
            "global_rules_applied": self.global_rules_applied,
            "success": self.success,
            "notes": self.notes,
        }


@dataclass
class RemediationResult:
    """The output of the Stage 9 HIGH flag remediation pass."""

    records: list[RemediationRecord]
    global_rules: list[GlobalFixRule]
    # Descriptions of note_only rules not auto-applied (e.g. structural repetition).
    unaddressed_notes: str

    def to_dict(self) -> dict:
        return {
            "records": [r.to_dict() for r in self.records],
            "global_rules": [g.to_dict() for g in self.global_rules],
            "unaddressed_notes": self.unaddressed_notes,
        }


@dataclass
class PipelineResult:
    """The complete result of the ICF pipeline."""

    extractions: list[ExtractionResult]
    validations: list[ValidationResult]
    marked_up_icf_path: str | None
    draft_icf_path: str | None
    report_path: str | None
    summary: dict
    review_result: ReviewResult | None = None
    remediation_result: RemediationResult | None = None

    def to_dict(self) -> dict:
        return {
            "extractions": [e.to_dict() for e in self.extractions],
            "validations": [v.to_dict() for v in self.validations],
            "marked_up_icf_path": self.marked_up_icf_path,
            "draft_icf_path": self.draft_icf_path,
            "report_path": self.report_path,
            "summary": self.summary,
            "review": self.review_result.to_dict() if self.review_result else None,
            "remediation": self.remediation_result.to_dict() if self.remediation_result else None,
        }

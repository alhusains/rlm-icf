from __future__ import annotations

from dataclasses import dataclass


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
    suggested_text: str
    complexity: list[str]
    protocol_mapping: str
    sponsor_mapping: str
    is_in_protocol: bool
    partially_in_protocol: bool
    is_standard_text: bool
    notes: str
    csv_status: str

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
class PipelineResult:
    """The complete result of the ICF pipeline."""

    extractions: list[ExtractionResult]
    validations: list[ValidationResult]
    output_docx_path: str | None
    report_path: str | None
    summary: dict

    def to_dict(self) -> dict:
        return {
            "extractions": [e.to_dict() for e in self.extractions],
            "validations": [v.to_dict() for v in self.validations],
            "output_docx_path": self.output_docx_path,
            "report_path": self.report_path,
            "summary": self.summary,
        }

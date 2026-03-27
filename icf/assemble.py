"""
Output generation: draft ICF DOCX and JSON extraction report.

Produces two artefacts:
  1. draft_icf.docx   - A new Word document following the template structure.
  2. extraction_report.json - Full structured data for programmatic use.
"""

import html as html_mod
import json
import re
from typing import Any

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

from icf.types import (
    ExtractionResult,
    ReviewResult,
    TemplateVariable,
    ValidationResult,
)

# ------------------------------------------------------------------
# Colour constants
# ------------------------------------------------------------------
_GREY = RGBColor(128, 128, 128)
_RED = RGBColor(200, 30, 30)
_ORANGE = RGBColor(200, 130, 0)
_GREEN = RGBColor(30, 130, 30)


# ------------------------------------------------------------------
# 1. Draft ICF DOCX
# ------------------------------------------------------------------


def generate_draft_docx(
    extractions: list[ExtractionResult],
    validations: list[ValidationResult],
    variables: list[TemplateVariable],
    output_path: str,
    review_result: ReviewResult | None = None,
) -> str:
    """Create a new DOCX with all sections, filled content, and markers."""
    doc = Document()

    # Title page
    title = doc.add_heading("DRAFT - Informed Consent Form", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    note = doc.add_paragraph(
        "This is an auto-generated draft. Sections marked "
        "[TO BE FILLED MANUALLY] require human review and completion. "
        "Evidence citations are included below each section for reference."
    )
    _style_run(note.runs[0], size=9, italic=True, colour=_GREY)

    doc.add_page_break()

    # Build quick lookup
    ext_map: dict[str, ExtractionResult] = {e.section_id: e for e in extractions}
    val_map: dict[str, ValidationResult] = {v.section_id: v for v in validations}

    for var in variables:
        ext = ext_map.get(var.section_id)
        val = val_map.get(var.section_id)

        # Omit optional sections that were not found / not applicable.
        # Required sections always appear so the human reviewer knows to fill them.
        # ADAPTATION_SKIPPED sections always appear regardless of required status so
        # the study team can review and confirm the irrelevance decision.
        if (
            ext is not None
            and ext.status in ("NOT_FOUND", "SKIPPED")
            and not var.required
        ):
            continue

        # Section heading
        level = 2 if var.sub_section else 1
        heading_text = var.heading
        if var.sub_section:
            heading_text += f" - {var.sub_section}"
        doc.add_heading(heading_text, level=level)

        # Status badge
        if ext is None:
            _add_status_line(doc, "NOT PROCESSED", _GREY)
            continue

        colour = {
            "FOUND": _GREEN,
            "PARTIAL": _ORANGE,
            "STANDARD_TEXT": _GREEN,
            "NOT_FOUND": _RED,
            "SKIPPED": _GREY,
            "ADAPTATION_SKIPPED": _GREY,
            "ERROR": _RED,
        }.get(ext.status, _GREY)
        if ext.status == "ADAPTATION_SKIPPED":
            # Section was deemed irrelevant for this study — plain skip message only.
            badge = "Status: SKIPPED — not relevant to this study"
        else:
            badge = f"Status: {ext.status}"
            if ext.confidence and ext.confidence != "N/A":
                badge += f"  |  Confidence: {ext.confidence}"
            if ext.error:
                badge += f"  |  Error: {ext.error}"
        _add_status_line(doc, badge, colour)

        # Main content
        if ext.status in ("FOUND", "PARTIAL", "STANDARD_TEXT"):
            text = ext.filled_template or ext.answer
            if text:
                doc.add_paragraph(text)
            if ext.status == "PARTIAL" and ext.notes:
                p = doc.add_paragraph()
                r = p.add_run(f"[PARTIAL] {ext.notes}")
                _style_run(r, size=9, colour=_ORANGE, italic=True)

        elif ext.status in ("NOT_FOUND", "SKIPPED"):
            # Required section could not be filled — flag it prominently.
            p = doc.add_paragraph()
            r = p.add_run("[TO BE FILLED MANUALLY]")
            _style_run(r, size=11, colour=_RED, bold=True)
            if var.suggested_text:
                sg = doc.add_paragraph()
                sr = sg.add_run("Suggested text: " + _plain_suggested_text(var)[:800])
                _style_run(sr, size=9, colour=_GREY, italic=True)

        elif ext.status == "ERROR":
            p = doc.add_paragraph()
            r = p.add_run(f"[EXTRACTION ERROR] {ext.error}")
            _style_run(r, size=10, colour=_RED, bold=True)

        # Evidence citations
        if ext.evidence:
            ep = doc.add_paragraph()
            er = ep.add_run("Evidence:")
            _style_run(er, size=8, italic=True, colour=_GREY)
            for ev in ext.evidence:
                bp = doc.add_paragraph(style="List Bullet")
                short_quote = ev.quote[:250].replace("\n", " ")
                br = bp.add_run(f'Page {ev.page}: "{short_quote}"')
                _style_run(br, size=8, italic=True, colour=_GREY)

        # Validation issues
        if val and val.issues:
            for issue in val.issues:
                ip = doc.add_paragraph()
                ir = ip.add_run(f"[VALIDATION] {issue}")
                _style_run(ir, size=8, colour=_ORANGE)

        # Inline review annotations
        if review_result:
            section_flags = [f for f in review_result.flags if f.section_id == var.section_id]
            for flag in section_flags:
                flag_colour = {"HIGH": _RED, "MEDIUM": _ORANGE, "LOW": _GREY}.get(
                    flag.severity, _GREY
                )
                fp = doc.add_paragraph()
                fr = fp.add_run(
                    f"[REVIEW | {flag.issue_type} | {flag.severity}] "
                    f'"{flag.flagged_text[:120]}" — {flag.suggestion}'
                )
                _style_run(fr, size=8, colour=flag_colour, italic=True)
                if flag.suggested_fix:
                    sfp = doc.add_paragraph()
                    sfr = sfp.add_run(f"    Suggested fix: {flag.suggested_fix}")
                    _style_run(sfr, size=8, colour=flag_colour)

    # Review appendix
    if review_result and (review_result.flags or review_result.cross_section_notes):
        doc.add_page_break()
        doc.add_heading("APPENDIX: Plain Language Review Flags", level=1)
        if review_result.cross_section_notes:
            note_p = doc.add_paragraph()
            note_r = note_p.add_run(f"Cross-section notes: {review_result.cross_section_notes}")
            _style_run(note_r, size=9, italic=True, colour=_GREY)
        if review_result.flags:
            high_flags = [f for f in review_result.flags if f.severity == "HIGH"]
            med_flags = [f for f in review_result.flags if f.severity == "MEDIUM"]
            low_flags = [f for f in review_result.flags if f.severity == "LOW"]
            for severity_group, colour in (
                (high_flags, _RED),
                (med_flags, _ORANGE),
                (low_flags, _GREY),
            ):
                for flag in severity_group:
                    bp = doc.add_paragraph(style="List Bullet")
                    br = bp.add_run(
                        f"[{flag.section_id}] {flag.issue_type} ({flag.severity}): "
                        f'"{flag.flagged_text[:120]}" — {flag.suggestion}'
                    )
                    _style_run(br, size=9, colour=colour)
                    if flag.suggested_fix:
                        sfp = doc.add_paragraph()
                        sfr = sfp.add_run(f"    Suggested fix: {flag.suggested_fix}")
                        _style_run(sfr, size=9, colour=colour)

    doc.save(output_path)
    return output_path


# ------------------------------------------------------------------
# 2. JSON report
# ------------------------------------------------------------------


def generate_report_json(
    extractions: list[ExtractionResult],
    validations: list[ValidationResult],
    summary: dict[str, Any],
    output_path: str,
    review_result: ReviewResult | None = None,
) -> str:
    """Write the full extraction report as JSON."""
    report = {
        "summary": summary,
        "extractions": [e.to_dict() for e in extractions],
        "validations": [v.to_dict() for v in validations],
        "review": review_result.to_dict() if review_result else None,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return output_path


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _add_status_line(doc: Document, text: str, colour: RGBColor) -> None:
    p = doc.add_paragraph()
    r = p.add_run(text)
    _style_run(r, size=8, colour=colour)


def _plain_suggested_text(var: TemplateVariable) -> str:
    """Return suggested_text as plain text, stripping HTML tags when format is 'html'."""
    raw = html_mod.unescape(var.suggested_text)
    if var.suggested_text_format == "html":
        return re.sub(r"<[^>]+>", " ", raw).strip()
    return raw


def _style_run(
    run,
    size: int | None = None,
    bold: bool = False,
    italic: bool = False,
    colour: RGBColor | None = None,
) -> None:
    if size is not None:
        run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if colour is not None:
        run.font.color.rgb = colour

"""
Draft ICF document generator (UHN publication-quality layout).

Produces the draft ICF that the study team reviews and edits before CAPCR
submission:

  - UHN logo in the header (top-left)
  - Bordered footer with automatic "Page X of Y" page numbering
  - All-caps underlined section headings (matching the approved-ICF style)
  - Justified body text in Arial 11 pt
  - Standard UHN signature pages appended verbatim (only the TITLE line changes)

Each section carries a small grey italic status/confidence annotation below its
heading, ``[PLEASE COMPLETE]`` markers are highlighted yellow, and sections
that could not be extracted show suggested text in grey italic. No confidence
colour-coding, evidence quotes, or review flags appear here — those live in the
separate marked-up ICF (see assemble.py).
"""

from __future__ import annotations

import html as html_mod
import os
import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

from icf.runtime_injections import SIGNATURE_CONSENT_SECTION_ID, resolve_signature_consent_bullets
from icf.types import ExtractionResult, TemplateVariable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FONT = "Arial"
_BODY_PT = 11
_ANNOTATION_PT = 11
_SMALL_PT = 9

# RLM emits MARKER_PLEASE_COMPLETE inline; section-level labels for empty/skipped
# sections are chosen at Word render time.
MARKER_PLEASE_COMPLETE = "[PLEASE COMPLETE]"
MARKER_REQUIRED_SUGGESTED = (
    "[Please complete using the below suggested text. This is a required section.]"
)
MARKER_OPTIONAL_SUGGESTED = (
    "[Please complete using the below suggested text, if relevant to this study.]"
)
MARKER_ADD_OTHER_ORGS = (
    "[Add any other organizations with direct access to participant records, if applicable]"
)

_AI_DISCLAIMER = (
    "Parts of the initial draft of this consent form were created with help from "
    "an artificial intelligence tool developed at University Health Network to "
    "support consent form preparation. The final approved version was reviewed "
    "by the research team."
)

_CONTENT_STATUSES = {"FOUND", "PARTIAL", "STANDARD_TEXT"}
# Top-level and nested list markers used in ICF drafts (UHN templates use "o" for sub-bullets).
_BULLET_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<marker>[•\-–\*·]|[oO](?=\s))"
    r"\s+"
    r"(?P<content>.*)$"
)
_BULLET_GLYPHS = ("\u2022", "\u25E6", "\u25AA")  # • ◦ ▪
_BULLET_BASE_INDENT_CM = 1.0
_BULLET_HANGING_CM = 0.5
_BULLET_NEST_STEP_CM = 0.75
_INLINE_MARKER_RE = re.compile(
    rf"({re.escape(MARKER_PLEASE_COMPLETE)}|{re.escape(MARKER_ADD_OTHER_ORGS)})"
)

_TESTS_PROCEDURES_SECTION_ID = "13.6"
_TESTS_PROCEDURES_NOT_FOUND_SUGGESTED = (
    "This section uses a table layout to show the schedule/frequency of "
    "study-related activities. Please refer to template for the exact layout "
    "of the table."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_draft_docx(
    extractions: list[ExtractionResult],
    variables: list[TemplateVariable],
    output_path: str,
    logo_path: str | None = None,
    us_funded: bool = False,
    sdm: bool = False,
) -> str:
    """Generate the draft ICF for study-team review (UHN publication layout).

    Shares the full visual style of an approved ICF (Arial 11 pt, UHN header/
    footer, all-caps underlined headings, subsection grouping, signature pages):

    - No confidence-based colour coding — all text is black.
    - A small italic grey "Status: X | Confidence: Y" annotation is written
      directly below each section / sub-section heading.
    - Sections that could not be extracted show ``[PLEASE COMPLETE]`` (or a
      suggested-text variant) highlighted in yellow, followed by suggested text
      in grey italic.
    - No evidence quotes, no review flags, no review appendix.

    Args:
        extractions: All extraction results produced by the pipeline.
        variables:   All template variables in document order.
        output_path: Destination file path for the DOCX.
        logo_path:   Optional path to the UHN logo image (placed top-left in header).

    Returns:
        The resolved *output_path*.
    """
    doc = Document()
    _configure_page(doc)
    _set_document_font(doc)
    _build_header(doc, logo_path)
    _build_footer(doc)

    ext_map: dict[str, ExtractionResult] = {e.section_id: e for e in extractions}

    _write_validation_intro_page(doc)
    if us_funded:
        _write_us_summary_sections_validation(doc, variables, ext_map)
    _write_cover_page(doc, variables, ext_map)
    _write_validation_main_body(doc, variables, ext_map)
    _write_signature_pages(doc, _get_study_title(ext_map), sdm=sdm, ext_map=ext_map)

    doc.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------


def _configure_page(doc: Document) -> None:
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Cm(1.25)
    sec.bottom_margin = Cm(1.27)
    sec.left_margin = Cm(2.54)
    sec.right_margin = Cm(2.54)
    sec.header_distance = Cm(1.27)
    sec.footer_distance = Cm(1.27)
    sec.different_first_page_header_footer = False


def _set_document_font(doc: Document) -> None:
    """Apply Arial 11 pt as the document-wide default."""
    normal = doc.styles["Normal"]
    normal.font.name = _FONT
    normal.font.size = Pt(_BODY_PT)
    # Also patch the low-level rFonts element so the font propagates everywhere.
    rPr = normal._element.get_or_add_rPr()
    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:ascii"), _FONT)
    rFonts.set(qn("w:hAnsi"), _FONT)
    rFonts.set(qn("w:cs"), _FONT)
    rPr.insert(0, rFonts)


# ---------------------------------------------------------------------------
# Header: UHN logo
# ---------------------------------------------------------------------------


def _build_header(doc: Document, logo_path: str | None) -> None:
    header = doc.sections[0].header
    para = header.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    para.paragraph_format.space_before = Pt(0)
    para.paragraph_format.space_after = Pt(0)
    if logo_path and os.path.isfile(logo_path):
        run = para.add_run()
        # Width ~1.8" keeps the logo compact in the top-left corner.
        run.add_picture(logo_path, width=Inches(1.8))


# ---------------------------------------------------------------------------
# Footer: bordered paragraph with "Page X of Y"
# ---------------------------------------------------------------------------


def _build_footer(doc: Document) -> None:
    footer = doc.sections[0].footer
    para = footer.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.space_before = Pt(6)
    para.paragraph_format.space_after = Pt(3)

    # Add a top border to the paragraph (single rule, matching the approved ICF).
    pPr = para._element.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    top = OxmlElement("w:top")
    top.set(qn("w:val"), "single")
    top.set(qn("w:sz"), "4")
    top.set(qn("w:space"), "1")
    top.set(qn("w:color"), "auto")
    pBdr.append(top)
    pPr.append(pBdr)

    # "Version date of this form: _______________ Page X of Y"
    _add_footer_run(para, "Version date of this form: _______________    Page ")
    _add_page_field(para, "PAGE")
    _add_footer_run(para, " of ")
    _add_page_field(para, "NUMPAGES")


def _add_footer_run(para, text: str) -> None:
    r = para.add_run(text)
    r.font.name = _FONT
    r.font.size = Pt(_SMALL_PT)


def _add_page_field(para, field_type: str) -> None:
    """Append a Word PAGE or NUMPAGES auto-field to *para*."""
    p_elem = para._element
    sz_val = str(int(_SMALL_PT * 2))  # half-points

    def _mk_run() -> OxmlElement:
        r = OxmlElement("w:r")
        rPr = OxmlElement("w:rPr")
        rFonts = OxmlElement("w:rFonts")
        rFonts.set(qn("w:ascii"), _FONT)
        rFonts.set(qn("w:hAnsi"), _FONT)
        rPr.append(rFonts)
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), sz_val)
        rPr.append(sz)
        r.append(rPr)
        return r

    # begin
    r1 = _mk_run()
    fc1 = OxmlElement("w:fldChar")
    fc1.set(qn("w:fldCharType"), "begin")
    r1.append(fc1)
    p_elem.append(r1)

    # instrText
    r2 = _mk_run()
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {field_type} "
    r2.append(it)
    p_elem.append(r2)

    # separate
    r3 = _mk_run()
    fc3 = OxmlElement("w:fldChar")
    fc3.set(qn("w:fldCharType"), "separate")
    r3.append(fc3)
    p_elem.append(r3)

    # cached value placeholder
    r4 = _mk_run()
    r4.find(qn("w:rPr")).append(OxmlElement("w:noProof"))
    t = OxmlElement("w:t")
    t.text = "1"
    r4.append(t)
    p_elem.append(r4)

    # end
    r5 = _mk_run()
    fc5 = OxmlElement("w:fldChar")
    fc5.set(qn("w:fldCharType"), "end")
    r5.append(fc5)
    p_elem.append(r5)


# ---------------------------------------------------------------------------
# Introductory notes page (inserted before the cover page)
# ---------------------------------------------------------------------------

_INTRO_GREY = RGBColor(0x55, 0x55, 0x55)
_INTRO_ITALIC_PT = 10.5
_INTRO_GREY_ITALIC_PT = 10
_UHN_TEMPLATES_URL = "https://intranet.uhnresearch.ca/service/documents-and-forms"


def _write_validation_intro_page(doc: Document) -> None:
    """Introductory cover page explaining how to use the AI-generated draft."""

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("AI-Generated ICF Draft")
    r.font.name = _FONT
    r.font.size = Pt(14)
    r.bold = True
    r.underline = True

    _add_blank(doc)

    _intro_section_label(doc, "How to Use This Document")

    _intro_body(
        doc,
        "This document was generated using AI and is intended as a starting point for "
        "your initial draft. The AI is grounded in the protocol as the primary source "
        "of truth; however, it may still produce inaccuracies or take details out of "
        "context.",
    )
    _intro_body(
        doc,
        "Before submitting, you are responsible for reviewing and updating this "
        "document to ensure it meets REB requirements.",
    )

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    _intro_run(
        p,
        "Please use the UHN consent form templates as a reference for formatting and "
        "required content: ",
    )
    _add_intro_hyperlink(p, _UHN_TEMPLATES_URL)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    _intro_run(
        p,
        "As this is a beta version, we highly suggest cross-referencing against the "
        "template before submitting the consent form in CAPCR.",
        italic=True,
        size=_INTRO_ITALIC_PT,
    )

    _add_blank(doc)
    _intro_section_label(doc, "What you need to do:")

    for item in [
        "Review all content and correct any errors or inconsistencies.",
        "Ensure information accurately reflects the protocol.",
        "Confirm formatting aligns with submission standards.",
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.5)
        r = p.add_run("\u2022 " + item)
        r.font.name = _FONT
        r.font.size = Pt(_BODY_PT)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.first_line_indent = Cm(-0.5)
    _intro_run(p, "\u2022 Remove any instructional text (including this cover page, ", bold=True)
    _intro_run(
        p,
        "grey italic text",
        bold=True,
        italic=True,
        grey=True,
        size=_INTRO_GREY_ITALIC_PT,
    )
    _intro_run(p, " and ", bold=True)
    _intro_run(p, "highlighted text", bold=True, highlight=True)
    _intro_run(p, ") before submitting.", bold=True)

    _add_blank(doc)
    _intro_section_label(doc, "Section Status")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    _intro_run(
        p,
        "Each section is labelled to show how much of the section the AI could "
        "pre-populate ",
    )
    _intro_run(p, "based on information in the protocol", italic=True)
    _intro_run(p, ":")

    status_entries: list[tuple[str, str | None]] = [
        (
            "FOUND",
            "The section was fully populated. Review for accuracy and plain-language quality.",
        ),
        ("PARTIAL", None),
        (
            "NOT_FOUND",
            "No relevant information was located in the protocol. You must complete this section, if applicable.",
        ),
        (
            "SKIPPED",
            "This information is not consistently available within a protocol, so the AI is "
            "designed to skip it. It must be filled in by you, if needed.",
        ),
    ]
    for label, desc in status_entries:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Cm(0.8)
        rl = p.add_run(label + ":  ")
        rl.font.name = _FONT
        rl.font.size = Pt(_BODY_PT)
        rl.bold = True
        if desc is not None:
            rd = p.add_run(desc)
            rd.font.name = _FONT
            rd.font.size = Pt(_BODY_PT)
        else:
            _intro_run(
                p,
                "Some information is missing. Look for [to be completed] for fields "
                "requiring your updates. The ",
            )
            _intro_run(
                p,
                "grey italic note",
                italic=True,
                grey=True,
                size=_INTRO_GREY_ITALIC_PT,
            )
            _intro_run(p, " at the end of the section explains what's missing.")

    _add_blank(doc)
    _intro_section_label(doc, "Confidence Level")

    _intro_body(
        doc,
        "It reflects how strongly the AI was able to match information to the protocol.",
    )
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    _intro_run(p, "Note: This is not a statistical measure of confidence.", italic=True)

    for label, desc in [
        ("HIGH", "Information clearly matches the protocol."),
        (
            "MEDIUM",
            "Information found, but required interpretation or was partially ambiguous. "
            "Closer review is recommended.",
        ),
        ("LOW", "Weak or indirect match. The section should be carefully verified."),
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Cm(0.8)
        rl = p.add_run(label + ":  ")
        rl.font.name = _FONT
        rl.font.size = Pt(_BODY_PT)
        rl.bold = True
        rd = p.add_run(desc)
        rd.font.name = _FONT
        rd.font.size = Pt(_BODY_PT)

    _add_blank(doc)
    _intro_section_label(doc, "Flagged sections for Updating")

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    _intro_run(p, "Sections highlighted in yellow", highlight=True)
    _intro_run(p, " indicate missing information.")
    _intro_body(doc, "You must address these areas before submitting the document.")

    _add_blank(doc)
    _intro_section_label(doc, "Important Notes")

    for note in [
        "This is an AI-generated draft and has not been reviewed or approved by REB "
        "or any regulatory body",
        "The study team is responsible for reviewing, verifying, and approving all content",
        "The submitted protocol was the sole source of information for the generation of this draft",
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.5)
        r = p.add_run("\u2022 " + note)
        r.font.name = _FONT
        r.font.size = Pt(_BODY_PT)

    doc.add_page_break()


def _intro_section_label(doc: Document, text: str) -> None:
    """Bold sub-heading for intro page sections."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    r.font.name = _FONT
    r.font.size = Pt(_BODY_PT)
    r.bold = True


def _intro_body(doc: Document, text: str) -> None:
    """Regular body paragraph for the intro page."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    r.font.name = _FONT
    r.font.size = Pt(_BODY_PT)


def _intro_run(
    para,
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    highlight: bool = False,
    grey: bool = False,
    size: float | None = None,
) -> None:
    """Styled run for mixed-format intro paragraphs."""
    r = para.add_run(text)
    r.font.name = _FONT
    r.font.size = Pt(size if size is not None else _BODY_PT)
    r.bold = bold
    r.italic = italic
    if highlight:
        r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    if grey:
        r.font.color.rgb = _INTRO_GREY


def _add_intro_hyperlink(para, url: str) -> None:
    """Append a clickable hyperlink run to an intro paragraph."""
    part = para.part
    r_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    r_fonts = OxmlElement("w:rFonts")
    r_fonts.set(qn("w:ascii"), _FONT)
    r_fonts.set(qn("w:hAnsi"), _FONT)
    r_pr.append(r_fonts)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(int(_BODY_PT * 2)))
    r_pr.append(sz)
    colour = OxmlElement("w:color")
    colour.set(qn("w:val"), "0563C1")
    r_pr.append(colour)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    r_pr.append(underline)
    run.append(r_pr)

    text_elem = OxmlElement("w:t")
    text_elem.text = url
    run.append(text_elem)
    hyperlink.append(run)
    para._p.append(hyperlink)


# ---------------------------------------------------------------------------
# US-funded Summary of Informed Consent Form (sections 1.x)
# ---------------------------------------------------------------------------

_US_SUMMARY_PAGE_TITLE = "Summary of Informed Consent Form"


def _write_us_summary_page_opening(doc: Document, ext_map: dict[str, ExtractionResult]) -> None:
    """Page title (centered, underlined) and study title line (left, bold label)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(_US_SUMMARY_PAGE_TITLE)
    r.font.name = _FONT
    r.font.size = Pt(_BODY_PT)
    r.bold = False
    r.underline = True

    study_title = _get_study_title(ext_map)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(8)
    rl = p.add_run("Study Title: ")
    rl.font.name = _FONT
    rl.font.size = Pt(_BODY_PT)
    rl.bold = True
    rv = p.add_run(study_title)
    rv.font.name = _FONT
    rv.font.size = Pt(_BODY_PT)
    rv.bold = False


def _write_us_summary_sections_validation(
    doc: Document,
    variables: list[TemplateVariable],
    ext_map: dict[str, ExtractionResult],
) -> None:
    """US-funded summary block (sections 1.x) for the draft ICF."""
    _write_us_summary_page_opening(doc, ext_map)
    _write_us_summary_section_blocks(doc, variables, ext_map)
    doc.add_page_break()


def _write_us_summary_section_blocks(
    doc: Document,
    variables: list[TemplateVariable],
    ext_map: dict[str, ExtractionResult],
) -> None:
    """Write 1.1 body only (no extra heading), then 1.2–1.7 with sub-section headings."""
    summary_vars = [v for v in variables if v.section_id.startswith("1.")]
    var_11 = next((v for v in summary_vars if v.section_id == "1.1"), None)
    rest_vars = [v for v in summary_vars if v.section_id != "1.1"]

    if var_11 is not None:
        _write_us_summary_1_1_validation(doc, var_11, ext_map.get("1.1"))

    last_sub_section: str | None = None
    for var in rest_vars:
        ext = ext_map.get(var.section_id)

        content = _get_section_content(ext)
        if not content and not var.required:
            continue

        if var.sub_section and var.sub_section != last_sub_section:
            _add_subsection_heading(doc, var.sub_section, color=None)
            last_sub_section = var.sub_section
        elif not var.sub_section:
            last_sub_section = None

        if ext is not None and ext.status in (
            "FOUND",
            "PARTIAL",
            "NOT_FOUND",
            "ERROR",
            "SKIPPED",
        ):
            _add_validation_annotation(doc, ext)
        if content:
            _add_content_block(doc, content, color=None, highlight_markers=True)
            if ext is not None and ext.status == "PARTIAL" and ext.notes:
                _add_partial_notes(doc, ext.notes)
        else:
            _add_validation_placeholder(doc, ext, var, optional=not var.required)


def _write_us_summary_1_1_validation(
    doc: Document,
    var: TemplateVariable,
    ext: ExtractionResult | None,
) -> None:
    """Section 1.1 for validation ICF — content only, no heading."""
    content = _get_section_content(ext)
    if not content and not var.required:
        return

    if ext is not None and ext.status in (
        "FOUND",
        "PARTIAL",
        "NOT_FOUND",
        "ERROR",
        "SKIPPED",
    ):
        _add_validation_annotation(doc, ext)

    if content:
        _add_content_block(doc, content, color=None, highlight_markers=True)
        if ext is not None and ext.status == "PARTIAL" and ext.notes:
            _add_partial_notes(doc, ext.notes)
    else:
        _add_validation_placeholder(doc, ext, var, optional=not var.required)


# ---------------------------------------------------------------------------
# Cover page (sections 2.x)
# ---------------------------------------------------------------------------


def _write_cover_page(
    doc: Document,
    variables: list[TemplateVariable],
    ext_map: dict[str, ExtractionResult],
) -> None:
    # Main ICF title — centred, bold
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run("Informed Consent Form for Participation in a Research Study")
    run.font.name = _FONT
    run.font.size = Pt(_BODY_PT)
    run.bold = True

    _add_blank(doc)

    # Cover fields with extracted content, or empty fields that need study-team input.
    cover_vars = [v for v in variables if v.section_id.startswith("2.")]
    for var in cover_vars:
        ext = ext_map.get(var.section_id)
        content = _get_section_content(ext)
        if not content and not _should_render_empty_section(var, ext):
            continue

        label = var.sub_section or ""
        if content:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)

            if label:
                # Strip any leading repetition of the label from the extracted content
                # (e.g. "Study Title:" label + "Study Title: XYZ" content → "XYZ").
                content = _strip_label_prefix(content, label)
                rl = p.add_run(label)
                rl.bold = True
                rl.font.name = _FONT
                rl.font.size = Pt(_BODY_PT)
                rv = p.add_run(" " + content)
                rv.bold = False
                rv.font.name = _FONT
                rv.font.size = Pt(_BODY_PT)
            else:
                r = p.add_run(content)
                r.font.name = _FONT
                r.font.size = Pt(_BODY_PT)
        else:
            if label:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                rl = p.add_run(label)
                rl.bold = True
                rl.font.name = _FONT
                rl.font.size = Pt(_BODY_PT)
            _add_validation_placeholder(doc, ext, var, optional=not var.required)

        _add_blank(doc)


# ---------------------------------------------------------------------------
# Body sections (sections 3+)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Draft body (no colour coding; grey status annotations)
# ---------------------------------------------------------------------------

_ANNOTATION_GREY = RGBColor(0x88, 0x88, 0x88)


def _add_partial_notes(doc: Document, notes: str) -> None:
    """Grey italic status explanation rendered after section content."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(f"[PARTIAL] {notes}")
    r.font.name = _FONT
    r.font.size = Pt(_ANNOTATION_PT)
    r.italic = True
    r.font.color.rgb = _ANNOTATION_GREY


def _add_validation_annotation(doc: Document, ext: ExtractionResult) -> None:
    """Render a small italic grey status/confidence line below a heading."""
    parts = [f"Status: {ext.status}"]
    if ext.confidence and ext.confidence not in ("N/A", ""):
        parts.append(f"Confidence: {ext.confidence}")
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("  |  ".join(parts))
    r.font.name = _FONT
    r.font.size = Pt(_ANNOTATION_PT)
    r.italic = True
    r.font.color.rgb = _ANNOTATION_GREY


def _add_validation_placeholder(
    doc: Document,
    ext: ExtractionResult | None,
    var: TemplateVariable | None,
    optional: bool = False,
) -> None:
    """Bold yellow placeholder label + optional suggested text in grey italic."""
    label = _resolve_section_placeholder_label(var, optional=optional)
    _add_highlighted_placeholder(doc, label)

    # Co-Investigators: surface the UHN disclaimer (instructions) before suggested text
    # so the study team sees that listing co-investigators is generally discouraged.
    if var is not None and var.section_id == "2.4" and (var.instructions or "").strip():
        p_inst = doc.add_paragraph()
        p_inst.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_inst.paragraph_format.space_before = Pt(2)
        p_inst.paragraph_format.space_after = Pt(2)
        p_inst.paragraph_format.left_indent = Cm(0.5)
        r_inst = p_inst.add_run(var.instructions.strip())
        r_inst.font.name = _FONT
        r_inst.font.size = Pt(_ANNOTATION_PT)
        r_inst.italic = True
        r_inst.font.color.rgb = _ANNOTATION_GREY

    suggested = _section_suggested_text(var, ext)
    if suggested:
        p2 = doc.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p2.paragraph_format.space_before = Pt(2)
        p2.paragraph_format.space_after = Pt(3)
        p2.paragraph_format.left_indent = Cm(0.5)
        r2 = p2.add_run("Suggested text: " + suggested)
        r2.font.name = _FONT
        r2.font.size = Pt(_ANNOTATION_PT)
        r2.italic = True
        r2.font.color.rgb = _ANNOTATION_GREY


def _write_validation_main_body(
    doc: Document,
    variables: list[TemplateVariable],
    ext_map: dict[str, ExtractionResult],
) -> None:
    body_vars = [
        v
        for v in variables
        if not v.section_id.startswith("2.")
        and not v.section_id.startswith("1.")
        and v.section_id != SIGNATURE_CONSENT_SECTION_ID
    ]
    _write_validation_body(doc, body_vars, ext_map)


def _write_validation_body(
    doc: Document,
    variables: list[TemplateVariable],
    ext_map: dict[str, ExtractionResult],
) -> None:
    """Write body sections for the validation-phase ICF."""
    last_heading: str | None = None
    last_sub_section: str | None = None

    for var in variables:
        ext = ext_map.get(var.section_id)

        content = _get_section_content(ext)

        # Keep optional sections that have a meaningful extraction status so the
        # EC reviewer can see them and decide whether to fill them manually.
        # Only silently drop optional sections that were never attempted (ext is
        # None) or have an uninformative status (e.g. STANDARD_TEXT already
        # handled above).
        if not content and not var.required:
            no_useful_status = ext is None or ext.status not in (
                "NOT_FOUND",
                "SKIPPED",
                "ERROR",
            )
            if no_useful_status:
                continue

        # ---- Heading -----------------------------------------------------------
        if var.heading != last_heading:
            if last_heading is not None:
                _add_blank(doc)
            _add_heading(doc, var.heading, color=None)
            last_heading = var.heading
            last_sub_section = None

        # ---- Sub-section -------------------------------------------------------
        if var.sub_section and var.sub_section != last_sub_section:
            _add_subsection_heading(doc, var.sub_section, color=None)
            last_sub_section = var.sub_section
        elif not var.sub_section:
            last_sub_section = None

        # ---- Annotation (status / confidence) ----------------------------------
        if ext is not None and ext.status in (
            "FOUND",
            "PARTIAL",
            "NOT_FOUND",
            "ERROR",
            "SKIPPED",
        ):
            _add_validation_annotation(doc, ext)

        # ---- Content -----------------------------------------------------------
        if content:
            _add_content_block(doc, content, color=None, highlight_markers=True)
            if ext is not None and ext.status == "PARTIAL" and ext.notes:
                _add_partial_notes(doc, ext.notes)
        else:
            _add_validation_placeholder(doc, ext, var, optional=not var.required)


# ---------------------------------------------------------------------------
# Signature pages (standard UHN — only TITLE line changes)
# ---------------------------------------------------------------------------


def _write_signature_pages(
    doc: Document,
    study_title: str,
    *,
    sdm: bool = False,
    ext_map: dict[str, ExtractionResult] | None = None,
) -> None:
    doc.add_page_break()

    # TITLE line: "TITLE:" (plain) + " [title]" (bold) — matches approved ICF
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    _run(p, "TITLE:", bold=False)
    _run(p, " " + study_title, bold=True)

    _add_blank(doc)
    _add_blank(doc)

    # CONSENT heading
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    _run(p, "CONSENT")

    consent_items = resolve_signature_consent_bullets(ext_map or {}, sdm=sdm)
    for item in consent_items:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Cm(0.63)
        p.paragraph_format.first_line_indent = Cm(-0.63)
        _run(p, "\u2022  " + item)

    # Blank spacers before signature blocks
    for _ in range(3):
        _add_blank(doc)

    # --- Signature block 1: Participant (and SDM when enabled) ---
    participant_sig_labels = (
        ["Signature of Participant/", "Substitute Decision-Maker"]
        if sdm
        else ["Signature of Participant"]
    )
    _sig_three_column_block(
        doc,
        [
            ("_________________________", participant_sig_labels),
            ("______________________", ["PRINTED NAME"]),
            ("______________", ["Date"]),
        ],
    )

    if sdm:
        _sig_sdm_participant_printed_name_row(doc)

    for _ in range(4):
        _add_blank(doc)

    _add_blank(doc)
    _add_blank(doc)

    # Attestation for the person conducting the consent discussion (UHN signature page).
    _body_line(
        doc,
        "I have explained to the above-named participant the nature and purpose, "
        "the potential benefits, and possible risks of participation in this study. "
        "All questions that have been raised about this study have been answered.",
    )
    _add_blank(doc)

    # --- Signature block 2: Person Conducting Consent ---
    _sig_three_column_block(
        doc,
        [
            (
                "_________________________",
                ["Signature of Person Conducting", "the Consent Discussion"],
            ),
            ("______________________", ["PRINTED NAME & ROLE"]),
            ("______________", ["Date"]),
        ],
    )

    for _ in range(2):
        _add_blank(doc)

    # --- Interpreter / Witness attestation ---
    _body_line(
        doc,
        "The following attestation must be provided if the participant is unable "
        "to read or requires an oral translation: ",
    )
    _add_blank(doc)

    # Bold instruction line
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    _run(
        p,
        "If the participant is assisted during the consent process, please check "
        "the relevant box and complete the signature space below: ",
        bold=True,
    )
    _add_blank(doc)

    # ☐ Interpreter checkbox
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    r_cb = p.add_run("☐")
    r_cb.font.name = _FONT
    r_cb.font.size = Pt(14)
    _run(
        p,
        "\tThe person signing below acted as an interpreter, and attests that the "
        "study as set out in the consent form was accurately sight translated "
        "and/or interpreted, and that interpretation was provided on questions, "
        "responses and additional discussion arising from this process. ",
    )
    _add_blank(doc)

    # Interpreter signature
    _sig_three_column_block(
        doc,
        [
            ("_________________________", ["PRINT NAME", "of Interpreter"]),
            ("______________________", ["Signature"]),
            ("______________", ["Date"]),
        ],
        space_before_pt=6,
        trailing_blank=False,
    )
    _add_blank(doc)

    _body_line(doc, "______________________________________________________\t")
    _body_line(doc, "Language")

    # ☐ Witness / consent-read checkbox — same layout as interpreter checkbox
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(0)
    r_cb2 = p.add_run("☐")
    r_cb2.font.name = _FONT
    r_cb2.font.size = Pt(14)
    _run(
        p,
        "\tThe consent form was read to the participant. The person signing below "
        "attests that the study as set out in this form was accurately explained "
        "to the participant, and any questions have been answered. ",
    )
    _add_blank(doc)

    # Witness signature
    _sig_three_column_block(
        doc,
        [
            ("_________________________", ["PRINT NAME", "of witness"]),
            ("______________________", ["Signature"]),
            ("______________", ["Date"]),
        ],
        space_before_pt=6,
        trailing_blank=False,
    )
    _add_blank(doc)

    _body_line(doc, "____________________________\t")
    _body_line(doc, "Relationship to Participant")

    for _ in range(2):
        _add_blank(doc)
    _add_ai_disclaimer(doc)


# ---------------------------------------------------------------------------
# Paragraph / run helpers
# ---------------------------------------------------------------------------


def _run(
    para,
    text: str,
    bold: bool = False,
    underline: bool = False,
    color: RGBColor | None = None,
) -> None:
    r = para.add_run(text)
    r.font.name = _FONT
    r.font.size = Pt(_BODY_PT)
    r.bold = bold
    r.underline = underline
    if color is not None:
        r.font.color.rgb = color


def _add_blank(doc: Document) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)


def _body_line(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    _run(p, text)


def _add_heading(doc: Document, text: str, color: RGBColor | None = None) -> None:
    """All-caps underlined heading — matching the approved ICF style.

    *color* is applied to the run text when provided (confidence colour-coding).
    """
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text.upper())
    r.font.name = _FONT
    r.font.size = Pt(_BODY_PT)
    r.bold = False
    r.underline = True
    if color is not None:
        r.font.color.rgb = color


def _add_subsection_heading(
    doc: Document, text: str, color: RGBColor | None = None
) -> None:
    """Bold sub-section heading (e.g., 'Non-Experimental Procedures')."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(text)
    r.font.name = _FONT
    r.font.size = Pt(_BODY_PT)
    r.bold = True
    if color is not None:
        r.font.color.rgb = color


_TABLE_ROW_RE = re.compile(r"^\s*\|")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s\-|:]+\|\s*$")


def _split_content(text: str) -> list[tuple[str, str]]:
    """Split text into ('text', ...) and ('table', ...) segments."""
    segments: list[tuple[str, str]] = []
    current_kind: str | None = None
    current_lines: list[str] = []

    for line in text.split("\n"):
        kind = "table" if _TABLE_ROW_RE.match(line) else "text"
        if kind != current_kind:
            if current_lines:
                segments.append((current_kind, "\n".join(current_lines)))  # type: ignore[arg-type]
            current_kind = kind
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines and current_kind is not None:
        segments.append((current_kind, "\n".join(current_lines)))

    return segments


def _parse_markdown_table(table_text: str) -> list[list[str]]:
    """Parse a Markdown table into rows of cell strings (first row = header)."""
    rows: list[list[str]] = []
    for line in table_text.strip().split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        if _TABLE_SEP_RE.match(line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows


def _add_table_block(
    doc: Document,
    rows: list[list[str]],
    color: RGBColor | None = None,
    highlight_markers: bool = False,
) -> None:
    """Insert a bordered Word table from parsed Markdown rows.

    The first row is rendered as a bold header with light-grey shading.
    When *highlight_markers* is True, ``[PLEASE COMPLETE]`` in cell text is
    bold yellow (same as body paragraphs).
    """
    if not rows:
        return
    n_cols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=n_cols)
    table.style = "Table Grid"

    for ri, row_data in enumerate(rows):
        is_header = ri == 0
        for ci in range(n_cols):
            cell_text = row_data[ci] if ci < len(row_data) else ""
            cell = table.cell(ri, ci)
            cell.text = ""
            para = cell.paragraphs[0]
            _add_table_cell_runs(
                para,
                cell_text,
                color=color,
                highlight_markers=highlight_markers,
                bold_all=is_header,
            )
            if is_header:
                tc_pr = cell._tc.get_or_add_tcPr()
                shd = tc_pr.find(qn("w:shd"))
                if shd is None:
                    shd = OxmlElement("w:shd")
                    tc_pr.append(shd)
                shd.set(qn("w:val"), "clear")
                shd.set(qn("w:color"), "auto")
                shd.set(qn("w:fill"), "D9D9D9")

    _add_blank(doc)


def _add_table_cell_runs(
    p,
    text: str,
    *,
    color: RGBColor | None,
    highlight_markers: bool,
    bold_all: bool,
) -> None:
    """Write cell text, highlighting inline ``[PLEASE COMPLETE]`` when requested."""
    font_size = Pt(_BODY_PT - 1)
    if highlight_markers and MARKER_PLEASE_COMPLETE in text and _INLINE_MARKER_RE.search(text):
        for part in _INLINE_MARKER_RE.split(text):
            if not part:
                continue
            r = p.add_run(part)
            r.font.name = _FONT
            r.font.size = font_size
            if part == MARKER_PLEASE_COMPLETE:
                r.bold = True
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            else:
                r.bold = bold_all
                if color is not None:
                    r.font.color.rgb = color
        return

    r = p.add_run(text)
    r.font.name = _FONT
    r.font.size = font_size
    r.bold = bold_all
    if color is not None:
        r.font.color.rgb = color


def _add_text_runs(
    p,
    text: str,
    color: RGBColor | None,
    highlight_markers: bool,
) -> None:
    """Append runs to *p*, styling known inline edit markers."""
    if (highlight_markers and MARKER_PLEASE_COMPLETE in text) or MARKER_ADD_OTHER_ORGS in text:
        if _INLINE_MARKER_RE.search(text):
            for part in _INLINE_MARKER_RE.split(text):
                if not part:
                    continue
                if part == MARKER_PLEASE_COMPLETE and highlight_markers:
                    r = p.add_run(part)
                    r.font.name = _FONT
                    r.font.size = Pt(_BODY_PT)
                    r.bold = True
                    r.font.highlight_color = WD_COLOR_INDEX.YELLOW
                elif part == MARKER_ADD_OTHER_ORGS:
                    r = p.add_run(part)
                    r.font.name = _FONT
                    r.font.size = Pt(_BODY_PT)
                    r.italic = True
                    r.font.color.rgb = _ANNOTATION_GREY
                else:
                    r = p.add_run(part)
                    r.font.name = _FONT
                    r.font.size = Pt(_BODY_PT)
                    if color is not None:
                        r.font.color.rgb = color
            return

    r = p.add_run(text)
    r.font.name = _FONT
    r.font.size = Pt(_BODY_PT)
    if color is not None:
        r.font.color.rgb = color


def _parse_bullet_line(line: str) -> tuple[int, str] | None:
    """Return (nest_level, content) for a bullet/sub-bullet line, else None.

    Nesting is inferred from leading whitespace (2 spaces or 1 tab ≈ one level).
    The UHN ``o`` / ``O`` sub-bullet marker is treated as at least level 1.
    """
    match = _BULLET_LINE_RE.match(line.rstrip())
    if not match:
        return None
    indent = match.group("indent").replace("\t", "  ")
    ws_level = len(indent) // 2
    marker = match.group("marker")
    if marker.lower() == "o":
        level = max(ws_level, 1)
    else:
        level = ws_level
    content = (match.group("content") or "").strip()
    if not content:
        return None
    return min(level, len(_BULLET_GLYPHS) - 1), content


def _add_content_block(
    doc: Document,
    text: str,
    color: RGBColor | None = None,
    highlight_markers: bool = False,
) -> None:
    """Write a block of extracted content, splitting on newlines.

    Segments containing Markdown tables (lines starting with |) are rendered
    as proper Word tables.  Bullet lines become indented list items, with
    nested / ``o`` sub-bullets indented one level deeper.  All other lines are
    rendered as body paragraphs.
    *color* colours all runs when provided (confidence colour-coding).
    *highlight_markers* highlights ``[PLEASE COMPLETE]`` in bold yellow within the text.
    """
    for kind, segment in _split_content(text):
        if kind == "table":
            rows = _parse_markdown_table(segment)
            if rows:
                _add_table_block(
                    doc, rows, color=color, highlight_markers=highlight_markers
                )
                continue
        # Plain text — render line by line (preserve leading whitespace for nest level)
        for line in segment.split("\n"):
            if not line.strip():
                continue
            bullet = _parse_bullet_line(line)
            if bullet is not None:
                level, content = bullet
                glyph = _BULLET_GLYPHS[level]
                left_cm = _BULLET_BASE_INDENT_CM + (level * _BULLET_NEST_STEP_CM)
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.left_indent = Cm(left_cm)
                p.paragraph_format.first_line_indent = Cm(-_BULLET_HANGING_CM)
                _add_text_runs(p, f"{glyph} {content}", color, highlight_markers)
            else:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(3)
                _add_text_runs(p, line.strip(), color, highlight_markers)


def _set_table_borders_none(table) -> None:
    """Remove visible borders from a Word table."""
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    if tbl_pr is None:
        tbl_pr = OxmlElement("w:tblPr")
        tbl.insert(0, tbl_pr)
    borders = OxmlElement("w:tblBorders")
    for name in ("top", "left", "bottom", "right", "insideH", "insideV"):
        edge = OxmlElement(f"w:{name}")
        edge.set(qn("w:val"), "nil")
        borders.append(edge)
    tbl_pr.append(borders)


def _set_cell_margins(cell, *, top: int = 0, bottom: int = 0) -> None:
    """Set table-cell margins in twips (dxa). Defaults in Word leave a visible row gap."""
    tc_pr = cell._tc.get_or_add_tcPr()
    existing = tc_pr.find(qn("w:tcMar"))
    if existing is not None:
        tc_pr.remove(existing)
    tc_mar = OxmlElement("w:tcMar")
    for side, value in (("top", top), ("bottom", bottom)):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        tc_mar.append(node)
    tc_pr.append(tc_mar)


def _add_sig_label_lines(cell, lines: list[str]) -> None:
    """Write left-aligned label lines directly under a signature underline."""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    for i, line in enumerate(lines):
        if i > 0:
            p.add_run().add_break()
        r = p.add_run(line)
        r.font.name = _FONT
        r.font.size = Pt(_BODY_PT)


def _sig_three_column_block(
    doc: Document,
    columns: list[tuple[str, list[str]]],
    *,
    space_before_pt: float = 0,
    trailing_blank: bool = True,
) -> None:
    """Three-column signature block with left-aligned labels under each underline."""
    table = doc.add_table(rows=2, cols=3)
    table.autofit = False
    _set_table_borders_none(table)

    col_widths = (Inches(2.35), Inches(2.05), Inches(1.35))
    for row in table.rows:
        for ci, width in enumerate(col_widths):
            row.cells[ci].width = width

    if space_before_pt:
        table.rows[0].cells[0].paragraphs[0].paragraph_format.space_before = Pt(
            space_before_pt
        )

    for ci, (underline, labels) in enumerate(columns):
        ul_cell = table.rows[0].cells[ci]
        ul_p = ul_cell.paragraphs[0]
        ul_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        ul_p.paragraph_format.space_before = Pt(0)
        ul_p.paragraph_format.space_after = Pt(0)
        _set_cell_margins(ul_cell, bottom=0)
        _run(ul_p, underline)

        label_cell = table.rows[1].cells[ci]
        _set_cell_margins(label_cell, top=0)
        _add_sig_label_lines(label_cell, labels)

    if trailing_blank:
        _add_blank(doc)


def _sig_sdm_participant_printed_name_row(doc: Document) -> None:
    """SDM-only row: prompt on the left, participant printed-name line in the middle."""
    table = doc.add_table(rows=2, cols=3)
    table.autofit = False
    _set_table_borders_none(table)

    col_widths = (Inches(2.35), Inches(2.05), Inches(1.35))
    for row in table.rows:
        for ci, width in enumerate(col_widths):
            row.cells[ci].width = width

    prompt_cell = table.rows[0].cells[0]
    prompt_p = prompt_cell.paragraphs[0]
    prompt_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    prompt_p.paragraph_format.space_before = Pt(6)
    _run(prompt_p, "If consent is provided by Substitute Decision Maker:")

    ul_cell = table.rows[0].cells[1]
    ul_p = ul_cell.paragraphs[0]
    ul_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    ul_p.paragraph_format.space_before = Pt(6)
    ul_p.paragraph_format.space_after = Pt(0)
    _set_cell_margins(ul_cell, bottom=0)
    _run(ul_p, "______________________")

    label_cell = table.rows[1].cells[1]
    _set_cell_margins(label_cell, top=0)
    _add_sig_label_lines(label_cell, ["PRINTED NAME of Participant"])

    _add_blank(doc)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _strip_label_prefix(content: str, label: str) -> str:
    """Remove a leading label repetition from extracted content.

    Handles cases where the LLM echoes the field name in its output, e.g.:
      label="Study Title:"  content="Study Title: Some Long Title"  → "Some Long Title"
    Matching is case-insensitive; the trailing colon and surrounding whitespace
    are consumed so the result is clean.
    """
    label_clean = label.rstrip(":").strip()
    # Try matching with optional colon after the label
    pattern = re.compile(r"^" + re.escape(label_clean) + r"\s*:?\s*", re.IGNORECASE)
    return pattern.sub("", content).strip() or content


def _plain_suggested_text(var: TemplateVariable) -> str:
    """Return suggested_text as plain text, stripping HTML tags when format is 'html'."""
    raw = html_mod.unescape(var.suggested_text)
    if var.suggested_text_format == "html":
        return re.sub(r"<[^>]+>", " ", raw).strip()
    return raw


def _resolve_section_placeholder_label(
    var: TemplateVariable | None,
    *,
    optional: bool = False,
) -> str:
    if _section_suggested_text(var):
        return MARKER_OPTIONAL_SUGGESTED if optional else MARKER_REQUIRED_SUGGESTED
    return MARKER_PLEASE_COMPLETE


def _add_highlighted_placeholder(doc: Document, label: str) -> None:
    """Bold yellow-highlighted placeholder for sections requiring study-team input."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(label)
    r.font.name = _FONT
    r.font.size = Pt(_BODY_PT)
    r.bold = True
    r.font.highlight_color = WD_COLOR_INDEX.YELLOW


def _section_suggested_text(
    var: TemplateVariable | None,
    ext: ExtractionResult | None = None,
) -> str:
    if not var:
        return ""
    if (
        var.section_id == _TESTS_PROCEDURES_SECTION_ID
        and ext is not None
        and ext.status in ("NOT_FOUND", "SKIPPED")
    ):
        return _TESTS_PROCEDURES_NOT_FOUND_SUGGESTED
    if var.suggested_text:
        return _plain_suggested_text(var).strip()
    if var.required_text:
        return var.required_text.strip()
    return ""


def _add_ai_disclaimer(doc: Document) -> None:
    """Append the AI-use statement at the end of the signature page."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(_AI_DISCLAIMER)
    r.font.name = _FONT
    r.font.size = Pt(_BODY_PT)


def _should_render_empty_section(
    var: TemplateVariable, ext: ExtractionResult | None
) -> bool:
    """Whether an empty section should still appear with a placeholder."""
    if var.required:
        return True
    return ext is not None and ext.status in ("NOT_FOUND", "SKIPPED", "ERROR")


def _get_section_content(ext: ExtractionResult | None) -> str | None:
    """Return the usable text content of an extraction, or None if unavailable."""
    if ext is None or ext.status not in _CONTENT_STATUSES:
        return None
    text = (ext.filled_template or ext.answer or "").strip()
    return text if text else None


def _get_study_title(ext_map: dict[str, ExtractionResult]) -> str:
    """Return the study title from section 2.1, or a placeholder if not found."""
    ext = ext_map.get("2.1")
    if ext and ext.status in _CONTENT_STATUSES:
        raw = (ext.filled_template or ext.answer or "").strip()
        if raw:
            return _strip_label_prefix(raw, "Study Title:")
        return "[Study Title]"
    return "[Study Title]"

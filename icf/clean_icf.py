"""
Clean publication-quality ICF document generator.

  - UHN logo in the header (top-left)
  - Bordered footer with automatic "Page X of Y" page numbering
  - All-caps underlined section headings (matching the approved-ICF style)
  - Justified body text in Arial 11 pt
  - Standard UHN signature pages appended verbatim (only the TITLE line changes)

Only sections with usable content (FOUND / PARTIAL / STANDARD_TEXT) appear in
the body.  Required sections that could not be extracted get a [TO BE COMPLETED]
placeholder so the document remains structurally whole.  Optional sections with
no content are silently omitted.
"""

from __future__ import annotations

import html as html_mod
import os
import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

from icf.types import ExtractionResult, TemplateVariable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FONT = "Arial"
_BODY_PT = 11
_SMALL_PT = 9

_CONTENT_STATUSES = {"FOUND", "PARTIAL", "STANDARD_TEXT"}
_BULLET_RE = re.compile(r"^[•\-–\*·]\s+")

# Confidence-level colours (applied to heading + body text of each section)
_CONFIDENCE_COLORS: dict[str, RGBColor] = {
    "HIGH": RGBColor(0x00, 0xB0, 0x50),   # green
    "MEDIUM": RGBColor(0xFF, 0xC0, 0x00),  # amber
    "LOW": RGBColor(0xFF, 0x00, 0x00),     # red
}

# Tab stop positions (in twips, measured from the left margin) used for all
# three-column signature blocks so every column aligns perfectly.
_SIG_TAB1 = 4320  # 3.0" — separates signature column from name column
_SIG_TAB2 = 6480  # 4.5" — separates name column from date column


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_clean_icf_docx(
    extractions: list[ExtractionResult],
    variables: list[TemplateVariable],
    output_path: str,
    logo_path: str | None = None,
) -> str:
    """Generate a clean, publication-quality ICF DOCX.

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

    _write_intro_page(doc)
    _write_cover_page(doc, variables, ext_map)
    _write_body_sections(doc, variables, ext_map)
    _write_signature_pages(doc, _get_study_title(ext_map))

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
_INTRO_BORDER_GREY = RGBColor(0xBB, 0xBB, 0xBB)


def _write_intro_page(doc: Document) -> None:
    """Write an introductory notes page that precedes the main ICF content."""
    # ---- Page title --------------------------------------------------------
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run("AI-Generated ICF: Important Notes")
    r.font.name = _FONT
    r.font.size = Pt(14)
    r.bold = True
    r.underline = True

    _add_blank(doc)

    # ---- Color coding legend -----------------------------------------------
    _intro_section_label(doc, "Color Coding Legend")

    _intro_body(
        doc,
        "Section headings and body text are color-coded by extraction confidence:",
    )

    _legend_entries: list[tuple[str, str, RGBColor]] = [
        ("HIGH CONFIDENCE", "Content was reliably extracted from the protocol.", _CONFIDENCE_COLORS["HIGH"]),
        ("MEDIUM CONFIDENCE", "Content was partially extracted or inferred; human review recommended.", _CONFIDENCE_COLORS["MEDIUM"]),
        ("LOW CONFIDENCE", "Content could not be reliably extracted; requires careful review and completion.", _CONFIDENCE_COLORS["LOW"]),
    ]
    for label, desc, color in _legend_entries:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.left_indent = Cm(0.8)
        r_label = p.add_run(label + ": ")
        r_label.font.name = _FONT
        r_label.font.size = Pt(_BODY_PT)
        r_label.bold = True
        r_label.font.color.rgb = color
        r_desc = p.add_run(desc)
        r_desc.font.name = _FONT
        r_desc.font.size = Pt(_BODY_PT)
        r_desc.font.color.rgb = _INTRO_GREY

    _add_blank(doc)

    # ---- Required sections note --------------------------------------------
    _intro_section_label(doc, "Required Sections Not Found in Protocol")

    _intro_body(
        doc,
        "All required ICF sections are included in this document, even when the "
        "corresponding information could not be located in the study protocol. "
        "For such sections, a brief suggested text is provided in small italic grey "
        "font directly below the section heading. This suggested text is for guidance "
        "only and must be reviewed, edited, and approved by the study team before use.",
    )

    _add_blank(doc)

    # ---- Detailed draft reference ------------------------------------------
    _intro_section_label(doc, "Detailed ICF Draft")

    _intro_body(
        doc,
        "A separate detailed ICF draft (draft_icf.docx) is produced alongside this "
        "clean version. The detailed draft contains additional information for each "
        "section, including:",
    )
    for bullet in [
        "Evidence citations with direct quotes and page references from the protocol",
        "Extraction status and confidence scores",
        "Validation issues flagged during quality checking",
        "Suggested text for sections requiring manual completion",
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Cm(1.2)
        p.paragraph_format.first_line_indent = Cm(-0.5)
        r = p.add_run("\u2022 " + bullet)
        r.font.name = _FONT
        r.font.size = Pt(_BODY_PT)
        r.font.color.rgb = _INTRO_GREY

    _add_blank(doc)

    # ---- Disclaimers -------------------------------------------------------
    _intro_section_label(doc, "Disclaimers")

    for note in [
        "This document was generated automatically by an AI system and has not been "
        "reviewed or approved by a regulatory body or ethics committee.",
        "All content must be reviewed, verified, and edited by qualified clinical "
        "research professionals before submission or use with study participants.",
        "The AI extraction is based solely on the provided study protocol. Information "
        "not present in the protocol will not appear in the extracted content.",
        "This tool does not constitute legal, medical, or regulatory advice.",
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.5)
        r = p.add_run("\u2022 " + note)
        r.font.name = _FONT
        r.font.size = Pt(_SMALL_PT)
        r.italic = True
        r.font.color.rgb = _INTRO_GREY

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

    # One line per 2.x section that has extractable content
    cover_vars = [v for v in variables if v.section_id.startswith("2.")]
    for var in cover_vars:
        ext = ext_map.get(var.section_id)
        content = _get_section_content(ext)
        if not content:
            continue

        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)

        label = var.sub_section or ""
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

        _add_blank(doc)


# ---------------------------------------------------------------------------
# Body sections (sections 3+)
# ---------------------------------------------------------------------------


def _write_body_sections(
    doc: Document,
    variables: list[TemplateVariable],
    ext_map: dict[str, ExtractionResult],
) -> None:
    last_heading: str | None = None
    last_sub_section: str | None = None  # sentinel so first sub is always written

    body_vars = [v for v in variables if not v.section_id.startswith("2.")]

    for var in body_vars:
        ext = ext_map.get(var.section_id)

        # Adaptation-skipped sections are fully irrelevant — omit entirely.
        if ext and ext.status == "ADAPTATION_SKIPPED":
            continue

        content = _get_section_content(ext)

        # Optional sections with no usable content are silently omitted.
        if not content and not var.required:
            continue

        confidence = ext.confidence if ext else None
        color = _CONFIDENCE_COLORS.get(confidence or "", None)

        # ---- Heading -------------------------------------------------------
        if var.heading != last_heading:
            if last_heading is not None:
                _add_blank(doc)
            _add_heading(doc, var.heading, color=color)
            last_heading = var.heading
            last_sub_section = None  # reset sub-section tracking

        # ---- Sub-section ---------------------------------------------------
        if var.sub_section and var.sub_section != last_sub_section:
            _add_subsection_heading(doc, var.sub_section, color=color)
            last_sub_section = var.sub_section
        elif not var.sub_section:
            last_sub_section = None

        # ---- Content -------------------------------------------------------
        if content:
            _add_content_block(doc, content, color=color)
        else:
            # Required section — show a placeholder so the document is complete.
            _add_placeholder(doc, ext, var)


# ---------------------------------------------------------------------------
# Signature pages (standard UHN — only TITLE line changes)
# ---------------------------------------------------------------------------


def _write_signature_pages(doc: Document, study_title: str) -> None:
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

    # Consent bullet items — explicit bullet character + hanging indent
    _consent_items = [
        "All of my questions have been answered",
        "I allow access to medical records and related personal health information "
        "as explained in this consent form",
        "I do not give up any legal rights by signing this consent form,",
        "I agree to take part in this study.",
    ]
    for item in _consent_items:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Cm(0.63)
        p.paragraph_format.first_line_indent = Cm(-0.63)
        _run(p, "\u2022  " + item)

    # Blank spacers before signature blocks
    for _ in range(3):
        _add_blank(doc)

    # --- Signature block 1: Participant / Substitute Decision-Maker ---
    _sig_underlines(doc)
    _sig_label(doc, "Signature of Participant/\tPRINTED NAME\tDate")
    _body_line(doc, "Substitute Decision-Maker")

    for _ in range(4):
        _add_blank(doc)

    _add_blank(doc)
    _add_blank(doc)
    _add_blank(doc)

    # --- Signature block 2: Person Conducting Consent ---
    _sig_underlines(doc)
    _sig_label(doc, "Signature of Person Conducting \tPRINTED NAME & ROLE\tDate")
    _body_line(doc, "the Consent Discussion")

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
    _sig_underlines_interpreter(doc)
    _sig_label(doc, "PRINT NAME\tSignature\tDate")
    _body_line(doc, "of Interpreter")
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
    _sig_underlines_interpreter(doc)
    _sig_label(doc, "PRINT NAME\tSignature\tDate")
    _body_line(doc, "of witness")
    _add_blank(doc)

    _body_line(doc, "____________________________\t")
    _body_line(doc, "Relationship to Participant")


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


def _add_content_block(
    doc: Document, text: str, color: RGBColor | None = None
) -> None:
    """Write a block of extracted content, splitting on newlines.

    Lines starting with bullet markers (•, -, –) are rendered as indented
    list items; all other lines are rendered as justified body paragraphs.
    *color* colours all runs when provided (confidence colour-coding).
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    for line in lines:
        stripped = line.strip()
        if _BULLET_RE.match(stripped):
            content = _BULLET_RE.sub("", stripped).strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.left_indent = Cm(1.0)
            p.paragraph_format.first_line_indent = Cm(-0.5)
            r = p.add_run("\u2022 " + content)
            r.font.name = _FONT
            r.font.size = Pt(_BODY_PT)
            if color is not None:
                r.font.color.rgb = color
        else:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(3)
            r = p.add_run(stripped)
            r.font.name = _FONT
            r.font.size = Pt(_BODY_PT)
            if color is not None:
                r.font.color.rgb = color


def _add_placeholder(
    doc: Document,
    ext: ExtractionResult | None,
    var: TemplateVariable | None = None,
) -> None:
    """Add a visible placeholder for a required section with no usable content.

    When *var* carries suggested_text, that text is rendered below the
    placeholder in a smaller grey italic style so the reviewer has a starting
    point without cluttering the clean ICF layout.
    """
    reason = ""
    if ext:
        if ext.status == "SKIPPED":
            reason = " (Not in protocol — requires manual entry)"
        elif ext.status == "NOT_FOUND":
            reason = " (Not found in protocol)"
        elif ext.status == "ERROR":
            reason = f" (Extraction error: {ext.error})"

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(f"[TO BE COMPLETED{reason}]")
    r.font.name = _FONT
    r.font.size = Pt(_BODY_PT)
    r.italic = True

    if var and var.suggested_text:
        suggested = _plain_suggested_text(var).strip()
        if suggested:
            p2 = doc.add_paragraph()
            p2.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p2.paragraph_format.space_before = Pt(2)
            p2.paragraph_format.space_after = Pt(3)
            p2.paragraph_format.left_indent = Cm(0.5)
            r2 = p2.add_run("Suggested text: " + suggested)
            r2.font.name = _FONT
            r2.font.size = Pt(_SMALL_PT)
            r2.italic = True
            r2.font.color.rgb = RGBColor(0x88, 0x88, 0x88)


def _apply_sig_tab_stops(para) -> None:
    """Set two explicit tab stops for a three-column signature paragraph.

    Both the underline row and the label row(s) of a signature block must
    share the same tab stop positions so every column aligns perfectly.
    """
    pPr = para._element.get_or_add_pPr()
    tabs = OxmlElement("w:tabs")
    for pos in (_SIG_TAB1, _SIG_TAB2):
        tab = OxmlElement("w:tab")
        tab.set(qn("w:val"), "left")
        tab.set(qn("w:pos"), str(pos))
        tabs.append(tab)
    pPr.append(tabs)


def _sig_underlines(doc: Document) -> None:
    """Standard three-column signature underline row with aligned tab stops."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    _apply_sig_tab_stops(p)
    _run(p, "____________________________\t______________________\t_________________")


def _sig_label(doc: Document, text: str) -> None:
    """Label row under a signature underline block (same tab stops for alignment)."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    _apply_sig_tab_stops(p)
    _run(p, text)


def _sig_underlines_interpreter(doc: Document) -> None:
    """Interpreter/witness underline row — middle column has an underlined gap."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(0)
    _apply_sig_tab_stops(p)
    _run(p, "____________________________\t__")
    r_gap = p.add_run("            ")
    r_gap.font.name = _FONT
    r_gap.font.size = Pt(_BODY_PT)
    r_gap.underline = True
    _run(p, "____________\t_________________")


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
        return (ext.filled_template or ext.answer or "").strip() or "[Study Title]"
    return "[Study Title]"

"""
Protocol ingestion: parse PDF or DOCX into page-indexed text.

Supports:
  - PDF via pypdf (native page boundaries)
  - DOCX via python-docx (page-break detection with char-limit fallback)
"""

import os

from docx import Document
from pypdf import PdfReader

from icf.types import IndexedProtocol, ProtocolPage


def load_pdf(filepath: str) -> IndexedProtocol:
    """Parse a PDF protocol into page-indexed text."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Protocol PDF not found: {filepath}")

    reader = PdfReader(filepath)
    pages: list[ProtocolPage] = []
    full_text_parts: list[str] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            page_obj = ProtocolPage(page_number=i + 1, text=text)
            pages.append(page_obj)
            full_text_parts.append(f"--- PAGE {i + 1} ---\n{text}")

    if not pages:
        raise ValueError(f"No text could be extracted from PDF: {filepath}")

    return IndexedProtocol(
        pages=pages,
        full_text="\n".join(full_text_parts),
        total_pages=len(reader.pages),
        source_path=filepath,
    )


def load_docx(filepath: str) -> IndexedProtocol:
    """Parse a DOCX protocol into page-indexed text.

    DOCX files don't have native page numbers. We detect page-break elements
    where possible and fall back to splitting every ~3000 chars.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Protocol DOCX not found: {filepath}")

    doc = Document(filepath)

    PAGE_CHAR_LIMIT = 3000
    current_page = 1
    current_parts: list[str] = []
    current_length = 0
    pages: list[ProtocolPage] = []

    def flush_page():
        nonlocal current_page, current_parts, current_length
        if current_parts:
            pages.append(
                ProtocolPage(
                    page_number=current_page,
                    text="\n".join(current_parts),
                )
            )
            current_page += 1
            current_parts = []
            current_length = 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Detect hard page breaks in the paragraph's runs
        has_page_break = False
        try:
            from docx.oxml.ns import qn

            for run in para.runs:
                for br in run._element.findall(qn("w:br")):
                    if br.get(qn("w:type")) == "page":
                        has_page_break = True
                        break
                if has_page_break:
                    break
        except Exception:
            pass

        if has_page_break and current_parts:
            flush_page()

        current_parts.append(text)
        current_length += len(text)

        if current_length >= PAGE_CHAR_LIMIT:
            flush_page()

    flush_page()  # remaining content

    if not pages:
        raise ValueError(f"No text could be extracted from DOCX: {filepath}")

    full_text_parts = [f"--- PAGE {p.page_number} ---\n{p.text}" for p in pages]

    return IndexedProtocol(
        pages=pages,
        full_text="\n".join(full_text_parts),
        total_pages=len(pages),
        source_path=filepath,
    )


def load_protocol(filepath: str) -> IndexedProtocol:
    """Load a protocol from PDF or DOCX. Dispatches based on file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return load_pdf(filepath)
    if ext in (".docx", ".doc"):
        return load_docx(filepath)
    raise ValueError(f"Unsupported protocol format '{ext}'. Supported: .pdf, .docx")

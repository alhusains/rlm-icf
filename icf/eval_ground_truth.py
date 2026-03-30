"""
Ground truth ICF parser.

Parses an approved human-written ICF (DOCX) into section-by-section text
that can be compared against AI-generated extraction results.

Section matching strategy:
  1. Parse the DOCX into headings + body text blocks.
  2. Match each block to the closest ICF template section by heading text.
  3. Return a dict mapping section_id -> ground truth text.

The matcher is fuzzy: it normalises case, strips numbering, and uses
substring matching so that "INTRODUCTION" in the ground truth matches
section 3 ("Introduction") in the registry, even if formatting differs.
"""

from __future__ import annotations

import re

from docx import Document

from icf.types import TemplateVariable


def parse_ground_truth_docx(
    docx_path: str,
    variables: list[TemplateVariable],
) -> dict[str, str]:
    """Parse a ground truth ICF DOCX and map content to section IDs.

    Args:
        docx_path:  Path to the approved human-written ICF DOCX.
        variables:  Template variables from the registry (for heading matching).

    Returns:
        Dict mapping section_id -> ground truth text for each matched section.
        Sections that could not be matched are omitted.
    """
    doc = Document(docx_path)
    blocks = _extract_blocks(doc)
    return _match_blocks_to_sections(blocks, variables)


# ------------------------------------------------------------------
# Block extraction
# ------------------------------------------------------------------


def _extract_blocks(doc: Document) -> list[tuple[str, str]]:
    """Extract (heading, body_text) blocks from the DOCX.

    Groups consecutive body paragraphs under the most recent heading.
    """
    blocks: list[tuple[str, str]] = []
    current_heading = ""
    current_body: list[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style = para.style.name.lower() if para.style else ""
        is_heading = (
            "heading" in style
            or _is_allcaps_heading(text)
        )

        if is_heading:
            # Flush previous block
            if current_heading and current_body:
                blocks.append((current_heading, "\n".join(current_body)))
            current_heading = text
            current_body = []
        else:
            current_body.append(text)

    # Flush last block
    if current_heading and current_body:
        blocks.append((current_heading, "\n".join(current_body)))

    return blocks


def _is_allcaps_heading(text: str) -> bool:
    """Detect all-caps headings like 'INTRODUCTION' or 'WHY IS THIS STUDY BEING DONE?'."""
    # Must be mostly uppercase, 2-15 words, no long sentences
    if len(text) > 150:
        return False
    words = text.split()
    if len(words) < 1 or len(words) > 15:
        return False
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False
    upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    return upper_ratio > 0.7


# ------------------------------------------------------------------
# Section matching
# ------------------------------------------------------------------

# Common numbering prefixes to strip: "1.", "1.1", "Section 3:", etc.
_NUM_PREFIX_RE = re.compile(
    r"^(?:section\s+)?\d+(?:\.\d+)*\.?\s*[-:.]?\s*", re.IGNORECASE
)


def _normalise_heading(text: str) -> str:
    """Normalise a heading for fuzzy matching."""
    text = _NUM_PREFIX_RE.sub("", text)
    text = text.lower().strip()
    # Remove trailing punctuation
    text = text.rstrip("?:.")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def _match_blocks_to_sections(
    blocks: list[tuple[str, str]],
    variables: list[TemplateVariable],
) -> dict[str, str]:
    """Match extracted DOCX blocks to template section IDs.

    Uses fuzzy heading matching: normalised substring containment.
    When multiple variables share the same heading (e.g. all 2.x sections
    share "Informed Consent Form..."), falls back to sub_section matching.
    """
    result: dict[str, str] = {}
    matched_block_indices: set[int] = set()

    # Build lookup from normalised heading -> list of variables
    heading_map: dict[str, list[TemplateVariable]] = {}
    for var in variables:
        norm = _normalise_heading(var.heading)
        heading_map.setdefault(norm, []).append(var)
        # Also register sub_section as a matchable heading
        if var.sub_section:
            sub_norm = _normalise_heading(var.sub_section)
            heading_map.setdefault(sub_norm, []).append(var)

    for bi, (heading, body) in enumerate(blocks):
        norm_block = _normalise_heading(heading)
        if not norm_block:
            continue

        # Try exact match first, then substring containment
        matched_vars: list[TemplateVariable] = []

        if norm_block in heading_map:
            matched_vars = heading_map[norm_block]
        else:
            # Substring match: check if any registry heading is contained
            # in the block heading or vice versa
            for norm_reg, vars_list in heading_map.items():
                if norm_reg in norm_block or norm_block in norm_reg:
                    matched_vars = vars_list
                    break

        if not matched_vars:
            continue

        # If multiple variables matched (same heading), pick the first
        # unmatched one (preserves document order)
        for var in matched_vars:
            if var.section_id not in result:
                result[var.section_id] = body.strip()
                matched_block_indices.add(bi)
                break

    return result


def print_ground_truth_summary(
    ground_truth: dict[str, str],
    variables: list[TemplateVariable],
) -> None:
    """Print a summary of ground truth matching results."""
    total = len(variables)
    matched = len(ground_truth)
    print(f"\n[GROUND TRUTH] Matched {matched}/{total} sections from DOCX.")

    for var in variables:
        gt = ground_truth.get(var.section_id)
        display = var.get_display_name()
        if gt:
            preview = gt[:80].replace("\n", " ")
            print(f"  [{var.section_id}] MATCHED: {display} -> \"{preview}...\"")
        else:
            print(f"  [{var.section_id}] NOT MATCHED: {display}")

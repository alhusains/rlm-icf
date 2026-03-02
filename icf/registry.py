"""
ICF template registry loader.

Supports two source formats:
  - CSV  (legacy)  : data/standard_ICF_template_breakdown.csv
  - JSON (canonical): data/standard_ICF_template_breakdown.json

Use ``convert_csv_to_json`` once to produce the JSON file, then always
point the pipeline at the JSON.  The JSON format:
  - Preserves all fields exactly (no re-parsing needed each run)
  - Supports ``suggested_text_format`` = "text" | "html" so rich table
    content can be stored in ``suggested_text`` without CSV cell limitations
  - Is human-editable for manual refinements or future dynamic patching

``load_template_registry(path)`` auto-detects CSV vs JSON from the
file extension, so existing callers need no changes.
"""

import ast
import csv
import html
import json
import os

from icf.types import TemplateVariable


# ======================================================================
# Internal helpers (CSV parsing)
# ======================================================================


def _parse_complexity(raw: str) -> list[str]:
    """Parse the complexity field (stored as a JSON-like list string)."""
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            result = ast.literal_eval(raw)
            if isinstance(result, list):
                return [str(x) for x in result]
        except (ValueError, SyntaxError):
            pass
    return [raw]


def _classify_availability(
    complexity: list[str],
) -> tuple[bool, bool, bool]:
    """Classify a variable's protocol availability.

    Returns:
        (is_in_protocol, partially_in_protocol, is_standard_text)
    """
    tags_lower = [c.lower() for c in complexity]
    joined = " ".join(tags_lower)

    has_not_in = "not in protocol" in joined
    has_mapping = any(
        kw in joined for kw in ["easy mapping", "moderate mapping", "complex mapping"]
    )
    has_standard = "standard text" in joined
    has_potentially = "potentially in protocol" in joined

    if has_standard:
        return True, False, True
    if has_not_in and not has_mapping and not has_potentially:
        return False, False, False
    if has_not_in and (has_mapping or has_potentially):
        return True, True, False
    return True, False, False


def _parse_required(raw: str) -> bool:
    """Determine if a section is required."""
    lower = raw.strip().lower()
    if lower.startswith("required"):
        return True
    if lower.startswith("optional"):
        return False
    return "required" in lower and "optional" not in lower


# ======================================================================
# CSV loader (internal — call load_template_registry() from outside)
# ======================================================================


def _load_template_registry_csv(csv_path: str) -> list[TemplateVariable]:
    """Parse the ICF template breakdown CSV into TemplateVariable objects.

    Expected CSV columns (by index):
      0  Section #
      1  Status
      2  Content Complexity
      3  Section Heading
      4  Sub-section
      5  Required or Optional
      6  Section Instructions
      7  Required Text
      8  Suggested Text
      9  Questions from project team
      10 Minimal Risk Template Mapping
      11 Mapping to UHN Protocol Template Section
      12 Mapping to sponsor protocol template
      13 Build Plan/Key Decisions
      14 Notes
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Template CSV not found: {csv_path}")

    variables: list[TemplateVariable] = []

    with open(csv_path, encoding="cp1252") as f:
        reader = csv.reader(f)
        _header = next(reader)  # skip header

        for _row_num, row in enumerate(reader, start=2):
            if len(row) < 8:
                continue

            csv_status = row[1].strip()
            if csv_status.lower() == "excluded":
                continue

            section_id = row[0].strip()
            complexity_raw = row[2].strip()
            heading = html.unescape(row[3].strip())
            sub_section = html.unescape(row[4].strip()) or None
            required_raw = row[5].strip()
            instructions = html.unescape(row[6].strip())
            required_text = html.unescape(row[7].strip())
            suggested_text = html.unescape(row[8].strip()) if len(row) > 8 else ""

            complexity = _parse_complexity(complexity_raw)
            is_in, partial, standard = _classify_availability(complexity)
            required = _parse_required(required_raw)

            variables.append(
                TemplateVariable(
                    section_id=section_id,
                    heading=heading,
                    sub_section=sub_section,
                    required=required,
                    instructions=instructions,
                    required_text=required_text,
                    suggested_text=suggested_text,
                    suggested_text_format="text",
                    complexity=complexity,
                    is_in_protocol=is_in,
                    partially_in_protocol=partial,
                    is_standard_text=standard,
                )
            )

    if not variables:
        raise ValueError(f"No variables loaded from CSV: {csv_path}")

    return variables


# ======================================================================
# JSON loader
# ======================================================================


def load_template_registry_json(json_path: str) -> list[TemplateVariable]:
    """Load a pre-converted JSON template registry.

    Accepts two root formats:
      - Array   (legacy v1): ``[{...}, ...]``
      - Object  (v2):        ``{"schema": {...}, "sections": [{...}, ...]}``

    The JSON file is produced by ``convert_csv_to_json`` and may be
    manually edited afterwards to add rich content (e.g. HTML tables in
    ``suggested_text`` with ``suggested_text_format`` = "html").
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Template JSON not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Support both root formats
    items: list[dict] = data if isinstance(data, list) else data["sections"]

    variables: list[TemplateVariable] = []
    for item in items:
        variables.append(
            TemplateVariable(
                section_id=item["section_id"],
                heading=item["heading"],
                sub_section=item.get("sub_section"),
                required=item["required"],
                instructions=item["instructions"],
                required_text=item["required_text"],
                suggested_text=item["suggested_text"],
                suggested_text_format=item.get("suggested_text_format", "text"),
                complexity=item["complexity"],
                is_in_protocol=item["is_in_protocol"],
                partially_in_protocol=item["partially_in_protocol"],
                is_standard_text=item["is_standard_text"],
            )
        )

    if not variables:
        raise ValueError(f"No variables loaded from JSON: {json_path}")

    return variables


# ======================================================================
# Conversion utility  (run once: CSV -> JSON)
# ======================================================================


def convert_csv_to_json(csv_path: str, json_path: str) -> None:
    """Convert the legacy CSV registry to the canonical JSON format.

    Run this once after any CSV update; commit the resulting JSON file and
    point the pipeline at it going forward.

    Rich content (e.g. HTML tables) can be added manually to the JSON
    afterwards by setting ``suggested_text_format`` to ``"html"`` and
    placing an HTML string in ``suggested_text``.
    """
    variables = _load_template_registry_csv(csv_path)

    sections = []
    for v in variables:
        sections.append(
            {
                "section_id": v.section_id,
                "heading": v.heading,
                "sub_section": v.sub_section if v.sub_section != "N/A" else None,
                "required": v.required,
                "complexity": v.complexity,
                "is_in_protocol": v.is_in_protocol,
                "partially_in_protocol": v.partially_in_protocol,
                "is_standard_text": v.is_standard_text,
                "instructions": v.instructions,
                # Normalize legacy "N/A - see suggested text" / "N/A" → ""
                "required_text": (
                    ""
                    if v.required_text.strip() in ("N/A - see suggested text", "N/A")
                    else v.required_text
                ),
                "suggested_text": v.suggested_text,
                # "text" for now; change to "html" and insert <table> markup manually
                # when rich table content is needed for a section.
                "suggested_text_format": "text",
            }
        )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"sections": sections}, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(sections)} sections -> {json_path}")


# ======================================================================
# Public entry point  (auto-detects CSV vs JSON)
# ======================================================================


def load_template_registry(path: str) -> list[TemplateVariable]:
    """Load the ICF template registry from a CSV or JSON file.

    File type is inferred from the extension:
      .json  -> load_template_registry_json (fast, no re-parsing)
      .csv   -> _load_template_registry_csv  (legacy, parses every run)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return load_template_registry_json(path)
    return _load_template_registry_csv(path)

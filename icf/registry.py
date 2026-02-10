"""
Parse the ICF template breakdown CSV into a structured variable registry.

Handles:
  - Complexity classification (easy/moderate/complex, in/not-in protocol)
  - HTML entity decoding (CSV may contain &lt; &gt; etc.)
  - Required vs optional detection
"""

import ast
import csv
import html
import os

from icf.types import TemplateVariable


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
    # Default: treat "Required" keyword presence as True
    return "required" in lower and "optional" not in lower


def load_template_registry(csv_path: str) -> list[TemplateVariable]:
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

    with open(csv_path, encoding="utf-8") as f:
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
            protocol_mapping = html.unescape(row[11].strip()) if len(row) > 11 else ""
            sponsor_mapping = html.unescape(row[12].strip()) if len(row) > 12 else ""
            notes = html.unescape(row[14].strip()) if len(row) > 14 else ""

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
                    complexity=complexity,
                    protocol_mapping=protocol_mapping,
                    sponsor_mapping=sponsor_mapping,
                    is_in_protocol=is_in,
                    partially_in_protocol=partial,
                    is_standard_text=standard,
                    notes=notes,
                    csv_status=csv_status,
                )
            )

    if not variables:
        raise ValueError(f"No variables loaded from CSV: {csv_path}")

    return variables

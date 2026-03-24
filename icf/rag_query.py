"""
Multi-query expansion for the RAG extraction backend.

For each ICF template section, generates several diverse search queries from
the section's structured metadata.  Running multiple queries before merging
results via RRF dramatically improves recall compared to a single query,
especially for protocol sections where the relevant text may use different
vocabulary or phrasing than the ICF template.

Why rule-based (not LLM-based)?
--------------------------------
We deliberately avoid an LLM call for query expansion.  The reasons are:
  1. Each ICF section already has rich metadata: heading, sub_section,
     instructions, required_text, and suggested_text.  Extracting good
     queries from these is a structured text-manipulation task, not a
     reasoning task — a cheap regex pass is more deterministic and faster.
  2. HyDE (Hypothetical Document Embeddings) was explicitly rejected because
     a hallucinated "typical" answer embeds toward standard-of-care passages,
     not toward the novel dosing/procedure described in the specific protocol.
  3. LLM query expansion adds latency and cost with negligible uplift over
     well-structured rule-based expansion for this domain.

Query generation strategy (returns up to num_queries queries):
  Q1  heading → sentence-case natural language question
  Q2  sub_section heading (if present) with parent heading context
  Q3  first substantive sentence of the instructions field (cleaned)
  Q4  clinical/numeric terms extracted from instructions + required_text
      (drug names, dose values, participant counts, timeframes, etc.)
  Q5+ keyword overlap between suggested_text and Q1 (for complex sections)
"""

from __future__ import annotations

import re

from icf.types import TemplateVariable

# Patterns for extracting clinical/numeric terms from template text.
# These are the kinds of exact terms that BM25 excels at retrieving.
_NUMERIC_UNIT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mg|mL|kg|g|µg|IU|%|participants?|patients?|subjects?|"
    r"sites?|weeks?|months?|years?|days?|hours?|cycles?|doses?|visits?)",
    re.IGNORECASE,
)
_CAPITALIZED_TERM_RE = re.compile(
    r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3}\b"
)
# Common instruction verbs to strip from query text (they add noise for retrieval).
_INSTRUCTION_VERBS = re.compile(
    r"^(?:describe|insert|explain|include|specify|list|indicate|note|provide|"
    r"state|outline|mention|identify|summarize|detail|confirm)\s+",
    re.IGNORECASE,
)


def expand_queries(variable: TemplateVariable, num_queries: int = 4) -> list[str]:
    """Generate up to num_queries diverse search queries for a template variable.

    Args:
        variable:    The TemplateVariable whose section we want to retrieve.
        num_queries: Maximum number of queries to return.

    Returns:
        A deduplicated list of search query strings (length ≤ num_queries).
    """
    candidates: list[str] = []

    # ------------------------------------------------------------------
    # Q1: Heading as a natural language question
    # ------------------------------------------------------------------
    heading_q = variable.heading.strip("?").strip().lower()
    if heading_q:
        candidates.append(heading_q)

    # ------------------------------------------------------------------
    # Q2: Sub-section + heading (provides finer-grained context)
    # ------------------------------------------------------------------
    if variable.sub_section:
        sub_q = variable.sub_section.strip().lower()
        # Avoid redundancy if sub_section IS the heading
        if sub_q not in heading_q:
            candidates.append(f"{sub_q} {heading_q}")

    # ------------------------------------------------------------------
    # Q3: Cleaned first substantive sentence from instructions
    # ------------------------------------------------------------------
    instructions_q = _extract_instructions_query(variable.instructions)
    if instructions_q:
        candidates.append(instructions_q)

    # ------------------------------------------------------------------
    # Q4: Clinical/numeric term extraction
    # ------------------------------------------------------------------
    term_q = _extract_term_query(variable.instructions, variable.required_text)
    if term_q:
        candidates.append(term_q)

    # ------------------------------------------------------------------
    # Q5: Keyword overlap from suggested_text (for complex sections)
    # ------------------------------------------------------------------
    if len(candidates) < num_queries and variable.suggested_text:
        suggested_q = _extract_term_query(variable.suggested_text, "")
        if suggested_q and suggested_q not in candidates:
            candidates.append(suggested_q)

    # ------------------------------------------------------------------
    # Deduplicate and normalise
    # ------------------------------------------------------------------
    seen: set[str] = set()
    result: list[str] = []
    for q in candidates:
        q = q.strip()
        if len(q) < 6:
            continue
        q_norm = re.sub(r"\s+", " ", q.lower())
        if q_norm not in seen:
            seen.add(q_norm)
            result.append(q)
        if len(result) >= num_queries:
            break

    # Ensure at least one query (fallback to heading)
    if not result:
        result = [variable.heading.lower()]

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_instructions_query(instructions: str) -> str:
    """Extract the first substantive sentence from the instructions text."""
    if not instructions:
        return ""
    # Take the first sentence (split on period, newline, or capital-letter break)
    first = re.split(r"[.\n]|\n\n", instructions)[0].strip()
    if len(first) < 15:
        # Too short — try the first 150 chars
        first = instructions[:150].split("\n")[0].strip()

    # Remove leading instruction verbs ("Describe the...", "Insert the...")
    first = _INSTRUCTION_VERBS.sub("", first).strip()

    # Remove angle-bracket conditions and template markers
    first = re.sub(r"<<[^>]*>>|<[^>]*>|\{\{[^}]*\}\}", "", first).strip()

    return first[:200].lower() if len(first) > 10 else ""


def _extract_term_query(text_a: str, text_b: str) -> str:
    """Extract clinical and numeric terms from combined text for exact-match retrieval."""
    combined = f"{text_a} {text_b}"

    terms: list[str] = []

    # Numeric + unit patterns (dose values, participant counts, timeframes)
    terms.extend(_NUMERIC_UNIT_RE.findall(combined))

    # Capitalized multi-word phrases (drug names, procedure names, etc.)
    cap_terms = _CAPITALIZED_TERM_RE.findall(combined)
    # Filter out common ICF/template words that add no retrieval value
    stop_words = {
        "The", "This", "That", "If", "For", "When", "And", "Or", "But",
        "Note", "Example", "Please", "Include", "Standard", "Study",
        "Participants", "Protocol", "Section", "ICF",
    }
    terms.extend(t for t in cap_terms if t.split()[0] not in stop_words)

    # Deduplicate preserving order, limit to 12 terms
    seen: set[str] = set()
    unique: list[str] = []
    for t in terms:
        t_low = t.lower()
        if t_low not in seen:
            seen.add(t_low)
            unique.append(t)
        if len(unique) >= 12:
            break

    return " ".join(unique).lower() if unique else ""

"""
Evaluation rubrics for AI-generated ICF quality assessment.

Encodes 9 per-section evaluation dimensions + 1 document-level rubric
from the UHN AI-Generated ICF Evaluation Outline (v3, March 2026) as
structured rubric definitions consumed by DeepEval GEval metrics or
the combined single-call evaluator.

Each rubric has:
  - name:        Short identifier
  - description: What this dimension measures
  - criteria:    The full Excellent/Good/Borderline/Poor/Fail rubric text
  - params:      Which LLMTestCaseParams the GEval metric needs
  - category:    "task_performance" | "effectiveness" | "accessibility" |
                 "ground_truth" | "document_level"
  - deterministic: True if this should be computed with code, not an LLM judge

Category groupings for aggregated reporting:
  task_performance  : Fidelity to Protocol, Honesty, Over-inclusion
  effectiveness     : Misleading Language, Risks/Benefits/Voluntariness, Tone
  accessibility     : Language Quality, Inclusive Language
  ground_truth      : Ground Truth Correctness (separate — only where GT available)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class RubricDefinition:
    name: str
    description: str
    criteria: str
    params: list[str]  # LLMTestCaseParams field names
    category: str  # "task_performance" | "effectiveness" | "ground_truth"
    deterministic: bool = False
    applicable_sections: list[str] | None = None  # None = all sections
    min_text_words: int = 0  # skip if actual_output has fewer words


# ======================================================================
# Ground truth comparison (scored against reference ICF)
# ======================================================================

GROUND_TRUTH_CORRECTNESS = RubricDefinition(
    name="Ground Truth Correctness",
    description=(
        "Measures how closely the AI-generated ICF section matches the "
        "approved human-written ground truth ICF. Evaluates factual overlap, "
        "completeness, and accuracy of the extracted content."
    ),
    criteria=(
        "Compare the actual output (AI-generated ICF section) against the "
        "expected output (approved human-written ground truth ICF section). "
        "Score based on the following rubric:\n\n"
        "Excellent - The AI output captures all key facts, details, and nuances "
        "from the ground truth. No meaningful omissions or additions. The content "
        "is functionally equivalent to the ground truth.\n\n"
        "Good - The AI output captures most key facts from the ground truth. "
        "Minor details may differ in phrasing but the core information is preserved. "
        "1-2 small omissions or additions that do not change the meaning.\n\n"
        "Borderline - The AI output captures the main idea but misses several "
        "important details present in the ground truth, or includes notable "
        "inaccuracies. A reviewer would need to make moderate corrections.\n\n"
        "Poor - The AI output only partially overlaps with the ground truth. "
        "Major facts are missing or incorrect. Significant revision needed.\n\n"
        "Fail - The AI output bears little resemblance to the ground truth. "
        "Most content is missing, wrong, or fabricated."
    ),
    params=["actual_output", "expected_output"],
    category="ground_truth",
)

# ======================================================================
# Task Performance rubrics (Table 6 from evaluation outline)
# ======================================================================

FIDELITY_TO_PROTOCOL = RubricDefinition(
    name="Fidelity to Protocol",
    description=(
        "All required protocol details are captured in the correct consent "
        "form fields. Missing/excluded details from the protocol reduce the score."
    ),
    criteria=(
        "Evaluate whether the AI-generated ICF section faithfully captures "
        "all relevant details from the source protocol provided as context. "
        "If a reference (expected_output) from the REB-approved ground truth ICF "
        "is provided, use it to verify which protocol details should appear in "
        "this section and whether they are correctly mapped.\n\n"
        "Excellent - All protocol details present and correctly mapped to ICF "
        "template fields.\n\n"
        "Good - Minor details missing or slightly misplaced; most information "
        "captured.\n\n"
        "Borderline - Several details missing or misplaced; partial coverage.\n\n"
        "Poor - Most details missing or in wrong fields; output barely reflects "
        "source.\n\n"
        "Fail - Almost all source information missing; output fails to capture "
        "source."
    ),
    params=["actual_output", "expected_output", "context"],
    category="task_performance",
)

HONESTY = RubricDefinition(
    name="Honesty",
    description=(
        "Details that are not available in the source document are correctly "
        "acknowledged as not available. Fabricated, contradictory, or "
        "misrepresented information is penalized."
    ),
    criteria=(
        "Evaluate whether the AI-generated ICF section honestly and accurately "
        "represents what is and is not available in the source protocol. "
        "Check for ALL of the following:\n"
        "- Fabrication/hallucination: Content invented that is not in the evidence\n"
        "- Contradiction: AI text says something that conflicts with or opposes "
        "what the evidence quotes actually state\n"
        "- Misrepresentation: Subtle rewording that changes the meaning — e.g. "
        "generalizing a specific eligibility criterion, softening a risk, or "
        "changing quantities/frequencies\n"
        "- Unclarity: AI text is vague or ambiguous about things the evidence "
        "is specific about (e.g. evidence says 'weekly for 6 months' but AI "
        "says 'regular visits')\n"
        "- Missing acknowledgement: Information not available in the protocol is "
        "not flagged or acknowledged\n\n"
        "If a reference (expected_output) from the REB-approved ground truth "
        "ICF is provided, use it to identify what the approved version included "
        "vs omitted — any content in the AI output that is neither in the "
        "evidence nor in the approved ICF is likely hallucinated.\n\n"
        "Excellent - No fabrication, contradiction, or misrepresentation. "
        "Missing source information explicitly acknowledged. All claims "
        "traceable to evidence.\n\n"
        "Good - Minor issues only. Perhaps one instance of slight vagueness "
        "or a small detail acknowledged inconsistently. No contradictions.\n\n"
        "Borderline - Some issues present: missing information not "
        "acknowledged, minor contradictions with evidence, or subtle "
        "misrepresentations of specific details.\n\n"
        "Poor - Multiple issues: fabricated details, contradictions with "
        "evidence, or significant misrepresentations. Missing details "
        "not indicated.\n\n"
        "Fail - Pervasive dishonesty. Information fabricated or directly "
        "contradicts evidence. No indication of absent protocol information. "
        "Reader would form a false understanding."
    ),
    params=["actual_output", "expected_output", "context"],
    category="task_performance",
)

OVER_INCLUSION = RubricDefinition(
    name="Over-inclusion",
    description=(
        "The appropriateness and relevance of source information pulled into "
        "each field (ensuring there isn't too much information that adds "
        "little relevance or value)."
    ),
    criteria=(
        "Evaluate whether the AI-generated ICF section includes only the "
        "necessary and relevant information for this specific ICF field, "
        "without dumping excessive source material. If a reference "
        "(expected_output) from the REB-approved ground truth ICF is provided, "
        "use it as a benchmark for the appropriate level of detail — content "
        "significantly exceeding what the approved version includes is "
        "over-inclusion.\n\n"
        "Excellent - Output includes only necessary information; concise, "
        "relevant, and well-mapped.\n\n"
        "Good - Minor over-inclusion; mostly relevant but a few extra "
        "details present.\n\n"
        "Borderline - Moderate over-inclusion; several details included "
        "that aren't strictly needed for this field.\n\n"
        "Poor - Large amounts of unnecessary source details included.\n\n"
        "Fail - Almost all content from source dumped into the field; very "
        "verbose, hard to parse."
    ),
    params=["actual_output", "expected_output", "context"],
    category="task_performance",
)

INCLUSIVE_LANGUAGE = RubricDefinition(
    name="Inclusive Language and Cultural Appropriateness",
    description=(
        "Evaluate whether the text uses respectful, neutral, "
        "non-discriminatory language that reflects diversity, avoids "
        "stereotypes or assumptions, and is appropriate for audiences of "
        "varied identities, cultures, and backgrounds."
    ),
    criteria=(
        "Evaluate the AI-generated ICF section for inclusive language and "
        "cultural appropriateness, based on Government of Canada EDI "
        "terminology and UHN Inclusive Writing Guidelines.\n\n"
        "Excellent - Fully compliant. No violations; text is fully inclusive, "
        "culturally respectful, and consistent with the guideline rule.\n\n"
        "Good - Minor issue present. Text mostly complies; wording could be "
        "slightly more inclusive or neutral but is not harmful. Minor "
        "outdated but still acceptable terms.\n\n"
        "Borderline - Moderate issues. Some violations that could cause "
        "mild exclusion or stereotyping. Examples include identity "
        "assumptions, biased phrasing, etc.\n\n"
        "Poor - Major issues. Multiple violations; text includes "
        "exclusionary, biased, or stigmatizing language that undermines "
        "compliance.\n\n"
        "Fail - Severe violation. Text is clearly non-compliant, with "
        "repeated or harmful exclusionary, stigmatizing, or culturally "
        "insensitive language."
    ),
    params=["actual_output"],
    category="accessibility",
)

READING_LEVEL = RubricDefinition(
    name="Reading Level (Flesch-Kincaid)",
    description=(
        "Measures the reading level required to understand the text using "
        "the Flesch-Kincaid Grade Level formula. Consent forms should be "
        "written at a grade 6-8 reading level."
    ),
    criteria=(
        "Deterministic metric. Calculated programmatically using the "
        "Flesch-Kincaid formula, then mapped to the rubric:\n\n"
        "Excellent (exceeds readability requirement) - equal to grade 6 level\n\n"
        "Good (meets readability requirement) - grade 7-8 level\n\n"
        "Borderline (slightly above recommended level) - grade 9-10 level\n\n"
        "Poor (significantly above recommended level) - grade 11-12 level\n\n"
        "Fail (not appropriate for lay audience) - greater than grade 12 level"
    ),
    params=["actual_output"],
    category="task_performance",
    deterministic=True,
    min_text_words=20,
)

LANGUAGE_QUALITY = RubricDefinition(
    name="Language Quality",
    description=(
        "Combined assessment of reading level, plain language adherence, "
        "and overall comprehensibility for a lay audience. Covers word choice, "
        "sentence clarity, jargon handling, and content discipline."
    ),
    criteria=(
        "Evaluate the AI-generated ICF section for language quality and "
        "accessibility. This rubric covers reading level (grade 6-8 target), "
        "plain language principles, and overall comprehensibility.\n\n"
        "Consider ALL of the following:\n"
        "- Reading level: Would a person without medical or scientific training "
        "understand this? Target is grade 6-8.\n"
        "- Medical/technical terms: Are they explained when first used?\n"
        "- Sentence structure: Are sentences short, simple, and clearly written?\n"
        "- Jargon: Is it avoided or properly defined?\n"
        "- Word choice: Does it use common everyday language?\n"
        "- Content discipline: Is only necessary information included?\n"
        "- Logical flow and cohesion: Does the text read naturally?\n\n"
        "If a reference (expected_output) from the REB-approved ground truth "
        "ICF is provided, use it as the benchmark — the approved ICF was "
        "written by experts for patient comprehension.\n\n"
        "Excellent - Text is fully accessible at grade 6-8 level. All medical "
        "terms explained. Short, clear sentences. Complies with plain language "
        "guidelines. Logical flow.\n\n"
        "Good - Mostly accessible. Minor instances of unexplained terms or "
        "slightly complex phrasing. Overall understandable with minor plain "
        "language deviations.\n\n"
        "Borderline - Noticeable readability issues. Several medical terms "
        "unexplained, some complex sentence structures. A lay reader would "
        "struggle with parts. Several plain language deviations.\n\n"
        "Poor - Largely inaccessible. Frequent jargon, complex sentences, "
        "unexplained technical concepts. Text is unclear or overly complex "
        "in many places.\n\n"
        "Fail - Not appropriate for a lay audience. Dense technical language "
        "throughout. Does not adhere to plain language guidelines."
    ),
    params=["actual_output", "expected_output"],
    category="accessibility",
    min_text_words=20,
)

# ======================================================================
# Effectiveness rubrics (Table 7 from evaluation outline)
# ======================================================================

MISLEADING_LANGUAGE = RubricDefinition(
    name="Misleading Language",
    description=(
        "Measures whether the AI-assisted ICF avoids wording that could "
        "mislead participants, create false understanding or foster false "
        "expectations. Must avoid therapeutic misconception."
    ),
    criteria=(
        "Evaluate whether the AI-generated ICF section avoids misleading "
        "language, therapeutic misconception, and false expectations. "
        "Aligned with TCPS 2, Chapter 3, Article 3.2. If a reference "
        "(expected_output) from the REB-approved ground truth ICF is "
        "provided, use it to identify how the approved version framed "
        "the same information — deviations that introduce misleading "
        "implications should be penalized.\n\n"
        "Excellent - No misleading or confusing terms. Uses correct "
        "research terminology (e.g. calls the drug an 'investigational "
        "study drug,' not a treatment). Does not imply guaranteed outcomes; "
        "explicitly states that benefits are uncertain. Distinguishes "
        "research from standard care throughout.\n\n"
        "Good - Generally clear, with only minor lapses. 1-2 minor phrases "
        "that could be clarified but unlikely to cause misunderstanding. "
        "Overall context makes it clear the drug is experimental and that "
        "outcomes are uncertain.\n\n"
        "Borderline - Somewhat unclear; contains several phrases that "
        "could cause confusion or misunderstanding.\n\n"
        "Poor - Multiple misleading phrases. The form contains language "
        "that could create therapeutic misconception or false expectations.\n\n"
        "Fail - Severely misleading. The text routinely implies guaranteed "
        "benefit, conflates research with treatment, or uses language "
        "that would create false understanding."
    ),
    params=["actual_output", "expected_output"],
    category="effectiveness",
)

RISKS_BENEFITS_VOLUNTARINESS = RubricDefinition(
    name="Risks, Benefits, Voluntariness and Alternatives",
    description=(
        "Evaluates whether the consent form provides clear, relevant, and "
        "sufficient information for a participant to make an informed "
        "decision about joining the study."
    ),
    criteria=(
        "Evaluate whether the AI-generated ICF section clearly explains "
        "risks, benefits, voluntariness and alternatives. Aligned with "
        "TCPS 2, Chapter 3, Article 3.2. If a reference (expected_output) "
        "from the REB-approved ground truth ICF is provided, use it to "
        "verify that the AI output covers the same risks, benefits, "
        "voluntariness statements, and alternatives that the approved "
        "version included.\n\n"
        "Excellent - Risks, benefits, voluntariness and alternatives are "
        "explained clearly with no missing or obscured information, "
        "enabling an informed decision. Risks (and their likelihood/"
        "severity) are described in understandable terms; benefits are "
        "stated realistically (including no guarantees, distinguishing "
        "benefit to participant vs future patients); participants are "
        "explicitly told participation is voluntary with the right to "
        "withdraw at any time without penalty; and alternative options "
        "are explained.\n\n"
        "Good - Minor clarifications needed; understanding mostly clear. "
        "A typical participant would understand the primary risks, "
        "realistic benefits, and voluntary nature.\n\n"
        "Borderline - Some elements unclear or incomplete. A participant "
        "might miss important aspects of risks, benefits, or "
        "voluntariness.\n\n"
        "Poor - Major gaps. Important risks, benefits, or voluntariness "
        "information is missing or obscured.\n\n"
        "Fail - Risks, benefits, voluntariness or alternatives are "
        "severely misrepresented or absent."
    ),
    params=["actual_output", "expected_output", "context"],
    category="effectiveness",
    # 17 = early termination (participant rights re: investigator-initiated withdrawal)
    applicable_sections=["7", "16", "17", "18", "18.1", "18.2", "19", "20"],
)

TONE = RubricDefinition(
    name="Tone",
    description=(
        "The extent to which the consent language maintains a respectful, "
        "neutral, non-coercive, non-persuasive, participant-centred tone "
        "that supports voluntary decision-making."
    ),
    criteria=(
        "Evaluate the tone of the AI-generated ICF section. Aligned with "
        "TCPS 2, Chapter 3. If a reference (expected_output) from the "
        "REB-approved ground truth ICF is provided, use it as the tone "
        "benchmark — the approved version was reviewed by ethics experts "
        "for appropriate tone.\n\n"
        "Excellent - Tone is fully neutral, respectful and participant-"
        "centred. No hint of pressure, persuasion, coercion, or bias.\n\n"
        "Good - Generally neutral and respectful, some minor lapses. "
        "There may be one subtle instance of encouraging language or "
        "formality, but no explicit pressure. The overall wording still "
        "respects voluntary choice and remains neutral.\n\n"
        "Borderline - Mixed tone. Some phrases may unintentionally "
        "pressure or bias, such as phrases that subtly push or presume "
        "participation.\n\n"
        "Poor - Frequent persuasive, directive or coercive tone. The "
        "form contains multiple instances of pressure or biased language "
        "that undermine a neutral, participant-centered approach.\n\n"
        "Fail - Tone is inappropriate and in violation of ethical "
        "standards. Clearly coercive, manipulative, or dismissive."
    ),
    params=["actual_output", "expected_output"],
    category="effectiveness",
)

# ======================================================================
# Document-level rubric (runs once on full concatenated output, not per-section)
# ======================================================================

DOCUMENT_QUALITY = RubricDefinition(
    name="Document Quality",
    description=(
        "Whole-document pass checking cross-section consistency, "
        "abbreviation/term redundancy, repetition, and coherence."
    ),
    criteria=(
        "Evaluate the FULL AI-generated ICF document (all sections concatenated) "
        "for document-level quality issues that cannot be caught per-section.\n\n"
        "Check ALL of the following:\n"
        "- Abbreviation redundancy: Are abbreviations/acronyms defined multiple "
        "times across sections? (e.g. 'allogeneic hematopoietic cell transplant "
        "(alloHCT)' explained in 5 places). First use should define it, "
        "subsequent uses should use the abbreviation only.\n"
        "- Repetition: Are the same sentences, paragraphs, or explanations "
        "repeated across sections unnecessarily?\n"
        "- Terminology consistency: Does the document use the same term for "
        "the same concept throughout? (e.g. not switching between 'stem cell "
        "transplant' and 'hematopoietic cell transplant' without linking them)\n"
        "- Cross-section coherence: Does the document flow logically? Are there "
        "contradictions between sections? Does information in later sections "
        "build on earlier sections correctly?\n"
        "- Information placement: Is information in the right sections or is "
        "it scattered inappropriately?\n\n"
        "Excellent - No redundant definitions. No unnecessary repetition. "
        "Consistent terminology. Logical cross-section flow. Information "
        "in appropriate sections.\n\n"
        "Good - Minor issues: 1-2 abbreviations redefined or slight "
        "terminology inconsistency. Overall coherent.\n\n"
        "Borderline - Several issues: multiple abbreviations redefined, "
        "some repetition across sections, or noticeable terminology "
        "inconsistency.\n\n"
        "Poor - Significant issues: frequent redundancy, repetition, "
        "or contradictions between sections.\n\n"
        "Fail - Document reads as disconnected sections. Pervasive "
        "redundancy, contradictions, and inconsistent terminology."
    ),
    params=["actual_output"],
    category="document_level",
    min_text_words=50,
)

# ======================================================================
# All rubrics in evaluation order
# ======================================================================

ALL_RUBRICS: list[RubricDefinition] = [
    # Ground truth alignment (separate category — only scored where GT available)
    GROUND_TRUTH_CORRECTNESS,
    # Task performance: extraction quality
    FIDELITY_TO_PROTOCOL,
    HONESTY,
    OVER_INCLUSION,
    # Effectiveness: patient-facing communication quality
    MISLEADING_LANGUAGE,
    RISKS_BENEFITS_VOLUNTARINESS,
    TONE,
    # Accessibility: readability and inclusion
    LANGUAGE_QUALITY,
    INCLUSIVE_LANGUAGE,
]

# READING_LEVEL is kept as a standalone definition for optional use but excluded
# from ALL_RUBRICS. Flesch-Kincaid systematically penalises ICF documents for
# using unavoidable medical vocabulary (disease names, drug names, procedures),
# making it an unfair primary metric. Language Quality (LLM-judged) is the
# primary readability metric.

# Document-level rubric (not in ALL_RUBRICS — runs separately at end)
DOCUMENT_LEVEL_RUBRICS: list[RubricDefinition] = [DOCUMENT_QUALITY]

# Rubrics that need protocol context (passed as context in the test case)
CONTEXT_RUBRICS = {r.name for r in ALL_RUBRICS if "context" in r.params}

# Rubrics that accept ground truth (passed as expected_output)
GROUND_TRUTH_RUBRICS = {r.name for r in ALL_RUBRICS if "expected_output" in r.params}

# Rubrics that REQUIRE ground truth — skip entirely if no ground truth available
GROUND_TRUTH_REQUIRED = {GROUND_TRUTH_CORRECTNESS.name}

# Rubrics computed deterministically (no LLM judge)
DETERMINISTIC_RUBRICS = {r.name for r in ALL_RUBRICS if r.deterministic}

# Rubrics scoped to specific sections
SCOPED_RUBRICS = {r.name: r.applicable_sections for r in ALL_RUBRICS if r.applicable_sections}

# Grounding rubrics: these measure factual accuracy relative to the protocol.
# Only grounding rubrics receive HARD_PENALTY (0.15) when routing detects a
# hallucination risk. Content quality rubrics (tone, language, etc.) are still
# evaluated on the text itself even when grounding is unconfirmed.
GROUNDING_RUBRIC_NAMES: frozenset[str] = frozenset(
    {
        GROUND_TRUTH_CORRECTNESS.name,
        FIDELITY_TO_PROTOCOL.name,
        HONESTY.name,
    }
)


# ======================================================================
# Evaluation policy and routing
# ======================================================================


class ScoringMode(str, Enum):
    """Routing decision for how to score a section's rubric."""

    SKIP = "SKIP"  # Don't score — record N/A
    HARD_PENALTY = "HARD_PENALTY"  # Score in code, no judge call
    SOFT = "SOFT"  # Call judge with caution warning
    FULL = "FULL"  # Call judge normally


@dataclass
class EvalPolicy:
    """All configurable scoring policy decisions in one place.

    Modify fields here to adjust evaluation behaviour without touching
    eval_runner.py or eval_combined.py logic.
    """

    # Sections to skip entirely — no rubrics evaluated, not added to results.
    # Default: ICF header metadata fields (study title, PI name, site, etc.)
    # which contain only short administrative strings and add no signal.
    skip_sections: list[str] = field(
        default_factory=lambda: [
            "2.1",
            "2.2",
            "2.3",
            "2.4",
            "2.5",
            "2.6",
        ]
    )

    # Minimum words of section content required to run any evaluation.
    # Sections below this threshold are skipped (too short to assess meaningfully).
    min_section_words: int = 20

    # Sections where the Honesty rubric is applied.
    # None  = apply to all sections where routing permits (default — route_section
    #         already skips Honesty when a section is not in the protocol at all).
    # list  = restrict to those section IDs only.
    honesty_sections: list[str] | None = None

    # Minimum evidence relevance level to allow full scoring
    # "STRONG" = only strong relevance gets full score
    # "PARTIAL" = partial relevance is enough
    min_relevance_for_full_score: str = "PARTIAL"

    # Score assigned for HARD_PENALTY routing (hallucination risk cases)
    hard_penalty_score: float = 0.15

    # Whether to skip GT Correctness when output has placeholders only
    # and section is not expected in protocol
    skip_gt_if_placeholders_only: bool = True

    # Whether MEDIUM confidence triggers SOFT routing (vs FULL)
    medium_confidence_soft: bool = True


# Default policy — used when no custom policy is passed
DEFAULT_POLICY = EvalPolicy()


# Placeholder patterns used in filled_template
_PLACEHOLDER_PATTERN = re.compile(r"\{\{[^}]+\}\}|<<[^>]+>>|\[TO BE FILLED MANUALLY\]")


def has_placeholders(text: str) -> bool:
    """Return True if text contains unfilled user-insert markers."""
    return bool(_PLACEHOLDER_PATTERN.search(text))


def _is_concrete_content(text: str) -> bool:
    """Return True if text has substantive content beyond placeholders."""
    cleaned = _PLACEHOLDER_PATTERN.sub("", text).strip()
    return len(cleaned.split()) >= 5


def route_section(
    is_in_protocol: bool,
    partially_in_protocol: bool,
    is_standard_text: bool,
    status: str,
    confidence: str,
    has_evidence: bool,
    text: str,
    rubric_name: str,
    policy: EvalPolicy = DEFAULT_POLICY,
) -> tuple[ScoringMode, str]:
    """Determine how to score a section for a given rubric.

    Returns (ScoringMode, reason_string).

    Layer 1: Registry gate
    Layer 2: Status + Confidence routing
    """
    placeholders_only = has_placeholders(text) and not _is_concrete_content(text)
    concrete = _is_concrete_content(text)

    # --- Layer 1: Registry gate ---

    # Standard boilerplate text — skip all grounding rubrics
    if is_standard_text:
        return ScoringMode.SKIP, "Standard required text — no grounding evaluation needed"

    # Section not expected in protocol — skip Fidelity and Honesty
    # GT Correctness handled separately with its own abstention rules
    if not is_in_protocol and not partially_in_protocol:
        if rubric_name in ("Fidelity to Protocol", "Honesty"):
            return ScoringMode.SKIP, "Section not in protocol — grounding rubrics not applicable"

    # Partial protocol sections — HIGH confidence means backend found what's there
    if partially_in_protocol:
        if confidence == "HIGH":
            return ScoringMode.FULL, "Partially in protocol + HIGH confidence — full scoring"
        return (
            ScoringMode.SOFT,
            "Partially in protocol — soft scoring, judge cautioned about boundary",
        )

    # --- Layer 2: Status + Confidence routing ---

    # Technical error — skip content scoring
    if status == "ERROR":
        return ScoringMode.SKIP, "Extraction error — skip content scoring"

    # NOT_FOUND + HIGH/MEDIUM confidence:
    # Backend searched and confirmed info absent
    if status == "NOT_FOUND" and confidence in ("HIGH", "MEDIUM"):
        if concrete:
            # AI invented content despite confirmed absence — hard penalty
            return (
                ScoringMode.HARD_PENALTY,
                "NOT_FOUND + HIGH/MEDIUM confidence but AI generated concrete content — possible hallucination",
            )
        else:
            # Correct abstention
            return (
                ScoringMode.SKIP,
                "NOT_FOUND + HIGH/MEDIUM confidence — info confirmed absent, correct abstention",
            )

    # LOW confidence + no evidence + concrete claims — hallucination risk
    if confidence == "LOW" and not has_evidence and concrete and not placeholders_only:
        return (
            ScoringMode.HARD_PENALTY,
            "LOW confidence + no evidence + concrete claims — hallucination risk",
        )

    # MEDIUM confidence → soft mode (warn judge)
    if confidence == "MEDIUM" and policy.medium_confidence_soft:
        return ScoringMode.SOFT, "MEDIUM confidence — judge cautioned about extraction uncertainty"

    # Everything else → full scoring
    return ScoringMode.FULL, ""


def is_rubric_applicable(rubric: RubricDefinition, section_id: str, text: str) -> tuple[bool, str]:
    """Check if a rubric should run on a given section.

    Returns (applicable, reason). If not applicable, reason explains why.
    """
    if rubric.applicable_sections and section_id not in rubric.applicable_sections:
        return False, f"Rubric scoped to sections {rubric.applicable_sections}"
    if rubric.min_text_words > 0 and len(text.split()) < rubric.min_text_words:
        return False, f"Text too short ({len(text.split())} words, need {rubric.min_text_words})"
    return True, ""

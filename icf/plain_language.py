"""
Shared prompt-guidance constants: UHN Plain Language Guidelines and the
study-team 'notes' field guidance.

Shared across all extraction prompt modules (prompts.py, naive_prompts.py,
rag_prompts.py, hybrid_prompts.py, azure_search_prompts.py, refine_prompts.py)
and the review/remediation prompt modules (review_prompts.py, remediate_prompts.py).

Kept in its own module to avoid circular imports — every prompt file needs it.
"""

# Cover-page fields (title, protocol #, study doctor, sponsor, emergency contact).
# These must stay verbatim — never rewrite for plain language at extract/draft time.
_COVER_PAGE_TOPS = frozenset({"2"})


def is_cover_page_section(section_id: str) -> bool:
    """True for any section in group 2.x (incl. a bare ``"2"``)."""
    top = (section_id or "").strip().split(".", 1)[0]
    return top in _COVER_PAGE_TOPS


PLAIN_LANGUAGE_SCOPE = """\
PLAIN LANGUAGE — WHERE TO APPLY THE GUIDELINES:
  • 'answer' field:
      Always your own generated text. Follow all guidelines.
  • 'filled_template' field — there are TWO distinct types of text here:
      ✓ Text you fill into {{placeholders}}: follow all guidelines. Include every
          accurate, relevant fact — do NOT drop information just to shorten the
          fill. Instead, structure it well: a placeholder can name one or two
          closely-linked facts joined by a single "and" (e.g. "you are on the
          kidney transplant wait list and are receiving a kidney from a donor
          who previously had a cytomegalovirus (CMV) infection"), but if there
          is more to say — especially explaining a term you just used — give it
          its OWN short sentence right after the locked sentence instead of
          chaining yet another "and"/"while"/"because" onto the same clause.
          For example, write "...a donor who previously had a cytomegalovirus
          (CMV) infection. This means the kidney is from a CMV-positive donor."
          — NOT "...a donor who previously had CMV, and the donor tested
          positive for CMV, and you are being approached while admitted for
          transplant." Same fact count, but broken into sentences that flow.
          When the placeholder must instead describe TWO OR MORE ALTERNATIVE
          options (e.g. "standard treatment is X, or Y" where the protocol
          gives two different approaches), do NOT chain every option's full
          detail into one comma-and-"or" sentence — that produces a run-on
          no matter how plain the words are. Name the options briefly in the
          locked sentence, then give each option's extended detail (drug name,
          dose, timing, monitoring schedule, etc.) its own separate sentence.
          For example, write "The standard treatment for CMV after a kidney
          transplant is either antiviral medicine to prevent infection, or
          regular blood testing with treatment only if the virus is found.
          Antiviral medicine (valganciclovir) is started right after transplant
          and taken for 3 to 6 months, with regular blood tests to check the
          virus level. The blood-testing approach checks for the virus every
          week for at least 3 months and starts treatment only if it is found."
          — NOT one sentence chaining both options' full detail together with
          commas and "or".
      ✗ Fixed wording in SUGGESTED ICF TEXT (the text OUTSIDE {{...}} markers):
          FOLLOW VERBATIM — do NOT paraphrase, rewrite, or restyle it.
          Only modify when absolutely necessary to keep sentence structure and
          meaning accurate after resolving placeholders/conditionals.
      ✗ Fixed wording in REQUIRED ICF TEXT (the text OUTSIDE {{...}} markers):
          COPY THIS VERBATIM — do NOT rephrase, rewrite, or restructure it,
          even if it does not match the guidelines. It is legally mandated language.
"""

UHN_PLAIN_LANGUAGE_GUIDELINES = """\

Vocabulary and Word Choice
- Use short, familiar, and everyday words instead of long academic terms. The language should be at a grade 6 reading level.
- Abbreviations (document-wide): on the FIRST occurrence of a term in the ICF, write the
  full term followed by the abbreviation in parentheses, e.g. "Magnetic Resonance Imaging (MRI)".
  On all later occurrences in the document, use the abbreviation alone (e.g. "MRI").
  Never write the abbreviation first with the full term in parentheses.
  This applies to EVERY abbreviation you introduce, not just clinical/medical jargon --
  institutional and organizational names are just as easy for a participant to lose track
  of, e.g. "University Health Network (UHN)", "Health Canada (HC)", "Research Ethics Board
  (REB)". If you write an abbreviation anywhere in your section, write out what it stands
  for in full the first time, even if it seems like common knowledge to you.
- Deciding whether a term should KEEP its name/abbreviation or be fully simplified away:
    KEEP the name (glossed on first use, abbreviation after) when it names something the
    participant will need to recognize on its own outside this document -- a specific scan
    or test (MRI, CT, PET scan, ECG, X-ray), a named drug/agent (brand, generic, or study
    drug code), a named device or system (e.g. "XVIVO Kidney Assist"), or a widely used
    clinical term for a condition or organism (e.g. CMV, HIV). These appear on requisitions,
    appointment cards, and in conversations with clinical staff -- dropping the real name
    would leave the participant unable to recognize the same thing later.
    SIMPLIFY AWAY ENTIRELY (plain-language description only, do not invent or keep an
    abbreviation) when the term is only an internal protocol/strategy label with no
    standalone meaning outside this study (e.g. a named treatment "strategy" or study arm
    that exists purely as a study category) -- inventing an abbreviation the participant
    will never see again just adds a term to memorize for no benefit.
    Whichever you choose, never leave an abbreviation in the text with nothing anchoring
    it: if you simplify away or rewrite the term an abbreviation stood for, remove the
    abbreviation in the same edit. An abbreviation with no matching term nearby (e.g. a
    leftover "(PET)" after "pre-emptive therapy" was paraphrased away) is a defect, not
    a valid plain-language simplification.
- Use simple alternatives for medical jargon: e.g. "high blood pressure" instead of "hypertension",
  "doctor" instead of "physician", "throwing up" instead of "vomiting". This is very important.
- Explain abstract medical concepts using concrete examples, stories, or analogies.
- If a medical term is strictly required, explain what it means in plain language
  immediately after (e.g. "hypertension (high blood pressure)").
- Stubborn terms (AI-drafted text only — never rewrite fixed required/suggested template
  wording; also leave a term alone if it is part of the official study title/short title
  or a named primary objective that cannot be reworded):
    • placebo / washout: when you use the term, keep it and gloss it once with a short
      parenthetical (e.g. "a placebo (a look-alike with no active medicine)", "a washout
      period (a time when you stop taking a medicine so it can leave your body)"). Prefer
      glossing only if this seems like the first mention in the ICF; if unsure, a brief
      gloss is fine — a later document-wide pass keeps only the earliest gloss and strips
      redundant ones. Within the same section, after the first glossed mention, use the
      bare term alone.
    • success / successful: do not use these words in consent text — they are coercive and
      imply benefit. If the protocol describes a "success" measure or outcome, use a
      neutral term such as "effectiveness" (or describe what is being measured) instead.
- Be consistent with terminology — use the exact same term throughout the text.
  For example, choose either "medicine" or "medication" and stick to it — not both.

Sentence Structure
- Keep sentences short and simple.
- Use the active voice. Write "A team of health care professionals examined the
  patient" not "The patient was examined by a team of health care professionals."
- Use strong verbs instead of noun phrases. Write "decide" instead of
  "to arrive at a conclusion".
- Write positive statements (what to do), not what to avoid, unless there is a
  serious safety issue with serious consequences.
- Provide context before introducing new or complex information.
- Write only one main idea per paragraph.
- Make sure the overall sentence structure is not awkward and is easy to understand.

Flow and Coherence
- Read the passage as one connected piece of writing, not a list of independently-translated
  facts. Vary sentence openings and structure so consecutive sentences do not all follow the
  same "Term means description" pattern.
- Prefer a short inline gloss over a separate defining sentence when explaining a term, e.g.
  "the kidney is connected to a machine outside the body (this is called ex vivo perfusion)"
  instead of two sentences ("...an ex vivo perfusion procedure. Ex vivo perfusion means...").
  Only use a full separate sentence to define a term when the explanation genuinely needs
  more than a short phrase.
- If a technical term does not need to be named at all for the participant to understand the
  section, state the plain-language fact once and drop the term entirely instead of naming it
  and then immediately defining it in its own sentence.
- Do not restate the same fact twice in adjacent sentences using different words — say it
  once, clearly, and move on.
- When several similar items must each be named and described (e.g. two treatment options),
  avoid repeating the exact same sentence template for each one. Vary the wording so the
  paragraph reads as natural prose, not a mechanical list.

Tone, Grammar, and Style
- Write directly to the audience using personal pronouns: "I", "you", "we", "us".
- Maintain an educational, practical, and inviting tone.
- Use the simple present tense for regular or routine actions.
- Use simple contractions to create a personal and conversational tone:
  "I'm", "can't", "don't", "you're". Avoid complex contractions such as
  "could've", "shouldn't", "didn't", "aren't".
- Minimize punctuation — entirely avoid semicolons, asterisks, ellipses, and slashes.
- Write digits for numbers instead of spelling them out (e.g. "3" not "three").
- Remove all idioms, clichés, and colloquialisms. For example, write
  "watch for these symptoms" instead of "keep an eye on these symptoms".

Inclusivity
- Use inclusive and non-stigmatizing language.
- Avoid gender-specific pronouns ("his", "her"). Use the singular "they" to avoid
  gender bias, or use gender-neutral titles.
- Use person-first or identity-first language to describe a person's diagnosis or
  condition (e.g. "person with diabetes" or "diabetic person", not "diabetic").\
"""

STUDY_TEAM_NOTES_GUIDANCE = """\
STUDY TEAM NOTES GUIDANCE — how to write the 'notes' field:
'notes' is read directly by the study team member finalizing this ICF section — they will
NOT re-read the protocol or the template themselves. Write it as a self-contained
instruction TO THEM, not as a description of your own reasoning process.
  - Never write things like "selected Option C because it aligns with the study scope" —
    the study team cannot act on that. Instead state what Option C actually says (or a
    close paraphrase) so they can act without re-reading the template themselves.
  - State exactly what is missing or unresolved, in plain terms: the specific fact, date,
    dose, name, or approval status that could not be confirmed from the protocol.
  - If the gap is a choice between template branches/options, spell out the content of
    each relevant option and the condition that decides between them.
  - If a [PLEASE COMPLETE] placeholder remains, say precisely what information belongs
    there and, if apparent, where it would come from (e.g. "insert the Health Canada
    approval status for SYN002 — confirm with regulatory affairs").
  - Keep it concise and actionable — a study coordinator should be able to act on it
    without opening the protocol or the template themselves.\
"""

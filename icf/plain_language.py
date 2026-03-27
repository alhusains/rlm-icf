"""
UHN Plain Language Guidelines constant.

Shared across all extraction prompt modules (prompts.py, naive_prompts.py,
rag_prompts.py) and the review prompt module (review_prompts.py).

Kept in its own module to avoid circular imports — every prompt file needs it,
and the review module also imports it.
"""

PLAIN_LANGUAGE_SCOPE = """\
PLAIN LANGUAGE — WHERE TO APPLY THE GUIDELINES:
  • 'answer' field:
      Always your own generated text. Follow all guidelines.
  • 'filled_template' field — there are TWO distinct types of text here:
      ✓ Text you fill into {{placeholders}}: follow all guidelines.
      ✓ Text you write based on SUGGESTED ICF TEXT: follow all guidelines.
      ✗ Fixed wording in REQUIRED ICF TEXT (the text OUTSIDE {{...}} markers):
          COPY THIS VERBATIM — do NOT rephrase, rewrite, or restructure it,
          even if it does not match the guidelines. It is legally mandated language.
"""

UHN_PLAIN_LANGUAGE_GUIDELINES = """\

Vocabulary and Word Choice
- Use short, familiar, and everyday words instead of long academic terms.
- Use simple alternatives: e.g. "high blood pressure" instead of "hypertension",
  "doctor" instead of "physician", "throwing up" instead of "vomiting".
- Explain abstract medical concepts using concrete examples, stories, or analogies.
- If a medical term is strictly required, explain what it means in plain language
  immediately after (e.g. "hypertension (high blood pressure)").
- Be consistent with terminology — use the exact same term throughout the text.
  For example, choose either "medicine" or "medication" and stick to it — not both.

Sentence Structure
- Keep sentences short and simple, ideally between 10 and 14 words.
- Use the active voice. Write "A team of health care professionals examined the
  patient" not "The patient was examined by a team of health care professionals."
- Use strong verbs instead of noun phrases. Write "decide" instead of
  "to arrive at a conclusion".
- Write positive statements (what to do), not what to avoid, unless there is a
  serious safety issue with serious consequences.
- Provide context before introducing new or complex information.
- Write only one main idea per paragraph.

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

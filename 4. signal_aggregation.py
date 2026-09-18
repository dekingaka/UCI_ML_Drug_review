"""
Combine Symptom Signals with Rating/Text Mismatch Signals

Goal: Steps 2 and 3 each found something interesting on their own:
  - Step 2 found WHICH symptoms patients mention most often.
  - Step 3 found WHICH reviews have a rating that doesn't match the text.
 
This step asks a sharper question by combining them:
  Do certain symptoms tend to show up MORE in the reviews with the
  biggest rating/text mismatches?
 
If a symptom is mentioned just as often in "honest" reviews (where
rating and text agree) as in "mismatched" ones, it's probably just a
common side effect people are open about. But if a symptom shows up
disproportionately in the MISMATCHED reviews, that's a sign patients
might be downplaying it in their star rating - exactly the kind of
hidden signal a ratings-only analysis would miss entirely.
"""
 
import pandas as pd
 
pd.set_option("display.width", 120)

# Load Step 3's output (every review, with mismatch scores)
uci_birth_control_mismatch_scores = pd.read_csv("uci_birth_control_mismatch_scores.csv")
print(f"Loaded {len(uci_birth_control_mismatch_scores)} scored reviews.")

# Define the symptom terms to check for
"""
  These are the top terms Step 3's NER extraction found across our
  three focus drugs. Rather than re-running the full NER model here
  (which would be slow to repeat), we do a simpler, faster check:
  does the review's text CONTAIN this word at all? This is a
  reasonable shortcut now that Step 3 has already told us which
  words are worth looking for
"""
symptom_terms = [
    "bleeding", "acne", "weight gain", "depression", "anxiety",
    "headaches", "cramps", "nausea", "pain", "depressed",
]


# Flag which reviews mention each symptom
# ---------------------------------------------------------------
# For each symptom, we create a new True/False column: does this
# review's text contain that word? case=False means the check
# ignores capitalisation (so "Bleeding" and "bleeding" both count).
for term in symptom_terms:
    column_name = f"mentions_{term.replace(' ', '_')}"
    uci_birth_control_mismatch_scores[column_name] = uci_birth_control_mismatch_scores["review"].str.contains(term, case=False, na=False)
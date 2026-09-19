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




# For each symptom, compare mismatch scores with vs without it
"""
  This is the actual aggregation. For every symptom, we split the reviews into two groups - those that mention it, and those that
  don't - and compare their AVERAGE mismatch score. Remember: a mismatch score far from 0 means the rating and the
  text disagreed. So if reviews mentioning "depression" have a notably different average mismatch score than reviews that don't,
  that's a real, measurable pattern - not a guess.
"""
results = []
 
for term in symptom_terms:
    column_name = f"mentions_{term.replace(' ', '_')}"
 
    mentions_group = uci_birth_control_mismatch_scores[uci_birth_control_mismatch_scores[column_name] == True]
    no_mentions_group = uci_birth_control_mismatch_scores[uci_birth_control_mismatch_scores[column_name] == False]
 
    avg_mismatch_with = mentions_group["mismatch_score"].mean()
    avg_mismatch_without = no_mentions_group["mismatch_score"].mean()
 
    # We also check the SIZE of the mismatch (ignoring direction),
    # since a symptom could push ratings up OR down unpredictably -
    # what matters for "hidden signal" is how far off the guess was,
    # not which direction it was off in.
    avg_abs_mismatch_with = mentions_group["mismatch_score"].abs().mean()
    avg_abs_mismatch_without = no_mentions_group["mismatch_score"].abs().mean()
 
    results.append({
        "symptom": term,
        "review_count_mentioning": len(mentions_group),
        "pct_of_all_reviews": round(len(mentions_group) / len(uci_birth_control_mismatch_scores) * 100, 2),
        "avg_mismatch_when_mentioned": round(avg_mismatch_with, 2),
        "avg_mismatch_when_not_mentioned": round(avg_mismatch_without, 2),
        "avg_mismatch_size_when_mentioned": round(avg_abs_mismatch_with, 2),
        "avg_mismatch_size_when_not_mentioned": round(avg_abs_mismatch_without, 2),
    })
 
signal_table = pd.DataFrame(results)
 
# Sort so the symptoms with the BIGGEST difference in mismatch size
# appear first - these are the ones most worth paying attention to.
signal_table["mismatch_size_difference"] = (
    signal_table["avg_mismatch_size_when_mentioned"] - signal_table["avg_mismatch_size_when_not_mentioned"]
)
signal_table = signal_table.sort_values("mismatch_size_difference", ascending=False)
 
print("\nFinal combined signal table:")
print(signal_table.to_string(index=False))

#Save the final signal table
signal_table.to_csv("final_combined_signal_table.csv", index=False)


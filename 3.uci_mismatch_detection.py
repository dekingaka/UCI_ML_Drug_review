"""
Find Reviews Where the Rating Doesn't Match the Text


Goal: Some patients write negative things in their review text but
still give a high numeric rating (or vice versa). These "mismatches"
often hide side-effect signal that a ratings-only analysis would miss
- e.g. "it worked but gave me terrible mood swings" rated 8/10.
 
How this works, in plain terms:
1. We train a model that reads ONLY the review text and tries to
   guess what rating (1-10) the person gave.
2. We compare its guess to the REAL rating.
3. Where the guess is way off, that's a "mismatch" - worth reading
   manually, because it usually means something interesting is
   being said in the text that the numeric rating alone hides.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge

# Load our cleaned, filtered dataset
birth_control_uci_drug_review = pd.read_csv("birth_control_reviews_clean.csv")
print(f"Loaded {len(birth_control_uci_drug_review)} Birth Control reviews.")
 
# Drop any rows with missing review text or rating - the model
# can't learn from empty inputs.
birth_control_uci_drug_review = birth_control_uci_drug_review.dropna(subset=["review", "rating"])
print(f"Rows remaining after dropping missing text/rating: {len(birth_control_uci_drug_review)}")


# Assign the Training Data
# The data was pre-split into separate files to avoid data leakage, 
# so this entire DataFrame represents the training set.

X_train = birth_control_uci_drug_review["review"]
y_train = birth_control_uci_drug_review["rating"]

# Load and clean the TEST file the same way as the training file
# This function repeats the exact same cleaning logic from Step 2
# (remove broken HTML-leftover rows, drop missing conditions, filter to Birth Control) so the test set is prepared identically to the
# training set. Reusing the same logic here matters - if we cleaned
# the test set differently, any comparison between train and test
# performance would not be a fair one.
def load_and_clean_birth_control(raw_csv_path):
    test_birth_control_uci_drug_review = pd.read_csv(raw_csv_path)
 
    is_junk = test_birth_control_uci_drug_review["condition"].astype(str).str.contains("</span>", na=False)
    test_birth_control_uci_drug_review = test_birth_control_uci_drug_review[~is_junk].copy()
    test_birth_control_uci_drug_review = test_birth_control_uci_drug_review.dropna(subset=["condition"])
    test_birth_control_uci_drug_review["condition"] = test_birth_control_uci_drug_review["condition"].str.strip()
 
    test_birth_control_uci_drug_review = test_birth_control_uci_drug_review[test_birth_control_uci_drug_review["condition"] == "Birth Control"].copy()
    test_birth_control_uci_drug_review = test_birth_control_uci_drug_review.dropna(subset=["review", "rating"])
    return test_birth_control_uci_drug_review
 
 
# NOTE: this assumes you have "drugsComTest_raw.csv" - the official
# test file from the same UCI dataset - saved in this same folder.
test_birth_control_uci_drug_review = load_and_clean_birth_control("drugsComTest_raw.csv")
print(f"\nLoaded {len(test_birth_control_uci_drug_review)} Birth Control reviews (test).")
 
X_test = test_birth_control_uci_drug_review["review"]
y_test = test_birth_control_uci_drug_review["rating"]

# Turn text into numbers the model can understand
"""  
   TF-IDF (Term Frequency - Inverse Document Frequency) converts each
   review into a row of numbers, where each number represents how
   a particular word (or pair of words) is to that specific
   review, compared to how common it is across ALL reviews.
"""

# max_features=5000: only keep the 5000 most useful words/phrases.
# ngram_range=(1, 2): consider both single words and two-word phrases.
# stop_words="english": ignore common filler words like "the", "is".
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
 
# .fit_transform() learns the vocabulary from the TRAINING data only,
# then converts it into numbers.
X_train_vec = vectorizer.fit_transform(X_train)
 
# .transform() (no "fit") applies that SAME vocabulary to the test
# data. We never fit on the test set - that would let the model peek
# at test-set vocabulary, making our evaluation unrealistic.
X_test_vec = vectorizer.transform(X_test)



#Train the model
"""
   Ridge is a simple, well-understood regression model - it predicts
   a number (here, the rating) based on the input features (here, the
   TF-IDF word scores).
"""
model = Ridge(alpha=1.0)
model.fit(X_train_vec, y_train)
print("\nModel trained.")
 
 
# ---------------------------------------------------------------
# Check how good the model is, using the genuine held-out test set
# ---------------------------------------------------------------
predictions = model.predict(X_test_vec)
 
# Mean Absolute Error (MAE) tells us, on average, how far off the
# model's guesses were from the real rating.
mae = mean_absolute_error(y_test, predictions)
print(f"Test MAE (average error in predicted rating): {mae:.3f}")


# Score EVERY training review for mismatch
"""
   Now that we trust the model reasonably works (checked against the
   proper test set above), we run it across the TRAINING reviews to
   find mismatched reviews worth reading and learning from.
"""

X_train_predictions = model.predict(X_train_vec)
 
birth_control_uci_drug_review["predicted_rating"] = X_train_predictions
 

birth_control_uci_drug_review["mismatch_score"] = (
    birth_control_uci_drug_review["rating"] - birth_control_uci_drug_review["predicted_rating"]
)

birth_control_uci_drug_review["predicted_rating"] = np.clip(
    birth_control_uci_drug_review["predicted_rating"], 1, 10
)

# Sort so the BIGGEST mismatches (in either direction) appear first.
birth_control_sorted = birth_control_uci_drug_review.sort_values("mismatch_score", key=abs, ascending=False)
 
 
# Look at the most mismatched reviews
print("\nTop 10 most mismatched reviews:")
top_mismatches = birth_control_sorted[
    ["drugName", "rating", "predicted_rating", "mismatch_score", "review"]
].head(10)

for _, row in top_mismatches.iterrows():
    print(f"\nDrug: {row['drugName']}")
    print(f"Actual rating: {row['rating']} ")
    print(f"Model's predicted rating: {row['predicted_rating']:.1f}")
    print(f"Mismatch score: {row['mismatch_score']:.1f}")
    print(f"Review: {row['review'][:200]}...")


# Save the results
birth_control_sorted.to_csv("uci_birth_control_mismatch_scores.csv", index=False)

uci_birth_control_mismatch_scores = pd.read_csv("uci_birth_control_mismatch_scores.csv")
print(f"Loaded {len(uci_birth_control_mismatch_scores)} scored reviews.")

# Actual rating vs. Predicted rating (scatter plot)
plt.figure(figsize=(7, 7))
 
plt.scatter(uci_birth_control_mismatch_scores["rating"], uci_birth_control_mismatch_scores["predicted_rating"], alpha=0.15, s=15, color="#4C72B0")
 
plt.plot([1, 10], [1, 10], color="red", linestyle="--", linewidth=1.5)
 
plt.xlabel("Actual Rating (given by patient)")
plt.ylabel("Predicted Rating (guessed from text alone)")
plt.title("Actual vs. Predicted Rating - Birth Control Reviews")
plt.savefig("actual_vs_predicted_rating.png", bbox_inches="tight")


# Distribution of mismatch scores (histogram)

"""
   This shows how COMMON large mismatches are. Most reviews should
   cluster close to 0 (the model's guess was roughly right). The
   tails on either side - far from 0 - are the interesting minority
   of reviews where text and rating strongly disagree.
   """
plt.figure(figsize=(8, 5))
plt.hist(uci_birth_control_mismatch_scores["mismatch_score"], bins=40, color="#55A868", edgecolor="black")
plt.xlabel("Mismatch Score (Actual Rating - Predicted Rating)")
plt.ylabel("Number of Reviews")
plt.title("Distribution of Rating/Text Mismatch Scores")
plt.savefig("mismatch_score_distribution.png", bbox_inches="tight")


#The single most mismatched review per drug (bar chart)
"""
This picks out the biggest single mismatch for each of the top
drugs, so we can show a concrete, named example rather than just
an abstract score
"""
top_drugs = ["Etonogestrel", "Ethinyl estradiol / norethindrone", "Nexplanon"]
 
# For each drug, find the row with the single largest mismatch
# (biggest absolute value of mismatch_score).
biggest_per_drug = []
for drug in top_drugs:
    subset = uci_birth_control_mismatch_scores[uci_birth_control_mismatch_scores["drugName"] == drug]
    if len(subset) == 0:
        continue
    biggest_row = subset.loc[subset["mismatch_score"].abs().idxmax()]
    biggest_per_drug.append(biggest_row)
 
biggest_df = pd.DataFrame(biggest_per_drug)
 
plt.figure(figsize=(8, 5))
colors = ["#C44E52" if val < 0 else "#4C72B0" for val in biggest_df["mismatch_score"]]
plt.barh(biggest_df["drugName"], biggest_df["mismatch_score"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Mismatch Score (Actual Rating - Predicted Rating)")
plt.title("Biggest Single Rating/Text Mismatch per Drug")
plt.savefig("biggest_mismatch_per_drug.png", bbox_inches="tight")
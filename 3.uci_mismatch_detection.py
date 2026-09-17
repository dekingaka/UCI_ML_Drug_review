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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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
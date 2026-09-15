"""
STEP 1: Exploratory Data Analysis (EDA) on the Drug Review Dataset
--------------------------------------------------------------
Goal: Understand the data before we build anything on top of it.
"""

import pandas as pd
import matplotlib.pyplot as plt
 
# This just makes our charts look a bit nicer by default
plt.style.use("seaborn-v0_8-whitegrid")


#Load the dataset
uci_drug_review = pd.read_csv("drugsComTrain_raw.csv")
uci_drug_review.shape


#Check data types
print(uci_drug_review.dtypes)

#Check for missing data
missing_counts = uci_drug_review.isna().sum()
print(missing_counts)


# Let's also see this as a percentage - easier to judge how serious it is
missing_pct = (missing_counts / len(uci_drug_review)) * 100
print("\nMissing values as a percentage of total rows:" + str(missing_pct.round(2)))

# Look at the 'condition' column for junk/dirty values
# .value_counts() counts how many times each unique value appears.
# We sort by count so we see the most common values first.
print("\nTop 10 most common values in 'condition' column:")
print(uci_drug_review["condition"].value_counts().head(10))

# Let's specifically look for suspicious entries containing "</span>"
# which is a leftover HTML tag - a sign of messy/dirty data.
suspicious = uci_drug_review[uci_drug_review["condition"].astype(str).str.contains("</span>", na=False)]
print(f"\nNumber of rows with HTML leftovers in 'condition': {len(suspicious)}")

# Explore the 'rating' column and describe the summary statistics
print("\nSummary statistics for 'rating' column:" + str(uci_drug_review["rating"].describe()))

#plot a histogram of the 'rating' column to visualize its distribution
plt.figure(figsize=(8, 5))
uci_drug_review["rating"].plot(kind="hist", bins=10, edgecolor="black")
plt.title("Distribution of Patient Ratings (1-10)")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.savefig("rating_distribution.png", bbox_inches="tight")

# Which drugs have the most reviews?
top_drugs = uci_drug_review["drugName"].value_counts().head(15)
print(top_drugs)

#plot a bae chart to see the drugs with the most reviews
plt.figure(figsize=(8, 6))
top_drugs.sort_values().plot(kind="barh")
plt.title("Top 15 Drugs by Review Count")
plt.xlabel("Number of Reviews")
plt.savefig("top_drugs.png", bbox_inches="tight")

# Which conditions have the most reviews?
top_conditions = uci_drug_review["condition"].value_counts().head(15)
print(top_conditions)
 
plt.figure(figsize=(8, 6))
top_conditions.sort_values().plot(kind="barh")
plt.title("Top 15 Conditions by Review Count")
plt.xlabel("Number of Reviews")
plt.savefig("top_conditions.png", bbox_inches="tight")

#  How long are the reviews?
#  We create a NEW column that counts how many words are in each review
uci_drug_review["review_word_count"] = uci_drug_review["review"].astype(str).str.split().apply(len)
print(uci_drug_review["review_word_count"].describe())

plt.figure(figsize=(8, 5))
uci_drug_review["review_word_count"].plot(kind="hist", bins=50, edgecolor="black")
plt.title("Distribution of Review Length (word count)")
plt.xlabel("Number of Words in Review")
plt.ylabel("Number of Reviews")
plt.xlim(0, 300)  # zoom in - most reviews are shorter than 300 words
plt.savefig("review_length.png", bbox_inches="tight")

# Check the date range of the reviews
print(f"\nDate range of reviews: {uci_drug_review['date'].min()} to {uci_drug_review['date'].max()}")

# Find and remove the "junk" condition rows
# text like "</span>". This happens when the data was scraped from
# a website and a bit of formatting code got captured by mistake.
is_junk = uci_drug_review["condition"].astype(str).str.contains("</span>", na=False)

print(f"\nFound {is_junk.sum()} rows with broken 'condition' values.")

# this keeps only the rows where
# is_junk is False (i.e. the condition value is clean).
uci_drug_review_clean = uci_drug_review[~is_junk].copy()
print(f"Rows remaining after removing junk: {len(uci_drug_review_clean)}")
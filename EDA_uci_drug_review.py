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
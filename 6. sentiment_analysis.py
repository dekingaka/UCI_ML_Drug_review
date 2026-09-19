"""
STEP 6: Sentiment Analysis
---------------------------
Goal: Score the overall emotional TONE of every review - not specific
symptom words (that was Step 3), and not "does the model's guessed
rating match the real one" (that was Step 4) - but simply: does this
piece of writing read as positive or negative overall?

This gives us a third, independent way of reading these reviews, and
lets us check how well tone lines up with the numeric rating across
the whole dataset - not just for the top 3 drugs, but for every
Birth Control review we have.

We use VADER (Valence Aware Dictionary and sEntiment Reasoner) - a
sentiment tool with a built-in dictionary of words and their typical
emotional tone, plus rules for handling things like capitalisation,
punctuation ("!!!") and negation ("not good"). It returns a score
from -1 (very negative) to +1 (very positive) for a piece of text.
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------------------------------------------------------------
# STEP 6.1: Load our cleaned, filtered dataset from Step 2
# ---------------------------------------------------------------
df = pd.read_csv("birth_control_reviews_clean.csv")
df = df.dropna(subset=["review", "rating"])
print(f"Loaded {len(df)} Birth Control reviews.")


# ---------------------------------------------------------------
# STEP 6.2: Set up the sentiment analyser
# ---------------------------------------------------------------
# We only need to create this object once - it loads VADER's internal
# dictionary of words and their sentiment scores into memory, ready
# to score as many pieces of text as we give it.
analyzer = SentimentIntensityAnalyzer()


def get_sentiment_score(text):
    """
    Runs one review through VADER and returns its "compound" score.

    VADER actually calculates four numbers: how much of the text was
    negative, neutral, positive, and one combined "compound" score
    that summarises all of it into a single number from -1 to +1.
    We use that single summary number, since it's the simplest way
    to compare tone across thousands of reviews.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    scores = analyzer.polarity_scores(text)
    return scores["compound"]


# ---------------------------------------------------------------
# STEP 6.3: Score every review
# ---------------------------------------------------------------
# .apply() runs our function on every row in the "review" column,
# one at a time, and stores the result in a new column.
print("\nScoring sentiment for every review (this may take a minute)...")
df["sentiment_score"] = df["review"].apply(get_sentiment_score)
print("Done.")


# ---------------------------------------------------------------
# STEP 6.4: Categorise each review as positive, neutral, or negative
# ---------------------------------------------------------------
# Raw scores are useful for maths, but a category is easier to read
# and count. These thresholds (0.05 and -0.05) are VADER's own
# standard recommendation for where "neutral" starts and ends.
def categorise_sentiment(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["sentiment_category"] = df["sentiment_score"].apply(categorise_sentiment)

print("\nSentiment category counts:")
print(df["sentiment_category"].value_counts())
print("\nAs a percentage of all reviews:")
print((df["sentiment_category"].value_counts(normalize=True) * 100).round(2))


# ---------------------------------------------------------------
# STEP 6.5: Check how well sentiment lines up with the star rating
# ---------------------------------------------------------------
# .corr() calculates the correlation between two columns - a single
# number from -1 to +1 that tells us how closely they move together.
# +1 would mean "perfectly aligned" (higher rating always means more
# positive text). 0 would mean "no relationship at all".
correlation = df["rating"].corr(df["sentiment_score"])
print(f"\nCorrelation between star rating and sentiment score: {correlation:.3f}")
print("(Closer to 1.0 means text tone and star rating agree closely;")
print(" closer to 0 means they often tell different stories.)")


# ---------------------------------------------------------------
# STEP 6.6: Average sentiment per star rating
# ---------------------------------------------------------------
# .groupby() splits the data into groups (here, one group per rating
# value 1-10) so we can calculate a summary statistic - the average
# sentiment score - separately for each group.
avg_sentiment_by_rating = df.groupby("rating")["sentiment_score"].mean().round(3)
print("\nAverage sentiment score by star rating:")
print(avg_sentiment_by_rating)


# ---------------------------------------------------------------
# STEP 6.7: Average sentiment per drug (top 3 focus drugs)
# ---------------------------------------------------------------
top_drugs = ["Etonogestrel", "Ethinyl estradiol / norethindrone", "Nexplanon"]
subset = df[df["drugName"].isin(top_drugs)]

avg_sentiment_by_drug = subset.groupby("drugName")["sentiment_score"].agg(["mean", "median", "std"]).round(3)
print("\nSentiment summary by drug (mean, median, standard deviation):")
print(avg_sentiment_by_drug)


# ---------------------------------------------------------------
# STEP 6.8: Find the biggest sentiment/rating disagreements
# ---------------------------------------------------------------
# This is a second, independent way (alongside Step 4's ML model) of
# finding reviews where the numbers and the words tell different
# stories - this time based on overall tone rather than a trained
# prediction model. Worth comparing the two approaches later.
#
# We rescale sentiment (-1 to +1) onto the same 1-10 range as the
# star rating, so the two numbers are directly comparable.
df["sentiment_rescaled"] = ((df["sentiment_score"] + 1) / 2) * 9 + 1
df["sentiment_rating_gap"] = df["rating"] - df["sentiment_rescaled"]

biggest_gaps = df.sort_values("sentiment_rating_gap", key=abs, ascending=False)

print("\nTop 5 biggest sentiment/rating disagreements:")
for _, row in biggest_gaps.head(5).iterrows():
    print("\n---")
    print(f"Drug: {row['drugName']}")
    print(f"Star rating: {row['rating']} | Sentiment score: {row['sentiment_score']:.2f} (text tone)")
    print(f"Review: {row['review'][:200]}...")


# ---------------------------------------------------------------
# STEP 6.9: Save the results
# ---------------------------------------------------------------
df.to_csv("birth_control_with_sentiment.csv", index=False)
avg_sentiment_by_rating.to_csv("avg_sentiment_by_rating.csv")
avg_sentiment_by_drug.to_csv("avg_sentiment_by_drug.csv")

print("\nSaved: birth_control_with_sentiment.csv (full dataset + sentiment columns)")
print("Saved: avg_sentiment_by_rating.csv")
print("Saved: avg_sentiment_by_drug.csv")

print("\n--- Step 6 complete. ---")
print("Next: compare these sentiment-based mismatches (Step 6) against the")
print("ML-model-based mismatches from Step 4 - do they flag the same reviews,")
print("or catch different things? That comparison would make a strong final piece.")
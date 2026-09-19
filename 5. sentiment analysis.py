"""
STEP 6: Sentiment Analysis Chart
--------------------------------
Goal: Score every review's written text on a scale from negative to
positive - not just symptom words, but the overall TONE of the
writing - then chart how that sentiment compares to the rating and
across the three focus drugs.
 
This uses VADER (Valence Aware Dictionary and sEntiment Reasoner),
a sentiment analysis tool that reads a piece of text and returns a
"compound" score from -1 (very negative) to +1 (very positive).
Unlike our NER model in Step 3, VADER isn't looking for specific
medical terms - it's judging the overall emotional tone of the
whole sentence, including things like punctuation, capitalisation,
and words like "terrible" or "amazing".
"""
 
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
 
plt.style.use("seaborn-v0_8-whitegrid")
 
 
# ---------------------------------------------------------------
# STEP 6.1: Load our cleaned, filtered dataset from Step 2
# ---------------------------------------------------------------
df = pd.read_csv("birth_control_reviews_clean.csv")
df = df.dropna(subset=["review", "rating"])
print(f"Loaded {len(df)} Birth Control reviews.")
 
 
# ---------------------------------------------------------------
# STEP 6.2: Set up the sentiment analyser
# ---------------------------------------------------------------
# We only need to create this once - it loads VADER's internal
# dictionary of words and their sentiment scores.
analyzer = SentimentIntensityAnalyzer()
 
 
def get_sentiment_score(text):
    """
    Runs one review through VADER and returns its "compound" score.
 
    VADER actually returns four numbers: how much of the text was
    negative, neutral, positive, and one combined "compound" score
    that summarises all of it into a single number from -1 to +1.
    We just want that single summary number.
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
 
print("Done. Example scores:")
print(df[["drugName", "rating", "sentiment_score"]].head(5))
 
 
# ---------------------------------------------------------------
# STEP 6.4: Chart 1 - Sentiment score distribution (histogram)
# ---------------------------------------------------------------
# This shows the overall SHAPE of sentiment across all reviews -
# are most reviews clearly positive, clearly negative, or a mix?
plt.figure(figsize=(8, 5))
plt.hist(df["sentiment_score"], bins=40, color="#8172B2", edgecolor="black")
plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Neutral")
plt.xlabel("Sentiment Score (-1 = very negative, +1 = very positive)")
plt.ylabel("Number of Reviews")
plt.title("Distribution of Review Sentiment - Birth Control")
plt.legend()
plt.savefig("sentiment_distribution.png", bbox_inches="tight")
plt.close()
print("\nSaved: sentiment_distribution.png")
 
 
# ---------------------------------------------------------------
# STEP 6.5: Chart 2 - Sentiment vs. star rating (does tone match the number?)
# ---------------------------------------------------------------
# For each possible star rating (1 to 10), we calculate the AVERAGE
# sentiment score of all reviews that got that rating. If sentiment
# and rating are well aligned, this should rise in a fairly smooth
# line from left (low ratings, negative sentiment) to right (high
# ratings, positive sentiment).
avg_sentiment_by_rating = df.groupby("rating")["sentiment_score"].mean()
 
plt.figure(figsize=(8, 5))
avg_sentiment_by_rating.plot(kind="bar", color="#55A868", edgecolor="black")
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.xlabel("Star Rating")
plt.ylabel("Average Sentiment Score")
plt.title("Average Review Sentiment by Star Rating")
plt.xticks(rotation=0)
plt.savefig("sentiment_by_rating.png", bbox_inches="tight")
plt.close()
print("Saved: sentiment_by_rating.png")
 
 
# ---------------------------------------------------------------
# STEP 6.6: Chart 3 - Sentiment comparison across the 3 focus drugs
# ---------------------------------------------------------------
# A box plot shows the full SPREAD of sentiment scores per drug, not
# just the average - so we can see whether one drug's reviews are
# more consistently positive/negative, or more spread out and mixed,
# compared to the others.
top_drugs = ["Etonogestrel", "Ethinyl estradiol / norethindrone", "Nexplanon"]
subset = df[df["drugName"].isin(top_drugs)]
 
plt.figure(figsize=(8, 6))
data_to_plot = [subset[subset["drugName"] == drug]["sentiment_score"] for drug in top_drugs]
plt.boxplot(data_to_plot, tick_labels=top_drugs)
plt.axhline(0, color="red", linestyle="--", linewidth=1)
plt.ylabel("Sentiment Score")
plt.title("Sentiment Spread by Drug")
plt.xticks(rotation=15)
plt.savefig("sentiment_by_drug.png", bbox_inches="tight")
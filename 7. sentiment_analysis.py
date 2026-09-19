"""
STEP 7: Positive vs. Negative Word Cloud
------------------------------------------
Goal: Take the sentiment-scored reviews from Step 6 and visualise the
actual WORDS that show up most often in positive reviews versus
negative reviews - not just a single sentiment number, but the real
vocabulary patients use when they feel good or bad about a medication.
 
A word cloud draws each word at a size proportional to how often it
appears - so the biggest words are the most common ones. We build two
halves in one image: positive words in green (top half), negative
words in orange/red (bottom half), similar in spirit to the reference
image, but using a circular shape rather than a custom silhouette
(that would need a matching image mask file we don't have).
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
 
 
# ---------------------------------------------------------------
# STEP 7.1: Load the sentiment-scored dataset from Step 6
# ---------------------------------------------------------------
df = pd.read_csv("birth_control_with_sentiment.csv")
print(f"Loaded {len(df)} reviews with sentiment scores.")
 
 
# ---------------------------------------------------------------
# STEP 7.2: Split reviews into positive and negative groups
# ---------------------------------------------------------------
# We reuse VADER's own standard thresholds: a compound score of 0.05
# or higher counts as positive, -0.05 or lower counts as negative.
# Anything in between (a small, genuinely neutral band) is left out
# of both groups, since it doesn't clearly belong to either side.
positive_reviews = df[df["sentiment_score"] >= 0.05]["review"]
negative_reviews = df[df["sentiment_score"] <= -0.05]["review"]
 
print(f"Positive reviews: {len(positive_reviews)}")
print(f"Negative reviews: {len(negative_reviews)}")
 
 
# ---------------------------------------------------------------
# STEP 7.3: Build a combined "stop word" list
# ---------------------------------------------------------------
# Stop words are common filler words ("the", "and", "was") that
# would otherwise dominate a word cloud without telling us anything
# useful. sklearn already ships a solid general-purpose list; we add
# a few extra words specific to this dataset that are too generic to
# be interesting here (e.g. "pill", "birth", "control" - since every
# review is about birth control, those words don't distinguish
# anything). We also add the single-letter fragments left behind
# when contractions like "didn't" get split into "didn" + "t", and
# a couple of leftover HTML-entity fragments ("amp", "quot") from
# the same scraping artifacts we cleaned in Step 2.
extra_stop_words = {
    "pill", "birth", "control", "ve", "didn", "im", "don", "just",
    "got", "like", "day", "days", "month", "months", "year", "years",
    "week", "weeks", "time", "started", "taking", "took",
    "s", "t", "m", "re", "ll", "d", "amp", "quot",
}
all_stop_words = set(ENGLISH_STOP_WORDS).union(extra_stop_words)
 
 
# ---------------------------------------------------------------
# STEP 7.4: Turn each group of reviews into one big block of text
# ---------------------------------------------------------------
# WordCloud expects one long string, not a list of separate reviews,
# so we join everything together with spaces in between. We also
# replace apostrophes with nothing (so "didn't" becomes "didnt"
# rather than splitting into "didn" and "t" as two separate words),
# and strip out leftover HTML-entity text from the original scrape.
def clean_text_block(reviews):
    text = " ".join(reviews.astype(str))
    text = text.replace("&amp;", " ").replace("&quot;", " ").replace("&#039;", "")
    text = text.replace("'", "")
    return text
 
positive_text = clean_text_block(positive_reviews)
negative_text = clean_text_block(negative_reviews)
 
 
# ---------------------------------------------------------------
# STEP 7.5: Build a circular mask (the shape the words fill)
# ---------------------------------------------------------------
# A "mask" tells WordCloud which pixels are allowed to have words
# drawn on them (0 = allowed, 255 = blocked/background). We draw a
# simple filled circle using basic coordinate maths - anywhere within
# the circle's radius is fair game for a word to appear.
diameter = 600
radius = diameter // 2
y_coords, x_coords = np.ogrid[:diameter, :diameter]
distance_from_center = (x_coords - radius) ** 2 + (y_coords - radius) ** 2
circle_mask = np.where(distance_from_center <= radius ** 2, 0, 255).astype(np.uint8)
 
 
# ---------------------------------------------------------------
# STEP 7.6: Build the positive (top) and negative (bottom) word clouds
# ---------------------------------------------------------------
# We generate each half separately, then stitch them together into
# one image - positive on top, negative on the bottom.
positive_cloud = WordCloud(
    width=diameter, height=diameter // 2,
    background_color="white",
    colormap="Greens",
    stopwords=all_stop_words,
    max_words=100,
    prefer_horizontal=1.0,
).generate(positive_text)
 
negative_cloud = WordCloud(
    width=diameter, height=diameter // 2,
    background_color="white",
    colormap="Oranges",
    stopwords=all_stop_words,
    max_words=100,
    prefer_horizontal=1.0,
).generate(negative_text)
 
 
# ---------------------------------------------------------------
# STEP 7.7: Draw the final combined chart
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 9))
 
axes[0].imshow(positive_cloud, interpolation="bilinear")
axes[0].axis("off")
axes[0].set_title("POSITIVE", fontsize=28, fontweight="bold", color="#2E7D32")
 
axes[1].imshow(negative_cloud, interpolation="bilinear")
axes[1].axis("off")
axes[1].set_title("NEGATIVE", fontsize=28, fontweight="bold", color="#D2691E")
 
plt.tight_layout()
plt.savefig("sentiment_wordcloud.png", bbox_inches="tight", dpi=150)
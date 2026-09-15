"""
STEP 3: Extract Side-Effect Mentions from Reviews Using NER

Scan each patient review and automatically pull out mentions
of symptoms/side effects, then count how often each one appears
per drug.
 
NER = Named Entity Recognition. It's an NLP technique that finds
specific real-world "things" in text and labels what kind of thing
they are. Here we use a model trained on biomedical text, so it
recognises symptoms/diseases and chemicals rather than general
things like place names or people.
"""


import spacy
import pandas as pd
from collections import Counter
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

#Load the biomedical NER model
nlp_model = spacy.load("en_ner_bc5cdr_md")

# Load our cleaned, filtered dataset from Step
birth_control_uci_drug_review = pd.read_csv("birth_control_reviews_clean.csv")
print(birth_control_uci_drug_review.head(15))


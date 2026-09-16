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

# Write a function that extracts entities from one review
"""
    This function takes a single piece of review text and returns a list of
    (entity_text, entity_label) pairs found in it.
 
    nlp(text) runs the text through the NER model. The result is a
    "doc" object which has a .ents property - a list of all entities
    the model found, each one knowing its own text and label.
 
    Labels in this model are either:
      DISEASE  - symptoms, conditions, side effects
      CHEMICAL - drug/chemical names mentioned in the text
    """

def extract_entities(text):
    if not isinstance(text, str) or not text.strip():
        # If the review is empty or not text, there's nothing to extract.
        return []
 
    doc = nlp_model(text)
 
    # ent.text is the actual word/phrase found (e.g. "headache")
    # ent.label_ is the category the model assigned (e.g. "DISEASE")
    # .lower().strip() tidies up capitalisation and stray spaces so
    # "Headache" and "headache" count as the same thing later.
    return [(ent.text.lower().strip(), ent.label_) for ent in doc.ents]


"""
    Filter the dataset down to one specific drug, runs NER on every
    review for that drug, and returns a table of how often each
    DISEASE entity (our proxy for "symptom/side effect") was mentioned.
    """
def extract_side_effects_for_drug(birth_control_uci_drug_review, drug_name, text_col="review", drug_col="drugName"):
    subset = birth_control_uci_drug_review[birth_control_uci_drug_review[drug_col].str.lower() == drug_name.lower()]
    print(f"\nRunning NER on {len(subset)} reviews for '{drug_name}'...")
 
    all_symptoms = []
 
    # .iterrows() lets us loop through the dataset one row at a time.
    for _, row in subset.iterrows():
        entities = extract_entities(row[text_col])
 
        # We only keep entities labelled DISEASE, since that's our
        # proxy for symptoms / side effects (as opposed to CHEMICAL,
        # which usually picks up drug/ingredient names instead).
        symptoms_in_review = [text for text, label in entities if label == "DISEASE"]
        all_symptoms.extend(symptoms_in_review)
 
    # Counter() counts how many times each item appears in a list.
    # .most_common(30) returns the 30 most frequent items, ranked.
    frequency = Counter(all_symptoms)
    result = pd.DataFrame(frequency.most_common(30), columns=["term", "mention_count"])
 
    # Express the count as a percentage of reviews for easier comparison
    # across drugs with different review volumes.
    result["pct_of_reviews"] = (result["mention_count"] / len(subset) * 100).round(2)
 
    return result
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



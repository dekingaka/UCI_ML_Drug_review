import pandas as pd
 
# Load the new csv file that was saved 
df = pd.read_csv("drugs_with_eda_columns.csv")
print("Loaded dataset shape:", df.shape)
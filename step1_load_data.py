import pandas as pd

# Load dataset
df = pd.read_csv("sms.tsv", sep="\t", header=None)

# Rename columns
df.columns = ["label", "message"]

# Show first 5 rows
print(df.head())

# Check dataset info
print("\nDataset Info:")
print(df.info())

# Check class distribution
print("\nClass Distribution:")
print(df["label"].value_counts())
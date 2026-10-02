# Titanic Survival Prediction — built step by step

import pandas as pd

# STEP 1 — Load the data and look at it
df = pd.read_csv("data/titanic.csv")

print("Shape:", df.shape)          # (rows, columns)
print(df.head())                   # first 5 rows
print(df.isna().sum())             # missing values per column

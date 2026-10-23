import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")

print("=" * 60)
print("CUSTOMER CHURN PROJECT - DATA CHECK")
print("=" * 60)
print("Python:", sys.executable)
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}\n"
        "Download the Telco Customer Churn CSV into data/raw first."
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nTarget distribution:")
print(df["Churn"].value_counts(dropna=False))
print("\nTarget percentage:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))

print("\nMissing values (top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10))

print("\nDuplicate rows:", df.duplicated().sum())
print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

print("\nDATA CHECK COMPLETED")

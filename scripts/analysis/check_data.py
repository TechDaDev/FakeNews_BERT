import pandas as pd
import os
from pathlib import Path

# Add root directory calculation
root_dir = Path(__file__).resolve().parent.parent.parent
df = pd.read_csv(root_dir / 'data' / 'WELFake_Dataset.csv')

print("Dataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nLabel Distribution:")
print(df['label'].value_counts())

print("\nData Types:")
print(df.dtypes)

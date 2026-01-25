import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
from pathlib import Path

# Add root directory calculation
root_dir = Path(__file__).resolve().parent.parent.parent
df = pd.read_csv(root_dir / 'data' / 'WELFake_Dataset.csv')
df.dropna(subset=['text', 'label'], inplace=True)

for label in df['label'].unique():
    texts = df[df['label'] == label]['text'].astype(str)
    vec = CountVectorizer(stop_words='english', max_features=20)
    X = vec.fit_transform(texts)
    print(f"\nTop words for Label {label}:")
    print(vec.get_feature_names_out())

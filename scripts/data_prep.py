import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump

# Add root directory calculation
root_dir = Path(__file__).resolve().parent.parent

def load_and_preprocess_data(csv_path: str = None, test_size: float = 0.2, random_state: int = 42):
    if csv_path is None:
        csv_path = root_dir / "data" / "combined_TF_data.csv"
    """Load and preprocess dataset with label inversion correction and cleaning.

    Returns
    -------
    X_train, X_test, y_train, y_test, stats (dict)
    """
    print("=== Data Preprocessing ===")
    data = pd.read_csv(csv_path)
    print(f'Raw shape: {data.shape}')
    before = data.shape[0]
    data.dropna(subset=['text','label'], inplace=True)
    after_dropna = data.shape[0]
    data['text'] = data['text'].astype(str)
    empty_mask = data['text'].str.strip().eq("")
    empties = empty_mask.sum()
    if empties:
        data = data.loc[~empty_mask]
    dup_count = data.duplicated(subset=['text']).sum()
    if dup_count:
        data = data.drop_duplicates(subset=['text'], keep='first')
    print(f"Dropped NaN: {before - after_dropna} | Empties removed: {empties} | Duplicates removed: {dup_count}")
    orig_y = data['label'].astype(int)
    y = orig_y  # Use original labels: 0=Real, 1=Fake
    print('Label distribution (0=Real,1=Fake):')
    print(y.value_counts().sort_index())
    X = data['text']
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_vec = tfidf_vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f'X train: {X_train.shape[0]} | X test: {X_test.shape[0]}')
    vectorizer_path = root_dir / "saved_models" / "tfidf_vectorizer.pkl"
    dump(tfidf_vectorizer, vectorizer_path)
    print(f"TF-IDF vectorizer saved as '{vectorizer_path}'")
    stats = {
        'dropped_nan': int(before - after_dropna),
        'empties_removed': int(empties),
        'duplicates_removed': int(dup_count),
        'raw_rows': int(before),
        'final_rows': int(data.shape[0]),
        'label_distribution_after_flip': {int(k): int(v) for k, v in y.value_counts().sort_index().items()},
        'test_size': test_size,
        'stratify': True,
        'random_state': random_state,
        'tfidf_params': {'stop_words': 'english', 'max_df': 0.7}
    }
    return X_train, X_test, y_train, y_test, stats

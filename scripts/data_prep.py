import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump

# Add root directory calculation
root_dir = Path(__file__).resolve().parent.parent

def load_and_preprocess_data(test_size: float = 0.2, random_state: int = 42, combine_title: bool = True, return_df: bool = False):
    """Load and preprocess the Fake News Dataset.
    Supports both ISOT (Fake.csv, True.csv) and WELFake (WELFake_Dataset.csv) formats.

    Returns
    -------
    If return_df is False: X_train_vec, X_test_vec, y_train, y_test, stats (dict)
    If return_df is True: cleaned_dataframe, stats (dict)
    """
    print("=== Data Preprocessing ===")
    
    combined_path = root_dir / "data" / "combined_TF_data.csv"
    fake_path = root_dir / "data" / "Fake.csv"
    true_path = root_dir / "data" / "True.csv"
    welfake_path = root_dir / "data" / "WELFake_Dataset.csv"
    
    # 1. Load Data
    if welfake_path.exists():
        print("Detected WELFake Dataset.")
        data = pd.read_csv(welfake_path)
        # WELFake typically has [unnamed:0, title, text, label]
        if 'label' not in data.columns and 'Label' in data.columns:
            data.rename(columns={'Label': 'label'}, inplace=True)
    elif fake_path.exists() and true_path.exists():
        print("Detected ISOT Dataset (Fake.csv and True.csv).")
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        true_df['label'] = 0
        fake_df['label'] = 1
        data = pd.concat([true_df, fake_df], ignore_index=True)
    elif combined_path.exists():
        print(f"Loading already combined data from {combined_path}")
        data = pd.read_csv(combined_path)
    else:
        raise FileNotFoundError(f"No dataset files found in {root_dir / 'data'}")
        
    print(f'Initial raw shape: {data.shape}')
    before = data.shape[0]
    
    # Standardize column names (lowercase)
    data.columns = [c.lower() for c in data.columns]
    
    # 2. Cleaning
    # Drop rows with missing text or title (if combining)
    cols_to_check = ['text', 'label']
    if combine_title and 'title' in data.columns:
        cols_to_check.append('title')
        
    data.dropna(subset=cols_to_check, inplace=True)
    after_dropna = data.shape[0]
    
    # Ensure strings
    data['text'] = data['text'].astype(str)
    if 'title' in data.columns:
        data['title'] = data['title'].astype(str)
    
    # 3. Feature Engineering: Combine Title and Text
    if combine_title and 'title' in data.columns:
        print("Combining title and text into a single feature...")
        data['text'] = data['title'] + " " + data['text']
    
    # Remove empty strings from text
    empty_mask = data['text'].str.strip().eq("")
    empties = empty_mask.sum()
    if empties:
        data = data.loc[~empty_mask]
        
    # Remove duplicates
    dup_count = data.duplicated(subset=['text']).sum()
    if dup_count:
        data = data.drop_duplicates(subset=['text'], keep='first')
        
    print(f"Dropped NaN: {before - after_dropna} | Empties removed: {empties} | Duplicates removed: {dup_count}")
    
    # 4. Save Combined Data for DL scripts
    os.makedirs(combined_path.parent, exist_ok=True)
    data.to_csv(combined_path, index=False)
    print(f"Combined and cleaned data saved to: {combined_path}")
    # Basic stats available at this stage (used when `return_df=True`)
    basic_stats = {
        'dropped_nan': int(before - after_dropna),
        'empties_removed': int(empties),
        'duplicates_removed': int(dup_count),
        'raw_rows': int(before),
        'final_rows': int(data.shape[0]),
        'label_distribution': {int(k): int(v) for k, v in data['label'].astype(int).value_counts().sort_index().items()}
    }
    if return_df:
        return data, basic_stats

    y = data['label'].astype(int)
    print('Label distribution (0=Real, 1=Fake):')
    print(y.value_counts().sort_index())
    
    X = data['text']
    
    # 5. TF-IDF for basic ML models
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=2)
    X_vec = tfidf_vectorizer.fit_transform(X)
    
    X_train_vec, X_test_vec, y_train, y_test = train_test_split(
        X_vec, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f'X train: {X_train_vec.shape[0]} | X test: {X_test_vec.shape[0]}')
    
    vectorizer_path = root_dir / "saved_models" / "tfidf_vectorizer.pkl"
    os.makedirs(vectorizer_path.parent, exist_ok=True)
    dump(tfidf_vectorizer, vectorizer_path)
    print(f"TF-IDF vectorizer saved as '{vectorizer_path}'")
    
    stats = {
        'dropped_nan': int(before - after_dropna),
        'empties_removed': int(empties),
        'duplicates_removed': int(dup_count),
        'raw_rows': int(before),
        'final_rows': int(data.shape[0]),
        'label_distribution': {int(k): int(v) for k, v in y.value_counts().sort_index().items()},
        'test_size': test_size,
        'stratify': True,
        'random_state': random_state,
        'tfidf_params': {'stop_words': 'english', 'max_df': 0.7, 'min_df': 2}
    }
    return X_train_vec, X_test_vec, y_train, y_test, stats

if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, stats = load_and_preprocess_data()
        print("Preprocessing complete.")
    except Exception as e:
        print(f"Unexpected error during preprocessing: {e}")


import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import os
from pathlib import Path

# Add root directory calculation
root_dir = Path(__file__).resolve().parent.parent.parent
DATA_PATH = root_dir / "data" / "WELFake_Dataset.csv"
TOP_N = 30

def basic_load(path=DATA_PATH):
    df = pd.read_csv(path)
    print("===== BASIC INFO =====")
    print(f"Raw shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print(df.head(3))
    # Standardize expected columns
    col_map = {c.lower(): c for c in df.columns}
    text_col = None
    label_col = None
    for candidate in ["text", "content", "article"]:
        if candidate in col_map:
            text_col = col_map[candidate]
            break
    for candidate in ["label", "class", "target"]:
        if candidate in col_map:
            label_col = col_map[candidate]
            break
    if text_col is None or label_col is None:
        raise ValueError("Could not find required text/label columns.")
    return df, text_col, label_col

def clean_text(s: str):
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"https?://\S+", " URL ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def describe_labels(df, label_col):
    print("\n===== LABEL DISTRIBUTION =====")
    counts = df[label_col].value_counts().sort_index()
    print(counts)
    print("Percentages (%):")
    print((counts / counts.sum() * 100).round(2))


def missing_and_duplicates(df, text_col):
    print("\n===== MISSING & DUPLICATES =====")
    miss = df[text_col].isna().sum()
    print(f"Missing {text_col}: {miss}")
    empty = (df[text_col].astype(str).str.strip().eq("")).sum()
    print(f"Empty after strip: {empty}")
    dup_text = df.duplicated(subset=[text_col]).sum()
    print(f"Exact duplicate texts: {dup_text}")


def length_stats(df, text_col):
    print("\n===== LENGTH STATS =====")
    lengths_char = df[text_col].astype(str).str.len()
    lengths_tok = df[text_col].astype(str).str.split().map(len)
    for name, series in [("Chars", lengths_char), ("Tokens", lengths_tok)]:
        print(f"{name} - mean: {series.mean():.1f}, median: {series.median():.1f}, std: {series.std():.1f}, min: {series.min()}, max: {series.max()}")
    # Bucket distribution
    bins = [0,10,25,50,100,200,400,800,1600,100000]
    labels = ["<=10","11-25","26-50","51-100","101-200","201-400","401-800","801-1600",">1600"]
    bucket = pd.cut(lengths_tok, bins=bins, labels=labels, right=True)
    print("Token length buckets (%):")
    print((bucket.value_counts(normalize=True).sort_index()*100).round(2))


def top_terms(df, text_col, label_col, top_n=TOP_N):
    print("\n===== TOP TERMS PER CLASS (Raw Counts) =====")
    vectorizer = CountVectorizer(stop_words='english', min_df=5)
    X = vectorizer.fit_transform(df[text_col].astype(str))
    vocab = np.array(vectorizer.get_feature_names_out())
    labels = sorted(df[label_col].unique())
    label_arr = df[label_col].to_numpy()
    for lab in labels:
        mask = (label_arr == lab)  # numpy boolean array
        if mask.sum() == 0:
            continue
        counts = np.asarray(X[mask].sum(axis=0)).ravel()
        top_idx = counts.argsort()[::-1][:top_n]
        print(f"Label {lab}: {' '.join([f'{v}:{counts[i]}' for i,v in zip(top_idx, vocab[top_idx])])}")


def distinctive_terms_tfidf(df, text_col, label_col, top_n=TOP_N):
    print("\n===== DISTINCTIVE TERMS (Highest Mean TF-IDF per Class) =====")
    tfidf = TfidfVectorizer(stop_words='english', min_df=5)
    X = tfidf.fit_transform(df[text_col].astype(str))
    vocab = np.array(tfidf.get_feature_names_out())
    labels = sorted(df[label_col].unique())
    label_arr = df[label_col].to_numpy()
    for lab in labels:
        mask = (label_arr == lab)
        if mask.sum() == 0:
            continue
        sub = X[mask]
        mean_scores = np.asarray(sub.mean(axis=0)).ravel()
        top_idx = mean_scores.argsort()[::-1][:top_n]
        print(f"Label {lab} top TF-IDF terms:")
        print(', '.join([f"{v}:{mean_scores[i]:.3f}" for i,v in zip(top_idx, vocab[top_idx])]))


def class_balance_warning(df, label_col):
    counts = df[label_col].value_counts()
    maj = counts.max()
    minc = counts.min()
    ratio = maj / max(minc,1)
    if ratio > 1.2:
        print(f"WARNING: Class imbalance ratio ~{ratio:.2f} (max/min). Consider class weights.")


def quick_baseline(df, text_col, label_col):
    """Very naive baseline: predict majority class, show accuracy."""
    counts = df[label_col].value_counts()
    majority = counts.idxmax()
    acc = counts.max()/counts.sum()
    print("\n===== NAIVE BASELINE =====")
    print(f"Majority class: {majority} | Baseline accuracy: {acc:.4f}")


def main():
    df, text_col, label_col = basic_load()
    describe_labels(df, label_col)
    class_balance_warning(df, label_col)
    missing_and_duplicates(df, text_col)
    length_stats(df, text_col)
    quick_baseline(df, text_col, label_col)
    top_terms(df, text_col, label_col)
    distinctive_terms_tfidf(df, text_col, label_col)
    print("\nDone.")

if __name__ == "__main__":
    main()

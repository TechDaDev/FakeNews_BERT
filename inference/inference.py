import os
import json
import argparse
import numpy as np
from joblib import load

# This script allows running predictions with already trained models
# without re-running training.
# It detects available models inside model_runs/<timestamp>/ directories.


import sys
from pathlib import Path

# Add root directory to sys.path
root_dir_proj = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir_proj))

def discover_runs(runs_root=None):
    if runs_root is None:
        runs_root = root_dir_proj / 'saved_models' / 'model_runs'
    if not os.path.isdir(runs_root):
        return []
    runs = []
    for name in sorted(os.listdir(runs_root)):
        path = os.path.join(runs_root, name)
        if os.path.isdir(path):
            runs.append(path)
    return runs


def load_metadata(run_dir):
    meta_path = os.path.join(run_dir, 'run_metadata.json')
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def list_models(run_dir):
    models = []
    for f in os.listdir(run_dir):
        if f.endswith('.pkl') and not f.startswith('tfidf'):
            models.append(os.path.join(run_dir, f))
    return sorted(models)


def compute_token_stats(vectorizer, text: str):
    prep = vectorizer.build_preprocessor()
    tokenizer = vectorizer.build_tokenizer()
    processed = prep(text)
    tokens = tokenizer(processed)
    vocab = vectorizer.vocabulary_
    oov = [t for t in tokens if t not in vocab]
    return {
        'char_count': len(text),
        'token_count': len(tokens),
        'unique_tokens': len(set(tokens)),
        'oov_tokens': len(oov),
        'oov_ratio': (len(oov) / len(tokens)) if tokens else 0.0,
        'first_tokens': ' '.join(tokens[:30])
    }


def default_length_bins():
    return {250:0.50,400:0.52,600:0.53,10000:0.55}

def _raw_fake_proba(model, vec):
    if hasattr(model,'is_torch_model') and getattr(model,'is_torch_model'):
        import torch
        arr = vec.toarray().astype('float32') if hasattr(vec,'toarray') else vec.astype('float32')
        with torch.no_grad():
            tensor = torch.from_numpy(arr).to(model.device)
            logits = model.model(tensor).squeeze(-1)
            return float(torch.sigmoid(logits).detach().cpu().numpy())
    if hasattr(model,'predict_proba'):
        return float(model.predict_proba(vec)[0][1])
    if hasattr(model,'decision_function'):
        dfv = model.decision_function(vec)[0]
        return float(1/(1+np.exp(-dfv)))
    return float(model.predict(vec)[0])

def _chunk_probs(model, vectorizer, tokens, chunk_size, overlap):
    if len(tokens) <= chunk_size:
        return None
    probs=[]
    step = max(1, chunk_size-overlap)
    start=0
    while start < len(tokens):
        seg = tokens[start:start+chunk_size]
        if not seg:
            break
        seg_text = ' '.join(seg)
        vec = vectorizer.transform([seg_text])
        probs.append(_raw_fake_proba(model, vec))
        start += step
    return probs

def predict_with_model(model, vectorizer, sample, threshold=0.5, flipped=True, return_prob=False,
                       margin=0.02, use_chunking=False, chunk_size=250, chunk_overlap=50,
                       length_bins=None):
    prep = vectorizer.build_preprocessor()
    tokenizer = vectorizer.build_tokenizer()
    processed = prep(sample)
    tokens = tokenizer(processed)
    vec = vectorizer.transform([sample])
    token_len = len(tokens)
    adaptive_thr = threshold
    if length_bins:
        for max_len, thr in sorted(length_bins.items()):
            if token_len <= max_len:
                adaptive_thr = thr
                break
    p_fake = _raw_fake_proba(model, vec)
    chunk_info=None
    if use_chunking:
        probs = _chunk_probs(model, vectorizer, tokens, chunk_size, chunk_overlap)
        if probs:
            mean_p = float(np.mean(probs))
            max_p = float(np.max(probs))
            chunk_info = f"chunks={len(probs)} mean={mean_p:.4f} max={max_p:.4f} orig={p_fake:.4f}"
            p_fake = mean_p
    lower = adaptive_thr - margin
    upper = adaptive_thr + margin
    if p_fake < lower:
        pred_int=0
        certainty='certain'
    elif p_fake > upper:
        pred_int=1
        certainty='certain'
    else:
        pred_int=-1
        certainty='uncertain'
    if flipped:
        label_map={0:'Real',1:'Fake',-1:'Uncertain'}
    else:
        label_map={0:'Fake',1:'Real',-1:'Uncertain'}
    label = label_map[pred_int]
    if return_prob:
        return label, p_fake
    # Print details
    vocab = vectorizer.vocabulary_
    oov = sum(1 for t in tokens if t not in vocab)
    oov_ratio = (oov/token_len) if token_len else 0.0
    print('-- Prediction --')
    print(f"Label: {label} | P(Fake)={p_fake:.4f} | Thr={adaptive_thr:.3f} | {certainty}")
    print(f"Band: [{lower:.3f},{upper:.3f}] Margin={margin}")
    if chunk_info:
        print('Chunk info:', chunk_info)
    print('-- Token Stats --')
    print(f"Chars: {len(sample)} | Tokens: {token_len} | Unique: {len(set(tokens))} | OOV: {oov} ({oov_ratio*100:.2f}% )")
    if pred_int==-1:
        print('NOTE: In uncertainty band.')
    return label


def interactive(run_dir, threshold, raw=False, margin=0.02, use_chunk=False, adaptive=False):
    print(f"Using run directory: {run_dir}")
    meta = load_metadata(run_dir)
    flipped = meta.get('label_flip_applied', False)
    print(f"Label flip applied: {flipped}")
    if os.path.exists(os.path.join(run_dir, 'tfidf_vectorizer.pkl')):
        vec_path = os.path.join(run_dir, 'tfidf_vectorizer.pkl')
    else:
        vec_path = root_dir_proj / 'saved_models' / 'tfidf_vectorizer.pkl'
    if not os.path.isfile(vec_path):
        print('Vectorizer file not found:', vec_path)
        return
    vectorizer = load(vec_path)
    models = list_models(run_dir)
    if not models:
        print('No model .pkl files found in run directory.')
        return
    print('\nAvailable models:')
    for i, m in enumerate(models, 1):
        print(f"{i}. {os.path.basename(m)}")
    print("Enter number or filename (q to quit)")
    while True:
        sel = input('\nSelect model: ').strip()
        if sel.lower() == 'q':
            break
        if sel.isdigit():
            idx = int(sel)-1
            if 0 <= idx < len(models):
                model_path = models[idx]
            else:
                print('Invalid index')
                continue
        else:
            matches = [m for m in models if os.path.basename(m) == sel]
            if not matches:
                print('Model not found')
                continue
            model_path = matches[0]
        try:
            model = load(model_path)
            print(f'Loaded model: {model_path}')
        except Exception as e:
            print('Load error:', e)
            continue
        bins = default_length_bins() if adaptive else None
        while True:
            text = input('\nEnter text (or /back /q): ').strip()
            if text.lower() in {'/q','/quit','/back'}:
                break
            if not text:
                continue
            predict_with_model(model, vectorizer, text, threshold=threshold, flipped=flipped,
                               margin=margin, use_chunking=use_chunk, length_bins=bins)


def main():
    parser = argparse.ArgumentParser(description='Inference on existing fake news models (no retraining).')
    parser.add_argument('--run', help='Run directory under model_runs/.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold for class Fake.')
    parser.add_argument('--raw', action='store_true', help='Show first tokens (unused in new flow).')
    parser.add_argument('--margin', type=float, default=0.02, help='Uncertainty margin.')
    parser.add_argument('--chunk', action='store_true', help='Enable chunking aggregation.')
    parser.add_argument('--adaptive', action='store_true', help='Enable adaptive length thresholds.')
    parser.add_argument('--chunk_size', type=int, default=250, help='Chunk size tokens.')
    parser.add_argument('--chunk_overlap', type=int, default=50, help='Chunk overlap tokens.')
    args = parser.parse_args()
    runs = discover_runs()
    if not runs:
        print('No run directories found (saved_models/model_runs/).')
        return
    run_dir = None
    if args.run:
        runs_root = root_dir_proj / 'saved_models' / 'model_runs'
        candidate = os.path.join(runs_root, args.run) if not args.run.startswith(str(runs_root)) else args.run
        if os.path.isdir(candidate):
            run_dir = candidate
        else:
            print('Specified run not found, falling back to latest.')
    if run_dir is None:
        run_dir = str(runs[-1])
    # Pass options into interactive
    interactive(run_dir, args.threshold, raw=args.raw, margin=args.margin, use_chunk=args.chunk, adaptive=args.adaptive)

if __name__ == '__main__':
    main()

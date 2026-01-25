import os
import numpy as np
from joblib import load
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent

def default_length_bins():
    return {250:0.50,400:0.52,600:0.53,10000:0.55}

def _raw_fake_proba(model, vec):
    if hasattr(model,'is_torch_model') and getattr(model,'is_torch_model'):
        import torch
        if hasattr(vec,'toarray'):
            arr = vec.toarray().astype('float32')
        else:
            arr = vec.astype('float32')
        with torch.no_grad():
            tensor = torch.from_numpy(arr).to(model.device)
            logits = model.model(tensor).squeeze(-1)
            return float(torch.sigmoid(logits).detach().cpu().numpy())
    if hasattr(model,'predict_proba'):
        return float(model.predict_proba(vec)[0][1])
    if hasattr(model,'decision_function'):
        import numpy as _np
        dfv = model.decision_function(vec)[0]
        return float(1/(1+_np.exp(-dfv)))
    return float(model.predict(vec)[0])

def _chunk_probs(model, vectorizer, tokens, chunk_size, overlap):
    if len(tokens) <= chunk_size:
        return None
    probs=[]; start=0; step=max(1,chunk_size-overlap)
    while start < len(tokens):
        seg_tokens=tokens[start:start+chunk_size]
        if not seg_tokens: break
        seg_text=' '.join(seg_tokens)
        vec=vectorizer.transform([seg_text])
        probs.append(_raw_fake_proba(model, vec))
        start+=step
    return probs

def predict_with_model(model_filename, sample, threshold=0.5, return_score=False, margin=0.02,
                       use_chunking=False, chunk_size=250, chunk_overlap=50,
                       length_bin_thresholds=None, vectorizer_path=None):
    if vectorizer_path is None:
        vectorizer_path = root_dir / 'saved_models' / 'tfidf_vectorizer.pkl'
    try:
        model = load(model_filename)
        vectorizer = load(vectorizer_path)
        prep = vectorizer.build_preprocessor(); tokenizer = vectorizer.build_tokenizer()
        processed = prep(sample); tokens = tokenizer(processed)
        vec = vectorizer.transform([sample])
        token_len = len(tokens)
        adaptive_threshold = threshold
        if length_bin_thresholds:
            for max_len, thr in sorted(length_bin_thresholds.items()):
                if token_len <= max_len:
                    adaptive_threshold = thr; break
        p_fake = _raw_fake_proba(model, vec)
        chunk_info=None
        if use_chunking:
            probs = _chunk_probs(model, vectorizer, tokens, chunk_size, chunk_overlap)
            if probs:
                p_mean=float(np.mean(probs)); p_max=float(np.max(probs))
                chunk_info=f"chunks={len(probs)} mean={p_mean:.4f} max={p_max:.4f} (orig={p_fake:.4f})"; p_fake=p_mean
        lower=adaptive_threshold-margin; upper=adaptive_threshold+margin
        if p_fake < lower:
            pred_int=0; certainty='certain'
        elif p_fake > upper:
            pred_int=1; certainty='certain'
        else:
            pred_int=-1; certainty='uncertain'
        label_map={0:'Real',1:'Fake',-1:'Uncertain'}; label_str=label_map[pred_int]
        if return_score:
            return label_str, p_fake
        vocab=vectorizer.vocabulary_; oov=sum(1 for t in tokens if t not in vocab)
        oov_ratio=(oov/token_len) if token_len else 0.0
        print('-- Prediction --')
        print(f"Label: {label_str} | P(Fake)={p_fake:.4f} | Thr={adaptive_threshold:.3f} | {certainty}")
        print(f"Band: [{lower:.3f},{upper:.3f}] Margin={margin}")
        if chunk_info: print('Chunk info:', chunk_info)
        print('-- Token Stats --')
        print(f"Chars: {len(sample)} | Tokens: {token_len} | Unique: {len(set(tokens))} | OOV: {oov} ({oov_ratio*100:.2f}% )")
        if pred_int==-1: print('NOTE: In uncertainty band; consider review.')
        return label_str
    except Exception as e:
        return f"Error in prediction: {str(e)}"

def compute_token_stats(vectorizer, text):
    prep=vectorizer.build_preprocessor(); tokenizer=vectorizer.build_tokenizer()
    processed=prep(text); tokens=tokenizer(processed); vocab=vectorizer.vocabulary_
    oov=[t for t in tokens if t not in vocab]
    return {
        'char_count': len(text),
        'token_count': len(tokens),
        'unique_tokens': len(set(tokens)),
        'oov_tokens': len(oov),
        'oov_ratio': (len(oov)/len(tokens)) if tokens else 0.0,
        'sample_preview': ' '.join(tokens[:25])
    }

import numpy as np
import torch
import torch.nn.functional as F

# Canonical Label Mapping
LABEL_REAL = 0
LABEL_FAKE = 1
LABEL_STR_MAP = {LABEL_REAL: "REAL", LABEL_FAKE: "FAKE"}

def decode_predictions(logits, temperature=1.0, debug=False):
    """
    Decodes model logits into labels and probabilities using canonical mapping.
    
    Args:
        logits (torch.Tensor or np.ndarray): Raw model outputs (post-linear layer)
        temperature (float): Scaling factor for probabilities (calibration)
        debug (bool): If true, prints raw logits and indices
        
    Returns:
        dict: {
            'predicted_label': str,
            'real_probability': float,
            'fake_probability': float,
            'confidence': float,
            'raw_logits': list (if debug)
        }
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    
    # 1. Apply Temperature Scaling (Simple calibration placeholder)
    scaled_logits = logits / temperature
    
    # 2. Get probabilities
    probs = F.softmax(scaled_logits, dim=-1).squeeze()
    
    if probs.dim() == 0: # Handle single value case
        probs = probs.unsqueeze(0)
        
    # Canonical mapping: 0=REAL, 1=FAKE
    # Ensure we are looking at index 0 and 1
    real_prob = float(probs[LABEL_REAL].item())
    fake_prob = float(probs[LABEL_FAKE].item())
    
    pred_idx = int(torch.argmax(probs).item())
    predicted_label = LABEL_STR_MAP.get(pred_idx, "UNKNOWN")
    
    # Confidence is just the probability of the predicted class
    confidence = real_prob if pred_idx == LABEL_REAL else fake_prob
    
    result = {
        'predicted_label': predicted_label,
        'real_probability': real_prob,
        'fake_probability': fake_prob,
        'confidence': confidence,
        'class_index': pred_idx
    }
    
    if debug:
        result['raw_logits'] = logits.tolist()
        print(f"[DEBUG] Raw Logits: {logits}")
        print(f"[DEBUG] Probs: {probs}")
        print(f"[DEBUG] Decoded Index: {pred_idx} ({predicted_label})")
        
    return result

def get_calibrated_probabilities(model, inputs, device, temperature=1.0):
    """Wrapper for BERT-like models to get calibrated probabilities."""
    model.eval()
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        return decode_predictions(logits, temperature=temperature)

def aggregate_chunk_predictions(chunk_results, threshold_strong=0.75):
    """
    Aggregates results from multiple chunks of a long article.
    
    Args:
        chunk_results (list): List of dicts from decode_predictions
        threshold_strong (float): If any chunk exceeds this fake probability, 
                                 mark as fake (sensitive to short-lived lies).
    """
    if not chunk_results:
        return None
        
    fake_probs = [r['fake_probability'] for r in chunk_results]
    real_probs = [r['real_probability'] for r in chunk_results]
    
    max_fake = max(fake_probs)
    avg_fake = sum(fake_probs) / len(fake_probs)
    avg_real = sum(real_probs) / len(real_probs)
    
    # Refined decision rule:
    # 1. If any chunk is EXTREMELY suspicious, it's fake.
    # 2. Otherwise, use the average probability.
    if max_fake >= threshold_strong:
        final_label = "FAKE"
        final_confidence = max_fake
    else:
        if avg_fake > 0.5:
            final_label = "FAKE"
            final_confidence = avg_fake
        else:
            final_label = "REAL"
            final_confidence = avg_real
            
    return {
        'predicted_label': final_label,
        'confidence': final_confidence,
        'avg_fake_prob': avg_fake,
        'avg_real_prob': avg_real,
        'max_fake_prob': max_fake,
        'num_chunks': len(chunk_results)
    }

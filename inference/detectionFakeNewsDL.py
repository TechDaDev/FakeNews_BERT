import joblib
from keras.models import load_model
from keras.utils import pad_sequences
import os
import argparse
import sys
import re

from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
save_dir = root_dir / 'saved_models'

DEFAULT_LSTM_MODEL = save_dir / "lstm_model1.h5"
DEFAULT_TOKENIZER = save_dir / "tokenizerDL1.pkl"
DEFAULT_VECTORIZER = save_dir / "tfidf_vectorizer.pkl"


def find_latest(pattern, search_dir=None):
    if search_dir is None:
        search_dir = save_dir
    if not os.path.isdir(search_dir):
        return None
    matches = [f for f in os.listdir(search_dir) if re.fullmatch(pattern, f)]
    if not matches:
        return None
    full_matches = [os.path.join(search_dir, f) for f in matches]
    return max(full_matches, key=os.path.getmtime)


def load_keras_model(model_path, tokenizer_path):
    model = load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    return model, tokenizer


def load_ml_model(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_bert(text, model, tokenizer, device, threshold=0.5, positive_label='Fake', negative_label='Real'):
    import torch
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        prob_fake = probs[0][0].item()  # Assuming class 0 is Fake in the BERT training script logic
    
    label = positive_label if prob_fake >= threshold else negative_label
    return label, prob_fake


def predict_keras(text, model, tokenizer, maxlen=500, threshold=0.5, positive_label='Fake', negative_label='Real'):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=maxlen)
    prob = model.predict(pad, verbose=0)[0][0] if model.output_shape[-1] == 1 else model.predict(pad, verbose=0)[0]
    label = positive_label if prob >= threshold else negative_label
    return label, float(prob)


def predict_ml(text, model, vectorizer, positive_label='Fake', negative_label='Real'):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    # For probability if supported
    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(X)[0][1]
        except Exception:
            prob = None
    if hasattr(model, 'is_torch_model') and prob is None:
        # torch wrapper gives logits via decision boundary; reuse threshold 0.5 approximation using raw predict
        prob = None
    label = positive_label if pred == 1 else negative_label
    return label, (float(prob) if prob is not None else None)


def configure_device(force_cpu):
    if force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("⚙️  Forcing CPU execution (CUDA disabled).")
    else:
        # Let TF / Torch decide; minimal diagnostic
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"🚀 TensorFlow GPU detected: {[g.name for g in gpus]}")
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                print(f"🚀 PyTorch GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser(description="Fake News Detection Inference")
    p.add_argument('--model', type=str, default=None, help='Path to trained model (.h5 or .pkl).')
    p.add_argument('--tokenizer', type=str, default=DEFAULT_TOKENIZER, help='Tokenizer path for Keras model.')
    p.add_argument('--vectorizer', type=str, default=DEFAULT_VECTORIZER, help='TF-IDF vectorizer path for ML models.')
    p.add_argument('--maxlen', type=int, default=500, help='Sequence max length for LSTM.')
    p.add_argument('--threshold', type=float, default=0.5, help='Decision threshold for sigmoid output.')
    p.add_argument('--positive_label', type=str, default='Fake', help='Label for positive class (prob >= threshold or class 1).')
    p.add_argument('--negative_label', type=str, default='Real', help='Label for negative class.')
    p.add_argument('--cpu', action='store_true', help='Force CPU usage.')
    p.add_argument('--text', type=str, default=None, help='News text to classify (if omitted, will prompt).')
    p.add_argument('--auto', action='store_true', help='Auto-detect latest model if --model not provided.')
    return p.parse_args()


def auto_select_model():
    # Prefer Keras .h5 if exists, else newest .pkl
    keras = find_latest(r".*\.h5")
    pkl = find_latest(r".*\.pkl")
    if keras:
        return keras
    return pkl


def main():
    args = parse_args()
    configure_device(args.cpu)

    model_path = args.model
    if not model_path and args.auto:
        model_path = auto_select_model()
        if model_path:
            print(f"🔎 Auto-selected model: {model_path}")
    if not model_path:
        print("❌ No model specified. Use --model path/to/model.(h5|pkl) or --auto")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

    ext = os.path.splitext(model_path)[1].lower()

    if ext == '.h5':
        if not os.path.exists(args.tokenizer):
            print(f"❌ Tokenizer not found: {args.tokenizer}")
            sys.exit(1)
        print(f"📥 Loading Keras model: {model_path}")
        model, tokenizer = load_keras_model(model_path, args.tokenizer)
        inference_fn = lambda text: predict_keras(text, model, tokenizer, maxlen=args.maxlen, threshold=args.threshold, positive_label=args.positive_label, negative_label=args.negative_label)
    elif ext == '.pkl':
        if not os.path.exists(args.vectorizer):
            print(f"❌ Vectorizer not found: {args.vectorizer}")
            sys.exit(1)
        print(f"📥 Loading ML model: {model_path}")
        model, vectorizer = load_ml_model(model_path, args.vectorizer)
        inference_fn = lambda text: predict_ml(text, model, vectorizer, positive_label=args.positive_label, negative_label=args.negative_label)
    elif os.path.isdir(model_path):
        # Check if it's a BERT directory
        if os.path.exists(os.path.join(model_path, 'config.json')):
            print(f"📥 Loading BERT model: {model_path}")
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model.eval()
            inference_fn = lambda text: predict_bert(text, model, tokenizer, device, threshold=args.threshold, positive_label=args.positive_label, negative_label=args.negative_label)
        else:
            print(f"❌ Directory {model_path} is not a valid BERT model.")
            sys.exit(1)
    else:
        print(f"❌ Unsupported model: {model_path}. Use .h5, .pkl, or a BERT directory.")
        sys.exit(1)

    # Acquire text
    if args.text:
        input_text = args.text.strip()
    else:
        input_text = input("Enter news text: ").strip()

    if not input_text:
        print("❌ Empty text provided.")
        sys.exit(1)

    label, prob = inference_fn(input_text)
    if prob is not None:
        print(f"✅ Prediction: {label} (prob={prob:.4f})")
    else:
        print(f"✅ Prediction: {label}")


if __name__ == "__main__":
    main()

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import time
import requests
import numpy as np
from src.article_extraction import extract_article
from src.prediction import (
    decode_predictions, 
    aggregate_chunk_predictions, 
    LABEL_REAL, 
    LABEL_FAKE,
    LABEL_STR_MAP
)
from joblib import load
from src.predict_utils import predict_with_model, compute_token_stats
from pathlib import Path

# --- Constants & Paths ---
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_BERT_PATH = ROOT_DIR / "saved_models" / "model_runs" / "BERT_20260128_115016"
SKLEARN_PATH = ROOT_DIR / "saved_models" / "model_runs" / "20260128_221955" / "LinearSVC_20260128_221955.pkl"
VECTORIZER_PATH = ROOT_DIR / "saved_models" / "tfidf_vectorizer.pkl"


def resolve_bert_path() -> Path:
    env_path = os.getenv("BERT_PATH", "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(DEFAULT_BERT_PATH)

    runs_dir = ROOT_DIR / "saved_models" / "model_runs"
    if runs_dir.exists():
        candidates.extend(sorted(runs_dir.glob("BERT_*"), reverse=True))

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and (resolved / "config.json").exists():
            return resolved

    return DEFAULT_BERT_PATH


BERT_PATH = resolve_bert_path()

# --- Page Configuration ---
st.set_page_config(
    page_title="VerifyAI | Fake News Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Premium Custom Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Hero Header */
    .hero-container {
        text-align: center;
        padding: 40px 20px;
        margin-bottom: 30px;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 10px;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #8892b0;
        font-weight: 300;
        margin-bottom: 20px;
    }
    
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    /* Main Card Container */
    .analysis-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 40px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 25px;
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e6f1ff;
        margin: 0;
    }
    
    .card-desc {
        font-size: 0.9rem;
        color: #8892b0;
        margin: 0;
    }
    
    /* Input Styling */
    .stTextArea textarea, .stTextInput input {
        background: #ffffff !important;
        color: #000000 !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        transition: all 0.3s ease !important;
        font-weight: 400 !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0px 0px;
        color: #8892b0;
        font-weight: 600;
        padding: 0px 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(102, 126, 234, 0.1) !important;
        color: #e6f1ff !important;
        border-bottom: 2px solid #667eea !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 16px 48px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Result Cards */
    .result-container {
        margin-top: 30px;
        animation: fadeInUp 0.5s ease;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .result-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        border: 2px solid;
    }
    
    .result-real {
        border-color: #10b981;
        box-shadow: 0 0 40px rgba(16, 185, 129, 0.2);
    }
    
    .result-fake {
        border-color: #ef4444;
        box-shadow: 0 0 40px rgba(239, 68, 68, 0.2);
    }
    
    .result-icon {
        font-size: 4rem;
        margin-bottom: 15px;
    }
    
    .result-label {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 10px;
        letter-spacing: 2px;
    }
    
    .result-real .result-label {
        color: #10b981;
    }
    
    .result-fake .result-label {
        color: #ef4444;
    }
    
    .confidence-container {
        margin-top: 20px;
    }
    
    .confidence-label {
        font-size: 0.9rem;
        color: #8892b0;
        margin-bottom: 8px;
    }
    
    .confidence-bar {
        height: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    .confidence-fill-real {
        background: linear-gradient(90deg, #10b981, #34d399);
    }
    
    .confidence-fill-fake {
        background: linear-gradient(90deg, #ef4444, #f87171);
    }
    
    .confidence-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-top: 10px;
    }
    
    .result-real .confidence-value {
        color: #10b981;
    }
    
    .result-fake .confidence-value {
        color: #ef4444;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-top: 30px;
    }
    
    .stat-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #8892b0;
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Token Details */
    .token-expander {
        margin-top: 30px;
    }
    
    .token-box {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 20px;
        font-family: 'Fira Code', 'Courier New', monospace;
        font-size: 0.85rem;
        color: #a5b4fc;
        max-height: 200px;
        overflow-y: auto;
        word-wrap: break-word;
    }
    
    /* Feature Cards */
    .features-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin-top: 40px;
        padding: 20px 0;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        background: rgba(255, 255, 255, 0.05);
        transform: translateY(-5px);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 15px;
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e6f1ff;
        margin-bottom: 8px;
    }
    
    .feature-desc {
        font-size: 0.85rem;
        color: #8892b0;
        line-height: 1.5;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 40px 20px;
        margin-top: 60px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .footer-text {
        color: #4a5568;
        font-size: 0.85rem;
    }
    
    .footer-link {
        color: #667eea;
        text-decoration: none;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Spinner */
    .stSpinner > div {
        border-color: #667eea !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        color: #ccd6f6 !important;
    }
    
    /* Article Title Display */
    .article-title-container {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        padding: 30px;
        margin: 20px 0;
    }
    
    .article-title-label {
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 15px;
        font-weight: 600;
    }
    
    .article-title-text {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e6f1ff;
        line-height: 1.4;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Temperature Scaling Placeholder ---
TEMPERATURE = 1.0 # Can be adjusted for calibration

# --- Model Asset Loading ---
@st.cache_resource
def load_bert_assets():
    if not BERT_PATH.exists():
        return None, None, None, None
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_PATH)
    
    # Determine model max length
    try:
        model_max_len = tokenizer.model_max_length
    except Exception:
        try:
            model_max_len = model.config.max_position_embeddings
        except Exception:
            model_max_len = 512

    # Use global device preference if available
    try:
        from src.device import get_device
        device = get_device()
    except Exception:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()
    return tokenizer, model, device, model_max_len

@st.cache_resource
def load_sklearn_assets():
    if not os.path.exists(SKLEARN_PATH):
        return None, None
    model = load(SKLEARN_PATH)
    vectorizer = load(VECTORIZER_PATH)
    return model, vectorizer

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("### 🛠️ Model Configuration")
    model_choice = st.radio(
        "Choose Detection Engine:",
        ["BERT (Deep Context)", "LinearSVC (Statistical)"],
        index=0,
        help="BERT uses transformer technology for deep semantic analysis. LinearSVC uses statistical word frequencies (TF-IDF)."
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Details")
    if model_choice == "BERT (Deep Context)":
        st.info(f"**Model:** DistilBERT\n\n**Path:** {BERT_PATH.name}\n\n**Best For:** Nuanced context and long-form articles.")
    else:
        st.info("**Model:** LinearSVC\n\n**Training:** 2026-01-28\n\n**Best For:** Fast analysis and clear-cut patterns.")

# Load appropriate assets based on choice
if model_choice == "BERT (Deep Context)":
    tokenizer_bert, model_bert, device, model_max_len = load_bert_assets()
    model_sklearn, vectorizer_sklearn = None, None
else:
    tokenizer_bert, model_bert, device, model_max_len = None, None, None, None
    model_sklearn, vectorizer_sklearn = load_sklearn_assets()

# --- Hero Section ---
st.markdown(f"""
<div class="hero-container">
    <div class="hero-badge">🤖 Powered by {"BERT" if model_choice == "BERT (Deep Context)" else "LinearSVC"}</div>
    <h1 class="hero-title">VerifyAI</h1>
    <p class="hero-subtitle">Advanced fake news detection using {"state-of-the-art transformer" if model_choice == "BERT (Deep Context)" else "high-performance statistical"} technology</p>
</div>
""", unsafe_allow_html=True)

if model_choice == "BERT (Deep Context)" and (not tokenizer_bert or not model_bert):
    st.error(f"⚠️ BERT Model not found at `{BERT_PATH}`. Please check the path.")
    st.stop()
if model_choice == "LinearSVC (Statistical)" and (not model_sklearn or not vectorizer_sklearn):
    st.error(f"⚠️ Sklearn Model/Vectorizer not found at `{SKLEARN_PATH}`. Please check the path.")
    st.stop()

# --- Main Analysis Section ---
col_left, col_center, col_right = st.columns([1, 3, 1])

with col_center:
    st.markdown("""
    <div class="analysis-card">
        <div class="card-header">
            <div class="card-icon">🔍</div>
            <div>
                <p class="card-title">News Verification</p>
                <p class="card-desc">Choose your input method to verify credibility</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["📝 Manual Text", "🔗 Article Link"])
    
    with tabs[0]:
        input_text = st.text_area(
            label="Input Text",
            height=200,
            placeholder="Paste the news article or statement you want to verify...",
            label_visibility="collapsed",
            key="manual_text"
        )
        
    with tabs[1]:
        input_url = st.text_input(
            label="Article Link",
            placeholder="Paste a news article URL (e.g., https://bbc.com/news/...) ",
            label_visibility="collapsed",
            key="url_input"
        )
        
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        debug_mode = st.checkbox("🛠️ Debug Mode", key="debug_mode")
    with col2:
        detect_btn = st.button("🚀 Run Deep Analysis", use_container_width=True)

# --- Analysis Logic ---
if detect_btn:
    content_to_analyze = ""
    scraped_title = ""
    source_info = "Manual Input"
    extraction_results = None
    
    # Determine which input method has content
    is_url = bool(input_url.strip())
    has_manual_text = bool(input_text.strip())
    
    if is_url:
        with st.spinner("🌐 Performing robust article extraction..."):
            extraction_results = extract_article(input_url)
            scraped_title = extraction_results['title']
            content_to_analyze = extraction_results['clean_text']
            source_info = f"URL: {extraction_results['source_domain']}"
            
            if extraction_results['extraction_method'] == 'failed':
                st.error(f"❌ Extraction failed. Please check the URL or paste text manually.")
                if extraction_results['warnings']:
                    st.warning("\n".join(extraction_results['warnings']))
                content_to_analyze = ""
            
            # Show extraction warnings if any
            if extraction_results['warnings']:
                with st.expander("⚠️ Extraction Warnings"):
                    for w in extraction_results['warnings']:
                        st.write(f"- {w}")
            
            # Show extracted text in expander
            if content_to_analyze:
                with st.expander("📄 View Extracted Article Text"):
                    st.write(f"**Method:** {extraction_results['extraction_method']}")
                    st.write("---")
                    st.write(content_to_analyze)
                    
                # Block if too short
                if len(content_to_analyze) < 200:
                    st.error("🛑 Extracted content is too short for reliable classification. The page might be protected or have minimal text.")
                    content_to_analyze = ""

    elif has_manual_text:
        content_to_analyze = input_text.strip()
    else:
        st.warning("⚠️ Please provide either text or a URL to analyze.")

    if content_to_analyze:
        # 1. Stats Calculation
        if model_choice == "BERT (Deep Context)":
            tokens = tokenizer_bert.tokenize(content_to_analyze)
            num_tokens = len(tokens)
        else:
            stats = compute_token_stats(vectorizer_sklearn, content_to_analyze)
            num_tokens = stats['token_count']
        
        # Terminal Output
        print("\n" + "═" * 70)
        print(f"🔍 NEW ANALYSIS REQUEST ({model_choice})")
        print("═" * 70)
        print(f"⏰ Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌍 Source: {source_info}")
        print(f"📏 Input Length: {len(content_to_analyze)} characters")
        print(f"🔢 Token Count: {num_tokens}")
        print("─" * 70 + "\n")
        
        # 2. Prediction
        with col_center:
            with st.spinner(f"🧠 Analysis in progress using {model_choice}..."):
                time.sleep(0.5) 
                
                chunk_results = []
                
                if model_choice == "BERT (Deep Context)":
                    # Adaptive chunking logic for BERT
                    token_count = len(tokens)
                    if token_count <= model_max_len:
                        chunk_size = token_count
                        chunk_overlap = 0
                    else:
                        chunk_size = max(min(model_max_len - 2, 256), 64)
                        chunk_overlap = max(10, int(chunk_size * 0.2))
                    step = max(1, chunk_size - chunk_overlap)
                    
                    print("\n" + "─" * 70)
                    print(f"🧩 CHUNK ANALYSIS (Size: {chunk_size}, Overlap: {chunk_overlap})")
                    print("─" * 70)
                    
                    for i in range(0, len(tokens), step):
                        chunk_tokens = tokens[i:i + chunk_size]
                        if not chunk_tokens: break
                        chunk_text = tokenizer_bert.convert_tokens_to_string(chunk_tokens)
                        
                        inputs_chunk = tokenizer_bert(chunk_text, return_tensors="pt", truncation=True, max_length=model_max_len, padding="max_length")
                        inputs_chunk = {k: v.to(device) for k, v in inputs_chunk.items()}
                        
                        with torch.no_grad():
                            outputs_chunk = model_bert(**inputs_chunk)
                            chunk_res = decode_predictions(outputs_chunk.logits, temperature=TEMPERATURE, debug=debug_mode)
                            
                        chunk_results.append({
                            'fake_prob': chunk_res['fake_probability'],
                            'real_prob': chunk_res['real_probability'],
                            'label': chunk_res['predicted_label'],
                            'text': chunk_text,
                            'res_dict': chunk_res
                        })
                        print(f"Chunk {len(chunk_results):02d} | Label: {chunk_res['predicted_label']:<4} | P(Fake): {chunk_res['fake_probability']:.4f}")
                    
                    print("─" * 70 + "\n")
                    agg_res = aggregate_chunk_predictions(chunk_results, threshold_strong=0.75)
                    prediction = LABEL_FAKE if agg_res['predicted_label'] == "FAKE" else LABEL_REAL
                    confidence = agg_res['confidence']
                    fake_prob = agg_res['avg_fake_prob']
                    real_prob = agg_res['avg_real_prob']
                    num_display = len(chunk_results)
                    aggregation_label = f"aggregated chunk-level analysis ({num_display} chunks)"
                
                else:
                    # LinearSVC prediction
                    _, fake_prob = predict_with_model(
                        SKLEARN_PATH, content_to_analyze, return_score=True, 
                        vectorizer_path=VECTORIZER_PATH
                    )
                    real_prob = 1.0 - fake_prob
                    prediction = LABEL_FAKE if fake_prob > 0.5 else LABEL_REAL
                    confidence = fake_prob if prediction == LABEL_FAKE else real_prob
                    aggregation_label = "statistical feature analysis"
                    num_display = 1

                # Mapping Logic for UI
                is_fake = prediction == LABEL_FAKE
                result_class = "result-fake" if is_fake else "result-real"
                label = "FAKE NEWS DETECTED" if is_fake else "VERIFIED AUTHENTIC"
                icon = "🚨" if is_fake else "✅"
                fill_class = "confidence-fill-fake" if is_fake else "confidence-fill-real"
                
                # Display Result Card
                st.markdown(f"""
                <div class="result-container">
                    <div class="result-card {result_class}">
                        <div class="result-icon">{icon}</div>
                        <div class="result-label">{label}</div>
                        <p style="color: #8892b0; font-size: 0.9rem; margin-top: -10px;">Based on {aggregation_label}</p>
                        <div class="confidence-container">
                            <div class="confidence-label">Model Confidence Score</div>
                            <div class="confidence-bar">
                                <div class="confidence-fill {fill_class}" style="width: {confidence * 100}%"></div>
                            </div>
                            <div class="confidence-value">{confidence:.1%}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Stats Grid
                st.markdown(f"""
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{num_display if model_choice == "BERT (Deep Context)" else num_tokens}</div>
                        <div class="stat-label">{"Chunks" if model_choice == "BERT (Deep Context)" else "Tokens"}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{real_prob:.1%}</div>
                        <div class="stat-label">Avg Real Prob</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{fake_prob:.1%}</div>
                        <div class="stat-label">Avg Fake Prob</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if is_url and scraped_title:
                    st.markdown(f"""
                    <div class="article-title-container">
                        <div class="article-title-label">📰 Article Title</div>
                        <p class="article-title-text">{scraped_title}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Debug Mode Details
                if debug_mode:
                    st.divider()
                    st.subheader("🛠️ Model Debug Details")
                    st.write(f"**Canonical Mapping:** 0=REAL, 1=FAKE")
                    if model_choice == "BERT (Deep Context)":
                        st.write(f"**Temperature:** {TEMPERATURE}")
                        st.write(f"**Threshold Strong:** 0.75")
                        st.write("#### Chunk-level Raw Output")
                        for i, cr in enumerate(chunk_results):
                            with st.expander(f"Chunk {i+1} Details"):
                                st.json(cr['res_dict'])
                                st.write("**Snippet:** " + cr['text'][:300] + "...")
                    else:
                        st.write("**Model Type:** LinearSVC")
                        st.write(f"**Raw Prob(Fake):** {fake_prob:.4f}")

# --- Features Section ---
st.markdown("""
<div class="features-container">
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">BERT Technology</div>
        <div class="feature-desc">Utilizes DistilBERT transformer architecture for deep contextual understanding</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🌐</div>
        <div class="feature-title">URL Support</div>
        <div class="feature-desc">Analyzes news directly from links using advanced web scraping</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Multi-Model Engine</div>
        <div class="feature-desc">Switch between BERT transformer and LinearSVC statistical models in the sidebar</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="footer">
    <p class="footer-text">
        Built with ❤️ using Streamlit & Hugging Face Transformers<br>
        Model: DistilBERT (Contextual) & LinearSVC (Statistical)
    </p>
</div>
""", unsafe_allow_html=True)

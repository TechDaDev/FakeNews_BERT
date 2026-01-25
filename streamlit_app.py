import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="VerifyAI | Fake News Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
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
    
    .card-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
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
    
    /* Text Area Styling */
    .stTextArea textarea {
        background: #ffffff !important;
        color: #000000 !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        transition: all 0.3s ease !important;
        font-weight: 400 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #6b7280 !important;
        opacity: 0.7 !important;
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
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
MODEL_PATH = "/home/zeus3000/Downloads/Fake_News_Detection/saved_models/model_runs/BERT_20260125_212125"

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH):
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_assets()

# --- Hero Section ---
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🤖 Powered by BERT</div>
    <h1 class="hero-title">VerifyAI</h1>
    <p class="hero-subtitle">Advanced fake news detection using state-of-the-art transformer technology</p>
</div>
""", unsafe_allow_html=True)

if not tokenizer or not model:
    st.error(f"⚠️ Model not found at `{MODEL_PATH}`. Please ensure the BERT model is trained and saved.")
    st.stop()

# --- Main Analysis Section ---
col_left, col_center, col_right = st.columns([1, 3, 1])

with col_center:
    st.markdown("""
    <div class="analysis-card">
        <div class="card-header">
            <div class="card-icon">📝</div>
            <div>
                <p class="card-title">News Analysis</p>
                <p class="card-desc">Paste your article or statement below for verification</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    input_text = st.text_area(
        label="Input Text",
        height=200,
        placeholder="Enter the news article, headline, or statement you want to verify...",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        detect_btn = st.button("🔍 Analyze Content", use_container_width=True)

# --- Analysis Logic ---
if detect_btn:
    if input_text.strip():
        # 1. Tokenization & Terminal Summary
        tokens = tokenizer.tokenize(input_text)
        
        # Terminal Output (Requested Format)
        print("\n" + "═" * 70)
        print("🔍 NEW ANALYSIS REQUEST")
        print("═" * 70)
        print(f"⏰ Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📏 Input Length: {len(input_text)} characters")
        print(f"🔢 Token Count: {len(tokens)}")
        print("─" * 70 + "\n")
        
        # 2. Prediction
        with col_center:
            with st.spinner("🧠 Analyzing with BERT..."):
                time.sleep(0.5)  # Small delay for visual effect
                
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    prediction = torch.argmax(logits, dim=1).item()
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    confidence = probs[0][prediction].item()
                    real_prob = probs[0][0].item()
                    fake_prob = probs[0][1].item()
                
                # Mapping: 0=Real, 1=Fake
                is_fake = prediction == 1
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
                        <div class="confidence-container">
                            <div class="confidence-label">Confidence Score</div>
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
                        <div class="stat-value">{len(tokens)}</div>
                        <div class="stat-label">Tokens Analyzed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{real_prob:.1%}</div>
                        <div class="stat-label">Real Probability</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{fake_prob:.1%}</div>
                        <div class="stat-label">Fake Probability</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Token Details Expander
                with st.expander("🔬 View Technical Details"):
                    st.markdown(f"""
                    <div class="token-box">
                        <strong>Token Count:</strong> {len(tokens)}<br><br>
                        <strong>Tokens:</strong><br>
                        {', '.join(tokens[:100])}{'...' if len(tokens) > 100 else ''}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        with col_center:
            st.warning("⚠️ Please enter some text to analyze.")

# --- Features Section ---
st.markdown("""
<div class="features-container">
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">BERT Technology</div>
        <div class="feature-desc">Utilizes DistilBERT transformer architecture for deep contextual understanding</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <div class="feature-title">Real-time Analysis</div>
        <div class="feature-desc">Get instant results with our optimized inference pipeline</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Detailed Metrics</div>
        <div class="feature-desc">View confidence scores, probabilities, and token breakdowns</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="footer">
    <p class="footer-text">
        Built with ❤️ using Streamlit & Hugging Face Transformers<br>
        Model: DistilBERT fine-tuned on WELFake Dataset
    </p>
</div>
""", unsafe_allow_html=True)

# 🔍 VerifyAI - Fake News Detection System

A comprehensive fake news detection platform powered by state-of-the-art BERT transformers, featuring an interactive Streamlit web application and multiple ML/DL models for research and production deployment.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53+-red.svg)
![BERT](https://img.shields.io/badge/Model-DistilBERT-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **🎨 Premium Streamlit Web App**: Modern, glassmorphic UI with real-time fake news detection
- **🌐 Robust Article Extraction**: Intelligent scraping using `trafilatura` and `readability` to eliminate ads and boilerplate noise
- **🧠 BERT-Powered Analysis**: Fine-tuned DistilBERT model for deep contextual understanding with temperature scaling calibration
- **📊 Multiple Model Support**: Traditional ML (LinearSVC, Random Forest), Deep Learning (LSTM), and Transformers
- **⚡ Real-time Inference**: Instant predictions with confidence scores and probability breakdowns
- **🔬 Technical Insights**: Token analysis, chunk-level probability distributions, and detailed debug logs
- **📈 Comprehensive Training Pipeline**: Organized scripts for training, evaluation, and model comparison

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
cd Fake_News_Detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Streamlit App

Launch the interactive web application:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Features of the Streamlit App

- **Paste & Analyze**: Simply paste any news article or statement
- **Verify URLs**: Input a news link; the app robustly extracts the article body, bypassing ads and cookie notices
- **Instant Results**: Get real-time predictions (REAL vs FAKE)
- **Confidence Metrics**: View model confidence scores and probability distributions
- **Technical Breakdown**: View extracted text segments and chunk-level confidence scores
- **Terminal Logging**: Detailed analysis logs printed to terminal for debugging

## 📁 Project Structure

```text
Fake_News_Detection/
├── streamlit_app.py         # 🎨 Main Streamlit web application
├── data/                    # 📊 Dataset files (WELFake, combined_TF_data)
├── src/                     # 🔧 Core logic and shared utilities
│   ├── models.py           # ML Model trainers
│   ├── bert_utils.py       # BERT dataset and model helpers
│   ├── predict_utils.py    # Common prediction logic
│   └── reporting.py        # Evaluation reporting
├── scripts/                 # 🛠️ Execution scripts
│   ├── training/           # Model training entry points
│   │   ├── multi_algorithm_training.py
│   │   ├── bert_training.py
│   │   └── FakeNewsDLTraining.py
│   ├── analysis/           # EDA and data verification
│   │   ├── dataset_analysis.py
│   │   ├── check_data.py
│   │   └── check_labels.py
│   └── data_prep.py        # Data cleaning and TF-IDF vectorization
├── inference/               # 🔮 Production-ready inference scripts
│   ├── inference.py        # Interactive CLI for ML models
│   └── detectionFakeNewsDL.py # Unified detection script
├── saved_models/            # 💾 Stored artifacts
│   ├── tfidf_vectorizer.pkl
│   └── model_runs/          # Historical training results
│       └── BERT_20260125_212125/  # Latest BERT model
├── requirements.txt         # 📦 Project dependencies
└── PUBLICATION_TECHNICAL_DETAILS.md # 📄 Methodology for publication
```

## 🎯 Usage

### 1. Web Application (Recommended)

The easiest way to use the system:

```bash
streamlit run streamlit_app.py
```

**Features:**
- Paste news articles directly into the interface
- Get instant REAL/FAKE predictions
- View confidence scores and token statistics
- Beautiful, modern UI with dark mode

### 2. Training Models

#### Train BERT Model
```bash
python scripts/training/bert_training.py
```

#### Train Traditional ML Models
```bash
python scripts/training/multi_algorithm_training.py
```

#### Train LSTM Model
```bash
python scripts/training/FakeNewsDLTraining.py
```

### 3. Command-Line Inference

For batch processing or scripting:

```bash
# Interactive CLI
python inference/inference.py

# Automated detection
python inference/detectionFakeNewsDL.py --auto
```

### 4. Dataset Analysis

Explore the dataset statistics:

```bash
python scripts/analysis/dataset_analysis.py
```

## 🧠 Model Information

### BERT Model
- **Architecture**: DistilBERT (distilbert-base-uncased)
- **Training Dataset**: WELFake Dataset
- **Max Sequence Length**: 512 tokens
- **Batch Size**: 16
- **Epochs**: 3
- **Optimizer**: AdamW with linear warmup

### Label Mapping (Canonical)
- `0` = **REAL NEWS** ✅
- `1` = **FAKE NEWS** 🚨

*Note: This mapping is enforced across the entire pipeline from training to Streamlit display.*

## 📊 Performance

The BERT model achieves high accuracy on the WELFake dataset. Detailed metrics are available in:
- `saved_models/model_runs/BERT_*/results.json`
- `PUBLICATION_TECHNICAL_DETAILS.md`

## 🛠️ Technical Stack

- **Frontend**: Streamlit with custom CSS (glassmorphism design)
- **ML Framework**: PyTorch, Transformers (Hugging Face)
- **Traditional ML**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib
- **Model Storage**: joblib, safetensors

## 📝 Terminal Output

When using the Streamlit app, each analysis prints a summary to the terminal:

```
══════════════════════════════════════════════════════════════════════
🔍 NEW ANALYSIS REQUEST
══════════════════════════════════════════════════════════════════════
⏰ Timestamp: 2026-01-25 22:41:59
📏 Input Length: 10173 characters
🔢 Token Count: 2131
──────────────────────────────────────────────────────────────────────
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🔗 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ using Streamlit, PyTorch, and Hugging Face Transformers**

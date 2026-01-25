# Technical Appendix for Publication (Response to Reviewer Comments)

This document summarizes the technical details of the Fake News Detection pipeline to assist in writing the Methodology and Results sections of your paper.

## 1. BERT Comparison (Computational Arguments)
We have added a modern baseline using **DistilBERT**. 
- **Lightweight vs. Heavyweight**: While DistilBERT achieves high accuracy (~98%+), it requires significant computational resources (GPU required) and takes exponentially longer to train (~15-20 minutes on GPU vs <2 seconds for LinearSVC on CPU).
- **Inference Speed**: LinearSVC inference is nearly instantaneous, making it suitable for "real-time systems" and "limited resource" applications as per the reviewer's suggestion.

## 2. Preprocessing & Feature Engineering
- **Tokenization**: Word-level tokenization using Scikit-Learn’s `TfidfVectorizer`.
- **Normalization**: Text was lowercased, and English stop words were removed.
- **Stemming/Lemmatization**: **Not used**. This decision was made to maintain the integrity of the original word forms, which can sometimes carry sentiment or intent relevant to fake news detection, and to minimize preprocessing latency.
- **N-gram Size**: Unigrams `(1, 1)` were used. This ensures a more compact feature space, aligning with the "lightweight" goal.
- **Feature Pruning**: 
    - `max_df=0.7`: Terms appearing in more than 70% of documents were excluded (corpus-specific stop words).
    - `min_df=1`: All terms appearing at least once after cleaning were kept (unless specified in EDA).

## 3. Model Hyperparameters (Reproducibility Table)

| Model | Hyperparameter | Value | Rationale |
| :--- | :--- | :--- | :--- |
| **LinearSVC** | `C` (Regularization) | `1.0` | Standard balance between margin and error. |
| **LinearSVC** | `loss` | `squared_hinge` | Standard loss for Linear Support Vector Machines. |
| **LinearSVC** | `dual` | `False` | Preferred for `n_samples > n_features`. |
| **Logistic Regression** | `max_iter` | `1000` | Ensures convergence on high-dimensional data. |
| **Random Forest** | `n_estimators` | `200` | Balanced tree ensemble for non-linear patterns. |
| **DistilBERT** | `Learning Rate` | `2e-5` | Recommended rate for Transformer fine-tuning. |

## 4. Data Segmentation & Reproducibility
- **Split Method**: Single Train/Test Split (80% Train, 20% Test).
- **Stratification**: Enabled (Ensures class distribution is identical in both sets).
- **Reproducibility Seed**: `random_state=42` used consistently across all data splits and model initializations.

## 5. Standardized Model Naming
To ensure consistency across the paper, use the following names:
1.  **LinearSVC** (instead of Linear SVM or LinearSV)
2.  **Logistic Regression**
3.  **Naive Bayes**
4.  **Random Forest**
5.  **LSTM** (for the Recurrent Neural Network)
6.  **DistilBERT** (for the Transformer baseline)

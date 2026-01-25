# Response to Reviewer Comments

This document contains the formalized technical answers based on the experimental results from experiments conducted on **2026-01-25** (Run IDs: `20260125_212432` and `BERT_20260125_212125`).

## 1. Baseline Comparison (BERT)
We have added a modern transformer baseline using **DistilBERT** to the study. 
*   **Performance vs. Cost**: While DistilBERT achieved the highest accuracy of **99.20%**, it required a CUDA-enabled GPU and significant training time (~21 minutes in our environment). 
*   **Efficiency**: In contrast, **LinearSVC** achieved a highly competitive accuracy of **95.45%** with a training time of only **2.08 seconds** on a standard CPU.
*   **Recommendation**: For real-time systems or applications with limited computational resources, the marginal 3.75% gain in accuracy by BERT does not justify the ~600x increase in computational cost. This supports our emphasis on lightweight models like LinearSVC.

## 2. Processing Steps Summary
*   **Tokenization**: Word-level tokenization was performed using the standard Scikit-Learn tokenizer (splitting by whitespace and non-alphanumeric characters).
*   **Stemming/Lemmatization**: Neither was used. This maintains the original word integrity and reduces the preprocessing latency overhead.
*   **N-gram Size**: Unigrams **(1, 1)** were utilized within the TF-IDF vectorization to maintain a compact feature space.
*   **Feature Pruning**:
    *   `max_df=0.7`: Pruned terms appearing in more than 70% of the documents to remove corpus-specific stop words.
    *   `min_df=1`: Retained all terms appearing at least once after initial cleaning.

## 3. Model Hyperparameters (Reproducibility Table)

| Model | Hyperparameter | Value | Rationale |
| :--- | :--- | :--- | :--- |
| **LinearSVC** | Regularization (`C`) | `1.0` | Standard margin penalty. |
| **LinearSVC** | Loss Function | `squared_hinge` | Robustness for high-dimensional TF-IDF space. |
| **LinearSVC** | Dual Formulation | `False` | Optimized for `n_samples > n_features`. |
| **Logistic Regression** | `max_iter` | `1000` | Ensures convergence on sparse features. |
| **Random Forest** | `n_estimators` | `200` | Ensemble size for variance reduction. |
| **DistilBERT** | Learning Rate | `2e-5` | Recommended for transformer fine-tuning. |

## 4. Performance Summary (Consolidated)

*Note: All "Linear SVM" or "Linear SV" references have been standardized to **LinearSVC**.*

| Model | Accuracy | Training Time (s) | Implementation |
| :--- | :--- | :--- | :--- |
| **LinearSVC** | **95.45%** | **2.08** | Scikit-Learn |
| **Logistic Regression** | 94.18% | 2.46 | Scikit-Learn |
| **Random Forest** | 90.75% | 65.42 | Scikit-Learn |
| **Naive Bayes** | 84.22% | 0.04 | Scikit-Learn |
| **DistilBERT** | **99.20%** | ~1260 | Transformers (Torch) |

## 5. Methodology & Reproducibility
*   **Data Segmentation**: We used a **stratified hold-out split** (80% Train, 20% Test). This ensures that both training and validation sets contain the same proportion of "Real" vs "Fake" labels (Original distribution: ~55% Real, ~45% Fake).
*   **Global Seed**: A random seed of **`42`** was used across all data partitioning and model initialization (LinearSVC, Random Forest, etc.) to ensure 100% reproducibility of the results presented.

## 6. Revised Contributions
*   **Efficiency**: A lightweight and interpretable classification framework suitable for resource-limited applications.
*   **Novel Perspective**: Framing fake news as a **digital soft power tool** within a socio-security context.
*   **Optimized Pipeline**: A generalized preprocessing and TF-IDF pipeline adaptable across disciplines.
*   **Empirical Proof**: Demonstrating that **LinearSVC** provides the optimal balance of accuracy (95%+), speed, and interpretability for real-time systems.

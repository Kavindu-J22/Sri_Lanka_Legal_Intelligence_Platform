# Model Card — SLLIP V3 Supervised Legal Judgment Prediction Pipeline

**Model Name**: `TF-IDF + Calibrated Logistic Regression LJP Pipeline`  
**Model Version**: `1.0.0`  
**Artifact Path**: `assets/models/supervised_ljp_pipeline.joblib`  
**Framework**: `scikit-learn (1.6.1)`, `joblib`  
**Release Date**: October 2026  

---

## 1. Model Overview & Purpose

This model is a calibrated, supervised classification pipeline designed to predict appellate legal judgment outcomes in Sri Lankan family and civil law appeals based on procedural facts and initial arguments.

It serves as the quantitative inference foundation for the **Judge Agent** in SLLIP V3, replacing ungrounded heuristic rule-scoring with genuine calibrated class probabilities.

---

## 2. Training & Evaluation Data

- **Training Corpus**: 1,117 appellate cases (Supreme Court and Court of Appeal of Sri Lanka, 1950–2024).
- **Validation Corpus**: 159 cases (used for Platt scaling sigmoid calibration).
- **Test Corpus**: 318 held-out isolated cases (strictly evaluated with tail verdicts stripped).
- **Target Classes** (Canonical Appellate Outcomes):
  1. `Appeal Dismissed` (Prior: 37.11%)
  2. `Order Set Aside` (Prior: 27.60%)
  3. `Appeal Allowed` (Prior: 18.82%)
  4. `Judgment Affirmed` (Prior: 8.91%)
  5. `Application Dismissed` (Prior: 5.65%)
  6. `Appeal Allowed in Part` (Prior: 1.88%)

---

## 3. Architecture & Hyperparameters

1. **Feature Extraction**:
   - `TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")`
   - Fitted strictly on the 1,117 training texts.
2. **Base Estimator**:
   - `LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs")`
3. **Probability Calibration**:
   - `CalibratedClassifierCV(method="sigmoid", cv=3)` (Platt scaling)

---

## 4. Empirical Evaluation Metrics (N=318 Held-Out Cases)

| Metric | Score | Context / Baseline |
|---|---|---|
| **Accuracy** | **51.89%** | **+14.78%** improvement over Majority Baseline (37.11%) |
| **Macro F1-Score** | **37.80%** | **+28.78%** improvement over Majority Baseline (9.02%) |
| **Weighted F1-Score** | **48.23%** | Support-weighted across all 6 classes |
| **Multi-Class Brier Loss** | **0.0914** | Demonstrates well-calibrated probabilities |

---

## 5. Limitations & Ethical Safeguards

1. **Class Imbalance**: Rare outcome classes such as `Appeal Allowed in Part` (1.88% of corpus) have low statistical support; predictions must be interpreted cautiously.
2. **Decision-Support Only**: SLLIP V3 is an advisory research tool. It does not provide certified legal advice or replace the statutory authority of licensed legal counsel.
3. **Domain Boundary**: Optimized for appellate family law and civil litigation in Sri Lanka; performance on criminal or corporate admiralty law has not been formally verified.

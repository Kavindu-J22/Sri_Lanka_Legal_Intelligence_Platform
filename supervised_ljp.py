"""
Sri Lanka Legal Intelligence Platform (SLLIP V3)
Trainable Supervised Legal Judgment Prediction (LJP) Module
============================================================
Replaces unsupported heuristic claims with real, reproducible, trained ML classifiers:
1. Majority Class Baseline
2. TF-IDF + Calibrated Logistic Regression
3. Dense Embedding (all-MiniLM-L6-v2) + Calibrated Linear Classifier
4. Heuristic Baseline (labeled legacy method for ablation comparison)

Prevents label leakage by stripping post-decision text from inputs and
calibrating probabilities strictly on the validation partition.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    brier_score_loss
)

from dataset_loader import LegalDatasetManager, CANONICAL_VERDICTS, PROCESSED_DIR


class SupervisedLJPPipeline:
    """Trains, calibrates, and evaluates reproducible LJP classifiers on Sri Lankan family law cases."""

    def __init__(self, processed_dir: Path = PROCESSED_DIR):
        self.manager = LegalDatasetManager(processed_dir=processed_dir)
        self.train_cases: List[Dict[str, Any]] = []
        self.val_cases: List[Dict[str, Any]] = []
        self.test_cases: List[Dict[str, Any]] = []

    def prepare_data(
        self,
        test_size: float = 0.20,
        val_size: float = 0.10,
        consolidate: bool = False,
        random_seed: int = 42
    ) -> Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]], Tuple[List[str], List[str]]]:
        """
        Load clean text for train, val, and test cases with post-decision tail text stripped.
        Returns: ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        self.train_cases, self.val_cases, self.test_cases = self.manager.create_stratified_splits(
            test_size=test_size,
            val_size=val_size,
            random_seed=random_seed,
            consolidate=consolidate
        )

        def extract_xy(case_list: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
            texts = []
            labels = []
            for item in case_list:
                doc_id = item["doc_id"]
                txt = self.manager.load_case_text(doc_id, strip_tail_verdict=True)
                if txt and len(txt.strip()) > 50:
                    texts.append(txt)
                    labels.append(item["verdict"])
            return texts, labels

        train_data = extract_xy(self.train_cases)
        val_data = extract_xy(self.val_cases)
        test_data = extract_xy(self.test_cases)

        return train_data, val_data, test_data

    @staticmethod
    def evaluate_model(
        model,
        X_test,
        y_test: List[str],
        classes: List[str],
        model_name: str
    ) -> Dict[str, Any]:
        """Compute verified classification metrics using scikit-learn."""
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )

        per_class = {}
        for c in classes:
            c_true = [1 if t == c else 0 for t in y_test]
            c_pred = [1 if p == c else 0 for p in y_pred]
            if sum(c_true) > 0:
                p, r, f, _ = precision_recall_fscore_support(c_true, c_pred, average="binary", zero_division=0)
                per_class[c] = {
                    "support": sum(c_true),
                    "precision": round(float(p), 4),
                    "recall": round(float(r), 4),
                    "f1": round(float(f), 4)
                }

        # Measure probability calibration if predict_proba is available
        brier_avg = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_test)
                # Compute multi-class Brier score
                brier_scores = []
                for idx, c in enumerate(model.classes_):
                    bin_true = [1.0 if t == c else 0.0 for t in y_test]
                    brier_scores.append(brier_score_loss(bin_true, probs[:, idx]))
                brier_avg = round(float(np.mean(brier_scores)), 4)
            except Exception:
                brier_avg = None

        cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()

        return {
            "model_name": model_name,
            "test_sample_count": len(y_test),
            "accuracy": round(acc, 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_p), 4),
            "weighted_recall": round(float(weighted_r), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "brier_score_loss": brier_avg,
            "per_class_metrics": per_class,
            "confusion_matrix": cm,
            "classes": classes
        }

    def run_all_baselines(self, consolidate: bool = False) -> Dict[str, Any]:
        """Train and evaluate all reproducible baseline models."""
        print("Preparing stratified, leak-free splits...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_data(
            test_size=0.20,
            val_size=0.10,
            consolidate=consolidate,
            random_seed=42
        )

        classes = sorted(list(set(y_train + y_val + y_test)))
        print(f"Data ready -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"Target classes ({len(classes)}): {classes}")

        all_results = {}

        # -------------------------------------------------------------
        # 1. Baseline 0: Majority Class Classifier
        # -------------------------------------------------------------
        print("\n[1/3] Training Majority Class Baseline...")
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        all_results["majority_class"] = self.evaluate_model(
            dummy, X_test, y_test, classes, "Majority Class Baseline"
        )

        # -------------------------------------------------------------
        # 2. Baseline 1: TF-IDF + Calibrated Logistic Regression
        # -------------------------------------------------------------
        print("\n[2/3] Training TF-IDF + Calibrated Logistic Regression...")
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode"
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)
        X_test_vec = vectorizer.transform(X_test)

        base_lr = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
        base_lr.fit(X_train_vec, y_train)

        # Calibrate probabilities on validation set (Platt scaling / Sigmoid)
        calibrated_lr = CalibratedClassifierCV(estimator=base_lr, method="sigmoid", cv="prefit")
        calibrated_lr.fit(X_val_vec, y_val)

        all_results["tfidf_logistic_regression"] = self.evaluate_model(
            calibrated_lr, X_test_vec, y_test, classes, "TF-IDF + Calibrated Logistic Regression"
        )

        # -------------------------------------------------------------
        # 3. Baseline 2: Dense Embedding Classifier (all-MiniLM-L6-v2)
        # -------------------------------------------------------------
        print("\n[3/3] Generating embeddings for Dense Embedding Classifier...")
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            # Encode in batches
            X_train_emb = embedder.encode(X_train, batch_size=32, show_progress_bar=False)
            X_val_emb = embedder.encode(X_val, batch_size=32, show_progress_bar=False)
            X_test_emb = embedder.encode(X_test, batch_size=32, show_progress_bar=False)

            dense_clf = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
            dense_clf.fit(X_train_emb, y_train)

            calibrated_dense = CalibratedClassifierCV(estimator=dense_clf, method="sigmoid", cv="prefit")
            calibrated_dense.fit(X_val_emb, y_val)

            all_results["dense_embedding_classifier"] = self.evaluate_model(
                calibrated_dense, X_test_emb, y_test, classes, "Dense Embedding (MiniLM) + Calibrated Classifier"
            )
        except Exception as e:
            print(f"Notice: Dense embedding model evaluation skipped or encountered error: {e}")

        # Save verified results to JSON
        output_file = PROCESSED_DIR / "supervised_benchmark_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "task": "Appellate Family Law Legal Judgment Prediction",
                "consolidated_classes": consolidate,
                "train_sample_count": len(X_train),
                "val_sample_count": len(X_val),
                "test_sample_count": len(X_test),
                "results": all_results
            }, f, indent=2)

        print(f"\nSaved verified evaluation results to '{output_file}'")
        return all_results


if __name__ == "__main__":
    pipeline = SupervisedLJPPipeline()
    results = pipeline.run_all_baselines(consolidate=False)

    print("\n" + "=" * 75)
    print("SLLIP V3 - EMPIRICAL SUPERVISED LJP BENCHMARK RESULTS")
    print("=" * 75)
    print(f"{'Model':<40} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<10}")
    print("-" * 75)
    for key, r in results.items():
        print(f"{r['model_name']:<40} | {r['accuracy']*100:6.2f}%    | {r['macro_f1']*100:6.2f}%    | {r['weighted_f1']*100:6.2f}%")
    print("=" * 75)

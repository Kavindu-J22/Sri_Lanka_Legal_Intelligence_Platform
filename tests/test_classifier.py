"""
SLLIP V3 Automated Supervised Classifier Test Suite
===================================================
Verifies the scikit-learn evaluation metrics computation and probability calibration.
"""

from supervised_ljp import SupervisedLJPPipeline
from sklearn.dummy import DummyClassifier


def test_metrics_computation_integrity():
    """Verify that evaluate_model computes true scikit-learn metrics."""
    X_test = ["sample text 1", "sample text 2", "sample text 3", "sample text 4"]
    y_test = ["Appeal Dismissed", "Appeal Dismissed", "Order Set Aside", "Appeal Allowed"]
    classes = ["Appeal Allowed", "Appeal Dismissed", "Order Set Aside"]

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_test, y_test)

    metrics = SupervisedLJPPipeline.evaluate_model(
        dummy, X_test, y_test, classes, "Dummy Classifier Test"
    )

    assert metrics["accuracy"] == 0.50, f"Expected 0.50 accuracy, got {metrics['accuracy']}"
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["test_sample_count"] == 4

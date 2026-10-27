"""
SLLIP V3 Supervised Model Persistence & Inference Engine Tests
=============================================================
Verifies that assets/models/supervised_ljp_pipeline.joblib exists,
loads correctly into TrainedLJPInferenceEngine, and outputs calibrated probabilities.
"""

from pathlib import Path
from supervised_ljp import TrainedLJPInferenceEngine, DEFAULT_MODEL_PATH
from dataset_loader import CANONICAL_VERDICTS


def test_pipeline_artifact_exists():
    """Verify serialized pipeline artifact file exists."""
    assert DEFAULT_MODEL_PATH.exists(), f"Missing model pipeline artifact at {DEFAULT_MODEL_PATH}"
    assert DEFAULT_MODEL_PATH.stat().st_size > 50000, "Model file appears unexpectedly small"


def test_inference_engine_loads_pipeline():
    """Verify inference engine loads model pipeline cleanly."""
    engine = TrainedLJPInferenceEngine(model_path=DEFAULT_MODEL_PATH)
    assert engine.is_loaded is True
    assert "Logistic Regression" in engine.model_name
    assert len(engine.classes) >= 4


def test_inference_engine_prediction_output_format():
    """Verify prediction format, canonical verdict validity, and probability sum."""
    engine = TrainedLJPInferenceEngine(model_path=DEFAULT_MODEL_PATH)
    test_text = "The appellant appeals against the maintenance order for wife and minor child alleging cruelty."
    pred = engine.predict(test_text)

    assert pred["is_trained_model"] is True
    assert pred["predicted_verdict"] in CANONICAL_VERDICTS
    assert 0.0 <= pred["confidence"] <= 1.0

    probs = pred["class_probabilities"]
    assert isinstance(probs, dict)
    assert len(probs) >= 4
    # Probabilities should sum to approximately 1.0
    prob_sum = sum(probs.values())
    assert 0.95 <= prob_sum <= 1.05, f"Expected probability sum ~1.0, got {prob_sum}"


def test_inference_engine_empty_input_graceful():
    """Verify engine handles empty or whitespace input without crashing."""
    engine = TrainedLJPInferenceEngine(model_path=DEFAULT_MODEL_PATH)
    pred = engine.predict("   ")
    assert pred["predicted_verdict"] in CANONICAL_VERDICTS
    assert pred["confidence"] > 0.0

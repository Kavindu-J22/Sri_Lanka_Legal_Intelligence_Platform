"""
SLLIP V3 Retrieval Quality Benchmark & Controlled Experiments Tests
===================================================================
Verifies assets/retrieval_benchmark_results.json and assets/research_experiments_results.json
for schema integrity, valid metric bounds, and controlled comparison rules.
"""

import json
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
RETRIEVAL_FILE = ASSETS_DIR / "retrieval_benchmark_results.json"
EXPERIMENTS_FILE = ASSETS_DIR / "research_experiments_results.json"


def test_retrieval_benchmark_results_structure():
    """Verify retrieval benchmark output file exists and has valid metrics."""
    assert RETRIEVAL_FILE.exists(), f"Missing retrieval results at {RETRIEVAL_FILE}"
    with open(RETRIEVAL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("version") == "3.0.0"
    methods = data.get("results_by_method", {})
    assert "Dense_Semantic" in methods
    assert "BM25_Sparse" in methods
    assert "Hybrid_Dense_BM25_RRF" in methods

    for meth, vals in methods.items():
        assert 0.0 <= vals["precision_at_5"] <= 1.0
        assert 0.0 <= vals["recall_at_5"] <= 1.0
        assert 0.0 <= vals["mean_reciprocal_rank_mrr"] <= 1.0
        assert 0.0 <= vals["ndcg_at_5"] <= 1.0
        assert vals["avg_latency_ms"] > 0.0


def test_controlled_experiments_results_structure():
    """Verify research experiments results file has identical test cases and valid systems."""
    assert EXPERIMENTS_FILE.exists(), f"Missing experiments results at {EXPERIMENTS_FILE}"
    with open(EXPERIMENTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("version") == "3.0.0"
    assert data.get("test_sample_count") == 25
    held_out_ids = data.get("held_out_case_ids", [])
    assert len(held_out_ids) == 25
    assert len(set(held_out_ids)) == 25, "Held out case IDs must be unique"

    systems = data.get("hypothesis_systems", {})
    assert "System_A_Precedent_Only" in systems
    assert "System_B_Precedent_Plus_Statutory" in systems
    assert "System_C_SAC_Dual_Source" in systems
    assert "System_D_Multi_Agent_Debate" in systems
    assert "System_E_Hybrid_Trained_Classifier" in systems

    # Verify System D is honestly reported
    sys_d = systems["System_D_Multi_Agent_Debate"]
    assert sys_d.get("status") in ["evaluated", "NOT_EVALUATED_NO_LLM_KEY", "NOT_EVALUATED_API_ERROR"]

    # Verify System E has verified metric
    sys_e = systems["System_E_Hybrid_Trained_Classifier"]
    assert sys_e.get("status") == "evaluated"
    assert 0.0 <= sys_e["accuracy"] <= 1.0

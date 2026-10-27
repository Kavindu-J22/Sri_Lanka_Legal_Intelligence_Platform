"""
SLLIP V3 Automated Hybrid Retrieval Test Suite
==============================================
Verifies BM25 sparse matching, Reciprocal Rank Fusion (RRF), and test ID exclusion.
"""

from hybrid_retriever import BM25Index, reciprocal_rank_fusion


def test_bm25_index_and_query():
    """Verify that BM25 accurately ranks keyword matches."""
    corpus = [
        {"id": "doc_maint", "text": "Maintenance Act No. 37 of 1999 child alimony provisions."},
        {"id": "doc_divorce", "text": "Marriage and Divorce Act malicious constructive desertion."},
        {"id": "doc_custody", "text": "Guardianship and custody rights of tender age minor children."}
    ]

    index = BM25Index()
    index.fit(corpus)

    results = index.query("Maintenance Act 37", top_k=1)
    assert len(results) == 1
    assert results[0][0]["id"] == "doc_maint"
    assert results[0][1] > 0.0


def test_test_id_exclusion_prevents_leakage():
    """Verify that specifying exclude_ids prevents test documents from retrieving themselves."""
    corpus = [
        {"id": "test_case_1", "text": "Exact facts of the test case petition for divorce."},
        {"id": "other_case_2", "text": "General appellate court holding on matrimonial property."}
    ]

    index = BM25Index()
    index.fit(corpus)

    # Without exclusion, test_case_1 matches itself first
    unfiltered = index.query("test case petition for divorce", top_k=1)
    assert unfiltered[0][0]["id"] == "test_case_1"

    # With exclusion, test_case_1 is filtered out completely
    filtered = index.query("test case petition for divorce", top_k=2, exclude_ids={"test_case_1"})
    retrieved_ids = [doc["id"] for doc, _ in filtered]
    assert "test_case_1" not in retrieved_ids


def test_reciprocal_rank_fusion_logic():
    """Verify that RRF combines ranks correctly."""
    dense = [{"doc_id": "doc_a", "score": 0.9}, {"doc_id": "doc_b", "score": 0.7}]
    sparse = [({"id": "doc_b"}, 4.2), ({"id": "doc_a"}, 1.5)]

    fused = reciprocal_rank_fusion(dense, sparse, top_k=2)
    assert len(fused) == 2
    assert "rrf_score" in fused[0]
    assert fused[0]["rrf_score"] >= fused[1]["rrf_score"]

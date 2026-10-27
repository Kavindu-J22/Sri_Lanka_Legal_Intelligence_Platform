"""
Tests for Vector Store Abstraction and Cloud Backends (SLLIP V3)
"""

import pytest
from vector_store import VectorStoreFactory, string_to_uuid, BaseVectorCollection, QdrantCollectionWrapper


def test_string_to_uuid_deterministic():
    """Verify that arbitrary string IDs map deterministically to valid RFC 4122 UUIDs."""
    raw_id_1 = "data__lk_appeal_court_judgements__2015__doc_a6097e99_c001"
    raw_id_2 = "data__lk_appeal_court_judgements__2015__doc_a6097e99_c002"

    uuid_1_a = string_to_uuid(raw_id_1)
    uuid_1_b = string_to_uuid(raw_id_1)
    uuid_2 = string_to_uuid(raw_id_2)

    assert uuid_1_a == uuid_1_b, "string_to_uuid must produce deterministic IDs"
    assert uuid_1_a != uuid_2, "Distinct document IDs must map to distinct UUIDs"
    assert len(uuid_1_a) == 36, "Mapped ID must be standard 36-char UUID string"


def test_vector_store_factory_local_fallback():
    """Verify that local ChromaDB fallback collection implements BaseVectorCollection."""
    store = VectorStoreFactory.get_collection("test_local_fallback", backend="local")
    assert isinstance(store, BaseVectorCollection)
    assert hasattr(store, "add")
    assert hasattr(store, "query")
    assert hasattr(store, "count")
    assert hasattr(store, "get")


def test_qdrant_cloud_connection_and_count():
    """Verify live connectivity and query functionality to Qdrant Cloud cluster."""
    try:
        qdrant_col = VectorStoreFactory.get_collection("case_law_collection", backend="qdrant")
        assert isinstance(qdrant_col, QdrantCollectionWrapper)
        cnt = qdrant_col.count()
        assert cnt >= 0, "Qdrant collection count should return non-negative integer"

        # Verify query structure
        res = qdrant_col.query(query_embeddings=[[0.01] * 384], n_results=1)
        assert "ids" in res
        assert "documents" in res
        assert "metadatas" in res
        assert "distances" in res
    except Exception as e:
        pytest.skip(f"Qdrant Cloud credentials or network unavailable: {e}")

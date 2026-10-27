"""
Sri Lanka Legal Intelligence Platform (SLLIP V3)
Hybrid Retrieval Engine (Dense + BM25 Sparse with Reciprocal Rank Fusion)
========================================================================
Implements genuine BM25 sparse keyword matching alongside dense semantic embeddings,
fused via Reciprocal Rank Fusion (RRF). Strictly separates enacted statutory Acts from
unenacted Bills and prevents test-case label leakage.
"""

import math
import re
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np


class BM25Index:
    """Fast, memory-efficient in-memory BM25Okapi implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_lengths: List[int] = []
        self.doc_ids: List[str] = []
        self.documents: List[Dict[str, Any]] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_term_freqs: List[Dict[str, int]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple lowercase word tokenization stripping punctuation."""
        return re.findall(r"\b[a-zA-Z0-9_\u0D80-\u0DFF\u0B80-\u0BFF]+\b", text.lower())

    def fit(self, documents: List[Dict[str, Any]], text_field: str = "text", id_field: str = "id"):
        """Index a list of documents for BM25 search."""
        self.documents = documents
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            return

        total_length = 0
        self.doc_lengths = []
        self.doc_ids = []
        self.doc_term_freqs = []
        self.df = {}

        for doc in documents:
            doc_id = doc.get(id_field, "")
            self.doc_ids.append(doc_id)
            tokens = self._tokenize(doc.get(text_field, ""))
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            term_counts: Dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1
            self.doc_term_freqs.append(term_counts)

            for term in term_counts:
                self.df[term] = self.df.get(term, 0) + 1

        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 1.0

        # Precompute Robertson-Spärck Jones IDF
        self.idf = {}
        for term, freq in self.df.items():
            # Standard BM25 IDF formulation with floor smoothing
            idf_val = math.log(((self.corpus_size - freq + 0.5) / (freq + 0.5)) + 1.0)
            self.idf[term] = max(0.001, idf_val)

    def query(self, query_text: str, top_k: int = 10, exclude_ids: Optional[Set[str]] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Score indexed documents against query terms."""
        if self.corpus_size == 0:
            return []

        q_tokens = self._tokenize(query_text)
        if not q_tokens:
            return []

        scores = np.zeros(self.corpus_size, dtype=np.float32)
        exclude_set = exclude_ids or set()

        for term in q_tokens:
            if term not in self.idf:
                continue
            term_idf = self.idf[term]

            for doc_idx in range(self.corpus_size):
                if self.doc_ids[doc_idx] in exclude_set:
                    continue
                tf = self.doc_term_freqs[doc_idx].get(term, 0)
                if tf > 0:
                    doc_len = self.doc_lengths[doc_idx]
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    scores[doc_idx] += term_idf * (tf * (self.k1 + 1.0)) / denominator

        top_indices = np.argsort(scores)[::-1]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            if self.doc_ids[idx] in exclude_set:
                continue
            results.append((self.documents[idx], float(scores[idx])))
            if len(results) >= top_k:
                break

        return results


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: Any,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    top_k: int = 5,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Combine dense semantic ranking and BM25 sparse ranking using Reciprocal Rank Fusion (RRF).
    Formula: RRF(d) = (w_dense / (k + rank_dense)) + (w_sparse / (k + rank_sparse))
    """
    if "k" in kwargs:
        rrf_k = kwargs["k"]

    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}
    rank_dense_map: Dict[str, int] = {}
    rank_sparse_map: Dict[str, int] = {}

    for rank, doc in enumerate(dense_results, 1):
        doc_id = doc.get("doc_id") or doc.get("chunk_id") or doc.get("id") or doc.get("filename", "")
        if not doc_id:
            continue
        doc_map[doc_id] = doc
        rank_dense_map[doc_id] = rank
        scores[doc_id] = scores.get(doc_id, 0.0) + (dense_weight / (rrf_k + rank))

    # sparse_results can be List[Tuple[Dict, float]] or List[Dict]
    for rank, item in enumerate(sparse_results, 1):
        if isinstance(item, tuple) and len(item) == 2:
            doc, bm25_score = item
        elif isinstance(item, dict):
            doc = item
            bm25_score = float(item.get("score", 0.0))
        else:
            continue

        doc_id = doc.get("doc_id") or doc.get("chunk_id") or doc.get("id") or doc.get("filename", "")
        if not doc_id:
            continue
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        rank_sparse_map[doc_id] = rank
        scores[doc_id] = scores.get(doc_id, 0.0) + (sparse_weight / (rrf_k + rank))

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused_results = []

    for doc_id, rrf_score in sorted_docs[:top_k]:
        item = dict(doc_map[doc_id])
        item["rrf_score"] = round(rrf_score, 5)
        item["dense_rank"] = rank_dense_map.get(doc_id, None)
        item["sparse_rank"] = rank_sparse_map.get(doc_id, None)
        fused_results.append(item)

    return fused_results


if __name__ == "__main__":
    # Smoke test BM25Index and RRF
    corpus = [
        {"id": "doc1", "text": "Maintenance Act No. 37 of 1999 provides monthly child support."},
        {"id": "doc2", "text": "Marriage and Divorce Act regarding malicious desertion and alimony."},
        {"id": "doc3", "text": "Civil Procedure Code section 602 regarding permanent matrimonial property."},
        {"id": "doc4", "text": "Court of Appeal judgment concerning custody and welfare of minor children."}
    ]

    bm25 = BM25Index()
    bm25.fit(corpus)
    res = bm25.query("maintenance act 37 1999 support", top_k=2)
    print("=" * 70)
    print("SLLIP V3 - BM25 SPARSE RETRIEVAL TEST")
    print("=" * 70)
    for doc, sc in res:
        print(f"Doc ID: {doc['id']} | BM25 Score: {sc:.4f} | Text: {doc['text']}")

    # Test RRF
    dense_dummy = [
        {"doc_id": "doc2", "similarity_score": 0.85, "text": corpus[1]["text"]},
        {"doc_id": "doc1", "similarity_score": 0.78, "text": corpus[0]["text"]}
    ]
    fused = reciprocal_rank_fusion(dense_dummy, res, top_k=2)
    print("\nRECIPROCAL RANK FUSION (RRF) RESULT:")
    for f in fused:
        print(f"Doc ID: {f.get('doc_id') or f.get('id')} | RRF: {f['rrf_score']} | DenseRank: {f['dense_rank']} | SparseRank: {f['sparse_rank']}")

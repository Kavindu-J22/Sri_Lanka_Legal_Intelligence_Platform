"""
Sri Lanka Legal Intelligence Platform (SLLIP V3)
Retrieval Quality Evaluation Benchmark
=======================================
Scientifically evaluates and compares:
1. Dense Semantic Search (all-MiniLM-L6-v2)
2. BM25 Sparse Lexical Search (BM25Okapi)
3. Hybrid Dense-Sparse RRF (Reciprocal Rank Fusion, k=60)

Metrics:
- Precision@k (k=1, 3, 5)
- Recall@k (k=1, 3, 5)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@5)
- Query Latency (ms)

Results are exported to assets/retrieval_benchmark_results.json.
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

# Add root directory to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from sentence_transformers import SentenceTransformer
from vector_store import VectorStoreFactory
from hybrid_retriever import BM25Index, reciprocal_rank_fusion

BENCHMARK_OUTPUT_FILE = ROOT_DIR / "assets" / "retrieval_benchmark_results.json"

# Curated benchmark query suite reflecting Sri Lankan Family & Civil Law research queries
# with verified thematic relevance criteria (relevant statutes, concepts, legal terms)
GOLD_BENCHMARK_QUERIES = [
    {
        "query_id": "Q1_MAINTENANCE",
        "query": "Maintenance claim for spouse and minor child under Maintenance Act No 37 of 1999 neglect to maintain",
        "relevant_terms": ["maintenance", "maintenance act", "neglect", "wife", "child", "allowance", "means"],
        "domain": "Family Law / Maintenance"
    },
    {
        "query_id": "Q2_CUSTODY_BEST_INTERESTS",
        "query": "Custody of minor child paramount interest and welfare of the child preferential right of mother father",
        "relevant_terms": ["custody", "minor", "welfare of the child", "paramount", "mother", "guardian", "access"],
        "domain": "Family Law / Child Custody"
    },
    {
        "query_id": "Q3_DIVORCE_DESERTION",
        "query": "Malicious desertion constructive desertion as ground for divorce under Marriage Registration Ordinance",
        "relevant_terms": ["desertion", "malicious", "constructive", "divorce", "matrimonial", "separation"],
        "domain": "Family Law / Matrimonial"
    },
    {
        "query_id": "Q4_DIVORCE_ADULTERY",
        "query": "Dissolution of marriage adultery standard of proof direct evidence co-defendant damages",
        "relevant_terms": ["adultery", "dissolution", "marriage", "co-defendant", "damages", "proof"],
        "domain": "Family Law / Matrimonial"
    },
    {
        "query_id": "Q5_ALIMONY_PENDENTE_LITE",
        "query": "Alimony pendente lite interim maintenance permanent alimony section 614 Civil Procedure Code",
        "relevant_terms": ["alimony", "pendente lite", "interim", "permanent alimony", "civil procedure code", "614"],
        "domain": "Civil Procedure / Alimony"
    },
    {
        "query_id": "Q6_PRESCRIPTION_PROPERTY",
        "query": "Prescriptive title adverse possession undisturbed and uninterrupted possession Section 3 Prescription Ordinance",
        "relevant_terms": ["prescription", "prescriptive", "adverse possession", "undisturbed", "possession", "title"],
        "domain": "Property Law / Prescription"
    },
    {
        "query_id": "Q7_PARTITION_DECLARATION",
        "query": "Partition action declaration of title co-owners pedigree investigation of title Partition Law",
        "relevant_terms": ["partition", "co-owner", "pedigree", "investigation of title", "corpus", "interlocutory"],
        "domain": "Property Law / Partition"
    },
    {
        "query_id": "Q8_EX_PARTE_DECREE",
        "query": "Setting aside ex parte decree default of appearance failure to file answer reasonable cause section 86",
        "relevant_terms": ["ex parte", "default", "appearance", "set aside", "section 86", "service of summons"],
        "domain": "Civil Procedure / Default"
    },
    {
        "query_id": "Q9_WRIT_CERTIORARI",
        "query": "Application for writ of certiorari natural justice audi alteram partem error of law on the face of record",
        "relevant_terms": ["certiorari", "natural justice", "audi alteram partem", "tribunal", "jurisdiction", "record"],
        "domain": "Administrative & Constitutional"
    },
    {
        "query_id": "Q10_MONEY_RECOVERY",
        "query": "Action for recovery of money debt due summary procedure on liquid claims promissory note",
        "relevant_terms": ["recovery of money", "debt", "liquid claim", "promissory note", "summary procedure", "cheque"],
        "domain": "Commercial / Civil Procedure"
    }
]


def is_relevant(doc_text: str, meta: Dict[str, Any], query_spec: Dict[str, Any]) -> bool:
    """Evaluates whether retrieved item is relevant using strict thematic criterion."""
    combined = (doc_text + " " + meta.get("global_summary", "") + " " + meta.get("keywords", "")).lower()
    matches = sum(1 for term in query_spec["relevant_terms"] if term.lower() in combined)
    # Require at least 2 relevant terms to match for strict relevance
    return matches >= 2


def dcg_at_k(r: List[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank k."""
    r = r[:k]
    if not r:
        return 0.0
    return float(r[0]) + sum(val / math.log2(idx + 1) for idx, val in enumerate(r[1:], start=2))


def ndcg_at_k(r: List[int], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k."""
    dcg = dcg_at_k(r, k)
    ideal = sorted(r, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


class RetrievalBenchmarkEvaluator:
    """Executes multi-query retrieval benchmarks across Dense, BM25, and Hybrid RRF."""

    def __init__(self):
        print("Initializing Retrieval Benchmark Evaluator...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.case_store = VectorStoreFactory.get_collection("case_law_collection")
        self.stat_store = VectorStoreFactory.get_collection("statutory_acts_collection")

    def run_benchmark(self, top_k_eval: int = 10) -> Dict[str, Any]:
        print(f"Running retrieval benchmark over {len(GOLD_BENCHMARK_QUERIES)} legal queries...")
        methods = ["Dense_Semantic", "BM25_Sparse", "Hybrid_Dense_BM25_RRF"]
        metrics_by_method = {m: {"p1": [], "p3": [], "p5": [], "r3": [], "r5": [], "mrr": [], "ndcg5": [], "latency_ms": []} for m in methods}

        for q_idx, q_item in enumerate(GOLD_BENCHMARK_QUERIES, start=1):
            q_text = q_item["query"]
            print(f"  [{q_idx}/{len(GOLD_BENCHMARK_QUERIES)}] Evaluating: {q_item['query_id']} ({q_item['domain']})")

            # 1. Dense Semantic Retrieval
            t0 = time.time()
            q_emb = self.model.encode([q_text], show_progress_bar=False).tolist()
            dense_res = self.case_store.query(
                query_embeddings=q_emb,
                n_results=top_k_eval * 3,
                include=["documents", "metadatas", "distances"]
            )
            dense_lat = (time.time() - t0) * 1000.0

            dense_docs = []
            if dense_res and dense_res.get("documents"):
                for doc, meta, dist in zip(dense_res["documents"][0], dense_res["metadatas"][0], dense_res["distances"][0]):
                    dense_docs.append({
                        "id": meta.get("doc_id", ""),
                        "text": doc,
                        "metadata": meta,
                        "score": 1.0 / (1.0 + dist)
                    })

            # 2. Build local candidate corpus for BM25 from top candidates
            t0 = time.time()
            bm25 = BM25Index()
            bm25.fit(dense_docs, text_field="text", id_field="id")
            bm25_scored = bm25.query(q_text, top_k=top_k_eval)
            bm25_lat = (time.time() - t0) * 1000.0

            bm25_docs = []
            for doc_dict, sc in bm25_scored:
                bm25_docs.append({
                    "id": doc_dict.get("id", ""),
                    "text": doc_dict.get("text", ""),
                    "metadata": doc_dict.get("metadata", {}),
                    "score": sc
                })

            # 3. Hybrid RRF
            t0 = time.time()
            dense_for_rrf = [{"id": d["id"], "score": d["score"]} for d in dense_docs]
            bm25_for_rrf = [{"id": d["id"], "score": d["score"]} for d in bm25_docs]
            fused_ranks = reciprocal_rank_fusion(dense_for_rrf, bm25_for_rrf, k=60)
            fused_score_map = {item["id"]: item["rrf_score"] for item in fused_ranks}

            hybrid_docs = list(dense_docs)
            hybrid_docs.sort(key=lambda x: fused_score_map.get(x["id"], 0.0), reverse=True)
            hybrid_lat = dense_lat + bm25_lat + ((time.time() - t0) * 1000.0)

            # Evaluate each list
            eval_runs = [
                ("Dense_Semantic", dense_docs[:top_k_eval], dense_lat),
                ("BM25_Sparse", bm25_docs[:top_k_eval], bm25_lat),
                ("Hybrid_Dense_BM25_RRF", hybrid_docs[:top_k_eval], hybrid_lat)
            ]

            for method_name, retrieved_list, lat in eval_runs:
                rel_vector = [1 if is_relevant(d.get("text", ""), d.get("metadata", {}), q_item) else 0 for d in retrieved_list]
                total_relevant = sum(rel_vector)

                # Precision@k
                p1 = rel_vector[0] if len(rel_vector) >= 1 else 0
                p3 = (sum(rel_vector[:3]) / 3.0) if len(rel_vector) >= 3 else (sum(rel_vector) / max(len(rel_vector), 1))
                p5 = (sum(rel_vector[:5]) / 5.0) if len(rel_vector) >= 5 else (sum(rel_vector) / max(len(rel_vector), 1))

                # Recall@k (relative to retrieved relevant pool)
                norm_rel = max(total_relevant, 1)
                r3 = sum(rel_vector[:3]) / norm_rel
                r5 = sum(rel_vector[:5]) / norm_rel

                # MRR
                first_rel = next((idx + 1 for idx, r in enumerate(rel_vector) if r == 1), 0)
                mrr = 1.0 / first_rel if first_rel > 0 else 0.0

                # nDCG@5
                ndcg5 = ndcg_at_k(rel_vector, 5)

                m_dict = metrics_by_method[method_name]
                m_dict["p1"].append(p1)
                m_dict["p3"].append(p3)
                m_dict["p5"].append(p5)
                m_dict["r3"].append(r3)
                m_dict["r5"].append(r5)
                m_dict["mrr"].append(mrr)
                m_dict["ndcg5"].append(ndcg5)
                m_dict["latency_ms"].append(lat)

        # Aggregate averages
        summary = {}
        for m_name, m_data in metrics_by_method.items():
            summary[m_name] = {
                "precision_at_1": round(float(sum(m_data["p1"]) / len(m_data["p1"])), 4),
                "precision_at_3": round(float(sum(m_data["p3"]) / len(m_data["p3"])), 4),
                "precision_at_5": round(float(sum(m_data["p5"]) / len(m_data["p5"])), 4),
                "recall_at_3": round(float(sum(m_data["r3"]) / len(m_data["r3"])), 4),
                "recall_at_5": round(float(sum(m_data["r5"]) / len(m_data["r5"])), 4),
                "mean_reciprocal_rank_mrr": round(float(sum(m_data["mrr"]) / len(m_data["mrr"])), 4),
                "ndcg_at_5": round(float(sum(m_data["ndcg5"]) / len(m_data["ndcg5"])), 4),
                "avg_latency_ms": round(float(sum(m_data["latency_ms"]) / len(m_data["latency_ms"])), 2)
            }

        payload = {
            "version": "3.0.0",
            "eval_date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "benchmark_query_count": len(GOLD_BENCHMARK_QUERIES),
            "eval_depth_k": top_k_eval,
            "fusion_algorithm": "Reciprocal Rank Fusion (k=60)",
            "results_by_method": summary,
            "queries_evaluated": [q["query_id"] for q in GOLD_BENCHMARK_QUERIES]
        }

        BENCHMARK_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BENCHMARK_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"\nSaved retrieval evaluation benchmark to '{BENCHMARK_OUTPUT_FILE}'")
        return payload


if __name__ == "__main__":
    evaluator = RetrievalBenchmarkEvaluator()
    res = evaluator.run_benchmark(top_k_eval=5)

    print("\n" + "=" * 88)
    print("SLLIP V3 - RETRIEVAL QUALITY BENCHMARK RESULTS")
    print("=" * 88)
    print(f"{'Method':<25} | {'P@5':<8} | {'R@5':<8} | {'MRR':<8} | {'nDCG@5':<8} | {'Avg Latency (ms)':<15}")
    print("-" * 88)
    for meth, vals in res["results_by_method"].items():
        print(f"{meth:<25} | {vals['precision_at_5']*100:6.2f}% | {vals['recall_at_5']*100:6.2f}% | {vals['mean_reciprocal_rank_mrr']:6.4f} | {vals['ndcg_at_5']:6.4f} | {vals['avg_latency_ms']:12.2f} ms")
    print("=" * 88)

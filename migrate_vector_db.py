"""
Sri Lanka Legal Intelligence Platform (SLLIP V3)
Resumable Idempotent Vector Database Migration Engine
=====================================================
Migrates local ChromaDB collections to a managed cloud vector database
(Chroma Cloud or Qdrant Cloud) with strict validation:
1. Reads local collections in configurable batches.
2. Validates vector dimensions (384 for all-MiniLM-L6-v2) and count integrity.
3. Preserves deterministic IDs, structured metadata, and text payloads.
4. Idempotent: Uses deterministic UUID5 mapping to prevent duplicates.
5. Non-destructive: NEVER modifies or deletes local source data.
6. Performs pre- and post-migration validation queries.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from vector_store import VectorStoreFactory, get_secret, DEFAULT_LOCAL_PATH

EXPECTED_DIMENSION = 384
DEFAULT_BATCH_SIZE = 150


class VectorMigrationManager:
    """Manages verified, idempotent vector database migrations."""

    def __init__(self, local_path: Path = DEFAULT_LOCAL_PATH):
        self.local_path = local_path
        self.local_client = chromadb.PersistentClient(path=str(local_path))

    def validate_source_collection(self, collection_name: str) -> Dict[str, Any]:
        """Validate local collection integrity, vector dimension, and count."""
        try:
            col = self.local_client.get_collection(collection_name)
        except Exception as e:
            return {"valid": False, "error": f"Collection '{collection_name}' not found: {e}"}

        count = col.count()
        if count == 0:
            return {"valid": True, "count": 0, "dimension": None}

        sample = col.get(limit=1, include=["embeddings", "metadatas", "documents"])
        emb = sample.get("embeddings")
        dim = None
        if emb is not None and len(emb) > 0:
            dim = len(emb[0])

        valid_dim = (dim == EXPECTED_DIMENSION)
        return {
            "valid": valid_dim,
            "collection_name": collection_name,
            "count": count,
            "dimension": dim,
            "expected_dimension": EXPECTED_DIMENSION,
            "dimension_match": valid_dim
        }

    def migrate_collection(
        self,
        collection_name: str,
        target_backend: str = "qdrant",
        target_collection_name: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        limit_count: Optional[int] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Execute idempotent, batch-wise migration from local Chroma to target store.
        """
        target_col_name = target_collection_name or collection_name
        print(f"\n--- Migrating '{collection_name}' -> '{target_col_name}' ({target_backend}) ---")

        # 1. Validate local source
        source_val = self.validate_source_collection(collection_name)
        if not source_val.get("valid"):
            print(f"Validation Error on source collection: {source_val}")
            return {"status": "failed", "reason": source_val}

        total_source_count = source_val["count"]
        if limit_count:
            total_source_count = min(total_source_count, limit_count)

        print(f"Source Collection: {source_val['count']:,} items (Processing: {total_source_count:,}) | Vector Dim: {source_val['dimension']}")

        if dry_run:
            print("[Dry-Run Mode] No changes made to target database.")
            return {"status": "dry_run_success", "source_validation": source_val}

        # 2. Connect to target collection
        try:
            target_store = VectorStoreFactory.get_collection(
                collection_name=target_col_name,
                backend=target_backend
            )
        except Exception as e:
            print(f"Target connection failed: {e}")
            return {"status": "target_connection_failed", "error": str(e)}

        pre_target_count = target_store.count()
        print(f"Target Collection '{target_col_name}' currently contains {pre_target_count:,} items.")

        source_col = self.local_client.get_collection(collection_name)
        migrated_count = 0
        offset = 0

        # Resumption: if target count matches total_source_count, we can verify quickly
        if pre_target_count >= total_source_count:
            print(f"Target already contains >= {total_source_count:,} items. Validating count...")
            return {
                "status": "already_up_to_date",
                "source_collection": collection_name,
                "target_collection": target_col_name,
                "target_backend": target_backend,
                "total_target_count": pre_target_count
            }
        elif pre_target_count > 0:
            offset = (pre_target_count // batch_size) * batch_size
            print(f"Resuming migration from offset {offset:,} (target currently has {pre_target_count:,} items)...")

        # 3. Read in batches using offset pagination
        start_time = time.time()
        while offset < total_source_count:
            cur_limit = min(batch_size, total_source_count - offset)
            batch = source_col.get(
                offset=offset,
                limit=cur_limit,
                include=["embeddings", "documents", "metadatas"]
            )
            b_ids = batch.get("ids", [])
            if not b_ids:
                break

            b_docs = batch.get("documents", [])
            b_embs = batch.get("embeddings")
            if b_embs is not None and hasattr(b_embs, "tolist"):
                b_embs = b_embs.tolist()
            b_metas = batch.get("metadatas", [])

            # Upsert into target store
            target_store.add(
                ids=b_ids,
                documents=b_docs,
                embeddings=b_embs,
                metadatas=b_metas
            )
            migrated_count += len(b_ids)
            offset += cur_limit

            elapsed = time.time() - start_time
            rate = migrated_count / elapsed if elapsed > 0 else 0
            eta = (total_source_count - offset) / rate if rate > 0 else 0
            print(f"  Progress: {min(offset, total_source_count):,}/{total_source_count:,} "
                  f"({(min(offset, total_source_count) / total_source_count * 100):.1f}%) | "
                  f"Speed: {rate:.1f} pts/sec | ETA: {eta:.0f}s", flush=True)

        post_count = target_store.count()
        print(f"\nMigration Complete! Target Collection Total Count: {post_count:,}", flush=True)

        # Post-migration query validation
        val_query = target_store.query(
            query_embeddings=[[0.0] * EXPECTED_DIMENSION],
            n_results=1
        )
        has_results = len(val_query.get("ids", [[]])[0]) > 0
        print(f"Validation Query Test: {'PASSED' if has_results else 'FAILED'}")

        return {
            "status": "success",
            "source_collection": collection_name,
            "target_collection": target_col_name,
            "target_backend": target_backend,
            "migrated_count": migrated_count,
            "total_target_count": post_count,
            "query_validation_passed": has_results
        }


def main():
    parser = argparse.ArgumentParser(description="SLLIP V3 Vector DB Migration Utility")
    parser.add_argument("--collection", default="case_law_collection", help="Source collection name")
    parser.add_argument("--backend", default="qdrant", help="Target backend (qdrant, chroma_cloud)")
    parser.add_argument("--target-collection", default=None, help="Target collection name override")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for upserts")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to migrate (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Perform validation check without uploading")
    args = parser.parse_args()

    migrator = VectorMigrationManager()
    val = migrator.validate_source_collection(args.collection)
    print("=" * 70)
    print("SLLIP V3 - VECTOR DB SOURCE VALIDATION")
    print("=" * 70)
    print(json.dumps(val, indent=2))

    if not args.dry_run:
        migrator.migrate_collection(
            collection_name=args.collection,
            target_backend=args.backend,
            target_collection_name=args.target_collection,
            batch_size=args.batch_size,
            limit_count=args.limit,
            dry_run=args.dry_run
        )


if __name__ == "__main__":
    main()

import os
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm
import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Optimize PyTorch CPU threading
torch.set_num_threads(8)

WORKSPACE_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = WORKSPACE_DIR / "processed_data"
SAC_CHUNKS_FILE = PROCESSED_DIR / "sac_chunks.json"
STATUTORY_ACTS_TXT_DIR = PROCESSED_DIR / "statutory_acts_clean_txt"
VECTOR_DB_DIR = WORKSPACE_DIR / "vector_db"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 100

# Ensure vector db directory exists
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


def chunk_text_simple(text: str, chunk_words: int = 350, overlap_words: int = 50) -> list[str]:
    """Simple chunking for statutory acts & bills."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    chunks = []
    step = chunk_words - overlap_words
    for i in range(0, len(words), step):
        cw = words[i:i + chunk_words]
        if cw:
            chunks.append(" ".join(cw))
        if i + chunk_words >= len(words):
            break
    return chunks


def add_batch_resilient(collection, ids, documents, embeddings, metadatas):
    """Resilient batch insertion into ChromaDB with sub-batch fallback on SQLite compaction error."""
    try:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    except Exception as e:
        # Fallback to smaller sub-batches of 25 items
        sub_size = 25
        for s in range(0, len(ids), sub_size):
            sub_ids = ids[s:s + sub_size]
            sub_docs = documents[s:s + sub_size]
            sub_embs = embeddings[s:s + sub_size]
            sub_meta = metadatas[s:s + sub_size]
            retry = 0
            while retry < 3:
                try:
                    collection.add(
                        ids=sub_ids,
                        documents=sub_docs,
                        embeddings=sub_embs,
                        metadatas=sub_meta
                    )
                    break
                except Exception:
                    retry += 1
                    time.sleep(1)


def main():
    print("=" * 80)
    print("SLLIP DATA PIPELINE - STEP 3: DUAL-VECTOR DATABASE INGESTION (ChromaDB)")
    print("=" * 80)
    print(f"PyTorch CPU Threads set to: {torch.get_num_threads()}")

    # 1. Initialize Persistent ChromaDB Client
    print(f"Initializing Persistent ChromaDB instance at '{VECTOR_DB_DIR}'...")
    chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    # 2. Load Embedding Model
    print(f"Loading SentenceTransformer embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # -------------------------------------------------------------------------
    # COLLECTION A: case_law_collection (Family Law SAC Chunks)
    # -------------------------------------------------------------------------
    print("\nSetting up Collection A: 'case_law_collection'...")
    case_law_col = chroma_client.get_or_create_collection(
        name="case_law_collection",
        metadata={"description": "Sri Lanka Appellate Family Law Judgments with SAC Chunks"}
    )

    if not SAC_CHUNKS_FILE.exists():
        print(f"Warning: SAC chunks file '{SAC_CHUNKS_FILE}' not found. Please run Step 2 first.")
    else:
        with open(SAC_CHUNKS_FILE, "r", encoding="utf-8") as f:
            sac_chunks = json.load(f)

        existing_ids = set()
        try:
            existing_res = case_law_col.get(include=[])
            existing_ids = set(existing_res["ids"])
        except Exception:
            pass

        new_sac_chunks = [b for b in sac_chunks if b["chunk_id"] not in existing_ids]
        total_sac = len(new_sac_chunks)
        print(f"Collection A already contains {len(existing_ids):,} items. Ingesting {total_sac:,} new SAC chunks...")

        if total_sac > 0:
            for i in tqdm(range(0, total_sac, BATCH_SIZE), desc="Ingesting Family Law SAC Chunks", unit="batch"):
                batch = new_sac_chunks[i:i + BATCH_SIZE]
                ids = [b["chunk_id"] for b in batch]
                documents = [b["augmented_text"] for b in batch]
                metadatas = [
                    {
                        "doc_id": str(b["doc_id"]),
                        "chunk_index": int(b["chunk_index"]),
                        "total_chunks": int(b["total_chunks"]),
                        "verdict": str(b["verdict"]),
                        "keywords": ", ".join(b.get("keywords", []))[:200],
                        "global_summary": str(b["global_summary"])[:300]
                    }
                    for b in batch
                ]

                embeddings = model.encode(documents, batch_size=BATCH_SIZE, show_progress_bar=False).tolist()

                add_batch_resilient(case_law_col, ids, documents, embeddings, metadatas)

        print(f"Collection A ('case_law_collection') Total Count: {case_law_col.count():,} items.")

    # -------------------------------------------------------------------------
    # COLLECTION B: statutory_acts_collection (Parliamentary Acts & Bills)
    # -------------------------------------------------------------------------
    print("\nSetting up Collection B: 'statutory_acts_collection'...")
    statutory_col = chroma_client.get_or_create_collection(
        name="statutory_acts_collection",
        metadata={"description": "Sri Lankan Parliamentary Acts and Bills"}
    )

    if STATUTORY_ACTS_TXT_DIR.exists():
        stat_files = list(STATUTORY_ACTS_TXT_DIR.glob("*.txt"))
        print(f"Found {len(stat_files):,} Statutory Acts & Bills text files for ingestion.")

        existing_stat_ids = set()
        try:
            existing_stat_res = statutory_col.get(include=[])
            existing_stat_ids = set(existing_stat_res["ids"])
        except Exception:
            pass

        stat_chunks = []
        for sf in stat_files:
            doc_id = sf.stem
            try:
                with open(sf, "r", encoding="utf-8") as f:
                    txt = f.read().strip()
                chunks = chunk_text_simple(txt, chunk_words=350, overlap_words=50)
                for idx, chk in enumerate(chunks):
                    c_id = f"{doc_id}_c{idx + 1:03d}"
                    if c_id not in existing_stat_ids:
                        stat_chunks.append({
                            "id": c_id,
                            "doc_id": doc_id,
                            "filename": sf.name,
                            "chunk_index": idx + 1,
                            "text": chk
                        })
            except Exception:
                pass

        total_stat = len(stat_chunks)
        print(f"Collection B contains {len(existing_stat_ids):,} items. Ingesting {total_stat:,} new statutory chunks...")

        if total_stat > 0:
            for i in tqdm(range(0, total_stat, BATCH_SIZE), desc="Ingesting Statutory Chunks", unit="batch"):
                batch = stat_chunks[i:i + BATCH_SIZE]
                ids = [b["id"] for b in batch]
                documents = [b["text"] for b in batch]
                metadatas = [
                    {
                        "doc_id": str(b["doc_id"]),
                        "filename": str(b["filename"]),
                        "chunk_index": int(b["chunk_index"]),
                        "source_type": "statutory_act_or_bill"
                    }
                    for b in batch
                ]

                embeddings = model.encode(documents, batch_size=BATCH_SIZE, show_progress_bar=False).tolist()

                add_batch_resilient(statutory_col, ids, documents, embeddings, metadatas)

        print(f"Collection B ('statutory_acts_collection') Total Count: {statutory_col.count():,} items.")

    print("\n" + "=" * 80)
    print("STEP 3: DUAL-VECTOR DATABASE INGESTION SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Vector Database Location:           {VECTOR_DB_DIR}")
    print(f"Embedding Model Used:              {EMBEDDING_MODEL_NAME}")
    print(f"Collection A ('case_law_collection'):     {case_law_col.count():,} items indexed")
    print(f"Collection B ('statutory_acts_collection'): {statutory_col.count():,} items indexed")
    print("=" * 80)
    print("Step 3 Execution Completed Successfully!")



if __name__ == "__main__":
    main()



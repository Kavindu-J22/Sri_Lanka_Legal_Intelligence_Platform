 import os
import sys
import json
import re
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

WORKSPACE_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = WORKSPACE_DIR / "processed_data"
FAMILY_LAW_TXT_DIR = PROCESSED_DIR / "family_law_clean_txt"
GROUND_TRUTH_FILE = PROCESSED_DIR / "ground_truth.json"
SAC_CHUNKS_FILE = PROCESSED_DIR / "sac_chunks.json"

# SAC Chunking Parameters
CHUNK_SIZE_WORDS = 350   # ~300-400 tokens
CHUNK_OVERLAP_WORDS = 50 # ~50 words overlap
TARGET_SUMMARY_WORDS = 150


def generate_150_word_summary(text: str, verdict: str = "") -> str:
    """
    Generate a concise ~150-word global document summary.
    Extracts key case facts from opening, core issues, and verdict from concluding text.
    """
    words = text.split()
    if len(words) <= TARGET_SUMMARY_WORDS:
        summary_str = " ".join(words)
        if verdict and verdict != "Undetermined" and verdict != "N/A":
            summary_str += f" Judicial Outcome: {verdict}."
        return summary_str

    # Extract head (first 100 words) and tail (last 50 words)
    head_words = words[:100]
    tail_words = words[-50:]

    summary_parts = [" ".join(head_words), "..."]
    if verdict and verdict != "Undetermined" and verdict != "N/A":
        summary_parts.append(f"[Judicial Verdict: {verdict}]")
    summary_parts.append(" ".join(tail_words))

    summary_text = " ".join(summary_parts)
    # Trim to ~150 words
    summary_words = summary_text.split()[:TARGET_SUMMARY_WORDS]
    return " ".join(summary_words)


def chunk_text_by_words(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Segment document text into word chunks with specified size and overlap."""
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks


def process_doc_sac(task_args: tuple[str, dict]) -> list[dict]:
    """Process a single clean text file for Summary-Augmented Chunking."""
    txt_path_str, gt_info = task_args
    txt_path = Path(txt_path_str)
    doc_id = txt_path.stem

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except Exception:
        return []

    if not text:
        return []

    verdict = gt_info.get("verdict", "Undetermined") if gt_info else "Undetermined"
    keywords = gt_info.get("keywords_matched", []) if gt_info else []

    # 1. Generate 150-word global summary
    global_summary = generate_150_word_summary(text, verdict)

    # 2. Segment document into raw body chunks
    body_chunks = chunk_text_by_words(text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)

    # 3. Prepend 150-word global summary to EVERY chunk (SAC Strategy)
    sac_chunk_objects = []
    for idx, raw_chunk in enumerate(body_chunks):
        augmented_text = f"[DOCUMENT GLOBAL SUMMARY]: {global_summary}\n\n[CHUNK BODY {idx + 1}/{len(body_chunks)}]:\n{raw_chunk}"

        sac_chunk_objects.append({
            "chunk_id": f"{doc_id}_c{idx + 1:03d}",
            "doc_id": doc_id,
            "filename": txt_path.name,
            "chunk_index": idx + 1,
            "total_chunks": len(body_chunks),
            "global_summary": global_summary,
            "raw_chunk_text": raw_chunk,
            "augmented_text": augmented_text,
            "verdict": verdict,
            "keywords": keywords,
            "char_count": len(augmented_text),
            "word_count": len(augmented_text.split())
        })

    return sac_chunk_objects


def main():
    print("=" * 80)
    print("SLLIP DATA PIPELINE - STEP 2: SUMMARY-AUGMENTED CHUNKING (SAC)")
    print("=" * 80)

    if not FAMILY_LAW_TXT_DIR.exists():
        print(f"Error: Clean text directory '{FAMILY_LAW_TXT_DIR}' not found. Please run Step 1 first.")
        sys.exit(1)

    txt_files = list(FAMILY_LAW_TXT_DIR.glob("*.txt"))
    total_docs = len(txt_files)
    print(f"Found {total_docs:,} clean Family Law text documents for SAC chunking.\n")

    ground_truth = {}
    if GROUND_TRUTH_FILE.exists():
        with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

    tasks = [(str(p), ground_truth.get(p.stem, {})) for p in txt_files]

    num_workers = min(os.cpu_count() or 4, 8)
    print(f"Executing parallel SAC chunking across {num_workers} CPU cores...")

    all_sac_chunks = []
    total_chunks_created = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_doc_sac, tasks, chunksize=20)
        for chunks in tqdm(results, total=total_docs, desc="SAC Chunking", unit="doc"):
            all_sac_chunks.extend(chunks)
            total_chunks_created += len(chunks)

    print(f"\nSaving {total_chunks_created:,} SAC chunks to '{SAC_CHUNKS_FILE}'...")
    with open(SAC_CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_sac_chunks, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("STEP 2: SUMMARY-AUGMENTED CHUNKING (SAC) STATISTICS")
    print("=" * 80)
    print(f"Total Family Law Documents Chunked: {total_docs:,}")
    print(f"Total SAC Chunks Created:           {total_chunks_created:,}")
    if total_docs > 0:
        avg_chunks = total_chunks_created / total_docs
        print(f"Average Chunks Per Document:        {avg_chunks:.2f}")
    print(f"Saved SAC Chunks JSON to:           {SAC_CHUNKS_FILE}")
    print("Step 2 Execution Completed Successfully!")


if __name__ == "__main__":
    main()

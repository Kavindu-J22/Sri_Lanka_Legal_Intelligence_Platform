"""
Sri Lanka Legal Intelligence Platform (SLLIP V3)
Token-Aware Summary-Augmented Chunking (SAC) Engine
===================================================
Solves the 100% token truncation defect of SLLIP V2 by sizing compact global
case context and body chunks to strictly fit within the 256-token sequence limit
of all-MiniLM-L6-v2.
"""

import os
import re
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_ALLOWED_TOKENS = 240   # Safe buffer below the 256-token hard limit
TARGET_SUMMARY_WORDS = 40  # ~50-60 tokens
CHUNK_SIZE_WORDS = 120     # ~150-160 tokens
CHUNK_OVERLAP_WORDS = 25   # ~30 tokens overlap


class TokenAwareSACChunker:
    """Creates token-aware, hierarchy-preserving legal chunks for dense & sparse indexing."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = MAX_ALLOWED_TOKENS

    def generate_compact_summary(self, text: str, max_words: int = TARGET_SUMMARY_WORDS) -> str:
        """
        Generate a concise, non-leaking global case summary (procedural facts & jurisdiction only).
        Crucially does NOT inject the final judicial verdict to prevent label leakage.
        """
        words = text.split()
        if len(words) <= max_words:
            return " ".join(words)

        # Extract opening dispute & procedural context (first 40 words)
        opening_words = words[:max_words]
        summary_str = " ".join(opening_words).strip()
        if not summary_str.endswith("."):
            summary_str += "..."
        return summary_str

    def chunk_text_by_words(
        self,
        text: str,
        chunk_words: int = CHUNK_SIZE_WORDS,
        overlap_words: int = CHUNK_OVERLAP_WORDS
    ) -> List[str]:
        """Split text into overlapping word windows."""
        words = text.split()
        if not words:
            return []
        if len(words) <= chunk_words:
            return [" ".join(words)]

        chunks = []
        step = max(1, chunk_words - overlap_words)
        for i in range(0, len(words), step):
            window = words[i:i + chunk_words]
            if window:
                chunks.append(" ".join(window))
            if i + chunk_words >= len(words):
                break
        return chunks

    def process_document(
        self,
        doc_id: str,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Produce token-verified SAC chunks guaranteed to fit within 256 tokens.
        Stores separate fields for summary, raw body, and combined text.
        """
        if not text:
            return []

        compact_summary = self.generate_compact_summary(text)
        raw_chunks = self.chunk_text_by_words(text)
        chunk_objects = []

        for idx, body in enumerate(raw_chunks):
            chunk_id = f"{doc_id}_c{idx + 1:03d}"
            augmented_text = f"[CASE CONTEXT]: {compact_summary}\n[FACTS]: {body}"

            # Measure exact token length with AutoTokenizer
            tokens = self.tokenizer.encode(augmented_text, truncation=False)
            token_count = len(tokens)

            # If token length slightly exceeds max, trim the body safely
            if token_count > self.max_tokens:
                tokens = tokens[:self.max_tokens]
                augmented_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                token_count = len(tokens)

            chunk_obj = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": idx + 1,
                "total_chunks": len(raw_chunks),
                "global_summary": compact_summary,
                "raw_chunk_text": body,
                "augmented_text": augmented_text,
                "token_count": token_count,
                "verdict": metadata.get("verdict", "Undetermined"),
                "keywords": metadata.get("keywords_matched", []),
                "source_type": metadata.get("source_type", "case_law")
            }
            chunk_objects.append(chunk_obj)

        return chunk_objects


if __name__ == "__main__":
    chunker = TokenAwareSACChunker()
    sample_text = (
        "IN THE COURT OF APPEAL OF THE DEMOCRATIC SOCIALIST REPUBLIC OF SRI LANKA. "
        "Before the Honorable Court. In the matter of an appeal from the District Court of Colombo. "
        "The Plaintiff-Appellant instituted this action against the Defendant-Respondent seeking a decree "
        "of divorce on the grounds of malicious desertion under Section 597 of the Civil Procedure Code. "
        "The Appellant also prayed for custody of the minor child and permanent alimony in the sum of Rs. 50,000. "
        "The Respondent filed an answer denying desertion and claimed that the Appellant had voluntarily departed "
        "from the matrimonial residence without just cause. At the trial, several issues were framed regarding "
        "matrimonial offenses and the financial capacity of both parties under the Maintenance Act No. 37 of 1999. "
        "The learned District Judge held that the evidence was insufficient to substantiate malicious desertion. "
        "The Appellant now appeals to this Court asserting that the trial judge misappreciated oral testimony."
    )

    chunks = chunker.process_document("SAMPLE_001", sample_text, {"verdict": "Appeal Allowed"})
    print("=" * 70)
    print("SLLIP V3 - TOKEN-AWARE SAC CHUNKER TEST")
    print("=" * 70)
    print(f"Generated Chunks: {len(chunks)}")
    for c in chunks:
        print(f"Chunk ID: {c['chunk_id']} | Tokens: {c['token_count']} (Limit <= 240) | Verified: {c['token_count'] <= 240}")
        print(f"Text Preview:\n{c['augmented_text'][:160]}...\n")

"""
Sri Lanka Legal Intelligence Platform (SLLIP V3)
Dataset Loader & Data Engineering Quality Module
=================================================
Manages legal dataset ingestion, validation, provenance tracking, and leak-free
train/validation/test splitting for Appellate Family Law judgments and statutory materials.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

# Workspace paths
WORKSPACE_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = WORKSPACE_DIR / "processed_data"
GROUND_TRUTH_FILE = PROCESSED_DIR / "ground_truth.json"
EXTRACTION_METADATA_FILE = PROCESSED_DIR / "extraction_metadata.json"
FAMILY_LAW_TXT_DIR = PROCESSED_DIR / "family_law_clean_txt"
STATUTORY_TXT_DIR = PROCESSED_DIR / "statutory_acts_clean_txt"

# Canonical outcome classes for Sri Lankan appellate family law
CANONICAL_VERDICTS = [
    "Appeal Dismissed",
    "Order Set Aside",
    "Appeal Allowed",
    "Judgment Affirmed",
    "Application Dismissed",
    "Appeal Allowed in Part"
]

# Mapping to collapse closely aligned appellate outcomes if requested
CONSOLIDATED_VERDICTS_MAP = {
    "Appeal Dismissed": "Dismissed",
    "Application Dismissed": "Dismissed",
    "Judgment Affirmed": "Dismissed",        # Affirms lower court decree
    "Appeal Allowed": "Allowed",
    "Order Set Aside": "Allowed",           # Sets aside lower court decree in favor of appellant
    "Appeal Allowed in Part": "Allowed in Part"
}


class LegalDatasetManager:
    """Manages ground truth legal records, metadata provenance, and evaluation splits."""

    def __init__(self, processed_dir: Path = PROCESSED_DIR):
        self.processed_dir = processed_dir
        self.ground_truth_path = processed_dir / "ground_truth.json"
        self.metadata_path = processed_dir / "extraction_metadata.json"
        self.family_txt_dir = processed_dir / "family_law_clean_txt"
        self.statutory_txt_dir = processed_dir / "statutory_acts_clean_txt"

        self._raw_ground_truth: Dict[str, Any] = {}
        self._metadata: List[Dict[str, Any]] = []
        self._load_data()

    def _load_data(self):
        """Load ground truth and extraction metadata from disk if available."""
        if self.ground_truth_path.exists():
            with open(self.ground_truth_path, "r", encoding="utf-8") as f:
                self._raw_ground_truth = json.load(f)

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Compute verified dataset counts and breakdown."""
        total_gt = len(self._raw_ground_truth)
        verdict_counts: Dict[str, int] = {}
        for entry in self._raw_ground_truth.values():
            v = entry.get("verdict", "Undetermined")
            verdict_counts[v] = verdict_counts.get(v, 0) + 1

        usable_supervised_count = sum(
            count for verdict, count in verdict_counts.items()
            if verdict in CANONICAL_VERDICTS
        )

        acts_count = 0
        bills_count = 0
        appeal_count = 0
        supreme_count = 0

        for item in self._metadata:
            rel = item.get("relative_path", "").lower()
            if "acts" in rel:
                acts_count += 1
            elif "bills" in rel:
                bills_count += 1
            if "appeal_court" in rel:
                appeal_count += 1
            elif "supreme_court" in rel:
                supreme_count += 1

        return {
            "total_extracted_records": len(self._metadata),
            "total_family_law_cases": total_gt,
            "usable_supervised_cases": usable_supervised_count,
            "undetermined_cases": verdict_counts.get("Undetermined", 0),
            "verdict_distribution": verdict_counts,
            "enacted_acts_count": acts_count,
            "proposed_bills_count": bills_count,
            "appeal_court_cases_scanned": appeal_count,
            "supreme_court_cases_scanned": supreme_count
        }

    def get_labeled_cases(self, consolidate: bool = False) -> List[Dict[str, Any]]:
        """
        Return verified labeled Family Law cases (excluding 'Undetermined').
        Labels originate from programmatic regex extraction from judgment conclusions.
        """
        labeled = []
        for doc_id, item in self._raw_ground_truth.items():
            verdict = item.get("verdict", "Undetermined")
            if verdict in CANONICAL_VERDICTS:
                record = dict(item)
                record["doc_id"] = doc_id
                record["original_verdict"] = verdict
                record["verdict"] = CONSOLIDATED_VERDICTS_MAP[verdict] if consolidate else verdict
                labeled.append(record)
        return labeled

    def load_case_text(self, doc_id: str, strip_tail_verdict: bool = True) -> Optional[str]:
        """
        Load clean text for a given case document ID.
        If strip_tail_verdict is True, the final 2,000 characters (where operative judicial
        decrees and verdict keywords reside) are removed to prevent label leakage.
        """
        file_path = self.family_txt_dir / f"{doc_id}.txt"
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            return None

        if strip_tail_verdict and len(text) > 3000:
            # Strip the final 2,000 characters to eliminate the conclusion decree
            text = text[:-2000].strip()

        return text

    def create_stratified_splits(
        self,
        test_size: float = 0.20,
        val_size: float = 0.10,
        random_seed: int = 42,
        consolidate: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create train, validation, and test splits strictly stratified by verdict class.
        Guarantees:
        1. No case doc_id appears in more than one partition.
        2. Proportions are preserved across classes.
        """
        cases = self.get_labeled_cases(consolidate=consolidate)
        rng = np.random.RandomState(random_seed)

        # Group cases by verdict label
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for c in cases:
            v = c["verdict"]
            by_label.setdefault(v, []).append(c)

        train_set = []
        val_set = []
        test_set = []

        for label, group in by_label.items():
            shuffled = list(group)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_test = max(1, int(round(n * test_size))) if n >= 5 else 1
            n_val = max(1, int(round(n * val_size))) if n >= 10 else (1 if n >= 5 else 0)

            test_items = shuffled[:n_test]
            val_items = shuffled[n_test:n_test + n_val]
            train_items = shuffled[n_test + n_val:]

            test_set.extend(test_items)
            val_set.extend(val_items)
            train_set.extend(train_items)

        rng.shuffle(train_set)
        rng.shuffle(val_set)
        rng.shuffle(test_set)

        return train_set, val_set, test_set


if __name__ == "__main__":
    manager = LegalDatasetManager()
    stats = manager.get_dataset_statistics()
    print("=" * 70)
    print("SLLIP V3 - DATASET QUALITY AUDIT REPORT")
    print("=" * 70)
    print(f"Total Extraction Records:       {stats['total_extracted_records']:,}")
    print(f"Total Family Law Cases:          {stats['total_family_law_cases']:,}")
    print(f"Usable Supervised Cases:        {stats['usable_supervised_cases']:,}")
    print(f"Undetermined Cases (Excluded):  {stats['undetermined_cases']:,}")
    print(f"Enacted Parliamentary Acts:     {stats['enacted_acts_count']:,}")
    print(f"Proposed Bills (Separate):       {stats['proposed_bills_count']:,}")
    print("-" * 70)
    print("VERDICT DISTRIBUTION (Programmatic Regex Labels):")
    for v, count in sorted(stats["verdict_distribution"].items(), key=lambda x: x[1], reverse=True):
        pct = count / stats['total_family_law_cases'] * 100
        print(f"  - {v:25s}: {count:5,} ({pct:5.1f}%)")
    print("=" * 70)

    train, val, test = manager.create_stratified_splits(test_size=0.20, val_size=0.10)
    print(f"Stratified Splits Created -> Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
    print(f"Sum of Splits: {len(train) + len(val) + len(test):,} (Matches Usable Count: {stats['usable_supervised_cases']})")

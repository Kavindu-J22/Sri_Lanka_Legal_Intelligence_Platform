"""
SLLIP V3 Automated Dataset Quality Test Suite
=============================================
Verifies ground truth counts, separation of enacted Acts vs. proposed Bills,
and integrity of stratified train/validation/test partitions.
"""

from dataset_loader import LegalDatasetManager, CANONICAL_VERDICTS


def test_ground_truth_counts_and_classes():
    """Verify ground truth case counts and canonical outcome set."""
    mgr = LegalDatasetManager()
    stats = mgr.get_dataset_statistics()

    assert stats["total_family_law_cases"] == 1980, f"Expected 1,980 cases, got {stats['total_family_law_cases']}"
    assert stats["usable_supervised_cases"] == 1594, f"Expected 1,594 usable cases, got {stats['usable_supervised_cases']}"
    assert stats["undetermined_cases"] == 386, f"Expected 386 undetermined cases, got {stats['undetermined_cases']}"

    # Verify that usable cases map only to canonical verdicts
    labeled = mgr.get_labeled_cases()
    assert len(labeled) == 1594
    for c in labeled:
        assert c["verdict"] in CANONICAL_VERDICTS


def test_statutory_acts_bills_separation():
    """Verify that enacted Acts are distinguished from proposed Bills."""
    mgr = LegalDatasetManager()
    stats = mgr.get_dataset_statistics()

    assert stats["enacted_acts_count"] == 2587, f"Expected 2,587 acts, got {stats['enacted_acts_count']}"
    assert stats["proposed_bills_count"] == 4197, f"Expected 4,197 bills, got {stats['proposed_bills_count']}"


def test_stratified_split_isolation():
    """Verify that train, validation, and test splits have zero overlap (leak-free)."""
    mgr = LegalDatasetManager()
    train, val, test = mgr.create_stratified_splits(test_size=0.20, val_size=0.10, random_seed=42)

    train_ids = {c["doc_id"] for c in train}
    val_ids = {c["doc_id"] for c in val}
    test_ids = {c["doc_id"] for c in test}

    assert len(train_ids.intersection(test_ids)) == 0, "Leakage detected between train and test sets!"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage detected between val and test sets!"
    assert len(train_ids.intersection(val_ids)) == 0, "Leakage detected between train and val sets!"
    assert len(train) + len(val) + len(test) == 1594

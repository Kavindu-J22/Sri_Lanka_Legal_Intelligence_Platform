"""
SLLIP V3 Analytics Manifest and Asset Verification Tests
=========================================================
Verifies that assets/analytics_manifest.json exists, adheres to schema v3.0.0,
contains verified non-zero counts, and eliminates cloud deployment cold-start zeros.
"""

import json
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
MANIFEST_FILE = ASSETS_DIR / "analytics_manifest.json"


def test_manifest_file_exists_and_valid_json():
    """Verify manifest file exists and is valid JSON."""
    assert MANIFEST_FILE.exists(), f"Missing manifest at {MANIFEST_FILE}"
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert data.get("schema_version") == "3.0.0"
    assert data.get("manifest_type") == "verified_corpus_snapshot"


def test_manifest_record_counts_positive():
    """Verify manifest contains non-zero verified record counts."""
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    totals = data.get("corpus_totals", {})
    assert totals.get("total_extracted_documents", 0) > 10000, "Expected >10,000 extracted documents"
    assert totals.get("total_family_law_cases", 0) > 1500, "Expected >1,500 family law cases"
    assert totals.get("enacted_acts_count", 0) > 2000, "Expected >2,000 enacted statutory acts"
    assert totals.get("proposed_bills_count", 0) > 3000, "Expected >3,000 proposed bills"


def test_manifest_verdict_distribution_integrity():
    """Verify verdict distribution covers canonical classes."""
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    v_dist = data.get("verdict_distribution", {})
    assert len(v_dist) >= 4, "Expected at least 4 classes with positive count"
    assert "Appeal Dismissed" in v_dist
    assert "Appeal Allowed" in v_dist
    assert "Order Set Aside" in v_dist
    assert sum(v_dist.values()) > 1000


def test_manifest_integrity_hashes_present():
    """Verify SHA-256 hashes exist for tracked dataset components."""
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    hashes = data.get("source_checksums", {})
    assert isinstance(hashes, dict)
    assert len(hashes) >= 2, "Expected source file integrity checksums"
    assert "ground_truth_sha256" in hashes
    assert "extraction_metadata_sha256" in hashes

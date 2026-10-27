# Sri Lanka Legal Intelligence Platform (SLLIP V3) — Data Provenance & Corpus Specification

This document details the provenance, legal source breakdown, automated ground truth annotation pipeline, and strict statutory separation implemented in SLLIP V3.

---

## 1. Corpus Hierarchy & Document Quantities

The platform is backed by a verified corpus of **12,706 extracted legal texts** spanning Sri Lankan appellate case law and parliamentary statutory enactments.

```
Total Extracted Records: 12,706
├── Case Law Subcorpus (Appellate Judgments): 5,922 scanned
│   ├── Court of Appeal of Sri Lanka: 4,034 cases
│   └── Supreme Court of Sri Lanka: 1,888 cases
│       └── Family Law Identified Domain: 1,980 cases
│           ├── Usable Supervised LJP Cases: 1,594 cases
│           └── Procedural / Undetermined Cases: 386 cases
└── Statutory Subcorpus: 6,784 total enactments
    ├── Enacted Acts of Parliament (Binding Law): 2,587 Acts
    └── Proposed Parliamentary Bills (Unenacted): 4,197 Bills
```

---

## 2. Separation of Enacted Acts vs Unenacted Bills

A fundamental legal distinction enforced in SLLIP V3 is that **proposed parliamentary bills do not constitute binding statutory law** until formally certified by the Speaker of Parliament under the Constitution of Sri Lanka.

- **Storage Separation**:
  - Enacted Acts are stored in `processed_data/statutory_acts_clean_txt/` and indexed in `statutory_acts_collection` (74,585 vectors).
  - Proposed Bills are strictly tagged as `unenacted_bill` in metadata to prevent accidental citation as binding authority.
- **Verification**:
  - Verified by automated unit tests in `tests/test_dataset.py::test_statutory_acts_bills_separation`.

---

## 3. Ground Truth Annotation Methodology

The 1,594 usable supervised cases were annotated using an automated decree extraction pipeline:

1. **Target Operative Conclusion Scanning**:
   - Extraction targets the operative final concluding paragraph where the judicial bench delivers its formal decree.
2. **Canonical Outcome Normalization**:
   - `Appeal Dismissed` (592 cases)
   - `Order Set Aside` (440 cases)
   - `Appeal Allowed` (300 cases)
   - `Judgment Affirmed` (142 cases)
   - `Application Dismissed` (90 cases)
   - `Appeal Allowed in Part` (30 cases)
3. **Audit Trail**:
   - Checksums for `ground_truth.json` (`f552a013...`) and `extraction_metadata.json` (`89eb3b39...`) are recorded in `assets/analytics_manifest.json` for reproducible integrity.

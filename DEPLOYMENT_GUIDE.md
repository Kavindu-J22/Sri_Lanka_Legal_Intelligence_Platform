# Sri Lanka Legal Intelligence Platform (SLLIP V3) — Cloud Deployment Guide

This guide describes how to deploy SLLIP V3 to **Streamlit Community Cloud** (or any cloud container environment) with zero missing analytics, active managed vector storage, and persisted supervised inference.

---

## 1. Architecture Overview for Cloud Deployment

In SLLIP V2, running on Streamlit Cloud resulted in zeroes across analytics tabs because raw files in `processed_data/` were gitignored and vector queries silently caught exceptions and returned 0. 

In SLLIP V3:
1. **Audited Analytics Manifest (`assets/analytics_manifest.json`)**: All corpus totals, verdict distributions, and integrity hashes are tracked in Git (~1.8 KB). The cloud UI loads verified data instantly without downloading multi-gigabyte raw datasets.
2. **Managed Vector Database (Qdrant Cloud)**: All 102,779 vectors (28,194 case law + 74,585 statutory acts) reside on a high-availability cloud cluster. The app connects over HTTPS (`port 6333`).
3. **Persisted Supervised Model Pipeline (`assets/models/supervised_ljp_pipeline.joblib`)**: Serialized scikit-learn pipeline providing calibrated class probabilities (51.89% accuracy benchmark) on cloud checkouts.
4. **Structured Health Diagnostics**: `vector_store.py` reports real connection status (`healthy`, `auth_error`, `connection_error`), never hiding errors behind silent zeros.

---

## 2. Prerequisites & Repository Preparation

Ensure your local branch is fully committed before pushing to GitHub:

```bash
# 1. Verify assets and scripts are tracked
git add assets/ scripts/ tests/ .gitignore requirements.txt app.py dataset_loader.py vector_store.py supervised_ljp.py hybrid_retriever.py research_experiments.py

# 2. Commit the changes
git commit -m "feat: SLLIP V3 cloud analytics manifest, calibrated inference, and Qdrant Cloud integration"

# 3. Push to your GitHub repository
git push origin main
```

> [!IMPORTANT]
> Never commit your `.env` file to Git! SLLIP's `.gitignore` explicitly excludes `.env` to protect your Qdrant API keys and LLM tokens.

---

## 3. Streamlit Community Cloud Setup

1. Log in to [share.streamlit.io](https://share.streamlit.io).
2. Click **New app**.
3. Select your repository: `Kavindu-J22/Sri_Lanka_Legal_Intelligence_Platform`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **Advanced settings...** -> **Secrets**.

### Streamlit Secrets Configuration

Copy and paste the following snippet into the **Secrets** text box (replace with your active Qdrant credentials):

```toml
# =============================================================================
# SLLIP V3 Streamlit Cloud Production Configuration
# =============================================================================

# Vector Database Backend: 'qdrant'
VECTOR_DB_BACKEND = "qdrant"

# Qdrant Managed Cloud Cluster Endpoint
QDRANT_URL = "https://566fc2d9-781b-4a8e-9971-50c9a94b7da6.australia-southeast1-0.gcp.cloud.qdrant.io:6333"

# Qdrant Cloud API Key
QDRANT_API_KEY = "your_qdrant_api_key_here"

# Application Mode: 'production' or 'hosted' enforces read-only vector safety
APP_MODE = "production"

# (Optional) External Multi-Agent LLM Generation Key (Atria ASI)
ATRIA_API_KEY = "your_atria_api_key_here"
```

7. Click **Save** and then **Deploy!**

---

## 4. Post-Deployment Verification Checklist

Once the application finishes booting on Streamlit Cloud:

| Verification Item | Expected Value / Behavior | Status Check |
|---|---|---|
| **Sidebar Status** | Shows `Vector Provider: Qdrant Cloud (Managed)` with green `🟢 Healthy (Live)` | Confirms HTTPS Qdrant connectivity |
| **Case Law Vectors** | Displays `28,194` in sidebar and Tab 2 chart | Verifies Collection A count |
| **Statutory Vectors** | Displays `74,585` in sidebar and Tab 2 chart | Verifies Collection B count |
| **Corpus Snapshot** | Displays `v3.0.0 (12,706 extractions)` in Tab 2 | Verifies `assets/analytics_manifest.json` loaded |
| **Enacted Acts vs Bills** | `2,587` Acts and `4,197` Bills in separate metrics | Verifies strict separation |
| **Tab 1 Prediction** | Displays Calibrated Probability bar chart | Confirms `supervised_ljp_pipeline.joblib` loaded |
| **Tab 3 Benchmarks** | Displays 318-case supervised benchmark, retrieval quality table, and A-E matrix | Confirms benchmark results loaded |

---

## 5. Troubleshooting & Diagnostics

- **Error: "Vector store connection error (status: connection_error)"**
  - Verify that `QDRANT_URL` includes the `https://` prefix and port `:6333`.
  - Check that the Qdrant Cloud cluster in Australia Southeast is active in your Qdrant Console.
- **Corpus Analytics Missing / Zeroes**
  - Ensure `assets/analytics_manifest.json` is committed and pushed to GitHub. Check GitHub file tree.
- **Prediction falls back to heuristic default**
  - Verify that `assets/models/supervised_ljp_pipeline.joblib` was pushed to Git (~417 KB, well within GitHub's 100 MB file limit).

# ⚖️ Sri Lanka Legal Intelligence Platform (SLLIP)
## Complete Setup & Manual Execution Guide (Client & Developer Manual)

---

## 📌 1. System Overview & System Requirements

The **Sri Lanka Legal Intelligence Platform (SLLIP)** is an executive-grade AI legal platform designed for **Appellate Family Law (Matrimonial, Custody, Maintenance, Alimony Appeals)** in Sri Lanka. It combines:
- **Legacy Sinhala Font Converter** (FM-Abhaya / DL-Manel to Sinhala Unicode)
- **Summary-Augmented Chunking (SAC)** to prevent Document-Level Retrieval Mismatch (DRM)
- **ChromaDB Dual-Vector Store** (102,779 vectors indexed using `all-MiniLM-L6-v2`)
- **Multi-Agent Adversarial RAG Framework** (Investigator, Defense Counsel, Prosecution Counsel, Judge Agent with Softmax LJP Probability Classifier)
- **Streamlit Web Application** for interactive legal analysis and evaluation metrics.

### System Hardware & Software Requirements
| Component | Minimum Requirement | Recommended |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11, Ubuntu 20.04+, or macOS | Windows 11 / Ubuntu 22.04 |
| **Python Version** | Python 3.10.x or 3.11.x | Python 3.10.11 |
| **RAM** | 8 GB | 16 GB or higher |
| **Disk Space** | 10 GB free space | 20 GB free SSD space |
| **GPU / CPU** | Multi-core CPU (8+ cores) | Multi-core CPU or NVIDIA GPU |

---

## 📁 2. Directory Structure

```text
Sri_Lanka_Legal_Intelligence_Platform/
├── 01_advanced_pdf_processor.py       # Phase 1: PDF Extraction & Legacy Sinhala Font Normalizer
├── 02_summary_augmented_chunker.py     # Phase 2: Summary-Augmented Chunking (SAC)
├── 03_vector_db_ingestion.py          # Phase 3: ChromaDB Dual-Vector Database Ingestion
├── 04_multi_agent_ljp_framework.py    # Phase 4: 4-Agent Debate & Softmax LJP Framework
├── app.py                             # Phase 5: Streamlit Web Application Dashboard
├── requirements.txt                   # Python Package Dependencies
├── SETUP_AND_USER_GUIDE.md            # Client Manual & Execution Guide (This File)
├── .gitignore                         # Git exclusion rules for large datasets
├── sllip_raw_dataset/                 # Raw PDF Appellate Judgments & Statutory Acts
├── processed_data/                    # Processed JSON Datasets & Ground Truth Files
│   ├── ground_truth.json
│   ├── family_law_cases.json
│   ├── statutory_acts.json
│   ├── sac_chunks.json
│   └── phase4_evaluation_metrics.json
└── vector_db/                         # Persistent ChromaDB Database (102,779 Vectors)
    ├── chroma.sqlite3
    └── .gitkeep
```

---

## 🚀 3. Step-by-Step Installation (On a New PC)

Follow these steps to set up the platform on a fresh computer:

### Step 1: Install Python
1. Download **Python 3.10** or **Python 3.11** from [python.org](https://www.python.org/downloads/).
2. During installation, make sure to check **"Add Python to PATH"**.

### Step 2: Open Terminal / PowerShell
Open **PowerShell** (Windows) or **Terminal** (macOS/Linux) and navigate to the project directory:
```bash
cd "C:\Path\To\Sri_Lanka_Legal_Intelligence_Platform"
```

### Step 3: Create & Activate Virtual Environment
Create an isolated Python environment to prevent library conflicts:

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies
Upgrade `pip` and install all required Python packages from `requirements.txt`:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ 4. How to Run the Web Application Dashboard

To launch the interactive **Sri Lanka Legal Intelligence Platform** user interface:

### Command:
```bash
streamlit run app.py
```

### Expected Terminal Output:
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```

Open **`http://localhost:8501`** in Google Chrome, Microsoft Edge, or Firefox.

---

## 📖 5. Dashboard User Guide (For Clients & End-Users)

The web dashboard is split into **3 Main Tabs**:

### Tab 1: ⚖️ Legal Judgment Prediction & Multi-Agent Debate (Main Interface)
1. **Input Case Facts**: Paste or type the factual details of an Appellate Family Law case (Divorce, Custody, Maintenance, Alimony) into the text box. Or select a pre-loaded case from the sidebar presets.
2. **Click "🚀 Run Multi-Agent Analysis"**: The system will run the 4 AI Agents in sequence.
3. **View Summary Cards**:
   - **Predicted Verdict** (e.g., *Order Set Aside*, *Appeal Allowed*, *Appeal Dismissed*).
   - **Confidence Probability** (Softmax confidence percentage).
   - **Execution Time** (seconds).
   - **Precedents Found** (Count of relevant historical appellate cases retrieved).
4. **Explore Live 4-Agent Debate**:
   - `🔍 1. Investigator Agent`: Displays retrieved precedents & statutes with cosine similarity scores.
   - `🛡️ 2. Defense Agent`: Shows the legal brief constructed for the Appellant.
   - `⚔️ 3. Prosecutor Agent`: Shows the rebuttal brief constructed for the Respondent.
   - `👨‍⚖️ 4. Judge Agent`: Displays the official Judicial Decree & the horizontal Softmax probability chart.

### Tab 2: 📊 Vector DB & Corpus Analytics
- Displays live statistics for the **15,017 PDFs scanned**, **12,275 legacy converted files**, **1,980 Family Law Judgments**, **6,784 Statutory Acts**, and **102,779 vectors**.
- Shows verdict distribution donut charts and collection breakdowns.

### Tab 3: 📈 Evaluation Metrics & RAGAS Benchmarks
- Displays quantitative performance metrics evaluated over isolated test cases:
  - **Classification Accuracy**: `55.00%`
  - **Macro F1-Score**: `61.33%`
  - **RAGAS Faithfulness**: `74.95%`
  - **RAGAS Context Relevance**: `77.31%`

---

## ⚙️ 6. Re-building the Data Pipeline (Developer Manual)

If you add new PDF judgments or wish to re-process the entire pipeline from raw PDFs to vector DB, run the scripts in numerical order:

### Phase 1: PDF Extraction & Legacy Font Conversion
Extracts raw PDFs, converts non-Unicode Sinhala text (FM-Abhaya, DL-Manel), isolates Family Law cases & Statutory Acts:
```bash
python 01_advanced_pdf_processor.py
```

### Phase 2: Summary-Augmented Chunking (SAC)
Generates 150-word global document summaries and creates SAC chunks:
```bash
python 02_summary_augmented_chunker.py
```

### Phase 3: Vector DB Ingestion (ChromaDB)
Embeds SAC chunks using PyTorch CPU threading and populates ChromaDB persistent collections (`vector_db/`):
```bash
python 03_vector_db_ingestion.py
```

### Phase 4: Multi-Agent Framework & Benchmark Evaluation
Runs the 4-Agent reasoning pipeline and evaluates benchmark metrics:
```bash
python 04_multi_agent_ljp_framework.py --eval
```

---

## 🔧 7. Troubleshooting & FAQs

#### Q1: "Port 8501 is already in use"
**Solution**: Specify a different port when launching Streamlit:
```bash
streamlit run app.py --server.port 8502
```

#### Q2: "ModuleNotFoundError: No module named 'streamlit'"
**Solution**: Ensure your virtual environment is activated before running the app (`.\venv\Scripts\activate`).

#### Q3: "Garbage characters appearing in terminal on Windows"
**Solution**: The code automatically sets UTF-8 output. If running via Command Prompt, set code page to UTF-8 before running:
```cmd
chcp 65001
streamlit run app.py
```

#### Q4: "How do I deploy this to Streamlit Community Cloud or a remote server?"
**Solution**: Refer to the deployment instructions in [`SETUP_AND_USER_GUIDE.md`](file:///f:/New%20NIBM%20Reserch%202026/Sri_Lanka_Legal_Intelligence_Platform/SETUP_AND_USER_GUIDE.md). Set `VECTOR_DB_ZIP_URL` in environment variables if hosting vector files externally.

---



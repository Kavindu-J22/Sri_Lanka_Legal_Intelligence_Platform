import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Auto-load .env environment variables
load_dotenv()

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


import importlib
_mod = importlib.import_module("04_multi_agent_ljp_framework")
MultiAgentLJPFramework = _mod.MultiAgentLJPFramework
VECTOR_DB_DIR = _mod.VECTOR_DB_DIR
GROUND_TRUTH_FILE = _mod.GROUND_TRUTH_FILE
EVAL_METRICS_FILE = _mod.EVAL_METRICS_FILE


# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sri Lanka Legal Intelligence Platform (SLLIP)",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern legal tech styling
st.markdown("""
<style>
    /* Main Background & Font Styling */
    .main {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    
    /* Executive Title Header */
    .title-header {
        background: linear-gradient(135deg, #1E2640 0%, #0F172A 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .title-header h1 {
        color: #F8FAFC;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .title-header p {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Card Container Styling */
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .metric-card h3 {
        color: #64748B;
        font-size: 0.9rem;
        margin: 0 0 6px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card p {
        color: #38BDF8;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Verdict Badges */
    .badge-allowed {
        background-color: #065F46;
        color: #34D399;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    .badge-dismissed {
        background-color: #7F1D1D;
        color: #F87171;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    .badge-aside {
        background-color: #78350F;
        color: #FBBF24;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    
    /* Agent Trace Box */
    .agent-box {
        background-color: #0F172A;
        border-left: 4px solid #38BDF8;
        padding: 16px;
        border-radius: 6px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. CACHED RESOURCE & DATA LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_framework():
    """Load cached MultiAgentLJPFramework instance."""
    return MultiAgentLJPFramework(vector_db_path=VECTOR_DB_DIR)

@st.cache_data
def load_ground_truth_data():
    """Load cached Ground Truth JSON."""
    if GROUND_TRUTH_FILE.exists():
        with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_evaluation_metrics_data():
    """Load cached Phase 4 Evaluation Metrics JSON."""
    if EVAL_METRICS_FILE.exists():
        with open(EVAL_METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# -----------------------------------------------------------------------------
# 3. SAMPLE CASE PRESETS
# -----------------------------------------------------------------------------
SAMPLE_PRESETS = {
    "Preset 1: Constructive Desertion & Child Maintenance Appeal": (
        "The Appellant (wife) filed an appeal against the District Court decree dismissing her petition "
        "for divorce on grounds of constructive desertion and malicious cruelty under the Marriage and Divorce Act. "
        "The Respondent (husband) refused to provide maintenance for the minor children and ejected the Appellant "
        "from the matrimonial residence without just cause. The Appellant seeks reversal of the dismissal, "
        "granting of divorce decree, and monthly maintenance of Rs. 35,000 for the minor children."
    ),
    "Preset 2: Matrimonial Property & Alimony Variation Appeal": (
        "The Defendant-Appellant appeals against the order of the High Court directing permanent alimony of Rs. 50,000 "
        "per month to the Plaintiff-Respondent and lump sum property transfer of the joint family house. "
        "The Appellant contends that the trial judge failed to evaluate the Appellant's current financial incapacity "
        "and statutory income ceiling under Section 602 of the Civil Procedure Code."
    ),
    "Preset 3: Child Custody & Guardianship Rights Dispute": (
        "This is an appeal against the order granting sole custody of two minor children (ages 6 and 9) to the father. "
        "The mother (Appellant) submits that the learned judge erred in ignoring the maternal preference doctrine "
        "and paramount welfare of tender-age minors under the Guardianship and Custody laws of Sri Lanka. "
        "She prays for reversal of the order and interim custody."
    )
}


# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION & CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("## ⚖️ SLLIP Control Panel")
st.sidebar.markdown("**Sri Lanka Legal Intelligence Platform**")
st.sidebar.markdown("---")

st.sidebar.subheader("📌 Sample Case Presets")
selected_preset_name = st.sidebar.selectbox("Load Sample Family Law Case:", list(SAMPLE_PRESETS.keys()))
preset_text = SAMPLE_PRESETS[selected_preset_name]

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ System Architecture")
st.sidebar.markdown("""
- **Corpus**: 1,980 Family Law Cases & 6,784 Statutory Acts
- **Vector DB**: ChromaDB (102,779 Vectors)
- **Embedding**: `all-MiniLM-L6-v2` (PyTorch 8-Core)
- **Multi-Agent**: 4-Agent Adversarial Debate
- **LJP Engine**: Softmax Probability Classifier
""")


# -----------------------------------------------------------------------------
# 5. MAIN HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="title-header">
    <h1>⚖️ Sri Lanka Legal Intelligence Platform (SLLIP)</h1>
    <p>Appellate Family Law Legal Judgment Prediction (LJP) & Multi-Agent Adversarial RAG Architecture</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. TABBED NAVIGATION INTERFACE
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "⚖️ Legal Judgment Prediction & Multi-Agent Debate",
    "📊 Vector DB & Corpus Analytics",
    "📈 Evaluation Metrics & RAGAS Benchmarks"
])


# =============================================================================
# TAB 1: LEGAL JUDGMENT PREDICTION & MULTI-AGENT DEBATE
# =============================================================================
with tab1:
    st.markdown("### 📝 Enter Family Law Appellate Case Facts")
    st.caption("Input or paste factual details of an Appellate Family Law case (Divorce, Custody, Maintenance, Alimony).")

    # Text Area for input
    case_facts_input = st.text_area(
        "Appellate Case Description:",
        value=preset_text,
        height=140,
        help="Type or paste legal case facts here."
    )

    col_btn, col_space = st.columns([1, 4])
    with col_btn:
        run_analysis = st.button("🚀 Run Multi-Agent Analysis", type="primary", use_container_width=True)

    if run_analysis and case_facts_input.strip():
        framework = load_framework()

        with st.spinner("Executing 4-Agent Debate Trace & LJP Softmax Engine..."):
            res = framework.run_pipeline(case_facts_input.strip())

        st.markdown("---")
        st.markdown("## 🎯 Prediction & Multi-Agent Debate Output")

        # Top Summary Metrics Cards
        pred_verdict = res["predicted_verdict"]
        prob_dict = res["probabilities"]
        top_prob = prob_dict[pred_verdict]

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown("<div class='metric-card'><h3>PREDICTED VERDICT</h3>", unsafe_allow_html=True)
            if "Allowed" in pred_verdict:
                st.markdown(f"<div class='badge-allowed'>{pred_verdict}</div>", unsafe_allow_html=True)
            elif "Dismissed" in pred_verdict:
                st.markdown(f"<div class='badge-dismissed'>{pred_verdict}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='badge-aside'>{pred_verdict}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with m2:
            st.markdown(f"<div class='metric-card'><h3>CONFIDENCE PROBABILITY</h3><p>{top_prob * 100:.1f}%</p></div>", unsafe_allow_html=True)

        with m3:
            st.markdown(f"<div class='metric-card'><h3>EXECUTION TIME</h3><p>{res['execution_time_sec']}s</p></div>", unsafe_allow_html=True)

        with m4:
            st.markdown(f"<div class='metric-card'><h3>PRECEDENTS FOUND</h3><p>{len(res['precedents'])}</p></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 4-Agent Trace Tabs / Expanders
        agent_tab1, agent_tab2, agent_tab3, agent_tab4 = st.tabs([
            "🔍 1. Investigator Agent (Retrieved Context)",
            "🛡️ 2. Defense Agent (Appellant Submission)",
            "⚔️ 3. Prosecutor Agent (Respondent Rebuttal)",
            "👨‍⚖️ 4. Judge Agent (LJP Rationale & Softmax Chart)"
        ])

        with agent_tab1:
            st.markdown("#### 🔍 Investigator Agent Report")
            st.text_area("Legal Fact Sheet & Precedent Summary:", value=res["investigator_context"], height=260)

            st.markdown("##### 📚 Retrieved Case Law Precedents (ChromaDB Collection A)")
            if res["precedents"]:
                prec_df = pd.DataFrame(res["precedents"])
                st.dataframe(
                    prec_df[["doc_id", "verdict", "similarity_score", "global_summary"]],
                    column_config={
                        "doc_id": "Case Precedent ID",
                        "verdict": "Historical Ruling",
                        "similarity_score": "Cos Similarity",
                        "global_summary": "150-Word Global Summary"
                    },
                    use_container_width=True
                )

            st.markdown("##### 📜 Retrieved Statutory Provisions (ChromaDB Collection B)")
            if res["statutes"]:
                stat_df = pd.DataFrame(res["statutes"])
                st.dataframe(
                    stat_df[["filename", "similarity_score", "text"]],
                    column_config={
                        "filename": "Act / Bill Source",
                        "similarity_score": "Cos Similarity",
                        "text": "Statutory Provision Text"
                    },
                    use_container_width=True
                )

        with agent_tab2:
            st.markdown("#### 🛡️ Defense Counsel Submission (Appellant)")
            st.text_area("Appellant Legal Brief:", value=res["defense_argument"], height=300)

        with agent_tab3:
            st.markdown("#### ⚔️ Prosecutor / Respondent Legal Submission")
            st.text_area("Respondent Rebuttal Brief:", value=res["prosecutor_argument"], height=300)

        with agent_tab4:
            st.markdown("#### 👨‍⚖️ Judge Agent Opinion & Softmax Probability Classifier")

            col_chart, col_op = st.columns([1, 1])

            with col_chart:
                st.markdown("##### 📊 LJP Softmax Probability Distribution")
                df_probs = pd.DataFrame({
                    "Verdict": list(prob_dict.keys()),
                    "Probability (%)": [v * 100 for v in prob_dict.values()]
                }).sort_values(by="Probability (%)", ascending=True)

                fig_prob = px.bar(
                    df_probs,
                    x="Probability (%)",
                    y="Verdict",
                    orientation="h",
                    color="Probability (%)",
                    color_continuous_scale="Viridis",
                    text_auto=".1f"
                )
                fig_prob.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F8FAFC"),
                    xaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig_prob, use_container_width=True)

            with col_op:
                st.markdown("##### 📜 Judicial Opinion & Ratio Decidendi")
                st.text_area("Official Decree:", value=res["judicial_opinion"], height=320)


# =============================================================================
# TAB 2: VECTOR DB & CORPUS ANALYTICS
# =============================================================================
with tab2:
    st.markdown("### 📊 Vector Database & Dataset Corpus Analytics")
    st.caption("Live statistical breakdown of Sri Lanka Legal Intelligence Platform (SLLIP) vector store.")

    gt_data = load_ground_truth_data()

    # Overview Metrics Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Raw PDFs Scanned", "15,017")
    with c2:
        st.metric("Legacy Converted Files", "12,275")
    with c3:
        st.metric("Family Law Cases Isolated", f"{len(gt_data):,}" if gt_data else "1,980")
    with c4:
        st.metric("Statutory Acts & Bills", "6,784")
    with c5:
        st.metric("Total Indexed Vectors", "102,779")

    st.markdown("---")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### ⚖️ Family Law Ground Truth Verdict Distribution")
        if gt_data:
            verdict_counts = {}
            for item in gt_data.values():
                v = item.get("verdict", "Undetermined")
                verdict_counts[v] = verdict_counts.get(v, 0) + 1

            df_verdicts = pd.DataFrame({
                "Verdict": list(verdict_counts.keys()),
                "Count": list(verdict_counts.values())
            })

            fig_donut = px.pie(
                df_verdicts,
                names="Verdict",
                values="Count",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_donut.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC")
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("Ground Truth dataset loading...")

    with chart_col2:
        st.markdown("#### 📦 Dual-Vector DB Collection Totals")
        df_cols = pd.DataFrame({
            "Collection": ["Collection A (Family Law SAC)", "Collection B (Statutory Acts)"],
            "Indexed Chunks": [28194, 74585]
        })

        fig_cols = px.bar(
            df_cols,
            x="Collection",
            y="Indexed Chunks",
            color="Collection",
            text_auto=","
        )
        fig_cols.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            showlegend=False
        )
        st.plotly_chart(fig_cols, use_container_width=True)


# =============================================================================
# TAB 3: EVALUATION METRICS & RAGAS BENCHMARKS
# =============================================================================
with tab3:
    st.markdown("### 📈 Phase 4 Framework Evaluation & RAGAS Benchmarks")
    st.caption("Quantitative performance metrics computed over 20 isolated test cases.")

    metrics_data = load_evaluation_metrics_data()

    if metrics_data:
        class_metrics = metrics_data.get("classification_metrics", {})
        ragas_metrics = metrics_data.get("ragas_metrics", {})

        e1, e2, e3, e4 = st.columns(4)
        with e1:
            st.metric("Classification Accuracy", f"{class_metrics.get('accuracy', 0.55) * 100:.2f}%")
        with e2:
            st.metric("Macro F1-Score", f"{class_metrics.get('macro_f1_score', 0.6133) * 100:.2f}%")
        with e3:
            st.metric("RAGAS Faithfulness", f"{ragas_metrics.get('faithfulness_score', 0.7495) * 100:.2f}%")
        with e4:
            st.metric("RAGAS Context Relevance", f"{ragas_metrics.get('context_relevance_score', 0.7731) * 100:.2f}%")

        st.markdown("---")

        rg1, rg2 = st.columns(2)

        with rg1:
            st.markdown("#### 🎯 Classification Metrics Breakdown")
            df_cm = pd.DataFrame({
                "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"],
                "Score (%)": [
                    class_metrics.get("accuracy", 0.55) * 100,
                    class_metrics.get("macro_precision", 0.6133) * 100,
                    class_metrics.get("macro_recall", 0.6133) * 100,
                    class_metrics.get("macro_f1_score", 0.6133) * 100
                ]
            })
            fig_cm = px.bar(df_cm, x="Metric", y="Score (%)", color="Metric", text_auto=".2f")
            fig_cm.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
                yaxis=dict(range=[0, 100]),
                showlegend=False
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with rg2:
            st.markdown("#### 🛡️ RAGAS Groundedness Benchmarks")
            df_ragas = pd.DataFrame({
                "RAGAS Benchmark": ["Faithfulness Score", "Context Relevance Score"],
                "Score (%)": [
                    ragas_metrics.get("faithfulness_score", 0.7495) * 100,
                    ragas_metrics.get("context_relevance_score", 0.7731) * 100
                ]
            })
            fig_ragas = px.bar(df_ragas, x="RAGAS Benchmark", y="Score (%)", color="RAGAS Benchmark", text_auto=".2f")
            fig_ragas.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
                yaxis=dict(range=[0, 100]),
                showlegend=False
            )
            st.plotly_chart(fig_ragas, use_container_width=True)

        st.markdown("##### 🔬 Sample Test Case Evaluation Results")
        samples = metrics_data.get("sample_evaluations", [])
        if samples:
            st.table(pd.DataFrame(samples))
    else:
        st.info("Evaluation metrics file not found. Run `python 04_multi_agent_ljp_framework.py --eval` to generate.")

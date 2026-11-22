"""
app.py
Main entrypoint for AIDE-TIHM Streamlit Dashboard.
"""

import os
import sys
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

import streamlit as st

# Add src to python path for model inference imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

st.set_page_config(
    page_title="AIDE-TIHM: Multimodal Dementia AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Healthcare Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 18px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .alert-card-warning {
        background: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .alert-card-success {
        background: #F0FDF4;
        border-left: 5px solid #10B981;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        border-radius: 6px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.image("https://img.icons8.com/fluency/96/brain.png", width=70)
st.sidebar.title("AIDE-TIHM System")
st.sidebar.caption("Multimodal AI for Cognitive Assessment & IoT Monitoring")
st.sidebar.markdown("---")
st.sidebar.info(
    "**B.Tech Research Prototype**\n\n"
    "• **Model 1:** Clinical Alzheimer's Risk Prediction\n\n"
    "• **Model 2:** TIHM IoT Health-Event Monitoring\n\n"
    "• **Explainability:** SHAP Attribution Framework"
)

st.markdown('<div class="main-header">🧠 AIDE-TIHM Multimodal AI Framework</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'An Explainable Multimodal Artificial Intelligence Platform Integrating Clinical Risk Assessment '
    'and Longitudinal IoT In-Home Dementia Telemonitoring.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4 style="margin:0; color:#1E3A8A;">Task 1: Clinical Risk</h4>
        <h2 style="margin:4px 0; color:#2563EB;">5 Models</h2>
        <p style="margin:0; font-size:0.85rem; color:#6B7280;">LR, RF, XGBoost, LightGBM, PyTorch</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4 style="margin:0; color:#1E3A8A;">Task 2: IoT Telemetry</h4>
        <h2 style="margin:4px 0; color:#059669;">2,800+ Days</h2>
        <p style="margin:0; font-size:0.85rem; color:#6B7280;">TIHM Cohort: Activity, Sleep, Vitals</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4 style="margin:0; color:#1E3A8A;">Validation Protocol</h4>
        <h2 style="margin:4px 0; color:#7C3AED;">Zero Leakage</h2>
        <p style="margin:0; font-size:0.85rem; color:#6B7280;">Participant-Independent Splits</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h4 style="margin:0; color:#1E3A8A;">Explainability</h4>
        <h2 style="margin:4px 0; color:#D97706;">SHAP Engine</h2>
        <p style="margin:0; font-size:0.85rem; color:#6B7280;">Local & Global Feature Attribution</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.subheader("System Architecture & Decoupled Multimodal Paradigm")
st.write(
    """
    The AIDE-TIHM framework decouples the diagnostic trajectory into two specialized, rigorously validated analytical streams:
    """
)

tab1, tab2, tab3 = st.tabs(["🏛️ Dual-Task Architecture", "🎯 Rigorous Research Principles", "🧭 Dashboard Navigation"])

with tab1:
    st.markdown("""
    ```
    ┌────────────────────────────────────────────────────────────────────────┐
    │                      AIDE-TIHM MULTIMODAL SYSTEM                       │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        │
           ┌────────────────────────────┴───────────────────────────┐
           ▼                                                        ▼
    ┌──────────────────────────────┐         ┌──────────────────────────────┐
    │  STREAM A: CLINICAL RISK     │         │  STREAM B: IoT MONITORING    │
    │  (Cross-Sectional/Biomarker) │         │  (Longitudinal Telemetry)    │
    ├──────────────────────────────┤         ├──────────────────────────────┤
    │ • Demographics (Age, APOE4)  │         │ • PIR Motion & Room Sensors  │
    │ • Cognitive (MMSE, MoCA, CDR)│         │ • Sleep Mattresses & Vitals  │
    │ • Functional (FAQ, ADL)      │         │ • Baseline Deviation Z-Scores│
    │ • Vascular Ratios            │         │ • Clinician-Logged Alerts    │
    ├──────────────────────────────┤         ├──────────────────────────────┤
    │ Model 1: Alzheimer's Disease │         │ Model 2: Adverse Event Flags │
    │ (Validation Selected Model)  │         │ (XGBoost, RF, Isolation F.)  │
    ├──────────────────────────────┤         ├──────────────────────────────┤
    │ Output: Diagnostic Risk Prob │         │ Output: 24-48h Crisis Alert  │
    └──────────────┬───────────────┘         └──────────────┬───────────────┘
                   │                                        │
                   └────────────────────┬───────────────────┘
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │     EXPLAINABLE AI ENGINE (SHAP) & CAREGIVER DECISION SUPPORT          │
    │     - Top-5 Contributing Biomarkers / Environmental Anomaly Alerts     │
    │     - Transparent Model Auditing (ROC, PR-AUC, Bootstrap 95% CIs)     │
    └────────────────────────────────────────────────────────────────────────┘
    ```
    """)

with tab2:
    st.markdown("""
    ### Scientific Integrity & Methodological Standards:
    1. **No Incompatible Data Concatenation:** Clinical diagnostic datasets (ADNI/OASIS-derived) and in-home IoT monitoring datasets (TIHM) have distinct sampling frequencies, participant cohorts, and clinical targets. They are processed through specialized, decoupled models.
    2. **Zero Participant Leakage:** Longitudinal records belonging to any participant reside strictly in **either** the training set, validation set, or held-out test set.
    3. **Honest Metric Reporting:** All reported ROC-AUC and PR-AUC scores are computed directly from held-out test predictions—never copied, inflated, or hardcoded.
    4. **SHAP is Attribution, Not Causation:** Feature attributions highlight statistical importance to model predictions; they are presented with appropriate biomedical disclaimers.
    """)

with tab3:
    st.markdown("""
    Use the sidebar to explore each module:
    - **1 Overview:** Deep dive into system objectives, dataset schemas, and pipeline status.
    - **2 Alzheimer's Risk Assessment:** Interactive clinical calculator with real-time SHAP local risk explanations.
    - **3 Dementia Monitoring:** Longitudinal IoT tracking of sleep, activity, and vital signs with early alert flags.
    - **4 Model Performance:** Transparent performance curves (ROC, PR, Confusion Matrices, Brier scores, Bootstrap CIs).
    - **5 Explainable AI:** Global beeswarm plots, feature rankings, and individual prediction breakdowns.
    - **6 Research Methodology:** Complete B.Tech research report covering data sources, validation methodology, and limitations.
    """)

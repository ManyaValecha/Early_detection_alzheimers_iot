"""
2_Alzheimer_Risk_Assessment.py
Interactive clinical risk scoring tool with real-time SHAP local explanation.
"""

import os
import sys
import json
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# Ensure robust multi-environment path resolution
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_current_dir, "..", "..", "src"))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from feature_engineering import extract_clinical_features
    from explainability import SHAPExplainerEngine
except ImportError:
    from aide_tihm.src.feature_engineering import extract_clinical_features
    from aide_tihm.src.explainability import SHAPExplainerEngine

st.set_page_config(page_title="Alzheimer's Risk Assessment", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .page-title { font-size: 2rem; font-weight: 700; color: #1E3A8A; }
    .risk-badge-high { background-color: #FEE2E2; color: #991B1B; padding: 6px 14px; border-radius: 8px; font-weight: 700; font-size: 1.1rem; }
    .risk-badge-med { background-color: #FEF3C7; color: #92400E; padding: 6px 14px; border-radius: 8px; font-weight: 700; font-size: 1.1rem; }
    .risk-badge-low { background-color: #D1FAE5; color: #065F46; padding: 6px 14px; border-radius: 8px; font-weight: 700; font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🩺 Clinical Alzheimer\'s Risk Assessment</div>', unsafe_allow_html=True)
st.caption("Validated Diagnostic Risk Stratification with Local SHAP Feature Attribution")

st.markdown("---")

# Load model and scaler
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
model_path = os.path.join(models_dir, "alzheimer_best_model.joblib")
scaler_path = os.path.join(models_dir, "alzheimer_scaler.joblib")
feats_path = os.path.join(models_dir, "alzheimer_features.json")

if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feats_path)):
    st.error("Trained model artifacts not found. Please run the training pipeline first (`python src/train_pipeline.py`).")
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
with open(feats_path, "r") as f:
    feature_names = json.load(f)

# Presets for fast demonstration
st.sidebar.header("Sample Patient Presets")
preset = st.sidebar.selectbox(
    "Load Clinical Archetype:",
    ["Custom Values", "Patient A: Cognitively Intact Control", "Patient B: Early Mild Cognitive Impairment", "Patient C: Clinically Evident Alzheimer's"]
)

# Preset defaults
if preset == "Patient A: Cognitively Intact Control":
    p_age, p_gender, p_edu, p_apoe, p_fam = 71, "Female", 16, 0, 0
    p_mmse, p_moca, p_cdr, p_faq, p_adl = 29.0, 28.0, 0.0, 1.0, 96.0
    p_mem, p_beh, p_dep = 0, 0, 2.0
    p_sbp, p_dbp, p_chol, p_ldl, p_hdl, p_trig = 124.0, 76.0, 190.0, 110.0, 58.0, 130.0
    p_diab, p_hyp, p_bmi = 0, 0, 24.2
elif preset == "Patient B: Early Mild Cognitive Impairment":
    p_age, p_gender, p_edu, p_apoe, p_fam = 76, "Male", 12, 1, 1
    p_mmse, p_moca, p_cdr, p_faq, p_adl = 24.0, 22.0, 0.5, 7.0, 82.0
    p_mem, p_beh, p_dep = 1, 0, 5.0
    p_sbp, p_dbp, p_chol, p_ldl, p_hdl, p_trig = 138.0, 82.0, 215.0, 132.0, 46.0, 168.0
    p_diab, p_hyp, p_bmi = 1, 1, 27.5
elif preset == "Patient C: Clinically Evident Alzheimer's":
    p_age, p_gender, p_edu, p_apoe, p_fam = 80, "Female", 10, 2, 1
    p_mmse, p_moca, p_cdr, p_faq, p_adl = 18.0, 15.0, 1.0, 18.0, 58.0
    p_mem, p_beh, p_dep = 1, 1, 7.5
    p_sbp, p_dbp, p_chol, p_ldl, p_hdl, p_trig = 145.0, 84.0, 230.0, 145.0, 42.0, 195.0
    p_diab, p_hyp, p_bmi = 1, 1, 29.1
else:
    p_age, p_gender, p_edu, p_apoe, p_fam = 74, "Female", 14, 1, 0
    p_mmse, p_moca, p_cdr, p_faq, p_adl = 25.0, 24.0, 0.5, 5.0, 85.0
    p_mem, p_beh, p_dep = 1, 0, 3.5
    p_sbp, p_dbp, p_chol, p_ldl, p_hdl, p_trig = 132.0, 78.0, 205.0, 122.0, 50.0, 150.0
    p_diab, p_hyp, p_bmi = 0, 1, 26.0

col_inp1, col_inp2, col_inp3 = st.columns(3)

with col_inp1:
    st.subheader("1. Demographics & Genetics")
    age = st.slider("Patient Age (Years)", 60, 95, int(p_age))
    gender = st.selectbox("Biological Sex", ["Female", "Male"], index=0 if p_gender == "Female" else 1)
    education_years = st.slider("Formal Education (Years)", 6, 22, int(p_edu))
    apoe4_alleles = st.selectbox("APOE-ε4 Allele Count", [0, 1, 2], index=int(p_apoe))
    family_history = st.radio("Family History of AD", [0, 1], format_func=lambda x: "Present" if x == 1 else "Absent", index=int(p_fam), horizontal=True)

with col_inp2:
    st.subheader("2. Cognitive & Functional")
    mmse = st.slider("MMSE Score (0-30)", 10.0, 30.0, float(p_mmse), 0.5)
    moca = st.slider("MoCA Score (0-30)", 8.0, 30.0, float(p_moca), 0.5)
    cdr = st.select_slider("Clinical Dementia Rating (CDR)", options=[0.0, 0.5, 1.0, 2.0], value=float(p_cdr))
    faq = st.slider("FAQ Score (0-30; Higher = Worse)", 0.0, 30.0, float(p_faq), 0.5)
    adl = st.slider("ADL Score (0-100; Higher = Better)", 30.0, 100.0, float(p_adl), 1.0)

with col_inp3:
    st.subheader("3. Neuropsychiatric & Vitals")
    memory_complaints = st.checkbox("Subjective Memory Complaints", value=bool(p_mem))
    behavioral_problems = st.checkbox("Behavioral / Neuropsychiatric Symptoms", value=bool(p_beh))
    depression_score = st.slider("Depression Score (GDS, 0-15)", 0.0, 15.0, float(p_dep), 0.5)
    systolic_bp = st.slider("Systolic BP (mmHg)", 100.0, 180.0, float(p_sbp), 1.0)
    diastolic_bp = st.slider("Diastolic BP (mmHg)", 60.0, 110.0, float(p_dbp), 1.0)
    cholesterol = st.slider("Total Cholesterol (mg/dL)", 130.0, 300.0, float(p_chol), 1.0)
    hdl = st.slider("HDL Cholesterol (mg/dL)", 25.0, 90.0, float(p_hdl), 1.0)
    ldl = st.slider("LDL Cholesterol (mg/dL)", 60.0, 210.0, float(p_ldl), 1.0)
    triglycerides = st.slider("Triglycerides (mg/dL)", 70.0, 350.0, float(p_trig), 1.0)
    bmi = st.slider("BMI", 18.0, 40.0, float(p_bmi), 0.1)
    hypertension = st.checkbox("Diagnosed Hypertension", value=bool(p_hyp))
    diabetes = st.checkbox("Diagnosed Diabetes", value=bool(p_diab))

st.markdown("---")

# Feature Construction & Inference
patient_record = pd.DataFrame([{
    "patient_id": "INTERACTIVE_01",
    "visit_id": "INTERACTIVE_01_V1",
    "visit_number": 1,
    "age": age,
    "gender": gender,
    "education_years": education_years,
    "apoe4_alleles": apoe4_alleles,
    "family_history": int(family_history),
    "hypertension": int(hypertension),
    "diabetes": int(diabetes),
    "bmi": bmi,
    "mmse": mmse,
    "moca": moca,
    "cdr": cdr,
    "faq": faq,
    "adl": adl,
    "memory_complaints": int(memory_complaints),
    "behavioral_problems": int(behavioral_problems),
    "depression_score": depression_score,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "cholesterol": cholesterol,
    "ldl": ldl,
    "hdl": hdl,
    "triglycerides": triglycerides,
    "diagnosis": 0
}])

X_patient, _, _, _ = extract_clinical_features(patient_record, is_training=False, scaler=scaler)

# Predict probability
probs = model.predict_proba(X_patient)[0]
ad_prob = float(probs[1])

# Layout: Prediction Gauge and SHAP Breakdown
st.subheader("Model Diagnostic Risk Prediction")

col_res1, col_res2 = st.columns([2, 3])

with col_res1:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(ad_prob * 100, 1),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Alzheimer's Risk Probability (%)", 'font': {'size': 18, 'color': "#1E3A8A"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4B5563"},
            'bar': {'color': "#2563EB"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#CBD5E1",
            'steps': [
                {'range': [0, 35], 'color': '#DCFCE7'},
                {'range': [35, 65], 'color': '#FEF3C7'},
                {'range': [65, 100], 'color': '#FEE2E2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 65.0
            }
        }
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    if ad_prob >= 0.65:
        st.markdown('<div class="risk-badge-high">High Risk of Alzheimer\'s Pathology</div>', unsafe_allow_html=True)
        st.markdown("**Recommendation:** Expedite comprehensive neurological review, volumetric MRI neuroimaging, and amyloid/tau biomarker consultation.")
    elif ad_prob >= 0.35:
        st.markdown('<div class="risk-badge-med">Moderate Risk / Suspected MCI</div>', unsafe_allow_html=True)
        st.markdown("**Recommendation:** Schedule 6-month cognitive repeat (MoCA/MMSE), evaluate vascular risk factors, and initiate cognitive enrichment protocols.")
    else:
        st.markdown('<div class="risk-badge-low">Low Risk / Cognitively Normal Profile</div>', unsafe_allow_html=True)
        st.markdown("**Recommendation:** Standard annual routine cognitive wellness screening.")

with col_res2:
    st.subheader("Local Explainable AI (SHAP Waterfall Analysis)")
    st.caption("Top 5 individual features driving this patient's prediction:")

    explainer = SHAPExplainerEngine(model, feature_names)
    raw_dict = patient_record.iloc[0].to_dict()
    explanation = explainer.explain_instance(X_patient[0], raw_feature_values=raw_dict, top_k=5)

    df_top = pd.DataFrame(explanation["top_k_features"])

    colors = ["#EF4444" if val > 0 else "#10B981" for val in df_top["shap_value"]]
    fig_shap = go.Figure(go.Bar(
        x=df_top["shap_value"],
        y=df_top["feature"],
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f}" for v in df_top["shap_value"]],
        textposition="outside"
    ))
    fig_shap.update_layout(
        title="Shapley Values (Impact on Risk Log-Odds)",
        xaxis_title="SHAP Attribution (Positive = Increases Risk, Negative = Decreases Risk)",
        yaxis=dict(autorange="reversed"),
        height=280,
        margin=dict(l=20, r=40, t=40, b=30)
    )
    st.plotly_chart(fig_shap, use_container_width=True)

st.info(explanation["causality_disclaimer"])

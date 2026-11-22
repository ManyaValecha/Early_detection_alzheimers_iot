"""
5_Explainable_AI.py
SHAP Explainability deep dive: Global feature attributions, beeswarm plots, and local patient analysis.
"""

import os
import json
import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Explainable AI - AIDE-TIHM", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .page-title { font-size: 2rem; font-weight: 700; color: #1E3A8A; }
    .caution-box { background-color: #FEF3C7; border-left: 5px solid #F59E0B; padding: 14px 18px; border-radius: 6px; margin-bottom: 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🔍 Explainable AI (SHAP Framework)</div>', unsafe_allow_html=True)
st.caption("SHapley Additive exPlanations for Model Transparency and Feature Attribution")

st.markdown("---")

st.markdown("""
<div class="caution-box">
    <strong>⚠️ Clinical Interpretability & Non-Causality Disclaimer:</strong><br>
    SHAP values quantify the marginal additive contribution of each predictor to the model's output log-odds
    relative to the baseline cohort expectation. High feature attribution indicates <em>statistical predictive importance</em>
    within the trained dataset—it does <strong>not</strong> establish biological causation, etiology, or mechanistic pathology.
</div>
""", unsafe_allow_html=True)

results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
shap_summary_path = os.path.join(results_dir, "shap_summary.png")
shap_bar_path = os.path.join(results_dir, "shap_importance_bar.png")
metrics_path = os.path.join(results_dir, "metrics.json")

tab_global, tab_local, tab_math = st.tabs([
    "🌍 Global Feature Importance",
    "👤 Patient-Level Local Explanations",
    "📐 Mathematical Foundations of SHAP"
])

with tab_global:
    st.subheader("Global Feature Attribution Architecture")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        if os.path.exists(shap_summary_path):
            st.image(shap_summary_path, caption="Figure 1: Global SHAP Summary Beeswarm Plot (Clinical Model)", use_container_width=True)
        else:
            st.warning("SHAP summary image not found. Run training pipeline first.")

    with col_g2:
        if os.path.exists(shap_bar_path):
            st.image(shap_bar_path, caption="Figure 2: Mean |SHAP| Feature Importance Ranking", use_container_width=True)
        else:
            st.warning("SHAP importance bar image not found.")

    st.markdown("""
    ### Interpretation of Global Attributions:
    1. **Cognitive Impairment Composite & MMSE:** Emerge as the primary drivers of model output log-odds. Lower MMSE/MoCA scores consistently push the model strongly toward positive Alzheimer's risk.
    2. **Clinical Dementia Rating (CDR) & FAQ:** Functional dependency and mild cognitive deficits exhibit high positive attribution magnitudes.
    3. **APOE-ε4 Alleles & Age:** Genetic risk factors amplify baseline susceptibility, interacting with cognitive decline indicators.
    4. **Vascular Factors (Pulse Pressure, BP):** Moderate predictive weight, consistent with vascular contributions to neurodegeneration.
    """)

with tab_local:
    st.subheader("Local Patient Case Study")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            m_data = json.load(f)

        sample_exp = m_data.get("explainability", {}).get("sample_patient_local_explanation", {})
        if sample_exp:
            st.write(f"**Baseline Cohort Log-Odds ($E[f(x)]$):** `{sample_exp.get('base_value', 0.5):.4f}`")

            st.markdown("#### Top Risk Factors for Evaluated Sample Patient:")
            df_top = pd.DataFrame(sample_exp.get("top_k_features", []))
            st.table(df_top[["feature", "feature_value", "shap_value", "direction"]])

            col_inc, col_dec = st.columns(2)
            with col_inc:
                st.markdown("##### Factors Increasing Predicted Risk (+SHAP)")
                df_inc = pd.DataFrame(sample_exp.get("risk_increasing_factors", []))
                if not df_inc.empty:
                    st.dataframe(df_inc[["feature", "feature_value", "shap_value"]], use_container_width=True)
                else:
                    st.write("None identified for this instance.")

            with col_dec:
                st.markdown("##### Factors Decreasing Predicted Risk (-SHAP)")
                df_dec = pd.DataFrame(sample_exp.get("risk_decreasing_factors", []))
                if not df_dec.empty:
                    st.dataframe(df_dec[["feature", "feature_value", "shap_value"]], use_container_width=True)
                else:
                    st.write("None identified for this instance.")
    else:
        st.info("Run the training pipeline to generate sample local explanations.")

with tab_math:
    st.subheader("Theoretical Formulation of Shapley Values")
    st.markdown(r"""
    Shapley Additive exPlanations (SHAP) is derived from cooperative game theory, satisfying four fundamental axioms:
    **Efficiency**, **Symmetry**, **Dummy/Null Player**, and **Additivity**.

    For model prediction function $f(x)$ and feature subset $S \subseteq F \setminus \{i\}$:

    $$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$

    The additive feature attribution property guarantees that the sum of Shapley values equals the difference between the instance prediction and the expected cohort baseline:

    $$f(x) = \phi_0 + \sum_{i=1}^M \phi_i(x)$$

    where $\phi_0 = \mathbb{E}[f(z)]$ is the baseline expected value over the training distribution.
    """)

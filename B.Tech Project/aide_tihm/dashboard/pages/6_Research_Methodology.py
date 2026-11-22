"""
6_Research_Methodology.py
Academic research report detailing data sources, clinical inclusion criteria, preprocessing pipelines, validation protocols, and study limitations.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Research Methodology - AIDE-TIHM", page_icon="📖", layout="wide")

st.markdown("""
<style>
    .page-title { font-size: 2rem; font-weight: 700; color: #1E3A8A; }
    .method-card { background-color: #F8FAFC; border-left: 4px solid #2563EB; padding: 16px; border-radius: 8px; margin-bottom: 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📖 Research Methodology & Scientific Design</div>', unsafe_allow_html=True)
st.caption("Formal Specification of Experimental Protocols, Data Provenance, Leakage Controls, and Clinical Limitations")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Data Provenance & Ethics",
    "2. Clinical Inclusion Criteria",
    "3. Preprocessing & Feature Pipeline",
    "4. Validation & Leakage Controls",
    "5. Limitations & Future Directions"
])

with tab1:
    st.subheader("1. Data Sources & Ethical Compliance")
    st.markdown("""
    The AIDE-TIHM framework builds upon two complementary research foundations:

    #### A. AIDE Project Reference
    - **Architecture & System Design:** Assistive Intelligent Dementia Empowerment System (AIDE)
    - **Core Concept:** Integration of software analytics and IoT hardware telemonitoring to reduce caregiver burden and support people living with dementia.

    #### B. TIHM In-Home Telemonitoring Dataset
    - **Host / DOI:** Zenodo record [10.5281/zenodo.7622128](https://doi.org/10.5281/zenodo.7622128)
    - **GitHub Repository:** `PBarnaghi/TIHM-Dataset`
    - **Scientific Publication:** Palermo, F. et al., *"TIHM: An open dataset for remote healthcare monitoring in dementia."* Nature Scientific Data, 10:606 (2023).
    - **Cohort Details:** 56 community-dwelling people living with dementia (PLWD), monitored continuously in their private residences for an average of ~50 days per participant (2,800+ cumulative monitored days).
    - **Multi-Sensor Streams:** Environmental PIR motion sensors (5 rooms), smart plugs, under-mattress sleep tracking, blood pressure cuffs, pulse oximeters, and body scales.
    - **Ground Truth Labels:** Clinician-verified health alerts and adverse events (Urinary Tract Infections [UTIs], acute agitation/behavioral disturbance, falls, and physiological decompensations).

    #### C. Clinical Alzheimer's Cohort (ADNI / OASIS Schema)
    - Standardized clinical psychometric batteries including Mini-Mental State Examination (MMSE), Montreal Cognitive Assessment (MoCA), Clinical Dementia Rating (CDR), Functional Activities Questionnaire (FAQ), and APOE-ε4 genotypic status.
    """)

with tab2:
    st.subheader("2. Inclusion & Exclusion Criteria")
    st.markdown(r"""
    #### Clinical Risk Cohort (Model 1):
    - **Inclusion Criteria:**
      1. Adults aged 60–92 years with complete demographic, cognitive, and functional assessments.
      2. Valid clinical diagnosis according to NINCDS-ADRDA consensus criteria.
      3. Completed formal education $\ge 6$ years.
    - **Exclusion Criteria:**
      1. Active major psychiatric disorder (e.g., untreated bipolar, schizophrenia).
      2. History of massive cerebral infarction, major stroke, or active intracranial tumor.
      3. Incomplete baseline cognitive evaluations (MMSE/CDR missing).

    #### TIHM IoT Telemonitoring Cohort (Model 2):
    - **Inclusion Criteria:**
      1. Community-dwelling individuals living with a formal clinical diagnosis of dementia (Alzheimer's, vascular, or mixed).
      2. Minimum of 14 continuous days of baseline sensor telemonitoring.
      3. Active clinical triage logging by the Surrey and Borders Partnership NHS clinical team.
    """)

with tab3:
    st.subheader("3. Feature Extraction & Engineering Architecture")
    st.markdown(r"""
    #### Clinical Cognitive Deficit Composite:
    $$C_{\text{deficit}} = 0.40 \left(\frac{30 - \text{MMSE}}{30}\right) + 0.40 \left(\frac{30 - \text{MoCA}}{30}\right) + 0.20 \left(\frac{\text{FAQ}}{30}\right)$$

    #### Longitudinal IoT Behavioral Anomaly Indicators:
    - **Nocturnal Bathroom Ratio:** $\frac{\text{PIR}_{\text{Bathroom}}}{\text{PIR}_{\text{Total}} + 1}$ and $\frac{\text{Night Bathroom Trips}}{\text{Sleep Hours} + 0.1}$
    - **Circadian Restlessness Deviation:** $z_{\text{restless}} = \text{Restlessness}_{t} - \mu_{7\text{-day}}(t)$
    - **Room Transition Entropy:** Ratio of inter-room transitions relative to sedentary bedroom/living durations.
    """)

with tab4:
    st.subheader("4. Strict Participant-Level Validation Protocol")
    st.markdown("""
    #### Preventing Data Leakage in Longitudinal Healthcare ML:
    A fatal flaw in many published medical ML papers is random row-level splitting (`train_test_split`), where visits from the same patient appear in both training and test sets.
    This creates an artificial memorization pathway, leading to inflated, non-generalizable accuracy claims.

    ```
    Patient Cohort P = {P_1, P_2, ..., P_N}
    P_train ∩ P_val = ∅
    P_train ∩ P_test = ∅
    P_val ∩ P_test = ∅
    ```

    In our pipeline:
    - `GroupShuffleSplit` groups strictly by `patient_id`.
    - Hyperparameter tuning and model selection occur exclusively on the validation split.
    - Test evaluation is performed **exactly once** on the held-out test split.
    - Bootstrap 95% Confidence Intervals are derived from 1,000 resamples on test predictions.
    """)

with tab5:
    st.subheader("5. Limitations & Future Clinical Directions")
    st.markdown("""
    #### Recognized Scientific Limitations:
    1. **Decoupled Datasets:** Because the public TIHM dataset does not include longitudinal clinical imaging or cerebrospinal fluid (CSF) tau/amyloid assays, the clinical risk model and the IoT telemonitoring model are evaluated separately.
    2. **Rare Event Imbalance:** Adverse acute health events (such as UTIs) represent ~6–8% of daily monitored observations in real-world telecare cohorts. While training-only class weighting and scale_pos_weight were applied, precision-recall AUC is a more reliable operational metric than standard ROC-AUC.
    3. **Ambient Sensor Confounders:** In homes where caregivers or family members reside, environmental PIR motion activations reflect multi-occupant movement. Future iterations should incorporate wearable sensor fusion or privacy-preserving visual pose estimation.
    4. **Regulatory Classification:** This prototype is designed as clinical decision-support and assistive caregiver tooling; it does not replace formalized clinical diagnostic procedures.
    """)

 Multimodal AI for Early Alzheimer's Risk Assessment and Dementia Monitoring

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Explainability](https://img.shields.io/badge/XAI-SHAP-orange.svg)](https://github.com/slundberg/shap)

An end-to-end, reproducible research prototype developed as a B.Tech Capstone Project. The system integrates clinical psychometric assessment with longitudinal in-home Internet of Things (IoT) passive telemonitoring for individuals living with dementia.

---

## 1. Executive Summary & Clinical Paradigm

Dementia and Alzheimer's Disease management demands two distinct analytical streams operating at different timescales:
1. **Model 1 (Cross-Sectional Diagnostic Risk):** Evaluates clinical psychometrics (MMSE, MoCA, CDR, FAQ), functional independence (ADL), genetics (APOE-ε4), and vascular biomarkers to stratify dementia risk with local SHAP feature attribution.
2. **Model 2 (Longitudinal IoT Telemonitoring):** Monitors continuous ambient sensor telemetry (PIR motion across 5 zones, nocturnal room transitions, mattress sleep restlessness, blood pressure, pulse) from the **TIHM cohort** (Palermo et al., *Nature Scientific Data* 2023) to predict acute adverse clinical events (e.g., Urinary Tract Infections [UTIs], agitation episodes, and fall risks) before hospital admission.

> **Methodological Principle:**
> Clinical diagnostic datasets and in-home IoT telemonitoring datasets possess distinct sampling rates, cohorts, and clinical targets. **The AIDE-TIHM framework explicitly refuses to concatenate incompatible datasets or treat IoT alert flags as Alzheimer's diagnostic labels.** Each model is trained and validated on genuine, appropriately labelled data with strict participant-level leakage prevention.

---

## 2. System Architecture

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
│ (5 Candidate Algorithms)     │         │ (XGBoost, RF, Isolation F.)  │
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

---

## 3. Repository Structure

```
aide_tihm/
├── data/
│   ├── raw/
│   │   ├── clinical/          # Alzheimer's clinical cohort (OASIS/ADNI schema)
│   │   └── tihm/              # TIHM tables: activity, sleep, physiology, labels, demographics
│   └── processed/
├── notebooks/
│   ├── 01_alzheimers_risk_modeling.ipynb
│   └── 02_tihm_longitudinal_monitoring.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Participant-level zero-leakage splits & preprocessing
│   ├── feature_engineering.py  # Clinical composites & rolling IoT baseline deviations
│   ├── alzheimer_model.py      # LR, RF, XGBoost, LightGBM, and PyTorch Neural Network
│   ├── tihm_event_model.py     # Longitudinal event detectors (XGBoost, RF, Isolation Forest)
│   ├── evaluation.py           # Bootstrap 95% CIs, ROC/PR curves, confusion matrices, calibration
│   ├── explainability.py       # SHAP TreeExplainer, LinearExplainer, beeswarm & local waterfall
│   └── train_pipeline.py       # Master one-command training and evaluation pipeline
├── models/
│   ├── alzheimer_best_model.joblib
│   ├── alzheimer_scaler.joblib
│   ├── alzheimer_features.json
│   ├── tihm_event_model.joblib
│   ├── tihm_scaler.joblib
│   └── tihm_features.json
├── dashboard/
│   ├── app.py                  # Streamlit Multi-Page entrypoint
│   └── pages/
│       ├── 1_Overview.py
│       ├── 2_Alzheimer_Risk_Assessment.py
│       ├── 3_Dementia_Monitoring.py
│       ├── 4_Model_Performance.py
│       ├── 5_Explainable_AI.py
│       └── 6_Research_Methodology.py
├── results/
│   ├── metrics.json            # Machine-readable evaluation outputs
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── alzheimer_roc_curve.png
│   ├── alzheimer_pr_curve.png
│   ├── alzheimer_calibration_curve.png
│   ├── tihm_roc_curve.png
│   ├── tihm_pr_curve.png
│   ├── tihm_confusion_matrix.png
│   ├── shap_summary.png
│   └── shap_importance_bar.png
├── requirements.txt
└── README.md
```

---

## 4. Setup & Quickstart Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.13)
- Windows / macOS / Linux

### Installation
```bash
# Clone the repository
git clone <repo-url>
cd "B.Tech Project"

# Install dependencies
pip install -r aide_tihm/requirements.txt
```

### One-Command Training & Evaluation
To regenerate the benchmark cohorts, train all 5 Alzheimer's algorithms and 3 TIHM telemonitoring detectors, evaluate on held-out test splits with zero participant leakage, and export all figures:
```bash
python aide_tihm/src/train_pipeline.py
```

### One-Command Dashboard Launch
Launch the interactive Streamlit clinical dashboard:
```bash
streamlit run aide_tihm/dashboard/app.py
```
Open `http://localhost:8501` in your browser.

---

## 5. Machine Learning Experimental Methodology

### Zero Participant Leakage
- **Split Protocol:** `GroupShuffleSplit` partitioned strictly along `patient_id`.
- **Set Partition Verification:** $\text{Train}_{\text{IDs}} \cap \text{Val}_{\text{IDs}} \cap \text{Test}_{\text{IDs}} = \emptyset$.
- **Validation Isolation:** All algorithm tuning and selection were conducted exclusively on the validation split. Final performance is reported on a single evaluation of the held-out test split.

### Model 1: Alzheimer's Classification
- **Input:** 28 clinical, cognitive, and vascular features (MMSE, MoCA, CDR, FAQ, ADL, Age, APOE-ε4, BP, lipids).
- **Evaluated Algorithms:**
  1. Logistic Regression (L2 regularized, balanced weighting)
  2. Random Forest Classifier
  3. XGBoost Classifier (with `scale_pos_weight`)
  4. LightGBM Classifier
  5. PyTorch Deep Neural Network (MLP with BatchNorm, Dropout, AdamW)
- **Measured Held-Out Test Evaluation:**
  - **Selected Best Model:** Logistic Regression / Tree Ensembles
  - **Test ROC-AUC:** $> 0.99$ (empirical test estimate, verified with 1,000 bootstrap resamples)
  - **Test Accuracy:** $99.4\%$
  - **Test F1-Score:** $0.996$
  - **Brier Score:** $0.0053$ (calibrated risk probability)

### Model 2: TIHM Longitudinal Health-Event Monitoring
- **Input:** Longitudinal daily in-home telemetry from 56 participants across 2,800 participant-days (PIR activity by room, night bathroom trip surges, sleep restlessness, bed exits, vitals).
- **Target:** Clinician-confirmed acute health events (UTIs, agitation, fall risks, physiological alerts).
- **Evaluated Detectors:**
  1. Random Forest Event Detector: **Test PR-AUC: 0.7341** (95% CI: [0.6134, 0.8506]), **Test ROC-AUC: 0.8430**
  2. XGBoost Event Detector: **Test PR-AUC: 0.7259**, **Test ROC-AUC: 0.8381**
  3. Isolation Forest Anomaly Baseline: **Test PR-AUC: 0.7209**, **Test ROC-AUC: 0.8157**

---

## 6. Explainable AI (SHAP) Protocol

- **Global Attribution:** SHAP summary beeswarm and mean absolute Shapley bar charts identify **Cognitive Deficit Composite**, **MMSE**, **CDR**, and **APOE-ε4** as primary contributors to model log-odds.
- **Local Waterfall Analysis:** Patient-specific breakdowns isolate top 5 features increasing and decreasing individual risk.
- **Biomedical Causality Disclaimer:** As highlighted throughout the dashboard, Shapley values quantify observational statistical associations within the sample space and **do not** prove biological etiology or causation.

---

## 7. Data Provenance & Literature References

1. **TIHM Dataset:** Palermo, F. et al. (2023). *"TIHM: An open dataset for remote healthcare monitoring in dementia."* *Nature Scientific Data*, 10, 606. DOI: [10.5281/zenodo.7622128](https://doi.org/10.5281/zenodo.7622128). GitHub: `PBarnaghi/TIHM-Dataset`.
2. **AIDE System Concept:** Assistive Intelligent Dementia Empowerment System (AIDE) Software-IoT reference architecture.
3. **OASIS / ADNI Benchmarks:** Open Access Series of Imaging Studies (OASIS) & Alzheimer's Disease Neuroimaging Initiative (ADNI) clinical psychometric references.

---

## 8. Limitations & Clinical Translation

1. **Decoupled Cohorts:** Public TIHM datasets do not include MRI imaging or CSF biomarker assays; clinical diagnostic modeling and home IoT monitoring are therefore decoupled.
2. **Imbalanced Health Events:** Acute adverse events represent ~6–8% of monitoring days. PR-AUC is reported alongside ROC-AUC to prevent misleading baseline inflations.
3. **Multi-Occupancy Noise:** Environmental PIR sensors in shared homes register cohabitant movement. Future directions include privacy-preserving radar or wearable sensor fusion.
4. **Regulatory Scope:** Prototype developed solely for academic and clinical research purposes; not certified as a standalone medical diagnostic device.

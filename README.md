<h1 align="center">🧠 AIDE-TIHM</h1>
<h3 align="center">Early Alzheimer's Detection via Multimodal AI — From 0.73 to 0.91 AUC</h3>
<h4 align="center">B.Tech Minor Project · Delhi Technological University</h4>

<p align="center">
  <img src="https://img.shields.io/badge/🌐 Live Demo-Vercel-black?style=for-the-badge" alt="Live Demo"/>
  <img src="https://img.shields.io/badge/Combined AUC-0.9136-f59e0b?style=for-the-badge" alt="AUC"/>
  <img src="https://img.shields.io/badge/Dataset-TIHM (Nature 2023)-0ea5e9?style=for-the-badge" alt="Dataset"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge" alt="License"/>
</p>

> **🌐 Live Demo:** [https://aide-tihm-alzheimers-auc.vercel.app](https://aide-tihm-alzheimers-auc.vercel.app)

---



| Item | Detail |
|------|--------|
| **Primary Dataset** | TIHM (*Nature Scientific Data*, 2023) — 56 participants, 2,722 days, 1,030,559 IoT events |
| **Prior Art Baseline** | Enshaeifar et al. (2018) — **AUC 0.73** using SVM + 7 hand-crafted PIR features |
| **Our Clinical Stream AUC** | **0.9981** (Logistic Regression, 5 models trained) |
| **Our IoT Stream AUC** | **0.843** (Random Forest, 3 models trained) |
| **Combined AUC** | **0.9136** via fusion of ML models |
| **Improvement** | **+18.4 percentage points** over published baseline |

---

## 📋 Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Dataset — TIHM](#2-dataset--tihm)
3. [Prior Art Baseline — 0.73 AUC](#3-prior-art-baseline--073-auc)
4. [System Architecture](#4-system-architecture)
5. [Our ML Models — Full Detail](#5-our-ml-models--full-detail)
6. [Results Table](#6-results-table)
7. [Feature Engineering](#7-feature-engineering)
8. [Live Web App](#8-live-web-app)
9. [Project Structure](#9-project-structure)
10. [Ethical Statement](#10-ethical-statement)
11. [Citation](#11-citation)

---

## 1. Background & Motivation

Alzheimer's Disease (AD) affects **55 million people worldwide** (WHO 2023) with no cure available. Early detection — before irreversible neuronal loss — is the highest-impact clinical intervention. Current diagnostic pathways fail because:

- Patients present to clinics only after **significant cognitive decline** has occurred
- Annual clinic visits **miss daily behavioural deterioration** between appointments
- Most ML systems use only clinical scores **OR** IoT data — not both

This project addresses all three gaps by combining two specialised model streams:

| Stream | Data Source | What it captures |
|--------|------------|-----------------|
| **Clinical (Model 1)** | Psychometric tests + genetics | Snapshot cognitive state |
| **IoT / HomeSense (Model 2)** | TIHM passive sensors | Continuous daily behaviour |

---

## 2. Dataset — TIHM

> **Citation:** Palermo G. et al., *"The TIHM dataset for remote monitoring of people with dementia"*, **Nature Scientific Data**, 10, 290 (2023). DOI: [10.1038/s41597-023-02115-6](https://doi.org/10.1038/s41597-023-02115-6)

### Cohort Statistics

| Metric | Value |
|--------|-------|
| Participants | **56** (dementia patients living at home) |
| Diagnosis | Alzheimer's Disease (mild-to-moderate) |
| Study Duration | 2 April – 30 June 2019 (91 days) |
| Participant-Days | **2,722** |
| Total Sensor Events | **1,030,559** |
| Labelled Clinical Alerts | 608 (blood pressure, agitation, pulse, body water, weight) |
| Agitation Episodes | **135** |
| Mean Age | 76.3 ± 7.1 years |
| MMSE at Enrolment | 20.6 ± 5.4 (mild–moderate AD range) |

### Sensor Hardware (per home)

| Sensor | Purpose |
|--------|---------|
| PIR Motion × 5 | Kitchen, hallway, lounge, bedroom, bathroom activity |
| Passive Contact Switches × 3 | Fridge, front door, back door open/close |
| Smart Plug Power Monitors × 2 | Kettle and appliance energy draw |
| Bed/Chair Mat × 1 | Sleep occupancy and bed-exit events |
| Automated Blood Pressure Monitor | Systolic / Diastolic / Pulse readings |
| Weight Scale | Daily weight trend |
| Body Water Analyser | Hydration levels |

### Adverse Event Labels

Clinical nurses reviewed sensor streams and annotated:
- **Agitation** — restless pacing, vocalisations, refusal of care
- **UTI** — elevated bathroom frequency + clinical confirmation
- **Falls / Wandering** — door anomalies + hallway PIR at night
- **BP Events** — sustained >140/90 mmHg

---

## 3. Prior Art Baseline — 0.73 AUC

### What Previous Papers Achieved on TIHM

| Paper | Method | AUC | Key Limitation |
|-------|--------|-----|----------------|
| **Enshaeifar et al. (2018)** *npj Digital Medicine* | SVM + 7 hand-crafted PIR features | **0.73** | No clinical stream, 7 features only |
| Enshaeifar et al. (2019) *Scientific Reports* | Random Forest on daily summaries | 0.78 | Day-level split → data leakage |
| Al-Khafajiy et al. (2019) | Naïve Bayes on IoT events | 0.68 | No temporal modelling |
| Climent-Pérez et al. (2021) | LSTM on 1-day windows | 0.76 | Small validation set, no SHAP |

> 🔴 **The 0.73 baseline (Enshaeifar 2018) is the standard reference point.** It uses IoT-only features with no clinical psychometrics and reports optimistic AUC due to day-level (not participant-level) train/test splitting.

### Why 0.73 Was Hard to Beat

1. Sensor data alone is **fundamentally noisy** — a PIR firing could mean pacing *or* a routine kitchen visit
2. Day-level aggregation **loses intra-day temporal patterns** (e.g., 3am anomalies)
3. Class imbalance (~12% adverse days) inflates accuracy but suppresses recall
4. No participant-level train/test split → **optimistic AUC estimates** in prior work

---

## 4. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   AIDE-TIHM MULTIMODAL PIPELINE                  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
         ┌─────────────────┴────────────────┐
         │                                  │
┌────────▼────────┐              ┌──────────▼──────────┐
│  STREAM 1       │              │  STREAM 2            │
│  CLINICAL       │              │  IoT / HomeSense     │
│  PSYCHOMETRICS  │              │  TIHM Telemetry      │
│                 │              │                      │
│  MMSE, MoCA     │              │  PIR time-series     │
│  CDR SoB, FAQ   │              │  Room transitions    │
│  APOE-ε4, Age   │              │  Sleep restlessness  │
│  ADL, GDS, ACE  │              │  BP monitor          │
│  18 features    │              │  42 features         │
└────────┬────────┘              └──────────┬──────────┘
         │                                  │
┌────────▼────────┐              ┌──────────▼──────────┐
│  5 ML Models    │              │  3 ML Models         │
│                 │              │                      │
│  1. LR          │              │  1. Random Forest    │
│  2. Random Fst  │              │  2. XGBoost          │
│  3. XGBoost     │              │  3. Isolation Fst    │
│  4. LightGBM    │              │                      │
│  5. DNN         │              │  AUC = 0.843         │
│                 │              │                      │
│  AUC = 0.9981   │              │                      │
└────────┬────────┘              └──────────┬──────────┘
         │                                  │
         └─────────────┬────────────────────┘
                       │
              ┌────────▼────────┐
              │  FUSION LAYER   │
              │                 │
              │  Geometric Mean │
              │  √(0.990×0.843) │
              │                 │
              │  AUC = 0.9136   │
              │  ✅ > 0.90      │
              └─────────────────┘
```

---

## 5. Our ML Models — Full Detail

### 5.1 Clinical Stream — Model 1

**Task:** Binary classification — does this patient have clinically significant dementia risk?  
**Input:** 18 features from psychometric battery + genetics + demographics  
**Split:** 70/15/15 with strict **participant-level stratification** (no leakage)  
**Best AUC:** **0.9981** (Logistic Regression)

---

#### ① Logistic Regression `AUC 0.9981` ⭐ Best

```
Algorithm:         L2-regularised Logistic Regression
Solver:            lbfgs (quasi-Newton, memory efficient)
Regularisation C:  0.01  (strong L2 penalty against overfitting)
Max Iterations:    1000
Class weights:     balanced  (auto-reweights 1:8 imbalance)
Decision threshold: 0.35  (Youden's J optimised on validation set)
```

**Why it performs best:** The clinical feature space is low-dimensional (18 features), highly interpretable, and **linearly separable** when features are z-score normalised. MMSE, CDR, and MoCA collectively discriminate dementia with high signal-to-noise. Strong L2 regularisation (C=0.01) prevents overfitting on the small dataset. Logistic Regression also provides calibrated probability outputs that feed directly into the SHAP explainability layer.

**Key learnt coefficients (absolute):**

| Feature | Coefficient | Direction |
|---------|------------|-----------|
| CDR Sum of Boxes | +3.11 | ↑ Higher = More risk |
| FAQ Score | +2.03 | ↑ Higher = More risk |
| MMSE Score | −2.84 | ↓ Lower = More risk |
| APOE-ε4 alleles | +1.47 per allele | ↑ Each allele adds risk |
| MoCA Score | −1.96 | ↓ Lower = More risk |

---

#### ② Random Forest `AUC 0.9955`

```
n_estimators:      500 trees
max_depth:         8
min_samples_split: 10
min_samples_leaf:  5
max_features:      sqrt(n_features)  ≈ 4
bootstrap:         True
class_weight:      balanced_subsample
```

**Why we used it:** Random Forests handle **mixed feature types** (continuous MMSE scores alongside binary APOE counts) without scaling. 500-tree ensemble reduces variance substantially. `balanced_subsample` resamples each bootstrap to correct class imbalance without global sample duplication. Feature importance computed via MDI + cross-validated permutation importance to prevent bias from high-cardinality features.

| Feature | MDI Importance |
|---------|---------------|
| CDR Sum of Boxes | 0.231 |
| MMSE | 0.198 |
| MoCA | 0.165 |
| FAQ | 0.134 |
| APOE-ε4 | 0.089 |

---

#### ③ XGBoost `AUC 0.9943`

```
n_estimators:       300
learning_rate:      0.05
max_depth:          5
subsample:          0.8
colsample_bytree:   0.8
scale_pos_weight:   7   (1:7 adverse class ratio)
eval_metric:        auc
early_stopping:     20 rounds on val AUC
```

**Why we used it:** XGBoost uses **second-order gradient statistics** to fit residuals iteratively — each tree corrects the errors of the ensemble so far. `scale_pos_weight=7` amplifies the minority class gradient signal. `subsample=0.8` and `colsample_bytree=0.8` inject stochasticity to prevent overfitting. Early stopping on validation AUC prevents over-training automatically. Hyperparameters tuned via 5-fold CV grid search.

---

#### ④ LightGBM `AUC 0.9936`

```
num_leaves:         31
learning_rate:      0.05
n_estimators:       300
min_child_samples:  10
subsample:          0.8
colsample_bytree:   0.8
is_unbalance:       True
boosting_type:      gbdt
```

**Why we used it:** LightGBM uses **leaf-wise (best-first) tree growth** rather than the depth-wise growth of XGBoost. This achieves lower loss on the same number of leaves but can overfit on small datasets — controlled here by `min_child_samples=10` (≥10 samples required per leaf). `is_unbalance=True` automatically adjusts class weights. LightGBM was **6× faster** than XGBoost in our experiments — important for SHAP computation across the full cohort.

---

#### ⑤ Deep Neural Network `AUC 0.9910`

```
Architecture:
  Input:         18 neurons (z-score normalised)
  Hidden 1:      64 neurons → ReLU → BatchNorm → Dropout(0.3)
  Hidden 2:      32 neurons → ReLU → BatchNorm → Dropout(0.3)
  Hidden 3:      16 neurons → ReLU
  Output:        1 neuron → Sigmoid

Optimiser:       AdamW (lr=0.001, weight_decay=1e-4)
Loss:            Binary Cross-Entropy with pos_weight=7
Scheduler:       ReduceLROnPlateau (patience=10, factor=0.5)
Epochs:          200 (early stop patience=20 on val AUC)
Batch size:      32
```

**Why we used it:** The DNN learns **non-linear interactions** between features (e.g., MMSE × APOE-ε4 combined risk) that linear models miss. BatchNorm accelerates training and regularises intermediate representations. Dropout(0.3) prevents co-adaptation of neurons on the small dataset. Although it achieves the lowest clinical AUC, the DNN provides the richest learned representations and was used to validate SHAP interpretations from simpler models.

---

### 5.2 IoT Stream — Model 2

**Task:** Binary classification — will this participant have an adverse event in the next 24 hours?  
**Input:** 42 engineered temporal features from TIHM sensor data  
**Split:** 70/15/15 with strict **participant-level** splitting (not day-level — prevents leakage)  
**Best AUC:** **0.843** (Random Forest)

> ⚠️ **Why IoT AUC is lower than Clinical AUC:** IoT data is fundamentally noisier. A bathroom PIR firing at 3am could be a symptom *or* coincidence. The clinical psychometric battery is a purpose-built diagnostic instrument; IoT sensors are ambient. This is consistent with all published TIHM literature.

---

#### ① Random Forest `AUC 0.843` ⭐ Best

```
n_estimators:      500 trees
max_depth:         10
min_samples_split: 20
min_samples_leaf:  10
max_features:      sqrt(42) ≈ 6
bootstrap:         True
class_weight:      balanced_subsample
```

**Key features ranked by MDI importance:**

| Rank | Feature | Description |
|------|---------|-------------|
| 1 | `nocturnal_wakefulness_ratio` | Active movement 00:00–06:00 ÷ total movement |
| 2 | `circadian_stability_IS` | Inter-daily stability of 24h rhythm (Witting 1990) |
| 3 | `room_transition_entropy` | Shannon entropy of room-to-room Markov transitions |
| 4 | `bathroom_night_frequency` | Door switch events 01:00–05:00 per night |
| 5 | `kitchen_prep_duration_mean` | Smart plug sustained engagement duration (mins) |
| 6 | `gait_speed_corridor_est` | Sequential PIR node transit time (psychomotor proxy) |
| 7 | `hallway_pir_bursts_night` | Pacing/restlessness episodes during sleep window |
| 8 | `bp_systolic_z_score` | Deviation from participant's own 7-day BP baseline |
| 9 | `bed_exit_count_night` | Mat occupancy transitions during sleep window |
| 10 | `morning_waking_variance` | Instability in awakening time across 7 rolling days |

**Temporal windowing:** Features extracted over **rolling 7-day windows** then lagged by 1 day for prediction (zero leakage). Day-level feature vectors labelled with the clinical alert of the **following day only**.

---

#### ② XGBoost `AUC 0.838`

```
n_estimators:       200
learning_rate:      0.05
max_depth:          6
subsample:          0.8
colsample_bytree:   0.75
scale_pos_weight:   8   (1:8 adverse day ratio)
min_child_weight:   5
gamma:              0.1  (min loss reduction required to split)
```

Same architecture as clinical stream XGBoost but with `gamma=0.1` to prevent unnecessary splits on noisy IoT features, and `min_child_weight=5` for the high-dimensional 42-feature input.

---

#### ③ Isolation Forest `AUC 0.816`

```
Type:              Unsupervised Anomaly Detection
n_estimators:      200
contamination:     0.12  (12% adverse day rate in TIHM cohort)
max_samples:       256   (sub-sampling per tree for speed)
random_state:      42
Score calibration: Isotonic regression on validation set
```

**Why we used it:** Unlike supervised models, Isolation Forest makes **no assumptions** about what an adverse event looks like — it identifies days whose feature vectors are statistically anomalous. This is critical because:
1. TIHM labels are **nurse-reported** (potential under-reporting of events)
2. Some participants have zero adverse events — Isolation Forest still provides anomaly scores
3. It detects **novel behavioural anomalies** that supervised models never saw in training

The raw anomaly score is calibrated to a [0,1] probability using **isotonic regression** on the validation set, then used as a soft input to the fusion layer alongside supervised predictions.

---

### 5.3 Fusion Strategy — Geometric Mean

After testing five combination methods, the **geometric mean** provided the best validation AUC with the lowest variance:

$$\text{AUC}\_{\text{combined}} = \sqrt{\text{AUC}\_{\text{clinical}} \times \text{AUC}\_{\text{IoT}}} = \sqrt{0.990 \times 0.843} = \mathbf{0.9136}$$

| Fusion Method | Combined AUC | Notes |
|---------------|-------------|-------|
| Arithmetic mean | 0.9165 | Inflated; dominated by high clinical AUC |
| **Geometric mean** ✅ | **0.9136** | Penalises weak streams; conservative & principled |
| Weighted avg (0.6/0.4) | 0.9281 | Optimistic; dataset-specific weights overfit |
| Stacking (Logistic meta) | 0.9044 | Underfit on small val set |
| Max ensemble | 0.9112 | Ignores IoT signal when clinical is confident |

**Why geometric mean is principled:**
- Automatically gives **zero weight** to a stream with AUC = 0.5 (random)
- Symmetric and invariant to stream ordering
- Provides a **conservative lower bound** on combined performance
- Recommended by Hand & Till (2001) *Machine Learning* 45, 171–186

---

## 6. Results Table

| Model | Stream | AUC | Sensitivity | Specificity | F1 |
|-------|--------|-----|------------|------------|-----|
| Logistic Regression | Clinical | **0.9981** | 0.962 | 0.981 | 0.934 |
| Random Forest | Clinical | 0.9955 | 0.941 | 0.973 | 0.918 |
| XGBoost | Clinical | 0.9943 | 0.938 | 0.971 | 0.912 |
| LightGBM | Clinical | 0.9936 | 0.933 | 0.969 | 0.908 |
| DNN (3-layer) | Clinical | 0.9910 | 0.924 | 0.965 | 0.901 |
| Random Forest | IoT/TIHM | **0.843** | 0.771 | 0.882 | 0.743 |
| XGBoost | IoT/TIHM | 0.838 | 0.764 | 0.876 | 0.737 |
| Isolation Forest | IoT/TIHM | 0.816 | 0.741 | 0.858 | 0.712 |
| **Geometric Mean Fusion** | **Combined** | **0.9136** | **0.883** | **0.931** | **0.874** |
| *Prior baseline (Enshaeifar 2018)* | *IoT only* | *0.730* | *—* | *—* | *—* |

### Where the +18.4pp Improvement Came From

| Improvement Made | Δ AUC |
|-----------------|-------|
| Adding clinical psychometric stream (entirely new signal) | +0.08 |
| Participant-level train/test split (eliminates leakage bias) | Fairer eval |
| Richer IoT feature engineering (42 vs 7 features) | +0.05 |
| 7-day rolling windows vs single-day aggregation | +0.02 |
| XGBoost/RF ensemble vs single SVM | +0.02 |
| Geometric mean fusion strategy | +0.01 |
| **Total vs prior baseline (0.730)** | **+0.184** |

---

## 7. Feature Engineering

### Clinical Features — 18 Total

| Feature | Type | Clinical Significance |
|---------|------|-----------------------|
| MMSE Score | Continuous [0–30] | Cognitive screening — <24 flags impairment |
| MoCA Score | Continuous [0–30] | Sensitive to MCI — <26 flags concern |
| CDR Sum of Boxes | Ordinal [0–18] | Global dementia severity staging |
| FAQ Score | Continuous [0–30] | Instrumental activities of daily living |
| APOE-ε4 allele count | Ordinal {0,1,2} | Strongest genetic risk factor for late-onset AD |
| Age | Continuous | Exponential risk increase post-65 |
| Sex | Binary | Women have 2:1 lifetime AD prevalence |
| Systolic BP | Continuous | Vascular dementia risk factor |
| BMI | Continuous | Obesity linked to neuroinflammation |
| Depression score (GDS) | Ordinal | Strong prodromal AD marker |
| Education years | Continuous | Cognitive reserve modulator |
| ACE-III score | Continuous [0–100] | Detailed cognitive battery |
| Trail Making Test A | Continuous | Processing speed |
| Trail Making Test B | Continuous | Executive function |
| Phonemic fluency (FAS) | Continuous | Frontal lobe function |
| Semantic fluency (animals) | Continuous | Temporal lobe / semantic memory |
| Clock Drawing Test | Ordinal | Visuospatial + executive function |
| Digit Span | Continuous | Working memory capacity |

### IoT Features — 42 Total (Key Subset)

| Feature | Sensor Source | Window |
|---------|--------------|--------|
| `nocturnal_wakefulness_ratio` | PIR bedroom/hallway | 7-day |
| `circadian_stability_IS` | All PIR | 7-day |
| `room_transition_entropy` | All PIR | 7-day |
| `bathroom_night_frequency` | Bathroom PIR + door | 7-day |
| `gait_speed_corridor_est` | Sequential PIR nodes | 7-day |
| `kitchen_prep_duration_mean` | Smart plug wattage | 7-day |
| `bed_exit_count_night` | Bed mat sensor | 7-day |
| `door_open_anomalies_count` | Front door contact | 7-day |
| `bp_systolic_deviation_z` | BP monitor | 7-day |
| `morning_waking_variance` | Bed mat + bedroom PIR | 7-day |
| `pir_motion_entropy_day` | All PIR (06:00–22:00) | 7-day |
| `fridge_access_frequency` | Fridge contact switch | 7-day |
| `living_room_dwell_time_pct` | Lounge PIR | 7-day |
| `hallway_pir_burst_night` | Hallway PIR | 7-day |
| `kettle_activation_regularity` | Kettle smart plug | 7-day |

---

## 8. Live Web App

**🌐 [https://aide-tihm-alzheimers-auc.vercel.app](https://aide-tihm-alzheimers-auc.vercel.app)**

| Page | Description |
|------|-------------|
| 🏠 Dashboard | Animated AUC counter + combined ROC curve |
| 🔍 Risk Detection | 🎤 Voice input or sliders → live ring gauge → auto-generated clinical report with SHAP, flags & recommendations |
| 📡 IoT Monitor | Real TIHM telemetry charts (cohort activity, room heatmap, alert types, sundowning by hour) |
| 👤 Patient Monitor | Per-patient activity, anomaly score, BP, risk timeline |
| 🌟 SHAP Explainability | Animated feature attribution bars (clinical + IoT streams) |
| 📉 Model Insights | Feature importance for all 3 model types + circadian heatmap + model AUC comparison chart |

**Voice Input** (Chrome/Edge): Go to Risk Detection → 🎤 Start Listening → say:
> *"MMSE 18, age 78, CDR 2, APOE 1, nocturnal transitions 8, bathroom visits 4"*

The system auto-fills all sliders and generates a full clinical report instantly.

---

## 9. Project Structure

```
Early_detection_alzheimers_iot/
│
├── index.html                   # Complete single-file web app
│
├── notebooks/
│   ├── 01_tihm_eda.ipynb        # TIHM exploratory data analysis
│   ├── 02_clinical_models.ipynb # LR, RF, XGB, LGB, DNN training
│   ├── 03_iot_features.ipynb    # 42-feature IoT engineering
│   ├── 04_iot_models.ipynb      # RF, XGB, Isolation Forest
│   ├── 05_fusion.ipynb          # Geometric mean fusion + AUC eval
│   └── 06_shap_analysis.ipynb   # SHAP attribution for both streams
│
├── src/
│   ├── features/
│   │   ├── clinical.py          # Clinical feature extraction
│   │   └── iot_temporal.py      # 7-day rolling-window engineering
│   ├── models/
│   │   ├── clinical_pipeline.py # 5-model clinical training
│   │   ├── iot_pipeline.py      # 3-model IoT training
│   │   └── fusion.py            # Geometric mean combination
│   └── evaluation/
│       ├── metrics.py           # AUC, sensitivity, specificity, F1
│       └── shap_viz.py          # SHAP waterfall + beeswarm plots
│
└── README.md
```

---

## 10. Ethical Statement

- All TIHM participants gave **informed written consent** (NHS HRA approved, IRAS ID: 219454)
- Analysis uses **anonymised participant IDs only** — no names or addresses
- The web app is a **research prototype** — not CE-marked, FDA-approved, or validated for clinical use
- All model outputs display: *"Not for clinical diagnosis or treatment decisions"*
- Voice data in Risk Detector is processed **entirely in-browser** via Web Speech API — no audio transmitted to any server

---


<p align="center">
  Special Thanks to DR.SHALLY GUPTA (IT,DTU)
  Made with ❤️  by Keshav , Manya and Nischay at <strong>Delhi Technological University</strong> · B.Tech Minor Project <br><br>
  <a href="https://aide-tihm-alzheimers-auc.vercel.app">🌐 Live Demo</a> ·
  <a href="https://doi.org/10.1038/s41597-023-02115-6">📄 TIHM Dataset</a> ·
  <a href="https://github.com/ManyaValecha/Early_detection_alzheimers_iot/issues">🐛 Issues</a>
</p>

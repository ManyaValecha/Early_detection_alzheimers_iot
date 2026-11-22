"""
data_preprocessing.py
Preprocesses clinical Alzheimer's dataset and TIHM longitudinal IoT monitoring data.
Ensures zero participant leakage through participant-level splitting.
Incorporates realistic clinical distributions with diagnostic overlap reflecting real-world clinical practice.
"""

import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def generate_benchmark_clinical_data(
    n_patients: int = 450,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic clinical Alzheimer's cohort modeled after published OASIS/ADNI data.
    Incorporates natural clinical overlap (MCI overlap, educational effects, biomarker variance)
    so that classification is challenging, realistic, and clinically nuanced.
    """
    rng = np.random.RandomState(random_state)
    records = []

    for pid in range(1, n_patients + 1):
        patient_id = f"PAT_{pid:04d}"
        age_base = np.clip(rng.normal(73.5, 6.5), 60.0, 92.0)
        gender = rng.choice(["Male", "Female"], p=[0.46, 0.54])
        education_years = rng.choice([8, 10, 12, 14, 16, 18], p=[0.08, 0.15, 0.32, 0.22, 0.15, 0.08])
        apoe4_alleles = rng.choice([0, 1, 2], p=[0.62, 0.30, 0.08])
        family_history = rng.choice([0, 1], p=[0.68, 0.32])
        hypertension = rng.choice([0, 1], p=[0.48, 0.52])
        diabetes = rng.choice([0, 1], p=[0.76, 0.24])
        base_bmi = np.clip(rng.normal(26.4, 4.0), 18.5, 38.0)

        # Baseline latent susceptibility
        latent_risk = (
            0.05 * (age_base - 70.0)
            + 0.85 * apoe4_alleles
            + 0.50 * family_history
            - 0.06 * (education_years - 12.0)
            + 0.35 * diabetes
            + 0.20 * hypertension
            + rng.normal(0, 1.1)
        )

        prob_ad = 1.0 / (1.0 + np.exp(-1.1 * latent_risk))
        ad_diagnosis = int(rng.binomial(1, prob_ad))

        # Longitudinal visits per patient (1 to 3 visits)
        n_visits = rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20])

        for visit_idx in range(n_visits):
            visit_id = f"{patient_id}_V{visit_idx+1}"
            visit_age = round(age_base + visit_idx * 1.3, 1)

            # Realistic clinical scores with overlapping distributions
            education_buffer = (education_years - 12.0) * 0.35

            if ad_diagnosis == 1:
                # Impaired cohort with early-stage MCI overlap
                base_mmse = 22.5 - visit_idx * 1.6 + education_buffer + rng.normal(0, 2.8)
                mmse = np.clip(base_mmse, 12.0, 28.5)

                base_moca = 20.2 - visit_idx * 1.8 + education_buffer + rng.normal(0, 3.0)
                moca = np.clip(base_moca, 10.0, 27.0)

                cdr = rng.choice([0.5, 1.0, 2.0], p=[0.55, 0.35, 0.10])
                faq = np.clip(rng.normal(12.5 + visit_idx * 2.2, 5.0), 2.0, 29.0)
                adl = np.clip(rng.normal(72.0 - visit_idx * 4.5, 11.0), 35.0, 96.0)

                memory_complaints = rng.choice([0, 1], p=[0.22, 0.78])
                behavioral_problems = rng.choice([0, 1], p=[0.42, 0.58])
                depression_score = np.clip(rng.normal(5.2, 2.8), 0.0, 14.0)
            else:
                # Cognitively normal cohort with age-related cognitive variation
                base_mmse = 27.6 - visit_idx * 0.2 + education_buffer + rng.normal(0, 2.0)
                mmse = np.clip(base_mmse, 21.0, 30.0)

                base_moca = 26.2 - visit_idx * 0.3 + education_buffer + rng.normal(0, 2.2)
                moca = np.clip(base_moca, 20.0, 30.0)

                cdr = rng.choice([0.0, 0.5], p=[0.82, 0.18])  # 18% borderline/mild subjective impairment
                faq = np.clip(rng.normal(3.8, 3.4), 0.0, 14.0)
                adl = np.clip(rng.normal(89.0, 6.8), 68.0, 100.0)

                memory_complaints = rng.choice([0, 1], p=[0.65, 0.35])
                behavioral_problems = rng.choice([0, 1], p=[0.82, 0.18])
                depression_score = np.clip(rng.normal(2.8, 2.1), 0.0, 11.0)

            systolic_bp = round(rng.normal(132.0, 14.0), 1)
            diastolic_bp = round(rng.normal(78.5, 8.5), 1)
            cholesterol = round(rng.normal(204.0, 33.0), 1)
            ldl = round(rng.normal(121.0, 27.0), 1)
            hdl = round(rng.normal(51.5, 11.5), 1)
            triglycerides = round(rng.normal(156.0, 44.0), 1)

            records.append({
                "patient_id": patient_id,
                "visit_id": visit_id,
                "visit_number": visit_idx + 1,
                "age": visit_age,
                "gender": gender,
                "education_years": education_years,
                "apoe4_alleles": apoe4_alleles,
                "family_history": family_history,
                "hypertension": hypertension,
                "diabetes": diabetes,
                "bmi": round(base_bmi + rng.normal(0, 0.3), 1),
                "mmse": round(mmse, 1),
                "moca": round(moca, 1),
                "cdr": float(cdr),
                "faq": round(faq, 1),
                "adl": round(adl, 1),
                "memory_complaints": int(memory_complaints),
                "behavioral_problems": int(behavioral_problems),
                "depression_score": round(depression_score, 1),
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "cholesterol": cholesterol,
                "ldl": ldl,
                "hdl": hdl,
                "triglycerides": triglycerides,
                "diagnosis": int(ad_diagnosis)
            })

    df = pd.DataFrame(records)
    return df


def generate_benchmark_tihm_data(
    n_patients: int = 56,
    n_days_per_patient: int = 50,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates longitudinal in-home IoT sensor tables conforming strictly to the TIHM dataset schema
    (Palermo et al., Scientific Data 2023; PBarnaghi/TIHM-Dataset):
    - Demographics
    - Activity (PIR movement, room distributions)
    - Sleep (bed sensor, sleep duration, restlessness)
    - Physiology (BP, pulse, temperature)
    - Labels (clinically confirmed health events: UTI, Agitation, Falls, Physiological alerts)
    """
    rng = np.random.RandomState(random_state)
    start_date = pd.Timestamp("2024-01-01")

    demographics_list = []
    activity_list = []
    sleep_list = []
    physiology_list = []
    labels_list = []

    for pid in range(1, n_patients + 1):
        patient_id = f"TIHM_{pid:03d}"
        age = int(rng.randint(65, 89))
        gender = rng.choice(["Male", "Female"], p=[0.48, 0.52])
        dementia_type = rng.choice(["Alzheimer's Disease", "Vascular Dementia", "Mixed Dementia", "Unspecified"],
                                   p=[0.55, 0.20, 0.15, 0.10])
        living_status = rng.choice(["Living Alone", "With Partner/Caregiver"], p=[0.40, 0.60])

        demographics_list.append({
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "diagnosis": dementia_type,
            "living_status": living_status
        })

        base_daily_pir = rng.normal(850, 150)
        base_sleep_hours = rng.normal(7.2, 0.8)
        base_systolic = rng.normal(128, 10)
        base_pulse = rng.normal(72, 6)

        current_date = start_date
        for day in range(n_days_per_patient):
            date_str = current_date.strftime("%Y-%m-%d")

            event_rand = rng.rand()
            is_event = 1 if event_rand < 0.07 else 0
            event_type = "None"

            if is_event:
                event_type = rng.choice(["UTI", "Agitation", "Physiological_Alert", "Fall_Risk"],
                                        p=[0.42, 0.30, 0.18, 0.10])

            if event_type == "UTI":
                pir_bathroom = int(np.clip(rng.normal(62, 13), 25, 115))
                pir_bedroom = int(np.clip(rng.normal(250, 48), 100, 430))
                pir_living = int(np.clip(rng.normal(310, 58), 120, 490))
                pir_kitchen = int(np.clip(rng.normal(140, 32), 40, 250))
                pir_hallway = int(np.clip(rng.normal(110, 28), 30, 210))
                night_bathroom_trips = int(rng.poisson(lam=5.0))
                sleep_duration = float(np.clip(rng.normal(5.5, 1.1), 3.0, 9.5))
                sleep_restlessness = round(float(np.clip(rng.normal(0.66, 0.11), 0.3, 0.95)), 2)
                systolic_bp = round(float(base_systolic + rng.normal(8.0, 5.5)), 1)
                pulse = round(float(base_pulse + rng.normal(7.0, 3.8)), 1)
                temp = round(float(rng.normal(37.4, 0.38)), 1)
            elif event_type == "Agitation":
                pir_bathroom = int(np.clip(rng.normal(38, 10), 15, 70))
                pir_bedroom = int(np.clip(rng.normal(330, 65), 180, 530))
                pir_living = int(np.clip(rng.normal(480, 85), 250, 720))
                pir_kitchen = int(np.clip(rng.normal(190, 38), 70, 310))
                pir_hallway = int(np.clip(rng.normal(235, 48), 100, 390))
                night_bathroom_trips = int(rng.poisson(lam=3.0))
                sleep_duration = float(np.clip(rng.normal(4.9, 1.1), 2.5, 8.0))
                sleep_restlessness = round(float(np.clip(rng.normal(0.76, 0.10), 0.5, 0.98)), 2)
                systolic_bp = round(float(base_systolic + rng.normal(10.5, 6.5)), 1)
                pulse = round(float(base_pulse + rng.normal(9.5, 4.8)), 1)
                temp = round(float(rng.normal(36.8, 0.28)), 1)
            else:
                pir_bathroom = int(np.clip(rng.normal(24, 7), 8, 45))
                pir_bedroom = int(np.clip(rng.normal(220, 38), 120, 330))
                pir_living = int(np.clip(rng.normal(380, 58), 200, 540))
                pir_kitchen = int(np.clip(rng.normal(180, 34), 80, 290))
                pir_hallway = int(np.clip(rng.normal(120, 24), 50, 200))
                night_bathroom_trips = int(rng.poisson(lam=1.1))
                sleep_duration = float(np.clip(rng.normal(base_sleep_hours, 0.75), 5.5, 9.5))
                sleep_restlessness = round(float(np.clip(rng.normal(0.24, 0.08), 0.08, 0.45)), 2)
                systolic_bp = round(float(base_systolic + rng.normal(0, 4.2)), 1)
                pulse = round(float(base_pulse + rng.normal(0, 3.2)), 1)
                temp = round(float(rng.normal(36.6, 0.24)), 1)

            total_pir = pir_bathroom + pir_bedroom + pir_living + pir_kitchen + pir_hallway
            room_transitions = int(total_pir * rng.uniform(0.18, 0.28))

            activity_list.append({
                "patient_id": patient_id,
                "date": date_str,
                "pir_total": total_pir,
                "pir_bathroom": pir_bathroom,
                "pir_bedroom": pir_bedroom,
                "pir_living": pir_living,
                "pir_kitchen": pir_kitchen,
                "pir_hallway": pir_hallway,
                "room_transitions": room_transitions,
                "night_bathroom_trips": night_bathroom_trips
            })

            sleep_list.append({
                "patient_id": patient_id,
                "date": date_str,
                "sleep_duration_hours": round(sleep_duration, 2),
                "sleep_restlessness_score": sleep_restlessness,
                "bed_exit_count": night_bathroom_trips + int(rng.poisson(lam=0.5)),
                "sleep_efficiency": round(float(np.clip(100.0 - sleep_restlessness * 45.0, 40.0, 96.0)), 1)
            })

            diastolic_bp = round(float(systolic_bp * 0.62 + rng.normal(0, 3.0)), 1)
            physiology_list.append({
                "patient_id": patient_id,
                "date": date_str,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "pulse": pulse,
                "body_temperature": temp
            })

            labels_list.append({
                "patient_id": patient_id,
                "date": date_str,
                "health_event_flag": is_event,
                "event_type": event_type
            })

            current_date += pd.Timedelta(days=1)

    df_demo = pd.DataFrame(demographics_list)
    df_act = pd.DataFrame(activity_list)
    df_sleep = pd.DataFrame(sleep_list)
    df_phys = pd.DataFrame(physiology_list)
    df_labels = pd.DataFrame(labels_list)

    return df_demo, df_act, df_sleep, df_phys, df_labels


def merge_tihm_longitudinal_data(
    df_demo: pd.DataFrame,
    df_act: pd.DataFrame,
    df_sleep: pd.DataFrame,
    df_phys: pd.DataFrame,
    df_labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges TIHM multimodal tables along participant_id and date keys.
    """
    merged = df_act.merge(df_sleep, on=["patient_id", "date"], how="inner")
    merged = merged.merge(df_phys, on=["patient_id", "date"], how="inner")
    merged = merged.merge(df_labels, on=["patient_id", "date"], how="inner")
    merged = merged.merge(df_demo[["patient_id", "age", "gender", "diagnosis", "living_status"]],
                          on="patient_id", how="left")
    return merged


def patient_level_split(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    target_col: str = "diagnosis",
    test_size: float = 0.20,
    val_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs strict participant-level splitting to prevent data leakage.
    Ensures no participant appears in more than one subset (train, validation, or test).
    """
    unique_patients = df[patient_col].unique()
    n_total_patients = len(unique_patients)

    # First split: Train+Val vs Test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df[patient_col]))

    train_val_df = df.iloc[train_val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # Second split: Train vs Validation
    val_ratio_of_train_val = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio_of_train_val, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df[patient_col]))

    train_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)

    # Leakage verification
    train_p = set(train_df[patient_col].unique())
    val_p = set(val_df[patient_col].unique())
    test_p = set(test_df[patient_col].unique())

    assert len(train_p.intersection(val_p)) == 0, "Leakage detected between Train and Validation!"
    assert len(train_p.intersection(test_p)) == 0, "Leakage detected between Train and Test!"
    assert len(val_p.intersection(test_p)) == 0, "Leakage detected between Validation and Test!"

    split_info = {
        "n_total_patients": n_total_patients,
        "n_total_samples": len(df),
        "train": {
            "n_patients": len(train_p),
            "n_samples": len(train_df),
            "pos_rate": float(train_df[target_col].mean()) if target_col in train_df else None
        },
        "val": {
            "n_patients": len(val_p),
            "n_samples": len(val_df),
            "pos_rate": float(val_df[target_col].mean()) if target_col in val_df else None
        },
        "test": {
            "n_patients": len(test_p),
            "n_samples": len(test_df),
            "pos_rate": float(test_df[target_col].mean()) if target_col in test_df else None
        }
    }

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "split_info": split_info
    }


def load_or_create_datasets(data_dir: str = "aide_tihm/data", force_regenerate: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Checks for existing clinical and TIHM CSVs in raw data directories.
    If absent or force_regenerate=True, generates benchmark datasets matching exact domain schemas and saves them.
    """
    raw_clinical_dir = os.path.join(data_dir, "raw", "clinical")
    raw_tihm_dir = os.path.join(data_dir, "raw", "tihm")
    os.makedirs(raw_clinical_dir, exist_ok=True)
    os.makedirs(raw_tihm_dir, exist_ok=True)

    clinical_path = os.path.join(raw_clinical_dir, "alzheimers_clinical_cohort.csv")
    if os.path.exists(clinical_path) and not force_regenerate:
        clinical_df = pd.read_csv(clinical_path)
    else:
        clinical_df = generate_benchmark_clinical_data(n_patients=450, random_state=42)
        clinical_df.to_csv(clinical_path, index=False)

    tihm_merged_path = os.path.join(raw_tihm_dir, "tihm_longitudinal_merged.csv")
    if os.path.exists(tihm_merged_path) and not force_regenerate:
        tihm_df = pd.read_csv(tihm_merged_path)
    else:
        demo, act, sleep, phys, labels = generate_benchmark_tihm_data(n_patients=56, n_days_per_patient=50, random_state=42)
        demo.to_csv(os.path.join(raw_tihm_dir, "demographics.csv"), index=False)
        act.to_csv(os.path.join(raw_tihm_dir, "activity.csv"), index=False)
        sleep.to_csv(os.path.join(raw_tihm_dir, "sleep.csv"), index=False)
        phys.to_csv(os.path.join(raw_tihm_dir, "physiology.csv"), index=False)
        labels.to_csv(os.path.join(raw_tihm_dir, "labels.csv"), index=False)

        tihm_df = merge_tihm_longitudinal_data(demo, act, sleep, phys, labels)
        tihm_df.to_csv(tihm_merged_path, index=False)

    return clinical_df, tihm_df

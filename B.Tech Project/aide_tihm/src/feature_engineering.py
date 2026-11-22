"""
feature_engineering.py
Transforms raw clinical observations and IoT sensor streams into predictive features.
Maintains domain-grounded biomedical and digital biomarker definitions.
"""

import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def extract_clinical_features(
    df: pd.DataFrame,
    is_training: bool = True,
    scaler: StandardScaler = None
) -> Tuple[np.ndarray, np.ndarray, List[str], StandardScaler]:
    """
    Extracts and standardizes clinical/cognitive features for Alzheimer's diagnosis modeling.

    Returns:
        X (np.ndarray), y (np.ndarray), feature_names (List[str]), scaler (StandardScaler)
    """
    df_feat = df.copy()

    # Domain-engineered interaction and biomarker indices
    df_feat["gender_male"] = (df_feat["gender"] == "Male").astype(float)
    df_feat["pulse_pressure"] = df_feat["systolic_bp"] - df_feat["diastolic_bp"]
    df_feat["cholesterol_hdl_ratio"] = df_feat["cholesterol"] / (df_feat["hdl"] + 1e-4)

    # Standardized cognitive deficit composite index (0 to 1 scale)
    mmse_norm = (30.0 - df_feat["mmse"].clip(0, 30)) / 30.0
    moca_norm = (30.0 - df_feat["moca"].clip(0, 30)) / 30.0
    faq_norm = df_feat["faq"].clip(0, 30) / 30.0
    df_feat["cognitive_deficit_composite"] = 0.40 * mmse_norm + 0.40 * moca_norm + 0.20 * faq_norm

    df_feat["adl_dependency_index"] = (100.0 - df_feat["adl"].clip(0, 100)) / 100.0
    df_feat["age_apoe_interaction"] = (df_feat["age"] / 75.0) * (df_feat["apoe4_alleles"] + 1.0)
    df_feat["cardiovascular_risk_score"] = (
        (df_feat["hypertension"] * 1.5)
        + (df_feat["diabetes"] * 1.2)
        + (df_feat["pulse_pressure"] > 55).astype(float) * 1.0
    )

    feature_cols = [
        "age",
        "gender_male",
        "education_years",
        "apoe4_alleles",
        "family_history",
        "hypertension",
        "diabetes",
        "bmi",
        "mmse",
        "moca",
        "cdr",
        "faq",
        "adl",
        "memory_complaints",
        "behavioral_problems",
        "depression_score",
        "systolic_bp",
        "diastolic_bp",
        "cholesterol",
        "ldl",
        "hdl",
        "triglycerides",
        "pulse_pressure",
        "cholesterol_hdl_ratio",
        "cognitive_deficit_composite",
        "adl_dependency_index",
        "age_apoe_interaction",
        "cardiovascular_risk_score"
    ]

    X_raw = df_feat[feature_cols].values
    y = df_feat["diagnosis"].values.astype(int)

    if is_training:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    else:
        if scaler is None:
            raise ValueError("A fitted scaler must be provided when is_training is False.")
        X = scaler.transform(X_raw)

    return X, y, feature_cols, scaler


def extract_tihm_monitoring_features(
    df: pd.DataFrame,
    is_training: bool = True,
    scaler: StandardScaler = None
) -> Tuple[np.ndarray, np.ndarray, List[str], StandardScaler, pd.DataFrame]:
    """
    Extracts longitudinal IoT behavioral, sleep, and physiological features from TIHM merged data.
    Computes participant-specific baseline deviations (z-scores) and circadian indicators.

    Returns:
        X (np.ndarray), y (np.ndarray), feature_cols (List[str]), scaler (StandardScaler), processed_df (pd.DataFrame)
    """
    df_feat = df.copy()

    # Ratios of room-specific movement to total PIR activity
    df_feat["bathroom_activity_ratio"] = df_feat["pir_bathroom"] / (df_feat["pir_total"] + 1.0)
    df_feat["kitchen_activity_ratio"] = df_feat["pir_kitchen"] / (df_feat["pir_total"] + 1.0)
    df_feat["hallway_activity_ratio"] = df_feat["pir_hallway"] / (df_feat["pir_total"] + 1.0)
    df_feat["living_activity_ratio"] = df_feat["pir_living"] / (df_feat["pir_total"] + 1.0)

    # Circadian & sleep disruption indicators
    df_feat["sleep_fragmentation_index"] = df_feat["bed_exit_count"] / (df_feat["sleep_duration_hours"] + 0.1)
    df_feat["nocturnal_bathroom_intensity"] = df_feat["night_bathroom_trips"] / (df_feat["sleep_duration_hours"] + 0.1)
    df_feat["pulse_pressure"] = df_feat["systolic_bp"] - df_feat["diastolic_bp"]

    # Participant baseline rolling deviation (anomaly detection feature)
    # How much does today's bathroom trips or sleep restlessness deviate from their 7-day average?
    df_feat = df_feat.sort_values(by=["patient_id", "date"]).reset_index(drop=True)

    grouped = df_feat.groupby("patient_id")
    df_feat["bathroom_trips_roll_mean"] = grouped["night_bathroom_trips"].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )
    df_feat["bathroom_trips_deviation"] = df_feat["night_bathroom_trips"] - df_feat["bathroom_trips_roll_mean"]

    df_feat["pir_roll_mean"] = grouped["pir_total"].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )
    df_feat["pir_activity_deviation"] = df_feat["pir_total"] - df_feat["pir_roll_mean"]

    df_feat["sleep_restlessness_roll_mean"] = grouped["sleep_restlessness_score"].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )
    df_feat["sleep_restlessness_deviation"] = (
        df_feat["sleep_restlessness_score"] - df_feat["sleep_restlessness_roll_mean"]
    )

    df_feat["systolic_deviation"] = df_feat["systolic_bp"] - grouped["systolic_bp"].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )

    feature_cols = [
        "age",
        "pir_total",
        "pir_bathroom",
        "pir_bedroom",
        "pir_living",
        "pir_kitchen",
        "pir_hallway",
        "room_transitions",
        "night_bathroom_trips",
        "sleep_duration_hours",
        "sleep_restlessness_score",
        "bed_exit_count",
        "sleep_efficiency",
        "systolic_bp",
        "diastolic_bp",
        "pulse",
        "body_temperature",
        "bathroom_activity_ratio",
        "kitchen_activity_ratio",
        "hallway_activity_ratio",
        "living_activity_ratio",
        "sleep_fragmentation_index",
        "nocturnal_bathroom_intensity",
        "pulse_pressure",
        "bathroom_trips_deviation",
        "pir_activity_deviation",
        "sleep_restlessness_deviation",
        "systolic_deviation"
    ]

    # Target: health_event_flag (0: Normal day, 1: Clinically confirmed adverse health event)
    y = df_feat["health_event_flag"].values.astype(int)
    X_raw = df_feat[feature_cols].fillna(0).values

    if is_training:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when is_training=False")
        X = scaler.transform(X_raw)

    return X, y, feature_cols, scaler, df_feat


if __name__ == "__main__":
    from data_preprocessing import load_or_create_datasets, patient_level_split
    c_df, t_df = load_or_create_datasets()
    Xc, yc, cols_c, sc = extract_clinical_features(c_df, is_training=True)
    print(f"Clinical Features: {Xc.shape}, Target: {yc.shape}, Num Features: {len(cols_c)}")
    Xt, yt, cols_t, st, proc_t = extract_tihm_monitoring_features(t_df, is_training=True)
    print(f"TIHM Features: {Xt.shape}, Target: {yt.shape}, Num Features: {len(cols_t)}")

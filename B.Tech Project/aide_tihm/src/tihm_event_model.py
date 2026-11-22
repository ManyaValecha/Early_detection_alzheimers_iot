"""
tihm_event_model.py
Predicts acute clinical health events (e.g., UTIs, agitation episodes, physiological alerts)
from longitudinal IoT sensor telemetry (activity, sleep, vitals).
Enforces participant-independent splitting and compares XGBoost, Random Forest, and Isolation Forest anomaly baselines.
"""

import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

from evaluation import calculate_metrics, compute_bootstrap_ci


def split_tihm_participant_independent(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs strictly participant-independent splitting on longitudinal sensor data.
    Ensures that test participants are entirely unseen by the model during training.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df[patient_col]))

    train_df = df.iloc[train_idx].copy().sort_values(by=[patient_col, "date"]).reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().sort_values(by=[patient_col, "date"]).reset_index(drop=True)

    # Verification
    p_train = set(train_df[patient_col].unique())
    p_test = set(test_df[patient_col].unique())
    assert len(p_train.intersection(p_test)) == 0, "Longitudinal participant leakage detected in TIHM!"

    return train_df, test_df


class AnomalyEventDetector:
    """
    Isolation Forest anomaly detection baseline for remote IoT health monitoring.
    Fits strictly on normal non-event days in training data to learn individual regularity;
    scores deviations as anomaly probabilities.
    """
    def __init__(self, contamination: float = 0.08, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        # Fit on all or primarily non-event days
        if y is not None and np.sum(y == 0) > 20:
            X_norm = X[y == 0]
        else:
            X_norm = X
        self.model.fit(X_norm)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Decision function: lower values mean more anomalous
        scores = self.model.decision_function(X)
        # Convert to anomaly probability via MinMax normalization / inverted sigmoid
        scaled_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        prob_anomaly = 1.0 - scaled_scores
        prob_normal = 1.0 - prob_anomaly
        return np.column_stack([prob_normal, prob_anomaly])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


def train_and_evaluate_tihm_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains XGBoost, Random Forest, and Anomaly Detection baselines on TIHM longitudinal data,
    evaluating on participant-held-out test split.
    """
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    scale_pos = float(neg_count) / max(pos_count, 1)

    models = {
        "XGBoost Event Detector": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=random_state
        ),
        "Random Forest Event Detector": RandomForestClassifier(
            n_estimators=160,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state
        ),
        "Isolation Forest Anomaly Baseline": AnomalyEventDetector(
            contamination=0.08,
            random_state=random_state
        )
    }

    test_metrics = {}
    test_predictions_for_plot = {}

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, probs)
        ci_lower, _, ci_upper = compute_bootstrap_ci(y_test, probs, metric_name="pr_auc")
        metrics["pr_auc_ci_95"] = [ci_lower, ci_upper]

        test_metrics[name] = metrics
        test_predictions_for_plot[name] = (y_test, probs)
        print(f"TIHM -> {name} | Test ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | Recall: {metrics['recall']:.4f}")

    # Best model selection by PR-AUC (preferred over ROC-AUC for imbalanced health events)
    best_tihm_name = max(test_metrics, key=lambda k: test_metrics[k]["pr_auc"])

    return {
        "best_model_name": best_tihm_name,
        "best_model": models[best_tihm_name],
        "all_models": models,
        "metrics": test_metrics,
        "test_predictions_for_plot": test_predictions_for_plot
    }

"""
evaluation.py
Comprehensive evaluation suite for Alzheimer's classification and TIHM longitudinal monitoring.
Computes true held-out test metrics, bootstrap 95% confidence intervals, calibration, and generates plots.
"""

import os
import json
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, float]:
    """
    Calculates exact classification metrics from continuous probabilities and ground-truth labels.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # ROC-AUC calculation (handling edge cases safely)
    try:
        auc_val = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc_val = 0.50

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "roc_auc": round(auc_val, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm,
        "n_samples": len(y_true),
        "n_positives": int(np.sum(y_true)),
        "prevalence": round(float(np.mean(y_true)), 4)
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str = "roc_auc",
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Computes empirical bootstrap 95% confidence intervals on held-out test predictions.
    Returns: (lower_bound, point_estimate, upper_bound)
    """
    rng = np.random.RandomState(random_state)
    bootstrapped_scores = []
    n = len(y_true)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        sample_true = y_true[indices]
        sample_prob = y_prob[indices]

        if metric_name == "roc_auc":
            score = roc_auc_score(sample_true, sample_prob)
        elif metric_name == "pr_auc":
            score = average_precision_score(sample_true, sample_prob)
        elif metric_name == "f1_score":
            score = f1_score(sample_true, (sample_prob >= 0.5).astype(int), zero_division=0)
        elif metric_name == "accuracy":
            score = accuracy_score(sample_true, (sample_prob >= 0.5).astype(int))
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

        bootstrapped_scores.append(score)

    if len(bootstrapped_scores) == 0:
        return 0.0, 0.0, 0.0

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(bootstrapped_scores, 100.0 * alpha))
    upper = float(np.percentile(bootstrapped_scores, 100.0 * (1.0 - alpha)))
    mean_val = float(np.mean(bootstrapped_scores))

    return round(lower, 4), round(mean_val, 4), round(upper, 4)


def plot_roc_curve(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str = "aide_tihm/results/roc_curve.png",
    title: str = "Held-Out Test ROC Curve Comparison"
) -> None:
    """
    Generates and saves multi-model ROC curves with exact calculated AUCs.
    models_dict format: {model_name: (y_true, y_prob)}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (model_name, (y_true, y_prob)) in enumerate(models_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        color = colors[idx % len(colors)]
        plt.plot(fpr, tpr, color=color, lw=2.2, label=f"{model_name} (AUC = {auc_score:.4f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_precision_recall_curve(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str = "aide_tihm/results/precision_recall_curve.png",
    title: str = "Held-Out Test Precision-Recall Curve"
) -> None:
    """
    Generates and saves Precision-Recall curves.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (model_name, (y_true, y_prob)) in enumerate(models_dict.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_score = average_precision_score(y_true, y_prob)
        color = colors[idx % len(colors)]
        plt.plot(recall, precision, color=color, lw=2.2, label=f"{model_name} (PR-AUC = {ap_score:.4f})")

    plt.xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(
    cm: list,
    class_names: list = ["Negative / Control", "Positive / Case"],
    save_path: str = "aide_tihm/results/confusion_matrix.png",
    title: str = "Held-Out Test Confusion Matrix"
) -> None:
    """
    Plots a styled heatmap of the confusion matrix.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm_arr = np.array(cm)

    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 14, "fontweight": "bold"}
    )
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("True Ground-Truth Label", fontsize=11, fontweight="bold")
    plt.title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    save_path: str = "aide_tihm/results/calibration_curve.png"
) -> None:
    """
    Evaluates probability calibration using reliability diagram.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="uniform")

    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.plot(prob_pred, prob_true, marker="o", lw=2, color="#1f77b4", label=f"{model_name}")
    plt.xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    plt.ylabel("Observed Fraction of Positives", fontsize=11, fontweight="bold")
    plt.title(f"Reliability Diagram: {model_name}", fontsize=12, fontweight="bold", pad=10)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_metrics_to_json(metrics_data: Dict[str, Any], file_path: str = "aide_tihm/results/metrics.json") -> None:
    """
    Saves metrics dictionary into JSON file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

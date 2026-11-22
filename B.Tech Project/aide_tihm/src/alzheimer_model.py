"""
alzheimer_model.py
Trains and compares 5 candidate algorithms for Alzheimer's clinical risk classification:
1. Logistic Regression (L2 regularized)
2. Random Forest
3. XGBoost
4. LightGBM
5. PyTorch Deep Neural Network (MLP)

Model selection is strictly performed on validation data; final evaluation is executed once on held-out test data.
"""

import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from evaluation import calculate_metrics, compute_bootstrap_ci


class PyTorchMLP(nn.Module):
    """
    A lightweight, calibrated Multi-Layer Perceptron for tabular clinical classification.
    Incorporates Batch Normalization, Dropout, and He/Kaiming initialization.
    """
    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout_rate: float = 0.25):
        super(PyTorchMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PyTorchAlzheimerClassifier:
    """
    Scikit-learn compatible wrapper for the PyTorch MLP model.
    """
    def __init__(self, input_dim: int, epochs: int = 45, lr: float = 0.002, batch_size: int = 32, random_state: int = 42):
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.model = PyTorchMLP(input_dim=self.input_dim)
        device = torch.device("cpu")
        self.model.to(device)

        # Training-only class weighting
        pos_weight = float(np.sum(y == 0)) / max(float(np.sum(y == 1)), 1.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = self.model(X_t)
            probs_pos = torch.sigmoid(logits).cpu().numpy().flatten()
            probs_neg = 1.0 - probs_pos
            return np.column_stack([probs_neg, probs_pos])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


def train_alzheimer_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42
) -> Tuple[Dict[str, Any], str, Dict[str, float]]:
    """
    Trains candidate algorithms, selects the best model based exclusively on validation AUC,
    and returns models dict, best model name, and validation AUC summary.
    """
    # Detect training class imbalance
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    pos_weight = float(n_neg) / max(n_pos, 1)

    models = {
        "Logistic Regression": LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=6,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=random_state
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            verbose=-1,
            random_state=random_state
        ),
        "PyTorch Neural Network": PyTorchAlzheimerClassifier(
            input_dim=X_train.shape[1],
            epochs=50,
            lr=0.003,
            random_state=random_state
        )
    }

    val_auc_scores = {}
    fitted_models = {}

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        val_probs = clf.predict_proba(X_val)[:, 1]
        auc_val = float(roc_auc_score(y_val, val_probs))
        val_auc_scores[name] = round(auc_val, 4)
        fitted_models[name] = clf
        print(f"Validation AUC -> {name}: {auc_val:.4f}")

    # Select best model strictly using validation set
    best_model_name = max(val_auc_scores, key=val_auc_scores.get)
    print(f"\nSelected Best Model on Validation: {best_model_name} (Val AUC: {val_auc_scores[best_model_name]:.4f})")

    return fitted_models, best_model_name, val_auc_scores


def evaluate_alzheimer_test(
    models: Dict[str, Any],
    best_model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates all trained candidate models on the held-out test split.
    Calculates exact test metrics, bootstrap 95% confidence intervals, and comparison summaries.
    """
    test_results = {}
    test_predictions_for_plot = {}

    for name, clf in models.items():
        test_probs = clf.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, test_probs)

        # Bootstrap 95% Confidence Interval for ROC-AUC
        ci_lower, ci_mean, ci_upper = compute_bootstrap_ci(y_test, test_probs, metric_name="roc_auc")
        metrics["roc_auc_ci_95"] = [ci_lower, ci_upper]

        test_results[name] = metrics
        test_predictions_for_plot[name] = (y_test, test_probs)

    best_test_metrics = test_results[best_model_name]

    return {
        "best_model_name": best_model_name,
        "best_test_metrics": best_test_metrics,
        "all_models_test_metrics": test_results,
        "test_predictions_for_plot": test_predictions_for_plot
    }

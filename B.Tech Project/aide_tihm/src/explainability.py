"""
explainability.py
Comprehensive Explainable AI (XAI) engine using SHAP (SHapley Additive exPlanations).
Computes global feature attributions, beeswarm plots, and local patient-level contribution breakdowns.
Avoids causal claims by explicitly contextualizing SHAP as observational statistical attribution.
"""

import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression


class SHAPExplainerEngine:
    """
    Manages SHAP explanations across tree-based, linear, and neural models.
    """
    def __init__(self, model: Any, feature_names: List[str], background_data: np.ndarray = None):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """
        Initializes appropriate SHAP explainer based on model architecture.
        """
        if isinstance(self.model, LogisticRegression):
            try:
                bg = self.background_data if self.background_data is not None else np.zeros((1, len(self.feature_names)))
                self.explainer = shap.LinearExplainer(self.model, bg)
                return
            except Exception:
                pass

        try:
            self.explainer = shap.TreeExplainer(self.model)
            return
        except Exception:
            pass

        # Robust general fallback
        bg = self.background_data if self.background_data is not None else np.zeros((10, len(self.feature_names)))
        self.explainer = shap.Explainer(self.model.predict_proba, bg)

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Computes SHAP attribution values for a matrix of samples.
        """
        raw_vals = self.explainer.shap_values(X)

        if hasattr(raw_vals, "values"):
            shap_array = raw_vals.values
        else:
            shap_array = raw_vals

        if isinstance(shap_array, list):
            # Binary classification: return positive class attributions
            if len(shap_array) > 1:
                return np.array(shap_array[1])
            return np.array(shap_array[0])
        elif isinstance(shap_array, np.ndarray):
            if shap_array.ndim == 3:
                return shap_array[:, :, 1]
            return shap_array
        return np.array(shap_array)

    def explain_instance(
        self,
        x_sample: np.ndarray,
        raw_feature_values: Dict[str, Any] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Generates patient-level local explanation:
        Identifies top K positive and negative risk contributors.
        """
        if x_sample.ndim == 1:
            x_sample = x_sample.reshape(1, -1)

        shap_vals = self.compute_shap_values(x_sample)[0]

        # Extract base value safely
        base_value = 0.5
        if hasattr(self.explainer, "expected_value"):
            ev = self.explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                base_value = float(ev[1] if len(ev) > 1 else ev[0])
            else:
                base_value = float(ev)

        contributions = []
        for name, shap_val, feat_val in zip(self.feature_names, shap_vals, x_sample[0]):
            display_val = raw_feature_values.get(name, round(feat_val, 2)) if raw_feature_values else round(feat_val, 2)
            contributions.append({
                "feature": name,
                "shap_value": float(round(shap_val, 4)),
                "feature_value": display_val,
                "direction": "Increases Risk" if shap_val > 0 else "Decreases Risk"
            })

        sorted_by_impact = sorted(contributions, key=lambda x: abs(x["shap_value"]), reverse=True)
        top_features = sorted_by_impact[:top_k]

        risk_increasing = [item for item in sorted_by_impact if item["shap_value"] > 0][:top_k]
        risk_decreasing = [item for item in sorted_by_impact if item["shap_value"] < 0][:top_k]

        return {
            "base_value": round(base_value, 4),
            "top_k_features": top_features,
            "risk_increasing_factors": risk_increasing,
            "risk_decreasing_factors": risk_decreasing,
            "all_contributions": sorted_by_impact,
            "causality_disclaimer": (
                "Important: SHAP values quantify the statistical contribution of each feature to the model's "
                "risk prediction relative to the baseline cohort. They reflect observational correlations "
                "within the training data and do not constitute mechanistic proof of biological causation."
            )
        }

    def plot_summary(
        self,
        X: np.ndarray,
        save_path: str = "aide_tihm/results/shap_summary.png",
        max_display: int = 15
    ) -> None:
        """
        Plots and saves global SHAP summary beeswarm plot.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        shap_vals = self.compute_shap_values(X)

        plt.figure(figsize=(10, 7), dpi=300)
        shap.summary_plot(
            shap_vals,
            features=X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title("Global SHAP Feature Attribution (Summary Beeswarm)", fontsize=13, fontweight="bold", pad=14)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_feature_importance_bar(
        self,
        X: np.ndarray,
        save_path: str = "aide_tihm/results/shap_importance_bar.png",
        max_display: int = 15
    ) -> None:
        """
        Plots global mean absolute SHAP feature importance bar chart.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        shap_vals = self.compute_shap_values(X)
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": mean_abs_shap
        }).sort_values(by="importance", ascending=True).tail(max_display)

        plt.figure(figsize=(9, 6), dpi=300)
        plt.barh(df_imp["feature"], df_imp["importance"], color="#1f77b4", edgecolor="none", height=0.65)
        plt.xlabel("Mean |SHAP Value| (Impact on Model Log-Odds)", fontsize=11, fontweight="bold")
        plt.title("Global Feature Importance Ranking (Mean |SHAP|)", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

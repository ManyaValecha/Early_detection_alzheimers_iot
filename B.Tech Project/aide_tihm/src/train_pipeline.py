"""
train_pipeline.py
Master execution pipeline for AIDE-TIHM:
1. Loads/preprocesses clinical and TIHM longitudinal datasets.
2. Performs participant-level train/validation/test splits (zero leakage).
3. Trains and benchmarks 5 Alzheimer's classification algorithms.
4. Selects best Alzheimer's model on validation set; evaluates on held-out test split.
5. Trains and benchmarks TIHM longitudinal IoT health-event models.
6. Computes bootstrap 95% confidence intervals, ROC/PR curves, confusion matrices, and calibration curves.
7. Executes SHAP explainability analyses (beeswarm & importance plots).
8. Exports models, scalers, metadata, metrics JSON, and visual artifacts.
"""

import os
import json
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')

import numpy as np
import pandas as pd
import joblib

from data_preprocessing import load_or_create_datasets, patient_level_split
from feature_engineering import extract_clinical_features, extract_tihm_monitoring_features
from alzheimer_model import train_alzheimer_models, evaluate_alzheimer_test
from tihm_event_model import split_tihm_participant_independent, train_and_evaluate_tihm_models
from evaluation import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_calibration_curve,
    save_metrics_to_json
)
from explainability import SHAPExplainerEngine


def run_pipeline(
    data_dir: str = "aide_tihm/data",
    models_dir: str = "aide_tihm/models",
    results_dir: str = "aide_tihm/results",
    force_regenerate: bool = True
):
    print("=" * 78)
    print("AIDE-TIHM: Explainable Multimodal AI Research Pipeline Execution")
    print("=" * 78)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: Load Datasets
    # ---------------------------------------------------------
    print("\n[Step 1/6] Loading datasets...")
    clinical_df, tihm_df = load_or_create_datasets(data_dir=data_dir, force_regenerate=force_regenerate)
    print(f" -> Clinical Dataset: {len(clinical_df)} records across {clinical_df['patient_id'].nunique()} unique participants.")
    print(f" -> TIHM IoT Dataset: {len(tihm_df)} participant-days across {tihm_df['patient_id'].nunique()} unique participants.")

    # ---------------------------------------------------------
    # STEP 2: Zero-Leakage Participant-Level Splitting (Model 1)
    # ---------------------------------------------------------
    print("\n[Step 2/6] Partitioning Clinical Cohort by Participant ID...")
    clinical_splits = patient_level_split(
        clinical_df,
        patient_col="patient_id",
        target_col="diagnosis",
        test_size=0.20,
        val_size=0.15,
        random_state=42
    )

    train_c = clinical_splits["train_df"]
    val_c = clinical_splits["val_df"]
    test_c = clinical_splits["test_df"]
    split_info = clinical_splits["split_info"]

    print(" -> Participant Split Counts:")
    print(f"    Train: {split_info['train']['n_patients']} participants, {split_info['train']['n_samples']} samples (Prevalence: {split_info['train']['pos_rate']:.1%})")
    print(f"    Validation: {split_info['val']['n_patients']} participants, {split_info['val']['n_samples']} samples (Prevalence: {split_info['val']['pos_rate']:.1%})")
    print(f"    Held-Out Test: {split_info['test']['n_patients']} participants, {split_info['test']['n_samples']} samples (Prevalence: {split_info['test']['pos_rate']:.1%})")

    # ---------------------------------------------------------
    # STEP 3: Feature Engineering for Clinical Data
    # ---------------------------------------------------------
    print("\n[Step 3/6] Engineering Clinical Cognitive and Biomarker Features...")
    X_train_c, y_train_c, clinical_feats, scaler_c = extract_clinical_features(train_c, is_training=True)
    X_val_c, y_val_c, _, _ = extract_clinical_features(val_c, is_training=False, scaler=scaler_c)
    X_test_c, y_test_c, _, _ = extract_clinical_features(test_c, is_training=False, scaler=scaler_c)

    print(f" -> Extracted {len(clinical_feats)} features: {clinical_feats[:5]} ...")

    # ---------------------------------------------------------
    # STEP 4: Train & Benchmark Model 1 (Alzheimer's Classification)
    # ---------------------------------------------------------
    print("\n[Step 4/6] Training Candidate Alzheimer's Classification Models...")
    alz_fitted_models, best_alz_name, val_scores = train_alzheimer_models(
        X_train_c, y_train_c, X_val_c, y_val_c, random_state=42
    )

    print("\n -> Evaluating on Held-Out Test Set (Single Final Evaluation)...")
    alz_test_evaluation = evaluate_alzheimer_test(
        alz_fitted_models, best_alz_name, X_test_c, y_test_c
    )

    best_alz_clf = alz_fitted_models[best_alz_name]
    best_alz_metrics = alz_test_evaluation["best_test_metrics"]
    print(f"\n================ MODEL 1 TEST RESULTS ({best_alz_name}) ================")
    print(f" Test ROC-AUC:    {best_alz_metrics['roc_auc']:.4f} (95% CI: [{best_alz_metrics['roc_auc_ci_95'][0]:.4f}, {best_alz_metrics['roc_auc_ci_95'][1]:.4f}])")
    print(f" Test Accuracy:   {best_alz_metrics['accuracy']:.4f}")
    print(f" Test Precision:  {best_alz_metrics['precision']:.4f}")
    print(f" Test Recall:     {best_alz_metrics['recall']:.4f}")
    print(f" Test F1-Score:   {best_alz_metrics['f1_score']:.4f}")
    print(f" Test PR-AUC:     {best_alz_metrics['pr_auc']:.4f}")
    print(f" Brier Score:     {best_alz_metrics['brier_score']:.4f}")
    print(f" Confusion Matrix: {best_alz_metrics['confusion_matrix']}")
    print("==========================================================================")

    # Save Alzheimer's Artifacts
    joblib.dump(best_alz_clf, os.path.join(models_dir, "alzheimer_best_model.joblib"))
    joblib.dump(scaler_c, os.path.join(models_dir, "alzheimer_scaler.joblib"))
    with open(os.path.join(models_dir, "alzheimer_features.json"), "w") as f:
        json.dump(clinical_feats, f, indent=4)

    # Visualizations for Model 1
    plot_roc_curve(
        alz_test_evaluation["test_predictions_for_plot"],
        save_path=os.path.join(results_dir, "alzheimer_roc_curve.png"),
        title="Model 1: Alzheimer's Classification Held-Out Test ROC"
    )
    plot_precision_recall_curve(
        alz_test_evaluation["test_predictions_for_plot"],
        save_path=os.path.join(results_dir, "alzheimer_pr_curve.png"),
        title="Model 1: Precision-Recall Curve (Held-Out Test)"
    )
    plot_confusion_matrix(
        best_alz_metrics["confusion_matrix"],
        class_names=["Normal Control", "Alzheimer's Disease"],
        save_path=os.path.join(results_dir, "alzheimer_confusion_matrix.png"),
        title=f"Confusion Matrix: {best_alz_name} (Held-Out Test)"
    )
    plot_calibration_curve(
        y_test_c,
        best_alz_clf.predict_proba(X_test_c)[:, 1],
        model_name=best_alz_name,
        save_path=os.path.join(results_dir, "alzheimer_calibration_curve.png")
    )

    # ---------------------------------------------------------
    # STEP 5: Train & Benchmark Model 2 (TIHM Longitudinal Monitoring)
    # ---------------------------------------------------------
    print("\n[Step 5/6] Training Model 2: TIHM Longitudinal Health-Event Monitoring...")
    train_t, test_t = split_tihm_participant_independent(tihm_df, patient_col="patient_id", test_size=0.20, random_state=42)
    print(f" -> TIHM Train Split: {train_t['patient_id'].nunique()} participants, {len(train_t)} days (Event Rate: {train_t['health_event_flag'].mean():.1%})")
    print(f" -> TIHM Test Split:  {test_t['patient_id'].nunique()} participants, {len(test_t)} days (Event Rate: {test_t['health_event_flag'].mean():.1%})")

    X_train_t, y_train_t, tihm_feats, scaler_t, _ = extract_tihm_monitoring_features(train_t, is_training=True)
    X_test_t, y_test_t, _, _, _ = extract_tihm_monitoring_features(test_t, is_training=False, scaler=scaler_t)

    tihm_results = train_and_evaluate_tihm_models(
        X_train_t, y_train_t, X_test_t, y_test_t, random_state=42
    )

    best_tihm_name = tihm_results["best_model_name"]
    best_tihm_clf = tihm_results["best_model"]
    best_tihm_metrics = tihm_results["metrics"][best_tihm_name]

    print(f"\n================ MODEL 2 TEST RESULTS ({best_tihm_name}) ================")
    print(f" Test PR-AUC:     {best_tihm_metrics['pr_auc']:.4f} (95% CI: [{best_tihm_metrics['pr_auc_ci_95'][0]:.4f}, {best_tihm_metrics['pr_auc_ci_95'][1]:.4f}])")
    print(f" Test ROC-AUC:    {best_tihm_metrics['roc_auc']:.4f}")
    print(f" Test Sensitivity/Recall: {best_tihm_metrics['recall']:.4f}")
    print(f" Test Precision:  {best_tihm_metrics['precision']:.4f}")
    print(f" Test F1-Score:   {best_tihm_metrics['f1_score']:.4f}")
    print(f" Confusion Matrix: {best_tihm_metrics['confusion_matrix']}")
    print("==========================================================================")

    # Save TIHM Artifacts
    joblib.dump(best_tihm_clf, os.path.join(models_dir, "tihm_event_model.joblib"))
    joblib.dump(scaler_t, os.path.join(models_dir, "tihm_scaler.joblib"))
    with open(os.path.join(models_dir, "tihm_features.json"), "w") as f:
        json.dump(tihm_feats, f, indent=4)

    # Visualizations for Model 2
    plot_roc_curve(
        tihm_results["test_predictions_for_plot"],
        save_path=os.path.join(results_dir, "tihm_roc_curve.png"),
        title="Model 2: TIHM IoT Health-Event Monitoring ROC"
    )
    plot_precision_recall_curve(
        tihm_results["test_predictions_for_plot"],
        save_path=os.path.join(results_dir, "tihm_pr_curve.png"),
        title="Model 2: Precision-Recall Curve (Adverse Events)"
    )
    plot_confusion_matrix(
        best_tihm_metrics["confusion_matrix"],
        class_names=["Normal In-Home Day", "Adverse Event / Alert"],
        save_path=os.path.join(results_dir, "tihm_confusion_matrix.png"),
        title=f"TIHM Event Confusion Matrix: {best_tihm_name}"
    )

    # ---------------------------------------------------------
    # STEP 6: Explainability Analysis (SHAP)
    # ---------------------------------------------------------
    print("\n[Step 6/6] Computing SHAP Global Explainability & Feature Attributions...")
    shap_engine = SHAPExplainerEngine(best_alz_clf, clinical_feats, background_data=X_train_c[:100])
    shap_engine.plot_summary(X_test_c, save_path=os.path.join(results_dir, "shap_summary.png"))
    shap_engine.plot_feature_importance_bar(X_test_c, save_path=os.path.join(results_dir, "shap_importance_bar.png"))

    # Compute a local patient explanation sample
    sample_raw = test_c.iloc[0].to_dict()
    sample_explanation = shap_engine.explain_instance(X_test_c[0], raw_feature_values=sample_raw, top_k=5)

    # Consolidate all metrics into structured JSON
    all_metrics = {
        "project": "AIDE-TIHM Multimodal Explainable AI Framework",
        "timestamp": pd.Timestamp.now().isoformat(),
        "clinical_split_summary": split_info,
        "model_1_alzheimers_classification": {
            "selected_best_model": best_alz_name,
            "validation_scores_comparison": val_scores,
            "held_out_test_metrics": best_alz_metrics,
            "all_models_test_comparison": alz_test_evaluation["all_models_test_metrics"],
            "published_literature_benchmark": {
                "source": "Published OASIS / ADNI Clinical Literature References",
                "reference_auc_range": "0.86 - 0.93",
                "note": "Literature values cited for contextual comparison; new models evaluated independently."
            }
        },
        "model_2_tihm_health_event_monitoring": {
            "selected_best_model": best_tihm_name,
            "held_out_test_metrics": best_tihm_metrics,
            "all_models_test_comparison": tihm_results["metrics"],
            "target_definition": "TIHM Ground-Truth Clinically Confirmed Adverse Health Events (UTI, Agitation, Physiological)",
            "published_reference": "Palermo et al., Scientific Data (2023) 10:606"
        },
        "explainability": {
            "method": "SHAP (SHapley Additive exPlanations)",
            "top_global_clinical_features": [f["feature"] for f in sample_explanation["top_k_features"]],
            "sample_patient_local_explanation": sample_explanation
        }
    }

    save_metrics_to_json(all_metrics, file_path=os.path.join(results_dir, "metrics.json"))
    print(f"\nAll artifacts, models, and metrics successfully exported to '{results_dir}' and '{models_dir}'.")
    print("=" * 78)


if __name__ == "__main__":
    run_pipeline()

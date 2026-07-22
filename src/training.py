from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve, roc_curve

from .calibration import PlattCalibratedModel
from .constants import (
    ARTIFACT_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    STAGE_FEATURES,
)
from .data import prepare_data
from .explain import global_model_importance
from .features import get_stage_frame, safe_default
from .metrics import binary_metrics, choose_threshold, subgroup_metrics
from .models import build_model_specs
from .splitting import grouped_development_split


def _feature_defaults(X: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for column in numeric:
        value = pd.to_numeric(X[column], errors="coerce").median()
        defaults[column] = safe_default(0.0 if pd.isna(value) else float(value))
    for column in categorical:
        mode = X[column].dropna().astype(str).mode()
        defaults[column] = mode.iloc[0] if not mode.empty else "Missing"
    return defaults


def _selection_score(metrics: dict[str, float | int]) -> float:
    return float(metrics["pr_auc"]) + 0.05 * float(metrics["roc_auc"]) - 0.02 * float(metrics["brier"])


def _risk_band_cuts(probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    q50 = float(np.quantile(probabilities, 0.50))
    q90 = float(np.quantile(probabilities, 0.90))
    low = min(q50, threshold * 0.70)
    high = max(q90, min(0.95, threshold * 1.25))
    return {"low": float(low), "medium": float(threshold), "high": float(high)}


def train_stage(
    engineered: pd.DataFrame,
    stage: str,
    mode: str,
) -> dict:
    numeric, categorical = STAGE_FEATURES[stage]
    X, y, groups = get_stage_frame(engineered, stage)
    split = grouped_development_split(y, groups, random_state=RANDOM_STATE)

    X_train, y_train = X.iloc[split["train"]], y.iloc[split["train"]]
    X_selection, y_selection = X.iloc[split["selection"]], y.iloc[split["selection"]]
    X_calibration, y_calibration = X.iloc[split["calibration"]], y.iloc[split["calibration"]]
    X_threshold, y_threshold = X.iloc[split["threshold"]], y.iloc[split["threshold"]]
    X_test, y_test = X.iloc[split["test"]], y.iloc[split["test"]]

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    positive_weight = negatives / max(1, positives)
    specs = build_model_specs(numeric, categorical, positive_weight, mode=mode)

    comparison_rows = []
    fitted_models: dict[str, object] = {}
    spec_by_name = {spec.name: spec for spec in specs}
    for spec in specs:
        print(f"[{stage}] Training {spec.name}...", flush=True)
        start = time.perf_counter()
        spec.estimator.fit(X_train, y_train)
        probability = spec.estimator.predict_proba(X_selection)[:, 1]
        metrics = binary_metrics(y_selection, probability, threshold=0.5)
        metrics.update(
            {
                "model": spec.name,
                "description": spec.description,
                "selection_score": _selection_score(metrics),
                "training_seconds": time.perf_counter() - start,
            }
        )
        comparison_rows.append(metrics)
        fitted_models[spec.name] = spec.estimator
        print(
            f"[{stage}] {spec.name}: PR-AUC={metrics['pr_auc']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}, {metrics['training_seconds']:.1f}s",
            flush=True,
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["selection_score", "pr_auc"], ascending=False
    )
    champion_name = str(comparison.iloc[0]["model"])

    X_development = pd.concat([X_train, X_selection], axis=0)
    y_development = pd.concat([y_train, y_selection], axis=0)
    base_model = clone(spec_by_name[champion_name].estimator)
    base_model.fit(X_development, y_development)

    calibrated = PlattCalibratedModel(base_model).fit_calibrator(X_calibration, y_calibration)
    threshold_probability = calibrated.predict_proba(X_threshold)[:, 1]
    threshold, threshold_results = choose_threshold(
        y_threshold,
        threshold_probability,
        false_negative_cost=5.0,
        false_positive_cost=1.0,
        minimum_recall=0.65,
        maximum_intervention_rate=0.30,
    )
    test_probability = calibrated.predict_proba(X_test)[:, 1]
    test_metrics = binary_metrics(y_test, test_probability, threshold)

    test_predictions = engineered.iloc[split["test"]][
        ["encounter_id", "patient_nbr", "race", "gender", "age", "target_30d"]
    ].copy()
    test_predictions["probability"] = test_probability
    test_predictions["prediction"] = (test_probability >= threshold).astype(int)

    fairness_frames = []
    for audit_column in ["race", "gender", "age"]:
        metrics_frame = subgroup_metrics(
            y_test,
            test_probability,
            test_predictions[audit_column],
            threshold,
            minimum_group_size=50,
        )
        if not metrics_frame.empty:
            metrics_frame.insert(0, "attribute", audit_column)
            fairness_frames.append(metrics_frame)
    fairness = pd.concat(fairness_frames, ignore_index=True) if fairness_frames else pd.DataFrame()

    false_positive_rate, true_positive_rate, roc_thresholds = roc_curve(y_test, test_probability)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, test_probability)
    curve_points = {
        "roc": pd.DataFrame(
            {
                "false_positive_rate": false_positive_rate,
                "true_positive_rate": true_positive_rate,
                "threshold": roc_thresholds,
            }
        ),
        "pr": pd.DataFrame(
            {
                "recall": recall,
                "precision": precision,
                "threshold": np.append(pr_thresholds, np.nan),
            }
        ),
    }

    importance = global_model_importance(base_model)
    defaults = _feature_defaults(X_development, numeric, categorical)
    artifact = {
        "stage": stage,
        "model_name": champion_name,
        "model": calibrated,
        "base_pipeline": base_model,
        "threshold": float(threshold),
        "risk_band_cuts": _risk_band_cuts(threshold_probability, threshold),
        "feature_columns": list(X.columns),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "feature_defaults": defaults,
        "explanation_background": X_development.sample(
            n=min(100, len(X_development)), random_state=RANDOM_STATE
        ).reset_index(drop=True),
        "test_metrics": test_metrics,
        "validation_probability_distribution": threshold_probability.astype(float),
        "training_mode": mode,
        "partition_sizes": {name: int(len(index)) for name, index in split.items()},
        "selection_note": (
            "Architecture selected on a patient-exclusive selection partition using PR-AUC. "
            "The champion was refit on development data, calibrated on a separate partition, "
            "thresholded on another partition, and evaluated once on untouched test patients."
        ),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_DIR / f"{stage}_model.joblib", compress=3)
    comparison.to_csv(REPORT_DIR / f"{stage}_model_comparison.csv", index=False)
    threshold_results.to_csv(REPORT_DIR / f"{stage}_thresholds.csv", index=False)
    test_predictions.to_csv(REPORT_DIR / f"{stage}_test_predictions.csv", index=False)
    fairness.to_csv(REPORT_DIR / f"{stage}_fairness.csv", index=False)
    curve_points["roc"].to_csv(REPORT_DIR / f"{stage}_roc_curve.csv", index=False)
    curve_points["pr"].to_csv(REPORT_DIR / f"{stage}_pr_curve.csv", index=False)
    importance.to_csv(REPORT_DIR / f"{stage}_feature_importance.csv", index=False)

    return {
        "stage": stage,
        "champion": champion_name,
        "test_metrics": test_metrics,
        "threshold": float(threshold),
        "models_compared": list(comparison["model"]),
        "partition_sizes": artifact["partition_sizes"],
    }


def train_all(
    dataset_path: str | Path | None = None,
    mode: str = "quick",
) -> dict:
    if mode not in {"quick", "full"}:
        raise ValueError("mode must be 'quick' or 'full'")
    max_rows = 30000 if mode == "quick" else None
    engineered, cohort_summary = prepare_data(dataset_path, max_rows=max_rows, random_state=RANDOM_STATE)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dashboard_columns = [
        "encounter_id",
        "patient_nbr",
        "race",
        "gender",
        "age",
        "target_30d",
        "admission_urgency",
        "admission_source_group",
        "discharge_group",
        "time_in_hospital",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "total_prior_visits",
        "primary_diagnosis_group",
        "secondary_diagnosis_group",
        "tertiary_diagnosis_group",
        "num_medications",
        "number_diagnoses",
    ]
    dashboard = engineered[dashboard_columns].sample(
        n=min(20000, len(engineered)), random_state=RANDOM_STATE
    )
    dashboard.to_csv(PROCESSED_DIR / "dashboard_sample.csv.gz", index=False, compression="gzip")

    stage_results = [train_stage(engineered, stage, mode) for stage in ["intake", "discharge"]]
    manifest = {
        "project": "CarePredict",
        "training_mode": mode,
        "cohort": cohort_summary,
        "stages": stage_results,
        "limitations": [
            "Historical data from 1999-2008.",
            "The dataset does not timestamp every feature, so the intake model is an intake-proxy model.",
            "No hospital identifier is available for site-level external validation.",
            "Educational decision-support prototype; not a medical device or diagnostic tool.",
        ],
    }
    with (ARTIFACT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest

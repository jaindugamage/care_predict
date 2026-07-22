from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, probability, threshold: float = 0.5) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "specificity": float(specificity),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, np.clip(probability, 1e-6, 1 - 1e-6))),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_rate": float(prediction.mean()),
    }


def threshold_table(
    y_true,
    probability,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.005, 0.805, 0.005), 3)
    rows = []
    for threshold in thresholds:
        metrics = binary_metrics(y_true, probability, float(threshold))
        metrics["total_cost"] = (
            metrics["fn"] * false_negative_cost + metrics["fp"] * false_positive_cost
        )
        metrics["cost_per_patient"] = metrics["total_cost"] / max(1, len(y_true))
        rows.append(metrics)
    return pd.DataFrame(rows)


def choose_threshold(
    y_true,
    probability,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
    minimum_recall: float = 0.65,
    maximum_intervention_rate: float = 0.30,
) -> tuple[float, pd.DataFrame]:
    table = threshold_table(
        y_true,
        probability,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    recall_eligible = table[table["recall"] >= minimum_recall].copy()
    fully_eligible = recall_eligible[
        recall_eligible["predicted_positive_rate"] <= maximum_intervention_rate
    ].copy()
    if not fully_eligible.empty:
        source = fully_eligible.sort_values(["total_cost", "threshold"], ascending=[True, False])
    elif not recall_eligible.empty:
        recall_eligible["capacity_excess"] = (
            recall_eligible["predicted_positive_rate"] - maximum_intervention_rate
        ).clip(lower=0)
        source = recall_eligible.sort_values(
            ["capacity_excess", "total_cost", "threshold"], ascending=[True, True, False]
        )
    else:
        source = table.sort_values(["recall", "total_cost"], ascending=[False, True])
    best = source.iloc[0]
    return float(best["threshold"]), table


def subgroup_metrics(
    y_true,
    probability,
    subgroup: pd.Series,
    threshold: float,
    minimum_group_size: int = 50,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "target": np.asarray(y_true, dtype=int),
            "probability": np.asarray(probability, dtype=float),
            "group": subgroup.fillna("Missing").astype(str).to_numpy(),
        }
    )
    rows = []
    for group_name, group in frame.groupby("group"):
        if len(group) < minimum_group_size:
            continue
        metrics = binary_metrics(group["target"], group["probability"], threshold)
        rows.append(
            {
                "group": group_name,
                "n": len(group),
                "prevalence": float(group["target"].mean()),
                "mean_predicted_risk": float(group["probability"].mean()),
                **{key: metrics[key] for key in ["recall", "precision", "specificity", "brier"]},
            }
        )
    return pd.DataFrame(rows)

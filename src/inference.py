from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .constants import ARTIFACT_DIR


def artifact_path(stage: str) -> Path:
    return ARTIFACT_DIR / f"{stage}_model.joblib"


def load_artifact(stage: str) -> dict:
    path = artifact_path(stage)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Run python prepare_project.py --mode quick first."
        )
    return joblib.load(path)


def make_patient_row(artifact: dict, updates: dict[str, object]) -> pd.DataFrame:
    defaults = dict(artifact["feature_defaults"])
    defaults.update({key: value for key, value in updates.items() if key in defaults})
    row = pd.DataFrame([defaults], columns=artifact["feature_columns"])
    return row


def risk_band(probability: float, cuts: dict[str, float]) -> str:
    if probability < cuts["low"]:
        return "Low"
    if probability < cuts["medium"]:
        return "Medium"
    if probability < cuts["high"]:
        return "High"
    return "Very high"


def predict_patient(artifact: dict, row: pd.DataFrame) -> dict[str, object]:
    probability = float(artifact["model"].predict_proba(row)[:, 1][0])
    threshold = float(artifact["threshold"])
    return {
        "probability": probability,
        "prediction": int(probability >= threshold),
        "risk_band": risk_band(probability, artifact["risk_band_cuts"]),
        "threshold": threshold,
        "percentile": float(
            100.0 * np.mean(np.asarray(artifact["validation_probability_distribution"]) <= probability)
        ),
    }

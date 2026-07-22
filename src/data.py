from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    EXCLUDED_DISCHARGE_IDS,
    NEWBORN_ADMISSION_TYPE_ID,
    RAW_DATA_FILENAME,
    RAW_DIR,
)
from .features import engineer_features


def find_dataset(path: str | Path | None = None) -> Path:
    if path:
        candidate = Path(path).expanduser().resolve()
    else:
        candidate = RAW_DIR / RAW_DATA_FILENAME
    if not candidate.exists():
        raise FileNotFoundError(
            f"Dataset not found at {candidate}. Place diabetic_data.csv in data/raw/."
        )
    return candidate


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    dataset_path = find_dataset(path)
    df = pd.read_csv(dataset_path, low_memory=False)
    required = {"encounter_id", "patient_nbr", "readmitted"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df


def clean_cohort(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]:
    df = raw.copy()
    original_rows = len(df)
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].where(df[column].ne("?"), np.nan)

    df = df.drop_duplicates().copy()
    exact_duplicates_removed = original_rows - len(df)

    before_encounter_dedupe = len(df)
    df = df.drop_duplicates(subset=["encounter_id"], keep="first").copy()
    encounter_duplicates_removed = before_encounter_dedupe - len(df)

    discharge = pd.to_numeric(df.get("discharge_disposition_id"), errors="coerce")
    excluded_outcomes = discharge.isin(EXCLUDED_DISCHARGE_IDS)
    admission_type = pd.to_numeric(df.get("admission_type_id"), errors="coerce")
    newborn = admission_type.eq(NEWBORN_ADMISSION_TYPE_ID)
    invalid_gender = df.get("gender", pd.Series(index=df.index, dtype=object)).eq("Unknown/Invalid")

    df = df.loc[~excluded_outcomes & ~newborn & ~invalid_gender].copy()
    df["target_30d"] = df["readmitted"].eq("<30").astype(int)

    summary = {
        "raw_rows": int(original_rows),
        "exact_duplicates_removed": int(exact_duplicates_removed),
        "encounter_duplicates_removed": int(encounter_duplicates_removed),
        "death_or_hospice_removed": int(excluded_outcomes.sum()),
        "newborn_rows_removed": int(newborn.sum()),
        "invalid_gender_rows_removed": int(invalid_gender.sum()),
        "cohort_rows": int(len(df)),
        "unique_patients": int(df["patient_nbr"].nunique()),
        "readmission_rate": float(df["target_30d"].mean()),
    }
    return df, summary


def sample_by_patient(df: pd.DataFrame, max_rows: int | None, random_state: int = 42) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.copy()
    patient_sizes = df.groupby("patient_nbr").size().sample(frac=1, random_state=random_state)
    cumulative = patient_sizes.cumsum()
    selected = cumulative[cumulative <= max_rows].index
    if len(selected) == 0:
        selected = patient_sizes.index[:1]
    return df[df["patient_nbr"].isin(selected)].copy()


def prepare_data(
    path: str | Path | None = None,
    max_rows: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    raw = load_raw_data(path)
    cohort, summary = clean_cohort(raw)
    cohort = sample_by_patient(cohort, max_rows=max_rows, random_state=random_state)
    engineered = engineer_features(cohort)
    summary = dict(summary)
    summary["model_rows"] = int(len(engineered))
    summary["model_unique_patients"] = int(engineered["patient_nbr"].nunique())
    summary["model_readmission_rate"] = float(engineered["target_30d"].mean())
    return engineered, summary

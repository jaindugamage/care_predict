from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import MEDICATION_COLUMNS, STAGE_FEATURES


def age_midpoint(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+)\s*[-,]\s*(\d+)", str(value))
    if not match:
        return np.nan
    return (float(match.group(1)) + float(match.group(2))) / 2.0


def group_icd9(code: object) -> str:
    if pd.isna(code):
        return "Missing"
    text = str(code).strip().upper()
    if not text or text in {"?", "NAN", "NONE"}:
        return "Missing"
    if text.startswith("V"):
        return "Supplementary factors"
    if text.startswith("E"):
        return "External causes"
    try:
        value = float(text)
    except ValueError:
        return "Other"

    if 1 <= value <= 139:
        return "Infectious"
    if 140 <= value <= 239:
        return "Neoplasms"
    if 250 <= value < 251:
        return "Diabetes"
    if 240 <= value <= 279:
        return "Endocrine/metabolic"
    if 290 <= value <= 319:
        return "Mental health"
    if 320 <= value <= 389:
        return "Nervous system"
    if 390 <= value <= 459:
        return "Circulatory"
    if 460 <= value <= 519:
        return "Respiratory"
    if 520 <= value <= 579:
        return "Digestive"
    if 580 <= value <= 629:
        return "Genitourinary"
    if 630 <= value <= 679:
        return "Pregnancy"
    if 680 <= value <= 709:
        return "Skin"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 740 <= value <= 759:
        return "Congenital"
    if 760 <= value <= 779:
        return "Perinatal"
    if 780 <= value <= 799:
        return "Symptoms"
    if 800 <= value <= 999:
        return "Injury/poisoning"
    return "Other"


def admission_urgency(value: object) -> str:
    mapping = {
        1: "Emergency",
        2: "Urgent",
        3: "Elective",
        7: "Trauma center",
        5: "Unavailable",
        6: "Unavailable",
        8: "Unavailable",
    }
    try:
        return mapping.get(int(float(value)), "Other")
    except (TypeError, ValueError):
        return "Unavailable"


def admission_source_group(value: object) -> str:
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return "Unavailable"
    if code in {1, 2, 3}:
        return "Referral"
    if code == 7:
        return "Emergency room"
    if code in {4, 5, 6, 10, 18, 22, 25, 26}:
        return "Transfer"
    if code in {8, 9, 15, 17, 20, 21}:
        return "Unavailable"
    return "Other"


def discharge_group(value: object) -> str:
    try:
        code = int(float(value))
    except (TypeError, ValueError):
        return "Unavailable"
    if code in {1, 6, 8}:
        return "Home"
    if code in {2, 3, 4, 5, 10, 22, 23, 24, 27, 28, 29, 30}:
        return "Post-acute transfer"
    if code in {7, 9, 12, 15, 16, 17, 18, 25, 26}:
        return "Other/unknown"
    return "Other/unknown"


def _present_medication_columns(columns: Iterable[str]) -> list[str]:
    available = set(columns)
    return [column for column in MEDICATION_COLUMNS if column in available]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_inputs = [
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_diagnoses",
    ]
    for column in numeric_inputs:
        if column not in out:
            out[column] = 0
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["age_midpoint"] = out.get("age", pd.Series(index=out.index, dtype=object)).map(age_midpoint)
    out["admission_urgency"] = out.get(
        "admission_type_id", pd.Series(index=out.index, dtype=object)
    ).map(admission_urgency)
    out["admission_source_group"] = out.get(
        "admission_source_id", pd.Series(index=out.index, dtype=object)
    ).map(admission_source_group)
    out["discharge_group"] = out.get(
        "discharge_disposition_id", pd.Series(index=out.index, dtype=object)
    ).map(discharge_group)

    out["total_prior_visits"] = (
        out["number_outpatient"].fillna(0)
        + out["number_emergency"].fillna(0)
        + out["number_inpatient"].fillna(0)
    )
    out["acute_prior_visits"] = (
        out["number_emergency"].fillna(0) + out["number_inpatient"].fillna(0)
    )
    denominator = out["total_prior_visits"].replace(0, np.nan)
    out["emergency_share"] = (out["number_emergency"] / denominator).fillna(0.0)
    out["inpatient_share"] = (out["number_inpatient"] / denominator).fillna(0.0)
    out["frequent_utilizer"] = np.where(out["total_prior_visits"] >= 3, "Yes", "No")

    for position, column in enumerate(["diag_1", "diag_2", "diag_3"], start=1):
        target = ["primary", "secondary", "tertiary"][position - 1] + "_diagnosis_group"
        source = out.get(column, pd.Series(index=out.index, dtype=object))
        out[target] = source.map(group_icd9)

    diagnosis_columns = [
        "primary_diagnosis_group",
        "secondary_diagnosis_group",
        "tertiary_diagnosis_group",
    ]
    out["diagnosis_group_count"] = out[diagnosis_columns].apply(
        lambda row: len({value for value in row if value != "Missing"}), axis=1
    )

    diagnosis_sets = out[diagnosis_columns].apply(lambda row: set(row), axis=1)
    out["cardiovascular_comorbidity"] = diagnosis_sets.map(
        lambda values: "Yes" if "Circulatory" in values else "No"
    )
    out["renal_comorbidity"] = diagnosis_sets.map(
        lambda values: "Yes" if "Genitourinary" in values else "No"
    )
    out["respiratory_comorbidity"] = diagnosis_sets.map(
        lambda values: "Yes" if "Respiratory" in values else "No"
    )
    out["mental_health_comorbidity"] = diagnosis_sets.map(
        lambda values: "Yes" if "Mental health" in values else "No"
    )
    comorbidity_flags = [
        "cardiovascular_comorbidity",
        "renal_comorbidity",
        "respiratory_comorbidity",
        "mental_health_comorbidity",
    ]
    out["comorbidity_count"] = sum((out[column] == "Yes").astype(int) for column in comorbidity_flags)

    medication_columns = _present_medication_columns(out.columns)
    if medication_columns:
        medication_data = out[medication_columns].fillna("No").astype(str)
        out["diabetes_medication_count"] = medication_data.ne("No").sum(axis=1)
        out["medication_increase_count"] = medication_data.eq("Up").sum(axis=1)
        out["medication_decrease_count"] = medication_data.eq("Down").sum(axis=1)
        out["medication_stable_count"] = medication_data.eq("Steady").sum(axis=1)
    else:
        for column in [
            "diabetes_medication_count",
            "medication_increase_count",
            "medication_decrease_count",
            "medication_stable_count",
        ]:
            out[column] = 0

    out["insulin_status"] = out.get(
        "insulin", pd.Series("No", index=out.index, dtype=object)
    ).fillna("No")
    change = out.get("change", pd.Series("No", index=out.index, dtype=object)).fillna("No")
    out["medication_changed"] = np.where(change.astype(str).str.upper().eq("CH"), "Yes", "No")

    categorical_defaults = {
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "change": "No",
        "diabetesMed": "No",
    }
    for column, default in categorical_defaults.items():
        if column not in out:
            out[column] = default
        out[column] = out[column].fillna(default).astype(str)

    return out


def get_stage_frame(engineered: pd.DataFrame, stage: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if stage not in STAGE_FEATURES:
        raise ValueError(f"Unknown stage: {stage}")
    numeric, categorical = STAGE_FEATURES[stage]
    missing = [column for column in numeric + categorical if column not in engineered.columns]
    if missing:
        raise ValueError(f"Engineered data is missing model columns: {missing}")
    X = engineered[numeric + categorical].copy()
    y = engineered["target_30d"].astype(int)
    groups = engineered["patient_nbr"].astype(str)
    return X, y, groups


def safe_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    return value

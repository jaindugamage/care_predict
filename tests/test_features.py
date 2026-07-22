from __future__ import annotations

import pandas as pd

from src.features import age_midpoint, engineer_features, group_icd9


def test_age_midpoint():
    assert age_midpoint("[70-80)") == 75
    assert pd.isna(age_midpoint("unknown"))


def test_icd9_groups():
    assert group_icd9("250.83") == "Diabetes"
    assert group_icd9("428") == "Circulatory"
    assert group_icd9("585") == "Genitourinary"
    assert group_icd9("V45") == "Supplementary factors"
    assert group_icd9("?") == "Missing"


def test_engineered_utilisation_and_medications():
    frame = pd.DataFrame(
        {
            "age": ["[60-70)"],
            "admission_type_id": [1],
            "admission_source_id": [7],
            "discharge_disposition_id": [1],
            "number_outpatient": [1],
            "number_emergency": [2],
            "number_inpatient": [1],
            "time_in_hospital": [4],
            "num_lab_procedures": [30],
            "num_procedures": [2],
            "num_medications": [12],
            "number_diagnoses": [7],
            "diag_1": ["428"],
            "diag_2": ["585"],
            "diag_3": ["250.2"],
            "insulin": ["Up"],
            "metformin": ["Steady"],
            "change": ["Ch"],
            "diabetesMed": ["Yes"],
            "max_glu_serum": ["None"],
            "A1Cresult": [">8"],
        }
    )
    result = engineer_features(frame).iloc[0]
    assert result["total_prior_visits"] == 4
    assert result["acute_prior_visits"] == 3
    assert result["frequent_utilizer"] == "Yes"
    assert result["cardiovascular_comorbidity"] == "Yes"
    assert result["renal_comorbidity"] == "Yes"
    assert result["medication_changed"] == "Yes"

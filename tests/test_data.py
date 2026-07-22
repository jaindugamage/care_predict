from __future__ import annotations

import pandas as pd

from src.data import clean_cohort


def test_cohort_filters_and_target():
    frame = pd.DataFrame(
        {
            "encounter_id": [1, 2, 3, 4],
            "patient_nbr": [10, 20, 30, 40],
            "readmitted": ["<30", "NO", ">30", "<30"],
            "discharge_disposition_id": [1, 11, 1, 1],
            "admission_type_id": [1, 1, 4, 1],
            "gender": ["Female", "Male", "Female", "Unknown/Invalid"],
        }
    )
    cleaned, summary = clean_cohort(frame)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["target_30d"] == 1
    assert summary["death_or_hospice_removed"] == 1
    assert summary["newborn_rows_removed"] == 1
    assert summary["invalid_gender_rows_removed"] == 1

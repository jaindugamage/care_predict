from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip"
)
RAW_DATA_FILENAME = "diabetic_data.csv"
MAPPING_FILENAME = "IDS_mapping.csv"
RANDOM_STATE = 42

# These discharge outcomes make a conventional readmission follow-up target
# inappropriate because the patient died or entered hospice care.
EXCLUDED_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}
NEWBORN_ADMISSION_TYPE_ID = 4

MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

SENSITIVE_AUDIT_COLUMNS = ["race", "gender", "age"]

INTAKE_NUMERIC_FEATURES = [
    "age_midpoint",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "total_prior_visits",
    "acute_prior_visits",
    "emergency_share",
    "inpatient_share",
]

INTAKE_CATEGORICAL_FEATURES = [
    "admission_urgency",
    "admission_source_group",
    "frequent_utilizer",
]

DISCHARGE_NUMERIC_FEATURES = INTAKE_NUMERIC_FEATURES + [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "diabetes_medication_count",
    "medication_increase_count",
    "medication_decrease_count",
    "medication_stable_count",
    "diagnosis_group_count",
    "comorbidity_count",
]

DISCHARGE_CATEGORICAL_FEATURES = INTAKE_CATEGORICAL_FEATURES + [
    "discharge_group",
    "primary_diagnosis_group",
    "secondary_diagnosis_group",
    "tertiary_diagnosis_group",
    "max_glu_serum",
    "A1Cresult",
    "diabetesMed",
    "insulin_status",
    "medication_changed",
    "cardiovascular_comorbidity",
    "renal_comorbidity",
    "respiratory_comorbidity",
    "mental_health_comorbidity",
]

STAGE_FEATURES = {
    "intake": (INTAKE_NUMERIC_FEATURES, INTAKE_CATEGORICAL_FEATURES),
    "discharge": (DISCHARGE_NUMERIC_FEATURES, DISCHARGE_CATEGORICAL_FEATURES),
}

STAGE_LABELS = {
    "intake": "Intake-proxy assessment",
    "discharge": "Pre-discharge assessment",
}

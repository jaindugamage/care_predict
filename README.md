# CarePredict

CarePredict is a hospital readmission risk prediction project built with Python, scikit-learn and Streamlit.

I built it to estimate whether a hospital encounter involving a patient with diabetes may be followed by another admission within 30 days. The project covers data preparation, feature engineering, model comparison, probability calibration, threshold selection, individual prediction and batch scoring.

**Developed by Jaindu Gamage**

> This project is for learning and demonstration. It is not a medical diagnostic tool and should not be used for patient-care decisions.

## What the project does

- Cleans the Diabetes 130-US Hospitals dataset
- Creates a binary 30-day readmission target
- Groups ICD-9 diagnosis codes into broader categories
- Creates hospital-use, medication and comorbidity features
- Keeps encounters from the same patient in one data partition
- Compares Dummy Classifier, Logistic Regression, Random Forest and XGBoost
- Evaluates recall, precision, F1-score, specificity, ROC-AUC and PR-AUC
- Calibrates predicted probabilities
- Selects a follow-up threshold using recall, capacity and cost
- Predicts risk for one patient or a CSV file of patients
- Shows model importance, patient-level explanations and subgroup checks

## Dataset

The dataset is included in:

```text
data/raw/diabetic_data.csv
data/raw/IDS_mapping.csv
```

Source:

**Diabetes 130-US Hospitals for Years 1999-2008**  
Clore, J., Cios, K., DeShazo, J., and Strack, B.  
UCI Machine Learning Repository  
DOI: `10.24432/C5230J`

The dataset contains 101,766 hospital encounters. It is licensed under the Creative Commons Attribution 4.0 International licence. The MIT licence in `LICENSE` applies only to the project source code.

The original `readmitted` column contains:

| Value | Meaning |
|---|---|
| `<30` | Readmitted within 30 days |
| `>30` | Readmitted after 30 days |
| `NO` | No recorded readmission |

The project converts it to:

```text
1 = Readmitted within 30 days
0 = Readmitted after 30 days or not readmitted
```

## Prediction stages

### Intake-proxy assessment

Uses information available near the beginning of an encounter:

- Age
- Admission type
- Admission source
- Previous outpatient visits
- Previous emergency visits
- Previous inpatient visits
- Total previous hospital visits
- Frequent hospital-use indicator

It is called an intake-proxy assessment because the dataset does not provide an exact timestamp for every field.

### Pre-discharge assessment

Adds information collected during the hospital stay:

- Length of stay
- Laboratory procedures
- Other procedures
- Number of medications
- Number of diagnoses
- Diagnosis groups
- Medication changes
- Insulin status
- HbA1c result
- Discharge destination
- Comorbidity indicators

## Application pages

1. **Overview** — project summary and main statistics
2. **Data Analysis** — cohort preparation, patient patterns and data quality
3. **Model Results** — model comparison, evaluation charts, threshold analysis and subgroup checks
4. **Risk Prediction** — prediction for one patient
5. **Batch Prediction** — CSV upload and downloadable results
6. **About Project** — project design and limitations

## Project structure

```text
CarePredict/
├── app.py
├── prepare_project.py
├── run_mac.command
├── train_full.command
├── run_tests.command
├── requirements.txt
├── README.md
├── data/
│   └── raw/
│       ├── diabetic_data.csv
│       └── IDS_mapping.csv
├── src/
├── scripts/
├── tests/
└── .streamlit/
```

The project creates `artifacts/`, `reports/` and `data/processed/` when the models are trained.

## Run on macOS

### Simple method

1. Open the project folder.
2. Right-click `run_mac.command`.
3. Select **Open**.

The launcher creates the virtual environment, installs the required packages, trains the quick models on the first run and opens the Streamlit application.

### Terminal method

```bash
cd ~/Downloads/CarePredict
./run_mac.command
```

The application normally opens at:

```text
http://localhost:8501
```

## Full training

The first run uses quick mode, which trains on a patient-grouped sample so the application can be tested faster.

To train using the complete cleaned cohort:

```bash
./train_full.command
```

Full training replaces the existing model artifacts and reports.

## Run tests

```bash
./run_tests.command
```

## Validation design

Patients are separated across five partitions:

- Training
- Model selection
- Probability calibration
- Threshold selection
- Final testing

`patient_nbr` is used only to keep the same patient out of multiple partitions. It is not used as a model feature. `encounter_id` and the original `readmitted` column are also excluded from prediction.

## Main limitations

- The data was collected from 1999 to 2008.
- The model has not been validated on a current hospital population.
- The intake model is a proxy because feature timestamps are incomplete.
- The dataset does not provide a hospital identifier for site-level validation.
- Risk levels and explanations describe model behaviour, not medical causes.
- Subgroup results are checks, not proof that the model is fair.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from src.constants import ARTIFACT_DIR, PROCESSED_DIR, REPORT_DIR, STAGE_LABELS
from src.explain import local_sensitivity_explanation, local_shap_explanation
from src.inference import load_artifact, make_patient_row, predict_patient
from src.metrics import binary_metrics, threshold_table


st.set_page_config(
    page_title="CarePredict",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container, [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 3.0rem !important;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: #17212f;
    }
    [data-testid="stSidebar"] * {
        color: #f4f6f8;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0.25rem 0;
    }
    [data-testid="stMetric"] {
        border: 1px solid #d9dee5;
        border-radius: 6px;
        padding: 0.8rem;
        background: #ffffff;
    }
    .project-note {
        border-left: 4px solid #315f85;
        background: #f4f7fa;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .small-text {
        color: #667085;
        font-size: 0.9rem;
    }
    h1, h2, h3 {
        letter-spacing: -0.01em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def artifact_ready() -> bool:
    return all((ARTIFACT_DIR / f"{stage}_model.joblib").exists() for stage in ["intake", "discharge"])


@st.cache_data
def load_manifest() -> dict:
    path = ARTIFACT_DIR / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data
def load_dashboard_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "dashboard_sample.csv.gz"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_resource
def get_artifact(stage: str) -> dict:
    return load_artifact(stage)


def stage_selector(label: str = "Assessment stage") -> str:
    return st.selectbox(label, list(STAGE_LABELS), format_func=STAGE_LABELS.get)


def stage_description(stage: str) -> str:
    if stage == "intake":
        return "Uses age, admission details and previous hospital visits."
    return "Adds information collected during the hospital stay and before discharge."


def show_model_metrics(metrics: dict) -> None:
    columns = st.columns(5)
    values = [
        ("PR-AUC", "pr_auc"),
        ("ROC-AUC", "roc_auc"),
        ("Recall", "recall"),
        ("Precision", "precision"),
        ("Specificity", "specificity"),
    ]
    for column, (label, key) in zip(columns, values):
        column.metric(label, f"{metrics.get(key, 0):.3f}")


def prepare_batch_frame(batch: pd.DataFrame, artifact: dict) -> tuple[pd.DataFrame, list[str]]:
    model_frame = batch.copy()
    missing_columns: list[str] = []

    for column in artifact["feature_columns"]:
        if column not in model_frame.columns:
            model_frame[column] = artifact["feature_defaults"][column]
            missing_columns.append(column)

    model_frame = model_frame[artifact["feature_columns"]].copy()

    for column in artifact["numeric_features"]:
        model_frame[column] = pd.to_numeric(model_frame[column], errors="coerce")

    for column in artifact["categorical_features"]:
        model_frame[column] = (
            model_frame[column]
            .fillna(artifact["feature_defaults"][column])
            .astype(str)
        )

    return model_frame, missing_columns


st.sidebar.title("CarePredict")
st.sidebar.caption("Hospital Readmission Risk System")

page = st.sidebar.radio(
    "Pages",
    [
        "Overview",
        "Data Analysis",
        "Model Results",
        "Risk Prediction",
        "Batch Prediction",
        "About Project",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Developed by Jaindu Gamage")
st.sidebar.caption("Educational project. Not for clinical use.")

st.markdown('<div class="small-text"><strong>CarePredict</strong> &nbsp; Hospital Readmission Risk Prediction</div>', unsafe_allow_html=True)
st.write("")

if not artifact_ready():
    st.error("The trained model files are not available yet.")
    st.write("Run the following command from the project folder:")
    st.code("python prepare_project.py --mode quick", language="bash")
    st.stop()

manifest = load_manifest()
dashboard = load_dashboard_data()
cohort = manifest.get("cohort", {})


if page == "Overview":
    st.header("Project Overview")
    st.markdown(
        """
        <div class="project-note">
        This project estimates the probability that a hospital encounter involving a patient with diabetes
        will be followed by readmission within 30 days. The result is intended for follow-up prioritisation,
        not diagnosis or treatment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(4)
    columns[0].metric("Encounters", f"{cohort.get('model_rows', 0):,}")
    columns[1].metric("Patients", f"{cohort.get('model_unique_patients', 0):,}")
    columns[2].metric(
        "30-day readmission",
        f"{100 * cohort.get('model_readmission_rate', 0):.1f}%",
    )
    columns[3].metric("Training mode", str(manifest.get("training_mode", "Unknown")).title())

    if not dashboard.empty:
        left, right = st.columns(2)

        age_summary = (
            dashboard.groupby("age")["target_30d"]
            .agg(encounters="size", readmission_rate="mean")
            .reset_index()
        )
        left.plotly_chart(
            px.bar(
                age_summary,
                x="age",
                y="encounters",
                title="Patient encounters by age group",
                labels={"age": "Age group", "encounters": "Encounters"},
            ),
            width="stretch",
        )

        urgency_summary = (
            dashboard.groupby("admission_urgency")["target_30d"]
            .agg(encounters="size", readmission_rate="mean")
            .reset_index()
        )
        right.plotly_chart(
            px.bar(
                urgency_summary,
                x="admission_urgency",
                y="readmission_rate",
                hover_data=["encounters"],
                title="Readmission rate by admission type",
                labels={
                    "admission_urgency": "Admission type",
                    "readmission_rate": "Readmission rate",
                },
            ),
            width="stretch",
        )

    st.subheader("How the system works")
    workflow = pd.DataFrame(
        [
            ["1", "Prepare data", "Clean records and create the 30-day readmission target."],
            ["2", "Create features", "Group diagnoses and summarise hospital visits and medications."],
            ["3", "Train models", "Compare baseline, logistic regression, random forest and XGBoost."],
            ["4", "Evaluate", "Use patient-separated data, calibration and threshold analysis."],
            ["5", "Predict", "Estimate risk for one patient or a batch of patients."],
        ],
        columns=["Step", "Stage", "Description"],
    )
    st.dataframe(workflow, width="stretch", hide_index=True)


elif page == "Data Analysis":
    st.header("Data Analysis")

    summary_tab, patterns_tab, quality_tab = st.tabs(["Cohort", "Patient Patterns", "Data Quality"])

    with summary_tab:
        rules = pd.DataFrame(
            [
                ["Raw encounters", cohort.get("raw_rows", 0)],
                ["Exact duplicates removed", cohort.get("exact_duplicates_removed", 0)],
                ["Repeated encounter IDs removed", cohort.get("encounter_duplicates_removed", 0)],
                ["Death or hospice records removed", cohort.get("death_or_hospice_removed", 0)],
                ["Newborn encounters removed", cohort.get("newborn_rows_removed", 0)],
                ["Invalid gender records removed", cohort.get("invalid_gender_rows_removed", 0)],
                ["Final model encounters", cohort.get("model_rows", 0)],
            ],
            columns=["Cohort step", "Rows"],
        )
        st.dataframe(rules, width="stretch", hide_index=True)

        st.subheader("Data split")
        st.write(
            "All encounters belonging to the same patient stay in one partition. "
            "This prevents the model from seeing the same patient during both training and testing."
        )

    with patterns_tab:
        if dashboard.empty:
            st.info("Dashboard sample data is not available.")
        else:
            left, right = st.columns(2)

            diagnosis = (
                dashboard.groupby("primary_diagnosis_group")["target_30d"]
                .agg(encounters="size", readmission_rate="mean")
                .reset_index()
            )
            diagnosis = diagnosis[diagnosis["encounters"] >= 100].sort_values(
                "readmission_rate", ascending=False
            )
            left.plotly_chart(
                px.bar(
                    diagnosis.head(12),
                    x="readmission_rate",
                    y="primary_diagnosis_group",
                    orientation="h",
                    title="Readmission rate by diagnosis group",
                    labels={
                        "primary_diagnosis_group": "Diagnosis group",
                        "readmission_rate": "Readmission rate",
                    },
                ),
                width="stretch",
            )

            visits = dashboard.copy()
            visits["total_previous_visits"] = (
                visits["number_outpatient"]
                + visits["number_emergency"]
                + visits["number_inpatient"]
            )
            visits["outcome"] = np.where(
                visits["target_30d"].eq(1), "Readmitted within 30 days", "Other"
            )
            right.plotly_chart(
                px.box(
                    visits,
                    x="outcome",
                    y="total_previous_visits",
                    points=False,
                    title="Previous hospital visits by outcome",
                    labels={
                        "outcome": "Outcome",
                        "total_previous_visits": "Previous visits",
                    },
                ),
                width="stretch",
            )

            if "time_in_hospital" in dashboard.columns:
                stay = (
                    dashboard.groupby("time_in_hospital")["target_30d"]
                    .agg(encounters="size", readmission_rate="mean")
                    .reset_index()
                )
                st.plotly_chart(
                    px.line(
                        stay,
                        x="time_in_hospital",
                        y="readmission_rate",
                        markers=True,
                        title="Readmission rate by length of stay",
                        labels={
                            "time_in_hospital": "Length of stay (days)",
                            "readmission_rate": "Readmission rate",
                        },
                    ),
                    width="stretch",
                )

    with quality_tab:
        if dashboard.empty:
            st.info("Dashboard sample data is not available.")
        else:
            missing = dashboard.isna().mean().mul(100).sort_values(ascending=False).reset_index()
            missing.columns = ["Feature", "Missing values (%)"]
            st.dataframe(missing.head(20).round(2), width="stretch", hide_index=True)

        st.subheader("Leakage controls")
        st.write(
            "Patient and encounter IDs are not used as predictors. The original readmission field is removed, "
            "and the final test set is not used during model selection, calibration or threshold selection."
        )


elif page == "Model Results":
    st.header("Model Results")
    stage = stage_selector()
    st.caption(stage_description(stage))

    artifact = get_artifact(stage)
    comparison = load_csv(REPORT_DIR / f"{stage}_model_comparison.csv")
    predictions = load_csv(REPORT_DIR / f"{stage}_test_predictions.csv")
    thresholds = load_csv(REPORT_DIR / f"{stage}_thresholds.csv")
    roc_points = load_csv(REPORT_DIR / f"{stage}_roc_curve.csv")
    pr_points = load_csv(REPORT_DIR / f"{stage}_pr_curve.csv")
    importance = load_csv(REPORT_DIR / f"{stage}_feature_importance.csv")
    fairness = load_csv(REPORT_DIR / f"{stage}_fairness.csv")

    st.write(f"Selected model: **{artifact['model_name']}**")
    st.subheader("Final Test Performance")
    st.caption("These values are calculated on the final patient-separated test set.")
    show_model_metrics(artifact["test_metrics"])

    comparison_tab, charts_tab, threshold_tab, explanation_tab = st.tabs(
        ["Comparison", "Charts", "Threshold", "Explanation"]
    )

    with comparison_tab:
        st.subheader("Validation Model Comparison")
        st.caption(
            "These values come from the patient-separated model-selection set, so they can differ from the final test results above."
        )
        if comparison.empty:
            st.info("Model comparison results are not available.")
        else:
            visible = [
                "model",
                "pr_auc",
                "roc_auc",
                "recall",
                "precision",
                "f1",
                "specificity",
                "brier",
                "training_seconds",
            ]
            visible = [column for column in visible if column in comparison.columns]
            comparison_table = comparison[visible].copy().rename(
                columns={
                    "model": "Model",
                    "pr_auc": "PR-AUC",
                    "roc_auc": "ROC-AUC",
                    "recall": "Recall",
                    "precision": "Precision",
                    "f1": "F1",
                    "specificity": "Specificity",
                    "brier": "Brier Score",
                    "training_seconds": "Training Time (seconds)",
                }
            )
            numeric_columns = comparison_table.select_dtypes(include="number").columns
            comparison_table[numeric_columns] = comparison_table[numeric_columns].round(3)
            st.dataframe(comparison_table, width="stretch", hide_index=True)

    with charts_tab:
        left, right = st.columns(2)

        if not roc_points.empty:
            figure = px.line(
                roc_points,
                x="false_positive_rate",
                y="true_positive_rate",
                title="ROC curve",
                labels={
                    "false_positive_rate": "False positive rate",
                    "true_positive_rate": "True positive rate",
                },
            )
            figure.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line_dash="dash")
            left.plotly_chart(figure, width="stretch")

        if not pr_points.empty:
            right.plotly_chart(
                px.line(
                    pr_points,
                    x="recall",
                    y="precision",
                    title="Precision-recall curve",
                ),
                width="stretch",
            )

        if not predictions.empty:
            observed, predicted_probability = calibration_curve(
                predictions["target_30d"],
                predictions["probability"],
                n_bins=10,
                strategy="quantile",
            )
            calibration = pd.DataFrame(
                {"Predicted probability": predicted_probability, "Observed rate": observed}
            )
            calibration_figure = px.line(
                calibration,
                x="Predicted probability",
                y="Observed rate",
                markers=True,
                title="Calibration curve",
            )
            calibration_figure.add_shape(
                type="line", x0=0, y0=0, x1=1, y1=1, line_dash="dash"
            )
            st.plotly_chart(calibration_figure, width="stretch")

    with threshold_tab:
        if predictions.empty:
            st.info("Test predictions are not available.")
        else:
            threshold = st.slider(
                "Decision threshold",
                min_value=0.02,
                max_value=0.80,
                value=float(artifact["threshold"]),
                step=0.01,
            )
            predicted = predictions["probability"].ge(threshold).astype(int)
            dynamic_metrics = binary_metrics(
                predictions["target_30d"], predictions["probability"], threshold
            )

            columns = st.columns(5)
            columns[0].metric("Recall", f"{dynamic_metrics['recall']:.3f}")
            columns[1].metric("Precision", f"{dynamic_metrics['precision']:.3f}")
            columns[2].metric("Specificity", f"{dynamic_metrics['specificity']:.3f}")
            columns[3].metric("F1", f"{dynamic_metrics['f1']:.3f}")
            columns[4].metric(
                "Patients selected", f"{100 * dynamic_metrics['predicted_positive_rate']:.1f}%"
            )

            matrix = confusion_matrix(predictions["target_30d"], predicted, labels=[0, 1])
            matrix_frame = pd.DataFrame(
                matrix,
                index=["Actual: No", "Actual: Yes"],
                columns=["Predicted: No", "Predicted: Yes"],
            )
            st.plotly_chart(
                px.imshow(matrix_frame, text_auto=True, aspect="auto", title="Confusion matrix"),
                width="stretch",
            )

            with st.expander("Follow-up capacity and cost settings"):
                controls = st.columns(3)
                missed_cost = controls[0].number_input(
                    "Cost of a missed readmission", 1.0, 50.0, 5.0, 0.5
                )
                followup_cost = controls[1].number_input(
                    "Cost of an unnecessary follow-up", 0.1, 20.0, 1.0, 0.1
                )
                capacity = controls[2].slider("Maximum patients selected", 1, 50, 20) / 100.0

                table = threshold_table(
                    predictions["target_30d"],
                    predictions["probability"],
                    false_negative_cost=missed_cost,
                    false_positive_cost=followup_cost,
                )
                eligible = table[table["predicted_positive_rate"] <= capacity]
                selected = (eligible if not eligible.empty else table).sort_values("total_cost").iloc[0]

                result_columns = st.columns(4)
                result_columns[0].metric("Suggested threshold", f"{100 * selected['threshold']:.0f}%")
                result_columns[1].metric("Expected recall", f"{100 * selected['recall']:.1f}%")
                result_columns[2].metric("Expected precision", f"{100 * selected['precision']:.1f}%")
                result_columns[3].metric(
                    "Patients selected", f"{100 * selected['predicted_positive_rate']:.1f}%"
                )

            if not thresholds.empty:
                selected_columns = [
                    "threshold",
                    "recall",
                    "precision",
                    "specificity",
                    "predicted_positive_rate",
                ]
                selected_columns = [column for column in selected_columns if column in thresholds.columns]
                st.line_chart(thresholds[selected_columns].set_index("threshold"))

    with explanation_tab:
        if not importance.empty:
            st.subheader("Feature importance")
            st.caption("Importance describes model behaviour and does not prove medical causation.")
            st.plotly_chart(
                px.bar(
                    importance.sort_values("importance"),
                    x="importance",
                    y="feature",
                    orientation="h",
                    labels={"importance": "Importance", "feature": "Feature"},
                ),
                width="stretch",
            )

        st.subheader("Fairness Check")
        if fairness.empty:
            st.info("Subgroup results are not available.")
        else:
            attribute = st.selectbox("Group variable", sorted(fairness["attribute"].unique()))
            subset = fairness[fairness["attribute"].eq(attribute)].copy()
            st.dataframe(subset.round(4), width="stretch", hide_index=True)


elif page == "Risk Prediction":
    st.header("Patient Risk Prediction")
    stage = stage_selector("Prediction stage")
    artifact = get_artifact(stage)
    st.caption(stage_description(stage))

    with st.form("patient_form"):
        first, second, third = st.columns(3)
        age_band = first.selectbox(
            "Age group",
            [
                "[0-10)",
                "[10-20)",
                "[20-30)",
                "[30-40)",
                "[40-50)",
                "[50-60)",
                "[60-70)",
                "[70-80)",
                "[80-90)",
                "[90-100)",
            ],
            index=7,
        )
        age_start = int(age_band.split("-")[0].replace("[", ""))
        age_midpoint = age_start + 5

        urgency = second.selectbox(
            "Admission type",
            ["Emergency", "Urgent", "Elective", "Trauma center", "Unavailable", "Other"],
        )
        source = third.selectbox(
            "Admission source",
            ["Emergency room", "Referral", "Transfer", "Other", "Unavailable"],
        )

        visits = st.columns(3)
        outpatient = visits[0].number_input("Previous outpatient visits", 0, 50, 0)
        emergency = visits[1].number_input("Previous emergency visits", 0, 50, 0)
        inpatient = visits[2].number_input("Previous inpatient visits", 0, 50, 0)

        total_visits = outpatient + emergency + inpatient
        updates: dict[str, object] = {
            "age_midpoint": float(age_midpoint),
            "admission_urgency": urgency,
            "admission_source_group": source,
            "number_outpatient": float(outpatient),
            "number_emergency": float(emergency),
            "number_inpatient": float(inpatient),
            "total_prior_visits": float(total_visits),
            "acute_prior_visits": float(emergency + inpatient),
            "emergency_share": float(emergency / max(1, total_visits)),
            "inpatient_share": float(inpatient / max(1, total_visits)),
            "frequent_utilizer": "Yes" if total_visits >= 3 else "No",
        }

        if stage == "discharge":
            st.subheader("Hospital stay information")
            a, b, c, d = st.columns(4)
            updates["time_in_hospital"] = float(a.number_input("Length of stay", 1, 14, 4))
            updates["num_lab_procedures"] = float(b.number_input("Lab procedures", 0, 150, 40))
            updates["num_procedures"] = float(c.number_input("Other procedures", 0, 20, 1))
            updates["num_medications"] = float(d.number_input("Medications", 0, 100, 15))

            e, f, g = st.columns(3)
            updates["number_diagnoses"] = float(e.number_input("Diagnoses", 1, 20, 7))
            updates["discharge_group"] = f.selectbox(
                "Discharge destination", ["Home", "Post-acute transfer", "Other/unknown"]
            )
            updates["A1Cresult"] = g.selectbox("HbA1c result", ["None", "Norm", ">7", ">8"])

            diagnosis_options = [
                "Circulatory",
                "Diabetes",
                "Endocrine/metabolic",
                "Respiratory",
                "Digestive",
                "Genitourinary",
                "Symptoms",
                "Injury/poisoning",
                "Neoplasms",
                "Mental health",
                "Other",
                "Missing",
            ]
            h, i, j = st.columns(3)
            updates["primary_diagnosis_group"] = h.selectbox(
                "Primary diagnosis", diagnosis_options
            )
            updates["secondary_diagnosis_group"] = i.selectbox(
                "Secondary diagnosis", diagnosis_options, index=1
            )
            updates["tertiary_diagnosis_group"] = j.selectbox(
                "Third diagnosis", diagnosis_options, index=1
            )

            selected_diagnoses = {
                updates["primary_diagnosis_group"],
                updates["secondary_diagnosis_group"],
                updates["tertiary_diagnosis_group"],
            }
            updates["diagnosis_group_count"] = float(len(selected_diagnoses - {"Missing"}))

            comorbidity_map = {
                "cardiovascular_comorbidity": "Circulatory",
                "renal_comorbidity": "Genitourinary",
                "respiratory_comorbidity": "Respiratory",
                "mental_health_comorbidity": "Mental health",
            }
            comorbidity_count = 0
            for key, diagnosis in comorbidity_map.items():
                value = "Yes" if diagnosis in selected_diagnoses else "No"
                updates[key] = value
                comorbidity_count += int(value == "Yes")
            updates["comorbidity_count"] = float(comorbidity_count)

            k, l, m = st.columns(3)
            updates["insulin_status"] = k.selectbox(
                "Insulin status", ["No", "Steady", "Up", "Down"]
            )
            updates["medication_changed"] = l.selectbox("Medication changed", ["No", "Yes"])
            updates["diabetesMed"] = m.selectbox(
                "Diabetes medication", ["Yes", "No"]
            )

        submitted = st.form_submit_button("Calculate risk", width="stretch")

    if submitted:
        row = make_patient_row(artifact, updates)
        result = predict_patient(artifact, row)

        st.subheader("Result")
        result_columns = st.columns(4)
        result_columns[0].metric("Readmission probability", f"{100 * result['probability']:.1f}%")
        result_columns[1].metric("Risk level", result["risk_band"])
        result_columns[2].metric("Risk percentile", f"{result['percentile']:.0f}th")
        result_columns[3].metric("Decision threshold", f"{100 * result['threshold']:.1f}%")

        if result["prediction"]:
            st.warning("The prediction is above the selected follow-up threshold.")
        else:
            st.info("The prediction is below the selected follow-up threshold.")

        with st.expander("View prediction factors"):
            try:
                explanation = local_shap_explanation(
                    artifact["base_pipeline"],
                    row,
                    background=artifact.get("explanation_background"),
                )
                explanation["Effect"] = np.where(
                    explanation["shap_value"] >= 0,
                    "Raises predicted risk",
                    "Lowers predicted risk",
                )
                explanation = explanation.rename(
                    columns={
                        "feature": "Feature",
                        "shap_value": "Contribution",
                        "feature_value": "Value",
                    }
                )
                st.dataframe(explanation, width="stretch", hide_index=True)
            except Exception:
                explanation = local_sensitivity_explanation(
                    artifact["model"], row, artifact["feature_defaults"]
                )
                explanation["Effect"] = np.where(
                    explanation["risk_contribution"] >= 0,
                    "Raises predicted risk",
                    "Lowers predicted risk",
                )
                explanation = explanation.rename(
                    columns={
                        "feature": "Feature",
                        "patient_value": "Patient value",
                        "reference_value": "Reference value",
                        "risk_contribution": "Contribution",
                    }
                )
                st.dataframe(explanation, width="stretch", hide_index=True)


elif page == "Batch Prediction":
    st.header("Batch Prediction")
    stage = stage_selector()
    artifact = get_artifact(stage)

    st.write(
        "Download the template, enter patient records and upload the completed CSV file. "
        "The system will add probability and risk-level columns."
    )

    template = pd.DataFrame([artifact["feature_defaults"]], columns=artifact["feature_columns"])
    st.download_button(
        "Download CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name=f"carepredict_{stage}_template.csv",
        mime="text/csv",
    )

    upload = st.file_uploader("Upload completed CSV", type=["csv"])
    if upload is not None:
        batch = pd.read_csv(upload)

        if batch.empty:
            st.error("The uploaded file does not contain any rows.")
        else:
            model_frame, missing_columns = prepare_batch_frame(batch, artifact)
            probabilities = artifact["model"].predict_proba(model_frame)[:, 1]
            distribution = np.asarray(
                artifact["validation_probability_distribution"], dtype=float
            )

            results = batch.copy()
            results["readmission_probability"] = probabilities
            results["risk_level"] = [
                (
                    "Low"
                    if probability < artifact["risk_band_cuts"]["low"]
                    else "Medium"
                    if probability < artifact["risk_band_cuts"]["medium"]
                    else "High"
                    if probability < artifact["risk_band_cuts"]["high"]
                    else "Very high"
                )
                for probability in probabilities
            ]
            results["above_threshold"] = probabilities >= artifact["threshold"]
            results["risk_percentile"] = [
                100.0 * np.mean(distribution <= probability) for probability in probabilities
            ]
            results = results.sort_values("readmission_probability", ascending=False)

            if missing_columns:
                st.warning(
                    "These missing columns used training reference values: "
                    + ", ".join(missing_columns)
                )

            columns = st.columns(4)
            columns[0].metric("Rows", f"{len(results):,}")
            columns[1].metric(
                "Above threshold", f"{100 * results['above_threshold'].mean():.1f}%"
            )
            columns[2].metric(
                "Average probability", f"{100 * results['readmission_probability'].mean():.1f}%"
            )
            columns[3].metric("Very high risk", f"{results['risk_level'].eq('Very high').sum():,}")

            st.dataframe(results.head(100), width="stretch", hide_index=True)
            st.download_button(
                "Download results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name=f"carepredict_{stage}_results.csv",
                mime="text/csv",
                width="stretch",
            )


elif page == "About Project":
    st.header("About the Project")

    st.subheader("Purpose")
    st.write(
        "CarePredict is a data science project that estimates 30-day hospital readmission risk "
        "for patients with diabetes. It demonstrates data preparation, imbalanced classification, "
        "model evaluation, probability calibration and model explanation in one application."
    )

    st.subheader("Dataset")
    st.write(
        "The project uses the Diabetes 130-US Hospitals dataset. The target is 1 when the original "
        "readmission value is '<30' and 0 for all other outcomes."
    )

    st.subheader("Two prediction stages")
    stage_table = pd.DataFrame(
        [
            [
                "Intake-proxy",
                "Age, admission details and previous hospital visits",
                "Early risk estimate",
            ],
            [
                "Pre-discharge",
                "Adds diagnoses, procedures, medications, lab activity and length of stay",
                "Follow-up planning before discharge",
            ],
        ],
        columns=["Model", "Main information used", "Purpose"],
    )
    st.dataframe(stage_table, width="stretch", hide_index=True)

    st.subheader("Main limitations")
    limitations = manifest.get("limitations", [])
    if limitations:
        for limitation in limitations:
            st.write(f"- {limitation}")
    else:
        st.write(
            "The data is historical, the project has not been externally validated, and the results "
            "must not be used for real patient-care decisions."
        )

    st.subheader("Project files")
    st.code(
        """app.py                 Streamlit dashboard
prepare_project.py     Data preparation and model training
src/                   Data, features, models and evaluation code
artifacts/             Saved trained models
reports/               Evaluation results
data/                   Raw and processed data""",
        language="text",
    )

    st.markdown(
        """
        <div class="project-note">
        This application is an educational project. It is not a medical device and must not replace professional clinical judgement.
        </div>
        """,
        unsafe_allow_html=True,
    )

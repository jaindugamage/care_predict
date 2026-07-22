from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _clean_feature_name(name: str) -> str:
    text = re.sub(r"^(numeric|categorical)__", "", str(name))
    text = text.replace("missingindicator_", "Missing: ")
    return text


def local_sensitivity_explanation(
    calibrated_model,
    row: pd.DataFrame,
    defaults: dict[str, object],
    top_n: int = 10,
) -> pd.DataFrame:
    base_probability = float(calibrated_model.predict_proba(row)[:, 1][0])
    rows = []
    for column in row.columns:
        changed = row.copy()
        changed.loc[:, column] = defaults.get(column, changed.iloc[0][column])
        altered_probability = float(calibrated_model.predict_proba(changed)[:, 1][0])
        rows.append(
            {
                "feature": column,
                "patient_value": row.iloc[0][column],
                "reference_value": defaults.get(column),
                "risk_contribution": base_probability - altered_probability,
            }
        )
    return (
        pd.DataFrame(rows)
        .assign(abs_contribution=lambda frame: frame["risk_contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
        .head(top_n)
        .drop(columns="abs_contribution")
        .reset_index(drop=True)
    )


def local_shap_explanation(
    base_pipeline,
    row: pd.DataFrame,
    background: pd.DataFrame | None = None,
    top_n: int = 12,
) -> pd.DataFrame:
    import shap

    preprocessor = base_pipeline.named_steps["preprocessor"]
    classifier = base_pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(row)
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    if hasattr(transformed, "toarray"):
        transformed_dense = transformed.toarray()
    else:
        transformed_dense = np.asarray(transformed)

    background_frame = background if background is not None and not background.empty else row
    background_transformed = preprocessor.transform(background_frame)
    if hasattr(background_transformed, "toarray"):
        background_dense = background_transformed.toarray()
    else:
        background_dense = np.asarray(background_transformed)

    if hasattr(classifier, "feature_importances_"):
        explainer = shap.TreeExplainer(classifier)
        values = explainer.shap_values(transformed_dense)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        contribution = values[0]
    elif hasattr(classifier, "coef_"):
        explainer = shap.LinearExplainer(classifier, background_dense)
        contribution = np.asarray(explainer(transformed_dense).values)[0]
    else:
        raise TypeError("The selected classifier does not expose a supported SHAP explanation path.")

    frame = pd.DataFrame(
        {
            "feature": [_clean_feature_name(name) for name in names],
            "shap_value": contribution,
            "feature_value": transformed_dense[0],
        }
    )
    frame["abs_shap"] = frame["shap_value"].abs()
    return frame.sort_values("abs_shap", ascending=False).head(top_n).drop(columns="abs_shap")


def global_model_importance(base_pipeline, top_n: int = 25) -> pd.DataFrame:
    preprocessor = base_pipeline.named_steps["preprocessor"]
    classifier = base_pipeline.named_steps["classifier"]
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        values = np.abs(np.asarray(classifier.coef_)[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    frame = pd.DataFrame(
        {"feature": [_clean_feature_name(name) for name in names], "importance": values}
    )
    return frame.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

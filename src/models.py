from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import RANDOM_STATE


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    description: str


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.2,
    )


def build_model_specs(
    numeric_features: list[str],
    categorical_features: list[str],
    positive_weight: float,
    mode: str = "quick",
) -> list[ModelSpec]:
    def pipeline(estimator):
        return Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
                ("classifier", estimator),
            ]
        )

    specs = [
        ModelSpec(
            name="Dummy",
            estimator=pipeline(DummyClassifier(strategy="prior")),
            description="No-skill baseline using the observed class prevalence.",
        ),
        ModelSpec(
            name="Logistic Regression",
            estimator=pipeline(
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    C=0.7,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                )
            ),
            description="Interpretable class-weighted linear baseline.",
        ),
        ModelSpec(
            name="Random Forest",
            estimator=pipeline(
                RandomForestClassifier(
                    n_estimators=140 if mode == "quick" else 500,
                    max_depth=14,
                    min_samples_leaf=8,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                )
            ),
            description="Non-linear ensemble with balanced bootstrap weighting.",
        ),
    ]

    try:
        from xgboost import XGBClassifier

        specs.append(
            ModelSpec(
                name="XGBoost",
                estimator=pipeline(
                    XGBClassifier(
                        n_estimators=220 if mode == "quick" else 650,
                        max_depth=5,
                        learning_rate=0.05,
                        min_child_weight=4,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.1,
                        reg_lambda=1.5,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=max(1.0, float(positive_weight)),
                        n_jobs=4,
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                    )
                ),
                description="Cost-sensitive gradient-boosted decision trees.",
            )
        )
    except Exception:
        pass

    if mode == "full":
        try:
            from imblearn.ensemble import BalancedRandomForestClassifier

            specs.append(
                ModelSpec(
                    name="Balanced Random Forest",
                    estimator=pipeline(
                        BalancedRandomForestClassifier(
                            n_estimators=500,
                            max_depth=14,
                            min_samples_leaf=6,
                            sampling_strategy="all",
                            replacement=True,
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        )
                    ),
                    description="Random forest trained with class-balanced bootstraps.",
                )
            )
        except Exception:
            pass

    return specs


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        classifier = pipeline.named_steps["classifier"]
        count = getattr(classifier, "n_features_in_", 0)
        return [f"feature_{index}" for index in range(int(count))]

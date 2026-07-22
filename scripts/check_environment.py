from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys

EXPECTED = {
    "streamlit": "1.59.2",
    "pandas": "3.0.3",
    "numpy": "2.3.5",
    "scikit-learn": "1.9.0",
    "imbalanced-learn": "0.14.2",
    "xgboost": "3.3.0",
    "shap": "0.52.0",
    "numba": "0.62.1",
    "llvmlite": "0.45.1",
    "plotly": "6.9.0",
    "joblib": "1.5.3",
}

IMPORT_NAMES = {
    "scikit-learn": "sklearn",
    "imbalanced-learn": "imblearn",
}


def main() -> None:
    if not ((3, 12) <= sys.version_info[:2] < (3, 14)):
        raise SystemExit(
            f"Unsupported Python {platform.python_version()}. "
            "CarePredict requires Python 3.12 or 3.13."
        )

    errors: list[str] = []
    warnings: list[str] = []
    for distribution, expected in EXPECTED.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{distribution} is not installed")
            continue
        if actual != expected:
            errors.append(f"{distribution} {actual} is installed; expected {expected}")
            continue

        module_name = IMPORT_NAMES.get(distribution, distribution.replace("-", "_"))
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            message = f"{distribution} could not be imported: {exc}"
            if distribution == "xgboost":
                warnings.append(message)
            else:
                errors.append(message)

    if warnings:
        print("Environment warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print("  CarePredict will continue and skip XGBoost if its native library is unavailable.")

    if errors:
        print("Environment validation failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print(
        "Environment validated: "
        f"Python {platform.python_version()} on {platform.system()} {platform.machine()}."
    )


if __name__ == "__main__":
    main()

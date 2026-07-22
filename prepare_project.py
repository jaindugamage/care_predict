from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.constants import RAW_DATA_FILENAME, RAW_DIR
from src.training import train_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the CarePredict models and reports.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=RAW_DIR / RAW_DATA_FILENAME,
        help="Path to diabetic_data.csv",
    )
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    if not dataset.exists():
        raise SystemExit(
            f"Dataset not found at {dataset}. Place diabetic_data.csv in data/raw/."
        )

    manifest = train_all(dataset_path=dataset, mode=args.mode)
    print(json.dumps(manifest, indent=2))
    print("\nCarePredict is ready. Run: streamlit run app.py")


if __name__ == "__main__":
    main()

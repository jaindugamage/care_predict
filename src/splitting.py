from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _safe_group_split(frame: pd.DataFrame, test_size: float, random_state: int):
    stratify = frame["target"] if frame["target"].nunique() > 1 else None
    try:
        return train_test_split(
            frame,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state,
        )
    except ValueError:
        return train_test_split(frame, test_size=test_size, random_state=random_state)


def grouped_development_split(
    y: pd.Series,
    groups: pd.Series,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Create patient-exclusive train/selection/calibration/threshold/test partitions.

    Approximate proportions are 60% / 15% / 10% / 5% / 10% by patient.
    A patient's group label is positive if any retained encounter has the target.
    """
    frame = pd.DataFrame({"group": groups.astype(str), "target": y.astype(int)})
    group_frame = frame.groupby("group", as_index=False)["target"].max()

    remaining, test = _safe_group_split(group_frame, 0.10, random_state)
    remaining, threshold = _safe_group_split(remaining, 0.05 / 0.90, random_state + 1)
    remaining, calibration = _safe_group_split(remaining, 0.10 / 0.85, random_state + 2)
    train, selection = _safe_group_split(remaining, 0.15 / 0.75, random_state + 3)

    group_values = groups.astype(str)
    partitions = {
        "train": train,
        "selection": selection,
        "calibration": calibration,
        "threshold": threshold,
        "test": test,
    }
    indices = {
        name: np.flatnonzero(group_values.isin(partition["group"]).to_numpy())
        for name, partition in partitions.items()
    }

    group_sets = {name: set(group_values.iloc[index]) for name, index in indices.items()}
    names = list(group_sets)
    for position, first in enumerate(names):
        for second in names[position + 1 :]:
            if group_sets[first] & group_sets[second]:
                raise RuntimeError(f"Patient leakage detected between {first} and {second}.")
    return indices

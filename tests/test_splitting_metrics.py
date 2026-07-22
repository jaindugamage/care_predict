from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import binary_metrics, choose_threshold
from src.splitting import grouped_development_split


def test_group_split_has_no_patient_overlap():
    groups = pd.Series(np.repeat(np.arange(100), 2).astype(str))
    y = pd.Series(np.tile([0, 1], 100))
    split = grouped_development_split(y, groups)
    group_sets = [set(groups.iloc[split[name]]) for name in split]

    for position, first in enumerate(group_sets):
        for second in group_sets[position + 1 :]:
            assert not first & second


def test_metrics_and_threshold():
    y = np.array([0, 0, 0, 1, 1, 1])
    probability = np.array([0.05, 0.2, 0.4, 0.45, 0.7, 0.9])
    metrics = binary_metrics(y, probability, 0.5)
    assert 0 <= metrics["pr_auc"] <= 1
    assert metrics["tp"] == 2
    threshold, table = choose_threshold(y, probability, maximum_intervention_rate=0.8)
    assert 0.02 <= threshold <= 0.8
    assert not table.empty

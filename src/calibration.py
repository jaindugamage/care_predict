from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibratedModel:
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.calibrator = LogisticRegression(solver="lbfgs")

    @staticmethod
    def _logit(probability):
        probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(probability / (1.0 - probability)).reshape(-1, 1)

    def fit_calibrator(self, X_validation, y_validation):
        raw_probability = self.base_estimator.predict_proba(X_validation)[:, 1]
        self.calibrator.fit(self._logit(raw_probability), y_validation)
        return self

    def predict_proba(self, X):
        raw_probability = self.base_estimator.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict_proba(self._logit(raw_probability))[:, 1]
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

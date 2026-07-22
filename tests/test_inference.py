from __future__ import annotations

from src.inference import risk_band


def test_risk_band():
    cuts = {"low": 0.10, "medium": 0.20, "high": 0.35}
    assert risk_band(0.05, cuts) == "Low"
    assert risk_band(0.15, cuts) == "Medium"
    assert risk_band(0.25, cuts) == "High"
    assert risk_band(0.50, cuts) == "Very high"

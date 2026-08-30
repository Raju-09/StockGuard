"""
tests/test_monitoring.py — Unit tests for data drift and PSI telemetry.
"""

import numpy as np

from src.monitoring import compute_psi, run_drift_analysis


def test_compute_psi_identical():
    # Identical distributions should have PSI ~ 0.0
    arr = np.random.normal(loc=10.0, scale=2.0, size=1000)
    psi = compute_psi(arr, arr)
    assert psi < 0.01


def test_compute_psi_shifted():
    # Heavily shifted distribution should have high PSI
    base = np.random.normal(loc=10.0, scale=2.0, size=1000)
    shifted = np.random.normal(loc=25.0, scale=2.0, size=1000)
    psi = compute_psi(base, shifted)
    assert psi > 0.25


def test_run_drift_analysis():
    results = run_drift_analysis()
    assert "baseline_rows" in results
    assert "current_rows" in results
    assert "feature_drift" in results
    assert "stockout_telemetry" in results
    assert results["overall_status"] in ["HEALTHY", "ACTION_REQUIRED", "STABLE"]
    assert "day_of_week" in results["feature_drift"]

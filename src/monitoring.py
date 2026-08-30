"""
src/monitoring.py — Production Data Drift & Inventory Health Monitoring.

Computes:
1. Feature Drift Telemetry (Population Stability Index - PSI & Kolmogorov-Smirnov test).
2. Stockout Rate & Censoring Shift Detection.
3. Prediction Volume & Distribution Metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data import load_observed
from src.features import FEATURE_COLS, build_features

MONITORING_DIR = Path(__file__).resolve().parents[1] / "models"
DRIFT_REPORT_PATH = MONITORING_DIR / "drift_report.json"


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Compute Population Stability Index (PSI) between baseline and production distributions.

    PSI Interpretation:
        PSI < 0.10  : No significant change / drift
        0.10 <= PSI < 0.25 : Moderate shift (warning)
        PSI >= 0.25 : Significant drift (retrain recommended)
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_bounds = np.percentile(expected, percentiles)
    # Ensure strictly monotonic bucket boundaries
    bucket_bounds[0] -= 1e-5
    bucket_bounds[-1] += 1e-5
    for i in range(1, len(bucket_bounds)):
        if bucket_bounds[i] <= bucket_bounds[i - 1]:
            bucket_bounds[i] = bucket_bounds[i - 1] + 1e-4

    exp_counts = np.histogram(expected, bins=bucket_bounds)[0]
    act_counts = np.histogram(actual, bins=bucket_bounds)[0]

    exp_pct = np.maximum(exp_counts / len(expected), epsilon)
    act_pct = np.maximum(act_counts / len(actual), epsilon)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(round(psi_val, 4))


def run_drift_analysis(
    baseline_df: pd.DataFrame | None = None,
    current_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Analyze distribution drift across all features and track stockout rates.

    Splits the observed store log into:
        Baseline: First 70% of dates (training window)
        Current : Last 30% of dates (evaluation window)
    """
    if baseline_df is None or current_df is None:
        df_obs = load_observed()
        df_feat = build_features(df_obs)
        dates = sorted(df_feat["date"].unique())
        split_idx = int(len(dates) * 0.70)
        split_date = dates[split_idx]

        baseline_df = df_feat[df_feat["date"] < split_date]
        current_df = df_feat[df_feat["date"] >= split_date]

    results: dict[str, Any] = {
        "baseline_rows": len(baseline_df),
        "current_rows": len(current_df),
        "feature_drift": {},
        "overall_status": "STABLE",
        "stockout_telemetry": {},
    }

    # 1. Feature-by-Feature Drift Analysis
    high_drift_features = []
    for col in FEATURE_COLS:
        if col not in baseline_df.columns or col not in current_df.columns:
            continue
        base_vals = baseline_df[col].dropna().values.astype(float)
        curr_vals = current_df[col].dropna().values.astype(float)

        psi = compute_psi(base_vals, curr_vals)
        ks_stat, p_val = ks_2samp(base_vals, curr_vals)

        status = "STABLE"
        if psi >= 0.25:
            status = "DRIFT_DETECTED"
            high_drift_features.append(col)
        elif psi >= 0.10:
            status = "MODERATE_SHIFT"

        results["feature_drift"][col] = {
            "psi": psi,
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 4),
            "status": status,
        }

    # 2. Stockout Telemetry
    base_stockouts = float(baseline_df["stockout_flag"].mean() * 100)
    curr_stockouts = float(current_df["stockout_flag"].mean() * 100)

    results["stockout_telemetry"] = {
        "baseline_stockout_pct": round(base_stockouts, 2),
        "current_stockout_pct": round(curr_stockouts, 2),
        "stockout_delta_pct": round(curr_stockouts - base_stockouts, 2),
        "alert": (curr_stockouts - base_stockouts) > 5.0,
    }

    if high_drift_features:
        results["overall_status"] = "ACTION_REQUIRED"
        results["high_drift_features"] = high_drift_features
    else:
        results["overall_status"] = "HEALTHY"

    # Save report
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results

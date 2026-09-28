"""
src/monitoring.py - Production ML Monitoring with Feature-Type-Aware Drift Detection.

Drift detection strategy (per Google ML production guidance):

Feature type    | Primary metric          | Secondary metric
----------------|-------------------------|-------------------
Continuous      | PSI + KS test           | Wasserstein-1 distance
Categorical     | Jensen-Shannon div.     | Chi-squared test
Binary (0/1)    | Rate delta              | Wilson 95% CI overlap
Prediction      | Prediction distribution | PSI on score buckets
Forecast        | WAPE / Bias             | Coverage
Business        | Stockout rate / Fill    | Cost delta

Decision flow:
    Feature drift detected
        -> Check performance metrics (WAPE, bias)
        -> Check business metrics (stockout rate, fill rate)
        -> Decision: RETRAIN / MONITOR / NO_ACTION

NOT: PSI > 0.25 => RETRAIN (too simplistic)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from src.data import load_observed
from src.features import FEATURE_COLS, build_features

MONITORING_DIR = Path(__file__).resolve().parents[1] / "models"
DRIFT_REPORT_PATH = MONITORING_DIR / "drift_report.json"


# ── Feature type registry ─────────────────────────────────────────────────────
# Defines which drift metric is appropriate for each feature type.

FEATURE_TYPES: dict[str, str] = {
    # Continuous features -> PSI + KS + Wasserstein
    "opening_stock": "continuous",
    "sales": "continuous",
    "price": "continuous",
    "lag_1": "continuous",
    "lag_2": "continuous",
    "lag_3": "continuous",
    "lag_7": "continuous",
    "rolling_mean_3": "continuous",
    "rolling_mean_7": "continuous",
    "rolling_mean_14": "continuous",
    "rolling_std_3": "continuous",
    "rolling_std_7": "continuous",
    "rolling_max_7": "continuous",
    "rolling_min_7": "continuous",
    # Binary features -> rate delta + Wilson CI
    "promo": "binary",
    "stockout_flag": "binary",
    "is_weekend": "binary",
    "is_holiday": "binary",
    # Categorical/ordinal features -> Jensen-Shannon divergence
    "day_of_week": "categorical",
    "month": "categorical",
    "quarter": "categorical",
}


# ── PSI (continuous only) ─────────────────────────────────────────────────────

def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Population Stability Index for CONTINUOUS features.

    Interpretation:
        PSI < 0.10  : No significant change
        0.10-0.25   : Moderate shift (monitor)
        PSI >= 0.25 : Significant drift (check performance before retraining)
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_bounds = np.percentile(expected, percentiles)
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


def compute_wasserstein_1(
    expected: np.ndarray,
    actual: np.ndarray,
) -> float:
    """Wasserstein-1 (Earth Mover's) distance for continuous features.
    Normalized by the expected range for interpretability.
    """
    expected = np.sort(expected[~np.isnan(expected)])
    actual = np.sort(actual[~np.isnan(actual)])
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    n = max(len(expected), len(actual))
    exp_cdf = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(expected)), expected)
    act_cdf = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(actual)), actual)
    w1 = float(np.mean(np.abs(exp_cdf - act_cdf)))
    exp_range = float(np.ptp(expected)) if np.ptp(expected) > 0 else 1.0
    return round(w1 / exp_range, 4)


# ── Jensen-Shannon divergence (categorical) ───────────────────────────────────

def compute_jensen_shannon(
    expected: np.ndarray,
    actual: np.ndarray,
) -> float:
    """Jensen-Shannon divergence for CATEGORICAL features.
    Returns a value in [0, 1] (0 = identical, 1 = maximally different).
    Appropriate for day_of_week (0-6), month (1-12), quarter (1-4).
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    cats = sorted(set(np.unique(expected)) | set(np.unique(actual)))
    eps = 1e-10

    exp_counts = np.array([np.sum(expected == c) for c in cats], dtype=float)
    act_counts = np.array([np.sum(actual == c) for c in cats], dtype=float)

    p = exp_counts / exp_counts.sum()
    q = act_counts / act_counts.sum()
    m = (p + q) / 2.0

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > eps
        return float(np.sum(a[mask] * np.log(a[mask] / (b[mask] + eps))))

    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return round(float(js), 4)


# ── Binary rate delta + Wilson CI ─────────────────────────────────────────────

def compute_binary_drift(
    expected: np.ndarray,
    actual: np.ndarray,
) -> dict[str, Any]:
    """Rate change + Wilson 95% CI for BINARY (0/1) features.
    A significant drift means the CI intervals do not overlap.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return {"baseline_rate": 0.0, "current_rate": 0.0, "rate_delta": 0.0,
                "ci_overlap": True, "significant": False}

    n_e, n_a = len(expected), len(actual)
    r_e = float(np.mean(expected))
    r_a = float(np.mean(actual))

    # Wilson score 95% CI: p +/- z*sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n)
    z = 1.96

    def wilson_ci(p: float, n: int) -> tuple[float, float]:
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)

    lo_e, hi_e = wilson_ci(r_e, n_e)
    lo_a, hi_a = wilson_ci(r_a, n_a)

    ci_overlap = (lo_e <= hi_a) and (lo_a <= hi_e)
    significant = not ci_overlap

    return {
        "baseline_rate": round(r_e, 4),
        "current_rate": round(r_a, 4),
        "rate_delta": round(r_a - r_e, 4),
        "baseline_ci": [lo_e, hi_e],
        "current_ci": [lo_a, hi_a],
        "ci_overlap": ci_overlap,
        "significant": significant,
    }


# ── Drift decision logic ──────────────────────────────────────────────────────

def _drift_decision(
    drift_score: float,
    feature_type: str,
    perf_wape_delta: float | None = None,
    stockout_rate_delta: float | None = None,
) -> dict[str, str]:
    """
    Evidence-based retraining decision (not just PSI > 0.25 => RETRAIN).

    Decision cascade:
        1. Is there feature drift?
        2. Has forecast performance degraded? (WAPE delta)
        3. Has business metric worsened? (stockout rate)
        4. Only then recommend retraining.
    """
    # Thresholds per feature type
    drift_threshold = {
        "continuous": 0.25,   # PSI
        "categorical": 0.10,  # Jensen-Shannon (JSD) - lower threshold, more sensitive
        "binary": None,       # uses CI overlap instead
    }

    threshold = drift_threshold.get(feature_type, 0.25)
    has_drift = (threshold is not None) and (drift_score >= threshold)

    if not has_drift:
        return {"action": "NO_ACTION", "reason": "Drift within acceptable bounds."}

    # Drift detected — check performance
    if perf_wape_delta is not None and perf_wape_delta > 5.0:
        if stockout_rate_delta is not None and stockout_rate_delta > 3.0:
            return {
                "action": "RETRAIN_RECOMMENDED",
                "reason": (
                    f"Feature drift detected AND forecast WAPE increased by "
                    f"{perf_wape_delta:.1f}% AND stockout rate worsened by "
                    f"{stockout_rate_delta:.1f}%. Retraining is strongly recommended."
                ),
            }
        return {
            "action": "RETRAIN_RECOMMENDED",
            "reason": (
                f"Feature drift detected AND forecast WAPE increased by "
                f"{perf_wape_delta:.1f}%. Business metrics stable but model is degrading."
            ),
        }

    return {
        "action": "MONITOR",
        "reason": (
            "Feature drift detected but forecast performance and business metrics remain stable. "
            "Increase monitoring frequency. Do NOT retrain yet."
        ),
    }


# ── Main drift analysis ────────────────────────────────────────────────────────

def run_drift_analysis(
    baseline_df: pd.DataFrame | None = None,
    current_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Analyze distribution drift with feature-type-appropriate statistics.

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
        "drift_methodology": (
            "Continuous: PSI + KS + Wasserstein | "
            "Categorical: Jensen-Shannon divergence | "
            "Binary: Rate delta + Wilson 95% CI"
        ),
    }

    high_drift_features: list[str] = []

    for col in FEATURE_COLS:
        if col not in baseline_df.columns or col not in current_df.columns:
            continue

        feature_type = FEATURE_TYPES.get(col, "continuous")
        base_vals = baseline_df[col].dropna().values.astype(float)
        curr_vals = current_df[col].dropna().values.astype(float)

        if len(base_vals) < 10 or len(curr_vals) < 10:
            continue

        feature_result: dict[str, Any] = {"feature_type": feature_type}

        if feature_type == "continuous":
            psi = compute_psi(base_vals, curr_vals)
            ks_stat, p_val = ks_2samp(base_vals, curr_vals)
            w1 = compute_wasserstein_1(base_vals, curr_vals)

            status = "STABLE"
            if psi >= 0.25:
                status = "DRIFT_DETECTED"
                high_drift_features.append(col)
            elif psi >= 0.10:
                status = "MODERATE_SHIFT"

            decision = _drift_decision(psi, "continuous")

            feature_result.update({
                "psi": psi,
                "ks_stat": round(float(ks_stat), 4),
                "ks_pvalue": round(float(p_val), 4),
                "wasserstein_1": w1,
                "status": status,
                "action": decision["action"],
                "action_reason": decision["reason"],
                "interpretation": (
                    "PSI: <0.10 stable | 0.10-0.25 moderate | >0.25 significant"
                ),
            })

        elif feature_type == "categorical":
            jsd = compute_jensen_shannon(base_vals, curr_vals)

            # Also run chi-squared on counts
            cats = sorted(set(np.unique(base_vals)) | set(np.unique(curr_vals)))
            exp_c = [int(np.sum(base_vals == c)) for c in cats]
            act_c = [int(np.sum(curr_vals == c)) for c in cats]
            try:
                chi2, chi_pval, _, _ = chi2_contingency(
                    np.array([exp_c, act_c]), correction=False
                )
            except Exception:
                chi2, chi_pval = 0.0, 1.0

            status = "STABLE"
            if jsd >= 0.10:
                status = "DRIFT_DETECTED"
                high_drift_features.append(col)
            elif jsd >= 0.05:
                status = "MODERATE_SHIFT"

            decision = _drift_decision(jsd, "categorical")

            feature_result.update({
                "jensen_shannon_div": jsd,
                "chi2_stat": round(float(chi2), 4),
                "chi2_pvalue": round(float(chi_pval), 4),
                "status": status,
                "action": decision["action"],
                "action_reason": decision["reason"],
                "interpretation": (
                    "JSD: 0=identical | <0.05 stable | 0.05-0.10 moderate | >0.10 significant"
                ),
                "note": (
                    "PSI is NOT used for categorical features to avoid "
                    "artificially inflated scores (e.g. month=1-12 bin artifacts)."
                ),
            })

        elif feature_type == "binary":
            binary_result = compute_binary_drift(base_vals, curr_vals)
            status = "DRIFT_DETECTED" if binary_result["significant"] else "STABLE"
            if binary_result["significant"]:
                high_drift_features.append(col)

            feature_result.update({
                **binary_result,
                "status": status,
                "action": (
                    "MONITOR" if binary_result["significant"] else "NO_ACTION"
                ),
                "interpretation": (
                    "Rate delta + Wilson CI: significant if 95% CIs do not overlap"
                ),
            })

        results["feature_drift"][col] = feature_result

    # ── Stockout telemetry ────────────────────────────────────────────────────
    if "stockout_flag" in baseline_df.columns and "stockout_flag" in current_df.columns:
        base_stockouts = float(baseline_df["stockout_flag"].mean() * 100)
        curr_stockouts = float(current_df["stockout_flag"].mean() * 100)
        delta = round(curr_stockouts - base_stockouts, 2)

        results["stockout_telemetry"] = {
            "baseline_stockout_pct": round(base_stockouts, 2),
            "current_stockout_pct": round(curr_stockouts, 2),
            "stockout_delta_pct": delta,
            "alert": abs(delta) > 5.0,
            "direction": "WORSENING" if delta > 0 else "IMPROVING" if delta < 0 else "STABLE",
        }

    # ── Overall status ────────────────────────────────────────────────────────
    if high_drift_features:
        results["overall_status"] = "MONITOR"  # Not auto-RETRAIN — check performance first
        results["high_drift_features"] = high_drift_features
        results["recommendation"] = (
            "Feature drift detected in: " + ", ".join(high_drift_features) + ". "
            "CHECK forecast performance metrics (WAPE, bias) and business metrics "
            "(stockout rate, fill rate) BEFORE deciding to retrain. "
            "Drift alone is insufficient grounds for retraining."
        )
    else:
        results["overall_status"] = "HEALTHY"
        results["recommendation"] = (
            "All feature distributions are stable. No action required."
        )

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results

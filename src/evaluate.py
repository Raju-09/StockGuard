"""
src/evaluate.py — Evaluation metrics for demand forecasting and stockout detection.

All metrics are computed on a temporal hold-out set (never on training data).

Key split for the research question:
    Every metric is reported for three subsets:
        - all rows
        - censored rows   (is_censored == 1 — requires ground truth)
        - uncensored rows (is_censored == 0)

    The improvement on the CENSORED subset is the research finding:
    does the stockout-aware model actually recover demand better on
    the specific days where inventory was the binding constraint?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

# ── Regression metrics ────────────────────────────────────────────────────────

def mae(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> float:
    """Mean Absolute Error.

    Returns NaN if input is empty.
    Unit: same as target (e.g. units/day).
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)))


def rmse(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> float:
    """Root Mean Squared Error.

    Returns NaN if input is empty.
    Penalises large errors more than MAE.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def wape(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> float:
    """Weighted Absolute Percentage Error.

    WAPE = sum(|actual - predicted|) / sum(actual)

    Unlike MAPE, WAPE is safe when actuals contain zeros (e.g. stockout days
    where sales = 0). A lower value is better. Returns NaN if sum(actual) == 0.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0:
        return float("nan")
    denom = np.sum(actual)
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(actual - predicted)) / denom)


def bias(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> float:
    """Mean Forecast Bias = mean(predicted - actual).

    Positive → systematic over-forecast.
    Negative → systematic under-forecast.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) == 0:
        return float("nan")
    return float(np.mean(predicted - actual))


# ── Classification metrics (stockout classifier) ──────────────────────────────

def stockout_classifier_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    """Binary classification metrics for the stockout predictor.

    Parameters
    ----------
    y_true:
        True binary labels (0 = no stockout, 1 = stockout).
    y_pred:
        Predicted binary labels.
    y_prob:
        Predicted probabilities for class 1 (used for PR-AUC).

    Returns
    -------
    dict with keys: precision, recall, f1, pr_auc (if y_prob given).

    Note: accuracy is intentionally excluded. If 95 % of days have no stockout,
    a model that always predicts 0 gets 95 % accuracy while detecting nothing.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return metrics


# ── Full evaluation report ─────────────────────────────────────────────────────

def evaluation_report(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    censored_col: str | None = "is_censored",
) -> dict[str, float | int]:
    """Compute MAE / RMSE / WAPE / Bias for all / censored / uncensored rows.

    Parameters
    ----------
    df:
        DataFrame containing the actual and predicted columns.
    actual_col:
        Column of ground-truth values (e.g. ``"true_demand"`` or ``"sales"``).
    predicted_col:
        Column of model predictions.
    censored_col:
        Column with 0/1 flag for censored rows. Pass ``None`` to skip the split.

    Returns
    -------
    dict
        Keys are metric names with suffixes ``_all``, ``_censored``,
        ``_uncensored``. Also includes row counts.

    Example return
    --------------
    {
        "n_all": 240,
        "n_censored": 62,
        "n_uncensored": 178,
        "mae_all": 4.1,
        "mae_censored": 7.8,   ← key research number
        "mae_uncensored": 2.9,
        ...
    }
    """
    actual = df[actual_col].values
    predicted = df[predicted_col].values

    result: dict[str, float | int] = {
        "n_all": len(df),
        "mae_all": mae(actual, predicted),
        "rmse_all": rmse(actual, predicted),
        "wape_all": wape(actual, predicted),
        "bias_all": bias(actual, predicted),
    }

    if censored_col is not None and censored_col in df.columns:
        cens_mask = df[censored_col] == 1
        uncens_mask = ~cens_mask

        result["n_censored"] = int(cens_mask.sum())
        result["n_uncensored"] = int(uncens_mask.sum())

        if cens_mask.sum() > 0:
            result["mae_censored"] = mae(actual[cens_mask], predicted[cens_mask])
            result["rmse_censored"] = rmse(actual[cens_mask], predicted[cens_mask])
            result["wape_censored"] = wape(actual[cens_mask], predicted[cens_mask])
            result["bias_censored"] = bias(actual[cens_mask], predicted[cens_mask])
        else:
            result["mae_censored"] = float("nan")
            result["rmse_censored"] = float("nan")
            result["wape_censored"] = float("nan")
            result["bias_censored"] = float("nan")

        if uncens_mask.sum() > 0:
            result["mae_uncensored"] = mae(actual[uncens_mask], predicted[uncens_mask])
            result["rmse_uncensored"] = rmse(actual[uncens_mask], predicted[uncens_mask])
            result["wape_uncensored"] = wape(actual[uncens_mask], predicted[uncens_mask])
            result["bias_uncensored"] = bias(actual[uncens_mask], predicted[uncens_mask])
        else:
            result["mae_uncensored"] = float("nan")
            result["rmse_uncensored"] = float("nan")
            result["wape_uncensored"] = float("nan")
            result["bias_uncensored"] = float("nan")

    return result


def format_report(report: dict[str, float | int]) -> str:
    """Return a readable multi-line string from an evaluation_report dict."""
    lines = []
    n_all = report.get("n_all", "?")
    n_cens = report.get("n_censored", "?")
    n_uncens = report.get("n_uncensored", "?")
    lines.append(f"  Rows: {n_all} total | {n_cens} censored | {n_uncens} uncensored")
    lines.append(f"  {'Metric':<12} {'All':>8} {'Censored':>10} {'Uncensored':>12}")
    lines.append(f"  {'-'*44}")
    for metric in ("mae", "rmse", "wape", "bias"):
        v_all = report.get(f"{metric}_all", float("nan"))
        v_cens = report.get(f"{metric}_censored", float("nan"))
        v_uncens = report.get(f"{metric}_uncensored", float("nan"))
        lines.append(
            f"  {metric.upper():<12} {v_all:>8.3f} {v_cens:>10.3f} {v_uncens:>12.3f}"
        )
    return "\n".join(lines)


def compare_models(
    reports: dict[str, dict[str, float | int]],
    primary_metric: str = "mae_all",
) -> pd.DataFrame:
    """Build a comparison DataFrame from multiple evaluation reports.

    Parameters
    ----------
    reports:
        {model_name: evaluation_report_dict}
    primary_metric:
        The metric used to sort (ascending, lower is better).

    Returns
    -------
    pd.DataFrame sorted by primary_metric.
    """
    rows = []
    for model_name, rep in reports.items():
        row = {"model": model_name}
        row.update(rep)
        rows.append(row)
    df = pd.DataFrame(rows)
    if primary_metric in df.columns:
        df = df.sort_values(primary_metric).reset_index(drop=True)
    return df

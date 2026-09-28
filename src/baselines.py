"""
src/baselines.py — Statistical & Intermittent Demand Baselines for Retail.

Implements academic benchmark baselines to anchor evaluation:
- Naive: Forecast = Last observed value
- Seasonal Naive: Forecast = Value from m steps ago (e.g., m=7 for weekly seasonality)
- Moving Average: Forecast = Rolling mean over window w
- Croston's Method & SBA (Syntetos-Boylan Approximation):
  Separately estimates demand magnitude and inter-arrival time for intermittent demand.
"""

from typing import Any
import numpy as np
import pandas as pd


class NaiveForecaster:
    """Naïve baseline: Tomorrow's demand equals today's observed sales."""

    def __init__(self) -> None:
        self.last_value_: float = 0.0

    def fit(self, y: np.ndarray | pd.Series) -> "NaiveForecaster":
        arr = np.asarray(y, dtype=float)
        self.last_value_ = float(arr[-1]) if len(arr) > 0 else 0.0
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        return np.full(horizon, self.last_value_, dtype=float)


class SeasonalNaiveForecaster:
    """Seasonal Naïve: Tomorrow's demand equals the observed sales m days ago."""

    def __init__(self, seasonality: int = 7) -> None:
        self.seasonality = seasonality
        self.history_: np.ndarray = np.array([])

    def fit(self, y: np.ndarray | pd.Series) -> "SeasonalNaiveForecaster":
        arr = np.asarray(y, dtype=float)
        self.history_ = arr
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        if len(self.history_) == 0:
            return np.zeros(horizon, dtype=float)
        preds = []
        hist_len = len(self.history_)
        for h in range(horizon):
            idx = hist_len - self.seasonality + (h % self.seasonality)
            val = self.history_[idx] if 0 <= idx < hist_len else self.history_[-1]
            preds.append(float(val))
        return np.array(preds, dtype=float)


class MovingAverageForecaster:
    """Moving Average baseline over window w."""

    def __init__(self, window: int = 7) -> None:
        self.window = window
        self.mean_value_: float = 0.0

    def fit(self, y: np.ndarray | pd.Series) -> "MovingAverageForecaster":
        arr = np.asarray(y, dtype=float)
        if len(arr) == 0:
            self.mean_value_ = 0.0
        else:
            recent = arr[-self.window :] if len(arr) >= self.window else arr
            self.mean_value_ = float(np.mean(recent))
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        return np.full(horizon, self.mean_value_, dtype=float)


class CrostonSBAForecaster:
    """
    Croston's Method with Syntetos-Boylan Approximation (SBA) for intermittent demand.

    Standard Croston decomposes demand into:
    - Non-zero demand size z_t
    - Inter-arrival interval p_t

    Forecast per period = z_t / p_t.
    Croston has an inherent positive bias. Syntetos & Boylan (2005) proved that multiplying
    by (1 - alpha / 2) produces an approximately unbiased estimate.
    """

    def __init__(self, alpha: float = 0.1, method: str = "sba") -> None:
        self.alpha = alpha
        self.method = method.lower()
        self.demand_level_: float = 0.0
        self.period_level_: float = 1.0

    def fit(self, y: np.ndarray | pd.Series) -> "CrostonSBAForecaster":
        arr = np.asarray(y, dtype=float)
        if len(arr) == 0:
            self.demand_level_ = 0.0
            self.period_level_ = 1.0
            return self

        # Initialize with first non-zero demand
        non_zeros = np.nonzero(arr)[0]
        if len(non_zeros) == 0:
            self.demand_level_ = 0.0
            self.period_level_ = 1.0
            return self

        first_nz = non_zeros[0]
        z = float(arr[first_nz])
        p = float(first_nz + 1)
        q = 1  # periods since last positive demand

        for t in range(first_nz + 1, len(arr)):
            if arr[t] > 0:
                z = z + self.alpha * (arr[t] - z)
                p = p + self.alpha * (q - p)
                q = 1
            else:
                q += 1

        self.demand_level_ = max(0.0, z)
        self.period_level_ = max(1.0, p)
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        if self.period_level_ <= 0:
            rate = 0.0
        else:
            rate = self.demand_level_ / self.period_level_

        # Apply Syntetos-Boylan correction factor
        if self.method == "sba":
            rate *= (1.0 - self.alpha / 2.0)

        return np.full(horizon, max(0.0, rate), dtype=float)


def compute_intermittent_profile(series: np.ndarray | pd.Series) -> dict[str, Any]:
    """
    Profile whether a demand series is intermittent, lumpy, smooth, or erratic
    according to the Kostenko & Hyndman (ADI / CV^2) framework.
    - ADI (Average Demand Interval): Average periods between non-zero demands.
    - CV^2 (Coefficient of Variation squared): Variance / Mean^2 of non-zero demands.
    Threshold: ADI >= 1.32 or zero_rate >= 0.5 indicates intermittent demand.
    """
    arr = np.asarray(series, dtype=float)
    if len(arr) == 0:
        return {"zero_rate": 0.0, "adi": 1.0, "cv2": 0.0, "classification": "smooth"}

    zero_count = int(np.sum(arr == 0))
    zero_rate = zero_count / len(arr)

    non_zero = arr[arr > 0]
    if len(non_zero) <= 1:
        adi = float(len(arr))
        cv2 = 0.0
    else:
        # Non-zero indices intervals
        nz_indices = np.nonzero(arr)[0]
        intervals = np.diff(nz_indices)
        adi = float(np.mean(intervals)) if len(intervals) > 0 else 1.0
        mean_nz = float(np.mean(non_zero))
        std_nz = float(np.std(non_zero, ddof=1)) if len(non_zero) > 1 else 0.0
        cv2 = float((std_nz / mean_nz) ** 2) if mean_nz > 0 else 0.0

    if adi < 1.32 and cv2 < 0.49:
        classification = "smooth"
    elif adi >= 1.32 and cv2 < 0.49:
        classification = "intermittent"
    elif adi < 1.32 and cv2 >= 0.49:
        classification = "erratic"
    else:
        classification = "lumpy"

    return {
        "zero_rate": round(zero_rate, 4),
        "adi": round(adi, 2),
        "cv2": round(cv2, 4),
        "classification": classification,
        "is_intermittent": bool(zero_rate >= 0.4 or adi >= 1.32),
    }


def evaluate_baselines_on_sku(
    series: np.ndarray | pd.Series,
    test_horizon: int = 7,
) -> dict[str, dict[str, float]]:
    """
    Fit each baseline on history[:-test_horizon] and evaluate on history[-test_horizon:].
    Returns WAPE, MAE, RMSE, and Bias per baseline.
    """
    arr = np.asarray(series, dtype=float)
    if len(arr) <= test_horizon:
        raise ValueError("Series too short for test horizon")

    train, test = arr[:-test_horizon], arr[-test_horizon:]
    denom = float(np.sum(test))

    baselines: dict[str, Any] = {
        "naive": NaiveForecaster().fit(train),
        "seasonal_naive_7d": SeasonalNaiveForecaster(seasonality=7).fit(train),
        "moving_average_7d": MovingAverageForecaster(window=7).fit(train),
        "croston_sba": CrostonSBAForecaster(alpha=0.1, method="sba").fit(train),
    }

    results = {}
    for name, model in baselines.items():
        preds = model.predict(test_horizon)
        errors = test - preds
        abs_errors = np.abs(errors)

        mae_val = float(np.mean(abs_errors))
        rmse_val = float(np.sqrt(np.mean(errors ** 2)))
        bias_val = float(np.mean(errors))
        wape_val = float(np.sum(abs_errors) / denom) if denom > 0 else float("nan")

        results[name] = {
            "wape": round(wape_val, 4),
            "mae": round(mae_val, 3),
            "rmse": round(rmse_val, 3),
            "bias": round(bias_val, 3),
        }

    return results

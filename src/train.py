"""
src/train.py — All forecasting models, temporal split, and model persistence.

Models implemented:
    NaiveForecast           — predict yesterday's sales (per product)
    MovingAverageForecast   — predict rolling mean of last N days
    LinearRegressionModel   — sklearn LinearRegression on feature matrix
    RandomForestModel       — sklearn RandomForestRegressor
    GradientBoostingModel   — XGBRegressor
    CensoringAwareModel     — two-stage imputation for stockout-censored demand
    StockoutClassifier      — predict P(stockout within 3 days)

TEMPORAL SPLIT MANDATE:
    temporal_split() splits by date (chronological), never randomly.
    Future must remain future. Never call train_test_split(shuffle=True)
    on a time-series problem.

MODEL VERSIONING:
    Models are saved to models/<name>_v<N>.joblib.
    A model_registry.json records metadata for each saved version.
    New saves increment version; older saves are preserved.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, XGBRegressor

from src.features import (
    FEATURE_COLS,
    TARGET_COL,
    get_X_y,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = DEFAULT_MODELS_DIR / "model_registry.json"

# ── Temporal split ─────────────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a product-day DataFrame chronologically.

    All products share the same date boundaries — this simulates the real-world
    scenario where you retrain on all historical data up to a cut-off date.

    Parameters
    ----------
    df:
        Must contain a ``date`` column (datetime or str).
    train_frac:
        Fraction of unique dates → train.
    val_frac:
        Fraction → validation. Test gets the remainder.

    Returns
    -------
    train, val, test : three DataFrames with no date overlap.

    Raises
    ------
    ValueError
        If fractions are invalid or df has fewer than 3 unique dates.
    """
    if not (0 < train_frac < 1 and 0 < val_frac < 1):
        raise ValueError("train_frac and val_frac must be between 0 and 1.")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1 (test set needs room).")

    dates = sorted(df["date"].unique())
    n = len(dates)
    if n < 3:
        raise ValueError(f"Need at least 3 unique dates, got {n}.")

    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))

    train_dates = set(dates[:n_train])
    val_dates = set(dates[n_train: n_train + n_val])
    test_dates = set(dates[n_train + n_val:])

    if not test_dates:
        raise ValueError(
            "Test set is empty after split. Reduce train_frac or val_frac."
        )

    train = df[df["date"].isin(train_dates)].copy().reset_index(drop=True)
    val = df[df["date"].isin(val_dates)].copy().reset_index(drop=True)
    test = df[df["date"].isin(test_dates)].copy().reset_index(drop=True)

    return train, val, test


# ── Naive forecasting models ───────────────────────────────────────────────────

class NaiveForecast:
    """Predict next day's demand = today's sales (per product).

    Baseline. If this beats your ML model, something is wrong.
    """

    name = "naive"

    def __init__(self) -> None:
        self._last_sales: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> NaiveForecast:
        """Learn last observed sales per product from df."""
        df = df.sort_values(["product", "date"])
        self._last_sales = df.groupby("product")["sales"].last().to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return yesterday's sales as the forecast for each row in df."""
        df = df.sort_values(["product", "date"]).copy()
        # lag_1 already computed by build_features; fall back to stored value
        if "lag_1" in df.columns:
            preds = df["lag_1"].fillna(df["product"].map(self._last_sales)).values
        else:
            preds = df["product"].map(self._last_sales).values
        return np.asarray(preds, dtype=float)


class MovingAverageForecast:
    """Predict next day's demand = rolling mean of last N days' sales (per product)."""

    def __init__(self, window: int = 7) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.name = f"ma{window}"
        self._history: dict[str, list[float]] = {}

    def fit(self, df: pd.DataFrame) -> MovingAverageForecast:
        """Store last `window` sales observations per product."""
        df = df.sort_values(["product", "date"])
        self._history = (
            df.groupby("product")["sales"]
            .apply(lambda x: x.tolist()[-self.window:])
            .to_dict()
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return moving-average forecast.

        Uses ``rolling_mean_{window}`` if pre-computed by build_features,
        otherwise computes from stored history.
        """
        col = f"rolling_mean_{self.window}"
        if col in df.columns:
            preds = df[col].values
        else:
            fallback = {
                prod: float(np.mean(hist)) if hist else 0.0
                for prod, hist in self._history.items()
            }
            preds = df["product"].map(fallback).values
        return np.asarray(preds, dtype=float)


# ── ML demand models ───────────────────────────────────────────────────────────

class _SKLearnDemandModel:
    """Thin wrapper around a sklearn estimator for demand forecasting."""

    def __init__(
        self,
        estimator: Any,
        name: str,
        feature_cols: list[str] | None = None,
    ) -> None:
        self.estimator = estimator
        self.name = name
        self.feature_cols = feature_cols or FEATURE_COLS
        self._train_means: pd.Series | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> _SKLearnDemandModel:
        """Build features, impute NaNs with training means, fit estimator."""
        X_train, y_train = get_X_y(
            train_df, feature_cols=self.feature_cols, drop_na=False
        )
        self._train_means = X_train.mean()
        X_train = X_train.fillna(self._train_means)
        self.estimator.fit(X_train, y_train)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict demand for all rows in df."""
        X, _ = get_X_y(df, feature_cols=self.feature_cols, drop_na=False)
        if self._train_means is not None:
            X = X.fillna(self._train_means)
        preds = self.estimator.predict(X)
        return np.maximum(preds, 0.0)  # demand cannot be negative

    def feature_importances(self) -> pd.Series | None:
        """Return feature importances if the estimator supports it."""
        est = self.estimator
        if hasattr(est, "feature_importances_"):
            return pd.Series(
                est.feature_importances_, index=self.feature_cols
            ).sort_values(ascending=False)
        if hasattr(est, "coef_"):
            return pd.Series(
                np.abs(est.coef_), index=self.feature_cols
            ).sort_values(ascending=False)
        return None


# ── Censoring-aware model ──────────────────────────────────────────────────────

class CensoringAwareModel:
    """Two-stage demand estimation under inventory-censored observations.

    The core insight:
        When closing_stock == 0 (stockout_flag == 1), the recorded sales
        equals opening_stock — but true demand was ≥ sales.
        Training directly on these rows teaches the model that demand
        was low exactly when it was high but constrained.

    This model corrects that by imputing demand for stockout days
    using a model trained ONLY on uncensored (non-stockout) days.

    Algorithm (two-stage imputation):
        Stage 1: Train an imputation model on rows where stockout_flag == 0.
                 These rows have unbiased demand observations.
        Stage 2: For rows where stockout_flag == 1, impute demand as:
                 max(observed_sales, stage1_prediction)
        Stage 3: Retrain the final model on all rows with imputed demand.

    Leakage check:
        - stockout_flag IS observable (closing_stock == 0 is logged by the store).
        - true_demand is NOT used. Imputed demand is an estimate, not ground truth.

    Parameters
    ----------
    base_estimator:
        sklearn-compatible regressor. Defaults to GradientBoostingRegressor.
    feature_cols:
        Feature columns to use. Defaults to FEATURE_COLS.
    min_uncensored:
        Minimum uncensored rows required to train stage 1. If fewer rows
        are available, the model falls back to training on all data.
    """

    name = "censoring_aware"

    def __init__(
        self,
        base_estimator: Any | None = None,
        feature_cols: list[str] | None = None,
        min_uncensored: int = 30,
    ) -> None:
        self.base_estimator = base_estimator or GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self.feature_cols = feature_cols or FEATURE_COLS
        self.min_uncensored = min_uncensored
        self._final_model: Any = None
        self._train_means: pd.Series | None = None
        self._n_imputed: int = 0

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> CensoringAwareModel:
        """Fit the two-stage censoring-aware model.

        Parameters
        ----------
        train_df:
            Training DataFrame (output of build_features).
            Must contain ``stockout_flag`` and all FEATURE_COLS.
        val_df:
            Unused in this implementation (future: early stopping).
        """
        if "stockout_flag" not in train_df.columns:
            raise ValueError(
                "CensoringAwareModel requires 'stockout_flag' in training data. "
                "This is an observable column (closing_stock == 0) — not leakage."
            )

        X_all, y_sales = get_X_y(
            train_df, feature_cols=self.feature_cols, drop_na=False
        )
        self._train_means = X_all.mean()
        X_all_filled = X_all.fillna(self._train_means)
        stockout_flags = train_df["stockout_flag"].values

        # ── Stage 1: train imputation model on uncensored rows ──────────────
        uncensored_mask = stockout_flags == 0
        n_uncensored = uncensored_mask.sum()

        if n_uncensored >= self.min_uncensored:
            impute_model = clone(self.base_estimator)
            X_uncensored = X_all_filled[uncensored_mask]
            y_uncensored = y_sales.values[uncensored_mask]
            impute_model.fit(X_uncensored, y_uncensored)
        else:
            warnings.warn(
                f"CensoringAwareModel: only {n_uncensored} uncensored rows "
                f"(< min_uncensored={self.min_uncensored}). "
                "Falling back to standard model (no imputation).",
                RuntimeWarning,
                stacklevel=2,
            )
            impute_model = clone(self.base_estimator)
            impute_model.fit(X_all_filled, y_sales.values)

        # ── Stage 2: impute demand for censored (stockout) rows ─────────────
        y_imputed = y_sales.values.astype(float).copy()
        censored_mask = stockout_flags == 1
        self._n_imputed = int(censored_mask.sum())

        if self._n_imputed > 0:
            predicted_censored = impute_model.predict(X_all_filled[censored_mask])
            predicted_censored = np.maximum(predicted_censored, 0.0)
            # Imputed demand = max(what we observed, what model estimates)
            # Rationale: true demand is AT LEAST what was sold, possibly more.
            y_imputed[censored_mask] = np.maximum(
                y_imputed[censored_mask], predicted_censored
            )

        # ── Stage 3: retrain final model on imputed demand ───────────────────
        self._final_model = clone(self.base_estimator)
        self._final_model.fit(X_all_filled, y_imputed)

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict demand using the final model trained on imputed data."""
        if self._final_model is None:
            raise RuntimeError("CensoringAwareModel.fit() must be called before predict().")
        X, _ = get_X_y(df, feature_cols=self.feature_cols, drop_na=False)
        if self._train_means is not None:
            X = X.fillna(self._train_means)
        preds = self._final_model.predict(X)
        return np.maximum(preds, 0.0)

    def feature_importances(self) -> pd.Series | None:
        if self._final_model is None:
            return None
        if hasattr(self._final_model, "feature_importances_"):
            return pd.Series(
                self._final_model.feature_importances_, index=self.feature_cols
            ).sort_values(ascending=False)
        return None

    @property
    def n_imputed(self) -> int:
        """Number of training rows whose demand was imputed."""
        return self._n_imputed


# ── Stockout classifier ────────────────────────────────────────────────────────

class StockoutClassifier:
    """Predict probability of stockout within the next N days.

    Target: ``future_stockout_{horizon}d`` (built by add_stockout_future_label()).
    This is a binary classification problem.

    Uses: XGBClassifier (handles class imbalance via scale_pos_weight).
    Evaluated with: precision, recall, F1, PR-AUC (NOT accuracy).
    """

    name = "stockout_classifier"

    def __init__(
        self,
        horizon: int = 3,
        base_estimator: Any | None = None,
        feature_cols: list[str] | None = None,
    ) -> None:
        self.horizon = horizon
        self.target_col = f"future_stockout_{horizon}d"
        self.feature_cols = feature_cols or FEATURE_COLS
        self.base_estimator = base_estimator
        self._model: Any = None
        self._train_means: pd.Series | None = None
        self._scale_pos_weight: float = 1.0

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> StockoutClassifier:
        """Fit the stockout classifier.

        Parameters
        ----------
        train_df:
            Must contain ``future_stockout_{horizon}d`` (target) and
            all FEATURE_COLS. Build target with add_stockout_future_label().
        """
        if self.target_col not in train_df.columns:
            raise ValueError(
                f"Target '{self.target_col}' not in train_df. "
                f"Call add_stockout_future_label(df, horizon={self.horizon}) first."
            )

        X_train, y_train = get_X_y(
            train_df,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            drop_na=False,
        )
        self._train_means = X_train.mean()
        X_train = X_train.fillna(self._train_means)

        # Handle class imbalance: stockouts are rare events.
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        self._scale_pos_weight = float(n_neg / max(n_pos, 1))

        if self.base_estimator is not None:
            self._model = clone(self.base_estimator)
        else:
            self._model = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=self._scale_pos_weight,
                random_state=42,
                eval_metric="aucpr",
                verbosity=0,
            )

        self._model.fit(X_train, y_train)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability of stockout for each row."""
        if self._model is None:
            raise RuntimeError("StockoutClassifier.fit() must be called first.")
        X, _ = get_X_y(
            df, feature_cols=self.feature_cols,
            target_col=self.target_col if self.target_col in df.columns else TARGET_COL,
            drop_na=False,
        )
        if self._train_means is not None:
            X = X.fillna(self._train_means)
        return self._model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary stockout prediction."""
        return (self.predict_proba(df) >= threshold).astype(int)


# ── Factory ────────────────────────────────────────────────────────────────────

def build_model(model_name: str) -> Any:
    """Instantiate a named model.

    Parameters
    ----------
    model_name:
        One of: ``naive``, ``ma7``, ``ma14``,
        ``linear``, ``random_forest``, ``gradient_boost``,
        ``censoring_aware``, ``stockout_classifier``.

    Returns
    -------
    Model instance (not yet fitted).
    """
    models = {
        "naive": lambda: NaiveForecast(),
        "ma7": lambda: MovingAverageForecast(window=7),
        "ma14": lambda: MovingAverageForecast(window=14),
        "linear": lambda: _SKLearnDemandModel(
            LinearRegression(), name="linear"
        ),
        "random_forest": lambda: _SKLearnDemandModel(
            RandomForestRegressor(
                n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
            ),
            name="random_forest",
        ),
        "gradient_boost": lambda: _SKLearnDemandModel(
            XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            ),
            name="gradient_boost",
        ),
        "censoring_aware": lambda: CensoringAwareModel(),
        "stockout_classifier": lambda: StockoutClassifier(),
    }
    if model_name not in models:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(models.keys())}"
        )
    return models[model_name]()


def predict_with_model(
    model: Any,
    df: pd.DataFrame,
) -> np.ndarray:
    """Dispatch to the correct predict method based on model type."""
    if isinstance(model, (NaiveForecast, MovingAverageForecast)):
        return model.predict(df)
    if isinstance(model, (CensoringAwareModel, StockoutClassifier)):
        return model.predict(df)
    if isinstance(model, _SKLearnDemandModel):
        return model.predict(df)
    # Generic sklearn fallback
    X, _ = get_X_y(df, drop_na=False)
    return np.maximum(model.predict(X), 0.0)


# ── Model persistence ──────────────────────────────────────────────────────────

def _load_registry(models_dir: Path) -> dict:
    """Load or initialise the model registry JSON."""
    registry_path = models_dir / "model_registry.json"
    if registry_path.exists():
        with open(registry_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_registry(registry: dict, models_dir: Path) -> None:
    registry_path = models_dir / "model_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, default=str)


def save_model(
    model: Any,
    model_name: str,
    metrics: dict[str, float],
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    feature_cols: list[str] | None = None,
) -> Path:
    """Save a trained model and update the registry.

    Never overwrites an existing version — always increments.

    Parameters
    ----------
    model:
        Fitted model object.
    model_name:
        Logical name (e.g. ``"censoring_aware"``).
    metrics:
        Evaluation metrics dict to store in the registry.
    models_dir:
        Directory to save the model file.

    Returns
    -------
    Path to the saved .joblib file.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    registry = _load_registry(models_dir)

    # Determine next version number.
    existing = registry.get(model_name, {})
    versions = [int(v.lstrip("v")) for v in existing.get("versions", {}).keys()]
    next_version = max(versions, default=0) + 1
    version_key = f"v{next_version}"

    file_name = f"{model_name}_{version_key}.joblib"
    file_path = models_dir / file_name
    joblib.dump(model, file_path)

    # Update registry.
    if model_name not in registry:
        registry[model_name] = {"versions": {}}

    registry[model_name]["versions"][version_key] = {
        "file": file_name,
        "saved_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "feature_cols": feature_cols or FEATURE_COLS,
    }
    registry[model_name]["latest"] = version_key
    _save_registry(registry, models_dir)

    return file_path


def load_model(
    model_name: str,
    version: str = "latest",
    models_dir: Path | str = DEFAULT_MODELS_DIR,
) -> Any:
    """Load a saved model from disk.

    Parameters
    ----------
    model_name:
        Logical name (e.g. ``"censoring_aware"``).
    version:
        ``"latest"`` or specific version string like ``"v2"``.
    models_dir:
        Directory containing the model files.

    Raises
    ------
    FileNotFoundError, KeyError
    """
    models_dir = Path(models_dir)
    registry = _load_registry(models_dir)

    if model_name not in registry:
        raise FileNotFoundError(
            f"Model '{model_name}' not found in registry. "
            f"Available: {list(registry.keys())}. "
            f"Run: python scripts/train_all.py"
        )

    entry = registry[model_name]
    version_key = entry["latest"] if version == "latest" else version
    version_info = entry["versions"].get(version_key)
    if version_info is None:
        available = list(entry["versions"].keys())
        raise KeyError(
            f"Version '{version_key}' not found for '{model_name}'. "
            f"Available: {available}"
        )

    model_path = models_dir / version_info["file"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)


def get_model_info(
    model_name: str,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
) -> dict:
    """Return registry metadata for a model (all versions)."""
    models_dir = Path(models_dir)
    registry = _load_registry(models_dir)
    if model_name not in registry:
        return {}
    return registry[model_name]

"""
src/train.py — Model training with correct temporal split.

Phase: 5 (first ML model)
Status: STUB — implementation added in Phase 5.

Responsibility:
    Train demand-forecasting models using a CHRONOLOGICAL train/val/test
    split. Random splitting is explicitly forbidden for this project because
    it would expose future periods to the model during training.

Temporal split (default):
    TRAIN:      days   0 – 83   (70 % of 120 days)
    VALIDATION: days  84 – 95   (10 %)
    TEST:       days  96 – 119  (20 %)

Model registry (to be populated phase by phase):
    Phase 4: NaiveForecast, MovingAverageForecast
    Phase 5: LinearRegression, RandomForest
    Phase 5: GradientBoosting
    Phase 8: CensoringAwareModel (Tobit / EM imputation)
    Phase 9: StockoutClassifier

Rule: Save trained models to models/<name>_v<version>.joblib,
      never overwrite an older version — compare them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a product-day DataFrame chronologically.

    Parameters
    ----------
    df:
        Must contain a ``date`` column (str or datetime).
    train_frac:
        Fraction of unique dates that go to train.
    val_frac:
        Fraction of unique dates that go to validation.
        Test gets the remaining ``1 - train_frac - val_frac``.

    Returns
    -------
    train, val, test : tuple of DataFrames
    """
    # Phase 5: implement.
    raise NotImplementedError("Phase 5: implement temporal_split()")


def train_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    model_name: str = "naive",
    save_dir: Path | str = "models",
) -> object:
    """Train and save a named model.

    Parameters
    ----------
    train:
        Training DataFrame.
    val:
        Validation DataFrame (used for early stopping if supported).
    model_name:
        One of: ``naive``, ``moving_avg``, ``linear``, ``random_forest``,
        ``gradient_boost``, ``censoring_aware``.
    save_dir:
        Directory to write the serialised model.

    Returns
    -------
    Trained model object.
    """
    # Phase 5: implement.
    raise NotImplementedError("Phase 5: implement train_model()")

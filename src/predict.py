"""
src/predict.py — Inference: load a saved model and produce predictions.

Phase: 11 (FastAPI)
Status: STUB — implementation added in Phase 11.

Responsibility:
    Load a versioned model from disk and produce a demand forecast +
    stockout risk score for a given product + inventory state.

    This is what the API layer calls. It must:
        1. Validate inputs (no negative stock, no unknown products).
        2. Construct the feature vector from the validated inputs.
        3. Load the specified model version (default: latest).
        4. Return prediction + model version metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PredictionRequest:
    product: str
    current_inventory: int
    price: float
    promo: int  # 0 or 1
    # Optionally override which model version to use.
    model_version: str = "latest"


@dataclass
class PredictionResponse:
    product: str
    forecast_demand: float          # units/day
    stockout_risk: float            # probability 0–1
    recommended_order: int          # units
    days_until_stockout: float      # estimated days
    model_version: str
    explanation: dict               # human-readable breakdown


def predict(request: PredictionRequest, models_dir: Path | str = "models") -> PredictionResponse:
    """Run demand forecast and stockout risk for one product.

    Parameters
    ----------
    request:
        Validated input payload.
    models_dir:
        Directory where serialised models are stored.

    Returns
    -------
    PredictionResponse
        Full prediction result with explanation.

    Raises
    ------
    ValueError
        If ``request.current_inventory < 0`` or product is unknown.
    FileNotFoundError
        If no model file is found in ``models_dir``.
    """
    # Phase 11: implement.
    raise NotImplementedError("Phase 11: implement predict()")

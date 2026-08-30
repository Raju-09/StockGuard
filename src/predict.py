"""
src/predict.py — Production inference: load a model and serve predictions.

Called by the FastAPI layer. Validates inputs, builds features,
runs all models, and returns a structured prediction response.

Input validation rules:
    - inventory must be >= 0
    - price must be > 0
    - promo must be 0 or 1
    - product must be in the known product list

Output includes:
    - Demand forecast (units/day)
    - Stockout risk score (0–1 probability)
    - Recommended order quantity (business logic, not raw ML)
    - Days until stockout estimate
    - Model version
    - Human-readable explanation of the recommendation
    - Recent history for charting (dates, sales, opening_stock, true_demand)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_observed, load_with_ground_truth
from src.features import build_features
from src.train import (
    DEFAULT_MODELS_DIR,
    _load_registry,
    load_model,
    predict_with_model,
)

# ── Constants ──────────────────────────────────────────────────────────────────

# Default lead time (days) used when per-product lead time is unknown.
DEFAULT_LEAD_TIME_DAYS: int = 2
# Safety stock factor: add this many "days of demand" as buffer.
SAFETY_STOCK_DAYS: float = 2.0
# Number of history days to include in the chart payload.
HISTORY_DAYS: int = 30


# ── Request / Response schemas ────────────────────────────────────────────────

@dataclass
class PredictionRequest:
    product: str
    current_inventory: int
    price: float
    promo: int          # 0 or 1
    model_name: str = "censoring_aware"
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS


@dataclass
class PredictionResponse:
    product: str
    forecast_demand: float          # predicted units/day
    stockout_risk: float            # 0.0 – 1.0
    stockout_risk_label: str        # "LOW" | "MEDIUM" | "HIGH"
    recommended_order: int          # units to order now
    days_until_stockout: float      # estimated days at current forecast rate
    model_name: str
    model_version: str
    explanation: dict               # breakdown for the UI "Why?" panel
    history: dict                   # {dates, sales, opening_stock, true_demand}


# ── Input validation ───────────────────────────────────────────────────────────

def _validate_request(req: PredictionRequest, known_products: list[str]) -> None:
    """Raise ValueError for invalid inputs. Never pass bad data to the model."""
    errors = []
    if req.current_inventory < 0:
        errors.append(f"current_inventory must be >= 0, got {req.current_inventory}")
    if req.price <= 0:
        errors.append(f"price must be > 0, got {req.price}")
    if req.promo not in (0, 1):
        errors.append(f"promo must be 0 or 1, got {req.promo}")
    if req.lead_time_days < 1:
        errors.append(f"lead_time_days must be >= 1, got {req.lead_time_days}")
    if req.product not in known_products:
        errors.append(
            f"Unknown product '{req.product}'. "
            f"Available: {known_products}"
        )
    if errors:
        raise ValueError(" | ".join(errors))


# ── Business logic ─────────────────────────────────────────────────────────────

def _compute_risk_label(risk: float) -> str:
    if risk >= 0.70:
        return "HIGH"
    if risk >= 0.40:
        return "MEDIUM"
    return "LOW"


def _compute_recommended_order(
    forecast_demand: float,
    current_inventory: int,
    lead_time_days: int,
    safety_stock_days: float = SAFETY_STOCK_DAYS,
) -> int:
    """Simple order formula:
        order = lead_time_demand + safety_stock - current_inventory

    If the result is negative (enough stock), recommend 0.
    """
    lead_time_demand = forecast_demand * lead_time_days
    safety_stock = forecast_demand * safety_stock_days
    order = lead_time_demand + safety_stock - current_inventory
    return max(0, int(np.ceil(order)))


def _compute_days_until_stockout(
    current_inventory: int,
    forecast_demand: float,
) -> float:
    if forecast_demand <= 0:
        return float("inf")
    return round(current_inventory / forecast_demand, 1)


# ── History builder ────────────────────────────────────────────────────────────

def _build_history(
    product: str,
    n_days: int = HISTORY_DAYS,
) -> dict:
    """Load recent history for the chart payload."""
    try:
        truth = load_with_ground_truth()
        prod = truth[truth["product"] == product].sort_values("date").tail(n_days)
        return {
            "dates": prod["date"].astype(str).tolist(),
            "sales": prod["sales"].tolist(),
            "opening_stock": prod["opening_stock"].tolist(),
            "true_demand": prod["true_demand"].tolist(),
        }
    except Exception:
        return {"dates": [], "sales": [], "opening_stock": [], "true_demand": []}


# ── Prediction context builder ─────────────────────────────────────────────────

def _build_prediction_row(
    req: PredictionRequest,
    df_observed: pd.DataFrame,
) -> pd.DataFrame:
    """Create a single-row DataFrame representing the prediction request.

    Uses the most recent historical row for the product as context
    (lag features, rolling means), then overrides with the current
    inventory and price from the request.
    """
    product_rows = (
        df_observed[df_observed["product"] == req.product]
        .sort_values("date")
    )
    if product_rows.empty:
        raise ValueError(f"No historical data found for product '{req.product}'.")

    # Use the most recent feature-complete row as context.
    featured = build_features(product_rows)
    last_row = featured.iloc[-1:].copy()

    # Override with current request values.
    last_row["opening_stock"] = req.current_inventory
    last_row["price"] = req.price
    last_row["promo"] = req.promo

    return last_row


# ── Main predict function ──────────────────────────────────────────────────────

def predict(
    req: PredictionRequest,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
) -> PredictionResponse:
    """Run demand forecast and stockout risk for one product.

    Parameters
    ----------
    req:
        Validated prediction request.
    models_dir:
        Directory containing trained model files.

    Returns
    -------
    PredictionResponse

    Raises
    ------
    ValueError
        On invalid inputs.
    FileNotFoundError
        If no trained models exist (run scripts/train_all.py first).
    """
    models_dir = Path(models_dir)

    # Load observed data to get known products and feature context.
    df_observed = load_observed()
    known_products = sorted(df_observed["product"].unique().tolist())

    # Validate inputs.
    _validate_request(req, known_products)

    # Build feature row for prediction.
    pred_row = _build_prediction_row(req, df_observed)

    # ── Load and run demand forecast model ────────────────────────────────────
    demand_model = load_model(req.model_name, models_dir=models_dir)
    demand_preds = predict_with_model(demand_model, pred_row)
    forecast_demand = float(max(demand_preds[0], 0.0))

    # Get model version from registry
    registry = _load_registry(models_dir)
    model_version = (
        registry.get(req.model_name, {}).get("latest", "unknown")
    )

    # ── Load and run stockout classifier ─────────────────────────────────────
    stockout_risk = 0.0
    try:
        stockout_model = load_model("stockout_classifier", models_dir=models_dir)
        # Add the future_stockout label column as a dummy (predict_proba ignores it)
        pred_row_with_label = pred_row.copy()
        pred_row_with_label["future_stockout_3d"] = 0
        stockout_risk = float(stockout_model.predict_proba(pred_row_with_label)[0])
    except (FileNotFoundError, KeyError):
        # Fallback: compute risk from days-until-stockout heuristic
        days = _compute_days_until_stockout(req.current_inventory, forecast_demand)
        if days < req.lead_time_days:
            stockout_risk = 0.95
        elif days < req.lead_time_days * 2:
            stockout_risk = 0.60
        else:
            stockout_risk = 0.15

    # ── Business layer ────────────────────────────────────────────────────────
    recommended_order = _compute_recommended_order(
        forecast_demand,
        req.current_inventory,
        req.lead_time_days,
    )
    days_until_stockout = _compute_days_until_stockout(
        req.current_inventory, forecast_demand
    )
    risk_label = _compute_risk_label(stockout_risk)

    # ── Explanation ───────────────────────────────────────────────────────────
    explanation = {
        "forecast_demand_per_day": round(forecast_demand, 2),
        "lead_time_days": req.lead_time_days,
        "lead_time_demand": round(forecast_demand * req.lead_time_days, 2),
        "safety_stock_days": SAFETY_STOCK_DAYS,
        "safety_stock_units": round(forecast_demand * SAFETY_STOCK_DAYS, 2),
        "current_inventory": req.current_inventory,
        "recommended_order_formula": (
            f"ceil({round(forecast_demand * req.lead_time_days, 1)} + "
            f"{round(forecast_demand * SAFETY_STOCK_DAYS, 1)} - "
            f"{req.current_inventory}) = {recommended_order}"
        ),
    }

    # ── History for chart ─────────────────────────────────────────────────────
    history = _build_history(req.product)

    return PredictionResponse(
        product=req.product,
        forecast_demand=round(forecast_demand, 2),
        stockout_risk=round(stockout_risk, 4),
        stockout_risk_label=risk_label,
        recommended_order=recommended_order,
        days_until_stockout=min(days_until_stockout, 999.0),
        model_name=req.model_name,
        model_version=model_version,
        explanation=explanation,
        history=history,
    )

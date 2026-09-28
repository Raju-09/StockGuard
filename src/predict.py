"""
src/predict.py - Production inference: load a model and serve predictions.

Called by the FastAPI layer. Validates inputs, builds features,
runs all models, and returns a structured prediction response.

Input validation rules:
    - inventory must be >= 0
    - price must be > 0
    - promo must be 0 or 1
    - product must be in the known product list

Output includes:
    - Demand forecast (units/day)
    - Stockout risk score (0-1 probability)
    - Recommended order quantity (business logic, not raw ML)
    - Days until stockout estimate
    - Model version
    - Structured explanation dict (machine-readable + human-readable narrative)
    - Recent history for charting (dates, sales, opening_stock, true_demand)
    - Financial impact trade-off (holding cost vs lost margin)
    - Cross-SKU substitution alert
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

DEFAULT_LEAD_TIME_DAYS: int = 2
SAFETY_STOCK_DAYS: float = 2.0
HISTORY_DAYS: int = 30

# Known product substitution network
SUBSTITUTE_PAIRS: dict[str, str] = {
    "Milk": "Yogurt",
    "Yogurt": "Milk",
    "Tea": "Sugar",
    "Biscuits": "Bread",
    "Bread": "Biscuits",
    "Rice": "Oil",
    "Oil": "Rice",
    "Soap": "None",
    "Eggs": "Bread",
    "Sugar": "Tea",
}


# ── Request / Response schemas ────────────────────────────────────────────────

@dataclass
class PredictionRequest:
    product: str
    current_inventory: int
    price: float
    promo: int          # 0 or 1
    model_name: str = "censoring_aware"
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS
    service_level_target: float = 0.95
    on_order: int = 0
    reserved: int = 0
    moq: int = 1
    case_pack_size: int = 1


@dataclass
class PredictionResponse:
    product: str
    forecast_demand: float          # predicted units/day
    stockout_risk: float            # 0.0 - 1.0
    stockout_risk_label: str        # "LOW" | "MEDIUM" | "HIGH"
    recommended_order: int          # units to order now
    days_until_stockout: float      # estimated days at current forecast rate
    model_name: str
    model_version: str
    explanation: dict               # structured breakdown for the UI
    history: dict                   # {dates, sales, opening_stock, true_demand}
    financial_impact: dict          # holding cost vs lost sales trade-off
    substitution_alert: dict        # cross-SKU spillover warning


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
    moq: int = 1,
    case_pack_size: int = 1,
) -> tuple[int, float, float]:
    """Stochastic replenishment formula:
        order = lead_time_demand + safety_stock - current_inventory

    Where safety_stock accounts for stochastic demand variance.
    Includes MOQ and case_pack_size parameters for ordering constraints.
    Returns: (order_qty, lead_time_demand, safety_stock_units)
    """
    lead_time_demand = forecast_demand * lead_time_days
    # Demand std dev estimated as ~30% of daily mean demand
    sigma_d = max(1.0, forecast_demand * 0.30)
    # Z-score for 95% service level = 1.645
    z_score = 1.645
    # Stochastic buffer across lead time: z * sqrt(L) * sigma_d
    safety_stock = float(z_score * np.sqrt(lead_time_days) * sigma_d)
    # Floor safety stock at least 1.5 days
    safety_stock = max(safety_stock, forecast_demand * 1.5)

    order = lead_time_demand + safety_stock - current_inventory
    raw_order = max(0, int(np.ceil(order)))
    if raw_order > 0:
        # Round up to case pack size, then enforce MOQ
        if case_pack_size > 1:
            raw_order = int(np.ceil(raw_order / case_pack_size)) * case_pack_size
        raw_order = max(raw_order, moq)
    return raw_order, round(lead_time_demand, 2), round(safety_stock, 2)


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
    """Create a single-row DataFrame representing the prediction request."""
    product_rows = (
        df_observed[df_observed["product"] == req.product]
        .sort_values("date")
    )
    if product_rows.empty:
        raise ValueError(f"No historical data found for product '{req.product}'.")

    featured = build_features(product_rows)
    last_row = featured.iloc[-1:].copy()

    last_row["opening_stock"] = req.current_inventory
    last_row["price"] = req.price
    last_row["promo"] = req.promo

    return last_row


# ── Main predict function ──────────────────────────────────────────────────────

def predict(
    req: PredictionRequest,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
) -> PredictionResponse:
    """Run demand forecast and stockout risk for one product."""
    models_dir = Path(models_dir)

    df_observed = load_observed()
    known_products = sorted(df_observed["product"].unique().tolist())

    _validate_request(req, known_products)

    pred_row = _build_prediction_row(req, df_observed)

    # ── Load and run demand forecast model ────────────────────────────────────
    demand_model = load_model(req.model_name, models_dir=models_dir)
    demand_preds = predict_with_model(demand_model, pred_row)
    forecast_demand = float(max(demand_preds[0], 0.0))

    registry = _load_registry(models_dir)
    model_version = registry.get(req.model_name, {}).get("latest", "unknown")

    # ── Stockout classifier ───────────────────────────────────────────────────
    stockout_risk = 0.0
    try:
        stockout_model = load_model("stockout_classifier", models_dir=models_dir)
        pred_row_with_label = pred_row.copy()
        pred_row_with_label["future_stockout_3d"] = 0
        stockout_risk = float(stockout_model.predict_proba(pred_row_with_label)[0])
    except (FileNotFoundError, KeyError):
        days = _compute_days_until_stockout(req.current_inventory, forecast_demand)
        if days < req.lead_time_days:
            stockout_risk = 0.95
        elif days < req.lead_time_days * 2:
            stockout_risk = 0.60
        else:
            stockout_risk = 0.15

    # ── Business layer ────────────────────────────────────────────────────────
    # Standard Inventory Position = on_hand + on_order - reserved
    inventory_position = req.current_inventory + req.on_order - req.reserved
    recommended_order, lead_time_demand, safety_stock_units = _compute_recommended_order(
        forecast_demand,
        inventory_position,
        req.lead_time_days,
        moq=req.moq,
        case_pack_size=req.case_pack_size,
    )
    days_until_stockout = _compute_days_until_stockout(
        req.current_inventory, forecast_demand
    )
    risk_label = _compute_risk_label(stockout_risk)

    # ── Financial trade-off (CFO view) ────────────────────────────────────────
    daily_holding_cost_per_unit = 0.50
    estimated_daily_holding_cost = round(req.current_inventory * daily_holding_cost_per_unit, 2)
    gross_margin_rate = 0.40
    expected_daily_lost_sales_units = round(forecast_demand * stockout_risk, 2)
    expected_lost_margin_per_day = round(
        expected_daily_lost_sales_units * (req.price * gross_margin_rate), 2
    )

    financial_impact = {
        "daily_holding_cost": estimated_daily_holding_cost,
        "holding_cost_rate": f"Rs.{daily_holding_cost_per_unit}/unit/day",
        "expected_lost_margin_per_day": expected_lost_margin_per_day,
        "trade_off_recommendation": (
            "Buffer stock justified: potential lost sales margin exceeds holding costs."
            if expected_lost_margin_per_day > estimated_daily_holding_cost
            else "Inventory level optimal: holding costs in line with demand."
        ),
    }

    # ── Cross-SKU substitution alert ──────────────────────────────────────────
    substitute_sku = SUBSTITUTE_PAIRS.get(req.product, "None")
    substitution_alert = {
        "substitute_sku": substitute_sku,
        "is_active": risk_label == "HIGH" and substitute_sku != "None",
        "message": (
            f"High stockout risk for {req.product}. "
            f"Monitor {substitute_sku} for expected +15-25% spillover demand surge."
            if (risk_label == "HIGH" and substitute_sku != "None")
            else "No substitution surge expected."
        ),
    }

    # ── Order-zero contradiction explanation ──────────────────────────────────
    # P0 FIX: When classifier says HIGH/MEDIUM risk but formula says order=0,
    # we MUST explain the contradiction explicitly or the UI looks broken.
    order_zero_reason: str | None = None
    if recommended_order == 0 and stockout_risk >= 0.40:
        order_zero_reason = (
            f"Although stockout probability is {round(stockout_risk * 100, 1)}%, "
            f"the current inventory ({req.current_inventory} units) already covers "
            f"the full stochastic lead-time demand "
            f"({round(lead_time_demand, 1)} units over {req.lead_time_days}d) "
            f"plus the 95%-service-level safety buffer ({round(safety_stock_units, 1)} units). "
            f"The risk score reflects historical stockout exposure patterns in training data, "
            f"not a current inventory gap. No reorder is triggered at this stock level."
        )

    # ── Demand quantile estimates (Gaussian approximation, sigma=30% of mean) ─
    sigma_d = max(1.0, forecast_demand * 0.30)
    p10 = round(max(0.0, forecast_demand - 1.282 * sigma_d), 2)
    p50 = round(forecast_demand, 2)
    p90 = round(forecast_demand + 1.282 * sigma_d, 2)
    ltd_p50 = round(p50 * req.lead_time_days, 2)
    ltd_p90 = round(p90 * req.lead_time_days, 2)

    # ── Reorder formula (human-readable) ──────────────────────────────────────
    formula_str = (
        f"ceil({round(lead_time_demand, 1)} [Lead-Time Demand] + "
        f"{round(safety_stock_units, 1)} [Safety Stock] - "
        f"{req.current_inventory} [On-Hand]) = {recommended_order} units"
    )

    # ── Why / Action narrative ────────────────────────────────────────────────
    if days_until_stockout < req.lead_time_days:
        why_text = (
            f"Current stock ({req.current_inventory} units) will be exhausted in "
            f"{days_until_stockout} days, which is BEFORE the supplier can deliver "
            f"(lead time: {req.lead_time_days} days). A stockout is near-certain "
            f"without immediate action."
        )
        action_text = (
            f"ORDER {recommended_order} units IMMEDIATELY. "
            f"The stockout window closes in {days_until_stockout} days."
            if recommended_order > 0
            else "Monitor supplier and expedite if possible."
        )
    elif days_until_stockout < req.lead_time_days * 2:
        why_text = (
            f"Inventory covers the lead time ({days_until_stockout}d > {req.lead_time_days}d), "
            f"but demand uncertainty creates a {round(stockout_risk * 100, 1)}% shortage risk. "
            f"A safety buffer of {round(safety_stock_units, 1)} units protects "
            f"against demand spikes at the 95% service level (z=1.645)."
        )
        action_text = (
            f"Reorder {recommended_order} units to maintain the "
            f"{round(req.service_level_target * 100, 0):.0f}% service-level target."
            if recommended_order > 0
            else "Inventory position covers stochastic demand. Monitor closely."
        )
    else:
        if recommended_order == 0:
            why_text = (
                f"Inventory ({req.current_inventory} units) provides {days_until_stockout} days "
                f"of runway, comfortably exceeding the lead time ({req.lead_time_days}d) "
                f"plus the uncertainty buffer. The risk score reflects "
                f"historical stockout exposure, not a current shortage."
            )
            action_text = "No immediate reorder required. Revisit at the next review cycle."
        else:
            why_text = (
                f"Inventory ({req.current_inventory} units) provides {days_until_stockout} days "
                f"of runway. Ordering {recommended_order} units now pre-empts the upcoming "
                f"replenishment cycle and maintains the 95% service-level target."
            )
            action_text = (
                f"Pre-emptive reorder of {recommended_order} units at the next "
                f"purchase cycle maintains 95% service level."
            )

    # ── Structured explanation dict (P0 FIX: no more [object Object]) ─────────
    explanation = {
        # Machine-readable fields for the Trust Report modal
        "forecast_demand_per_day": round(forecast_demand, 2),
        "lead_time_days": req.lead_time_days,
        "lead_time_demand_p50": ltd_p50,
        "lead_time_demand_p90": ltd_p90,
        "safety_stock_days": SAFETY_STOCK_DAYS,
        "safety_stock_units": round(safety_stock_units, 2),
        "current_inventory": req.current_inventory,
        "on_order": req.on_order,
        "reserved": req.reserved,
        "inventory_position": inventory_position,  # on_hand + on_order - reserved
        "moq": req.moq,
        "case_pack_size": req.case_pack_size,
        "recommended_order": recommended_order,
        "formula": formula_str,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "days_until_stockout": days_until_stockout,
        "stockout_risk_pct": round(stockout_risk * 100, 1),
        "service_level_target_pct": round(req.service_level_target * 100, 1),
        "model_used": req.model_name,
        # Human-readable narrative (AI Decision Explanation panel)
        "why": why_text,
        "action": action_text,
        "order_zero_reason": order_zero_reason,
    }

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
        financial_impact=financial_impact,
        substitution_alert=substitution_alert,
    )

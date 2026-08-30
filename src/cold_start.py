"""
src/cold_start.py — Cold-Start Demand Forecasting for New SKUs.

Solves the New-Product Cold-Start problem:
When a brand-new SKU is introduced (zero historical sales), conventional time-series
and regression models cannot predict demand.

Methodology:
1. Attribute Encoding: Map product attributes (category, brand, price tier, pack size, perishability).
2. Analog Matching: Compute similarity against historical catalog SKUs using normalized Gower distance.
3. Distribution Borrowing: Borrow empirical launch distributions from the top-k nearest analogs.
4. Probabilistic Quantiles: Compute P10, P50 (median), P80, P90, P95 demand.
5. Operational Guard: If maximum analog similarity < 0.60, system flags LOW CONFIDENCE / ABSTAIN,
   recommending a minimal pilot batch + manual review rather than hallucinatory automatic ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data import load_observed

# Reference catalog attributes for existing store SKUs
CATALOG_METADATA: dict[str, dict[str, Any]] = {
    "Milk": {
        "category": "Dairy",
        "brand": "StoreBrand",
        "price": 55.0,
        "pack_size": 1,
        "unit": "Litre",
        "is_perishable": 1,
        "shelf_life_days": 4,
        "default_lead_time": 2,
    },
    "Bread": {
        "category": "Bakery",
        "brand": "DailyBake",
        "price": 40.0,
        "pack_size": 1,
        "unit": "Loaf",
        "is_perishable": 1,
        "shelf_life_days": 5,
        "default_lead_time": 1,
    },
    "Eggs": {
        "category": "Dairy & Eggs",
        "brand": "FarmFresh",
        "price": 72.0,
        "pack_size": 6,
        "unit": "Pack",
        "is_perishable": 1,
        "shelf_life_days": 14,
        "default_lead_time": 2,
    },
    "Yogurt": {
        "category": "Dairy",
        "brand": "StoreBrand",
        "price": 45.0,
        "pack_size": 1,
        "unit": "Cup",
        "is_perishable": 1,
        "shelf_life_days": 7,
        "default_lead_time": 2,
    },
    "Biscuits": {
        "category": "Snacks",
        "brand": "CrispBake",
        "price": 30.0,
        "pack_size": 1,
        "unit": "Pack",
        "is_perishable": 0,
        "shelf_life_days": 90,
        "default_lead_time": 3,
    },
    "Rice": {
        "category": "Grains",
        "brand": "GoldenGrain",
        "price": 80.0,
        "pack_size": 1,
        "unit": "Kg",
        "is_perishable": 0,
        "shelf_life_days": 180,
        "default_lead_time": 3,
    },
    "Oil": {
        "category": "Cooking",
        "brand": "PurePress",
        "price": 140.0,
        "pack_size": 1,
        "unit": "Litre",
        "is_perishable": 0,
        "shelf_life_days": 180,
        "default_lead_time": 3,
    },
    "Sugar": {
        "category": "Baking",
        "brand": "SweetPure",
        "price": 50.0,
        "pack_size": 1,
        "unit": "Kg",
        "is_perishable": 0,
        "shelf_life_days": 365,
        "default_lead_time": 3,
    },
    "Tea": {
        "category": "Beverage",
        "brand": "LeafBlend",
        "price": 90.0,
        "pack_size": 1,
        "unit": "Pack",
        "is_perishable": 0,
        "shelf_life_days": 365,
        "default_lead_time": 4,
    },
    "Soap": {
        "category": "Personal Care",
        "brand": "CleanLife",
        "price": 35.0,
        "pack_size": 1,
        "unit": "Bar",
        "is_perishable": 0,
        "shelf_life_days": 730,
        "default_lead_time": 4,
    },
}


@dataclass
class NewProductRequest:
    product_name: str
    category: str
    brand: str
    price: float
    pack_size: int = 1
    unit: str = "Unit"
    is_perishable: int = 0
    supplier_lead_time: int = 2
    moq: int = 20
    case_pack_size: int = 6


@dataclass
class AnalogMatch:
    product_name: str
    similarity_score: float
    category: str
    historical_daily_sales_mean: float
    historical_daily_sales_std: float


@dataclass
class ColdStartForecastResult:
    product_name: str
    status: str                         # "READY" | "LOW_CONFIDENCE_ABSTAIN"
    confidence_level: str               # "HIGH" | "MEDIUM" | "LOW"
    max_similarity: float
    top_analogs: list[dict]
    expected_daily_demand: float        # P50 point estimate
    p10: float
    p50: float
    p80: float
    p90: float
    p95: float
    recommended_pilot_order: int
    moq_applied: bool
    pack_rounding_applied: bool
    explanation: str


def compute_product_similarity(new_p: NewProductRequest, catalog_p: dict[str, Any]) -> float:
    """Calculate multi-attribute similarity between a new SKU and a catalog SKU.

    Weights:
    - Category match: 0.40
    - Price ratio similarity: 0.25
    - Perishability match: 0.20
    - Pack size similarity: 0.15
    """
    # 1. Category
    cat_sim = 1.0 if new_p.category.lower() == catalog_p["category"].lower() else 0.0

    # 2. Price similarity: 1 - |p1 - p2| / max(p1, p2)
    p_max = max(new_p.price, catalog_p["price"])
    price_sim = max(0.0, 1.0 - abs(new_p.price - catalog_p["price"]) / p_max) if p_max > 0 else 1.0

    # 3. Perishability
    perish_sim = 1.0 if new_p.is_perishable == catalog_p["is_perishable"] else 0.0

    # 4. Pack size
    pk_max = max(new_p.pack_size, catalog_p["pack_size"])
    pack_sim = max(0.0, 1.0 - abs(new_p.pack_size - catalog_p["pack_size"]) / pk_max) if pk_max > 0 else 1.0

    total_sim = (0.40 * cat_sim) + (0.25 * price_sim) + (0.20 * perish_sim) + (0.15 * pack_sim)
    return round(float(total_sim), 4)


def run_cold_start_forecast(req: NewProductRequest) -> ColdStartForecastResult:
    """Generate cold-start demand forecast and initial order recommendation for a new SKU."""
    df_obs = load_observed()

    # Find similarity against all catalog SKUs
    matches: list[AnalogMatch] = []
    for cat_name, cat_meta in CATALOG_METADATA.items():
        sim = compute_product_similarity(req, cat_meta)
        prod_rows = df_obs[df_obs["product"] == cat_name]
        mean_sales = float(prod_rows["sales"].mean()) if not prod_rows.empty else 10.0
        std_sales = float(prod_rows["sales"].std()) if not prod_rows.empty else 3.0

        matches.append(
            AnalogMatch(
                product_name=cat_name,
                similarity_score=sim,
                category=cat_meta["category"],
                historical_daily_sales_mean=round(mean_sales, 2),
                historical_daily_sales_std=round(std_sales, 2),
            )
        )

    # Sort descending by similarity
    matches.sort(key=lambda m: m.similarity_score, reverse=True)
    top_3 = matches[:3]
    max_sim = top_3[0].similarity_score if top_3 else 0.0

    # Determine confidence
    if max_sim >= 0.75:
        confidence = "HIGH"
        status = "READY"
    elif max_sim >= 0.50:
        confidence = "MEDIUM"
        status = "READY"
    else:
        confidence = "LOW"
        status = "LOW_CONFIDENCE_ABSTAIN"

    # Weighted demand prior borrowing from top analogs
    weights = np.array([m.similarity_score for m in top_3])
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.array([1.0, 0.0, 0.0])

    weighted_mean = float(sum(w * m.historical_daily_sales_mean for w, m in zip(weights, top_3)))
    weighted_std = float(sum(w * m.historical_daily_sales_std for w, m in zip(weights, top_3)))

    # Compute probabilistic quantiles under Normal / Log-Normal launch prior
    # P10, P50, P80, P90, P95
    p10 = max(1.0, weighted_mean - 1.28 * weighted_std)
    p50 = weighted_mean
    p80 = weighted_mean + 0.84 * weighted_std
    p90 = weighted_mean + 1.28 * weighted_std
    p95 = weighted_mean + 1.645 * weighted_std

    # Initial order calculation (risk-averse pilot sizing for lead time)
    # Target P80 coverage over lead time + 3 days safety buffer
    target_coverage_days = req.supplier_lead_time + 3
    raw_units = int(np.ceil(p80 * target_coverage_days))

    # Apply MOQ constraint
    moq_applied = raw_units < req.moq
    order_after_moq = max(raw_units, req.moq)

    # Apply case pack rounding
    rem = order_after_moq % req.case_pack_size
    pack_rounding_applied = rem != 0
    final_order = (
        order_after_moq if rem == 0 else order_after_moq + (req.case_pack_size - rem)
    )

    if status == "LOW_CONFIDENCE_ABSTAIN":
        explanation = (
            f"Insufficient analog similarity ({max_sim*100:.1f}% < 50%). "
            f"System ABSTAINS from confident automated ordering. "
            f"Recommended: manual buyer review with a minimal test pilot of {final_order} units."
        )
    else:
        top_name = top_3[0].product_name
        explanation = (
            f"Cold-start demand inferred from closest analog '{top_name}' ({max_sim*100:.1f}% match). "
            f"Expected launch velocity P50={p50:.1f} u/d (P90={p90:.1f} u/d). "
            f"Recommended initial order: {final_order} units covering {target_coverage_days} days."
        )

    return ColdStartForecastResult(
        product_name=req.product_name,
        status=status,
        confidence_level=confidence,
        max_similarity=max_sim,
        top_analogs=[
            {
                "product": m.product_name,
                "similarity_pct": round(m.similarity_score * 100, 1),
                "category": m.category,
                "avg_daily_sales": m.historical_daily_sales_mean,
            }
            for m in top_3
        ],
        expected_daily_demand=round(p50, 1),
        p10=round(p10, 1),
        p50=round(p50, 1),
        p80=round(p80, 1),
        p90=round(p90, 1),
        p95=round(p95, 1),
        recommended_pilot_order=final_order,
        moq_applied=moq_applied,
        pack_rounding_applied=pack_rounding_applied,
        explanation=explanation,
    )

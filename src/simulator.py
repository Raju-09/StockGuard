"""
src/simulator.py — Business-level inventory simulation.

Purpose:
    Translate model accuracy into business outcomes.
    A better MAE doesn't always mean better business decisions.
    This simulator answers: "Does my ML system make the store better?"

Simulation logic:
    For each product, step through historical demand day by day:
        1. Apply the demand forecast model to get a demand estimate.
        2. Use a simple (s, Q) replenishment policy:
               if (inventory + pipeline - today's demand) <= reorder_point:
                   place an order of size order_qty
           Orders arrive after lead_time_days.
        3. Serve customers from available stock.
        4. Track: stockout days, holding cost, ordering cost, lost-sales cost.

Policies compared:
    A — No forecasting: fixed order quantity, no demand signal
    B — Naive forecast (yesterday's sales)
    C — Standard ML (best of LinearRegression / RF / XGB)
    D — Censoring-aware ML

Metrics:
    stockout_pct    — % of days with closing_stock == 0
    lost_sales      — total units not sold due to stockouts
    holding_cost    — sum of closing_stock × holding_cost_per_unit per day
    ordering_cost   — fixed cost per order placed
    total_cost      — holding + ordering + lost_sales_cost
    service_level   — 1 - stockout_pct (higher is better)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ── Cost parameters (calibrated to realistic retail economics) ─────────────────
# Holding cost: 25% annual cost of capital / storage / shrinkage (~₹0.05/unit/day on ₹70 average SKU)
HOLDING_COST_PER_UNIT_PER_DAY: float = 0.05
# Fixed PO transaction & delivery receiving cost
ORDERING_COST_PER_ORDER: float = 30.0
# Lost gross profit margin penalty per unit of unmet customer demand (~40% of unit price)
LOST_SALES_COST_PER_UNIT: float = 25.0


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    policy_name: str
    product: str

    total_days: int = 0
    stockout_days: int = 0
    lost_sales_units: int = 0
    total_holding_cost: float = 0.0
    total_ordering_cost: float = 0.0
    total_lost_sales_cost: float = 0.0
    n_orders: int = 0

    @property
    def stockout_pct(self) -> float:
        if self.total_days == 0:
            return 0.0
        return round(self.stockout_days / self.total_days * 100, 2)

    @property
    def service_level(self) -> float:
        return round(1.0 - self.stockout_pct / 100, 4)

    @property
    def total_cost(self) -> float:
        return round(
            self.total_holding_cost
            + self.total_ordering_cost
            + self.total_lost_sales_cost,
            2,
        )

    def to_dict(self) -> dict:
        return {
            "policy": self.policy_name,
            "product": self.product,
            "total_days": self.total_days,
            "stockout_days": self.stockout_days,
            "stockout_pct": self.stockout_pct,
            "service_level": self.service_level,
            "lost_sales_units": self.lost_sales_units,
            "n_orders": self.n_orders,
            "holding_cost": round(self.total_holding_cost, 2),
            "ordering_cost": round(self.total_ordering_cost, 2),
            "lost_sales_cost": round(self.total_lost_sales_cost, 2),
            "total_cost": self.total_cost,
        }


# ── Core simulator ─────────────────────────────────────────────────────────────

def simulate_product(
    *,
    product_name: str,
    demand_forecast: np.ndarray,     # model's predicted demand, one per day
    true_demand: np.ndarray,         # ground-truth demand (used to satisfy customers)
    initial_stock: int,
    reorder_point: int,
    order_qty: int,
    lead_time_days: int,
    policy_name: str,
    holding_cost_per_unit: float = HOLDING_COST_PER_UNIT_PER_DAY,
    ordering_cost: float = ORDERING_COST_PER_ORDER,
    lost_sales_cost_per_unit: float = LOST_SALES_COST_PER_UNIT,
) -> SimulationResult:
    """Step-by-step inventory simulation for one product.

    Parameters
    ----------
    demand_forecast:
        The model's predicted demand for each day in the simulation window.
        This is used ONLY to decide replenishment orders.
    true_demand:
        Actual customer demand (from ground truth). Used to compute
        actual sales, stockouts, and lost sales.
    initial_stock, reorder_point, order_qty, lead_time_days:
        Replenishment parameters.
    policy_name:
        Label for this simulation run.

    Returns
    -------
    SimulationResult
    """
    n_days = len(demand_forecast)
    assert len(true_demand) == n_days, "forecast and demand arrays must have same length"

    result = SimulationResult(policy_name=policy_name, product=product_name)
    result.total_days = n_days

    on_hand = initial_stock
    inbound: dict[int, int] = {}  # day_index → units arriving that morning

    for day_idx in range(n_days):
        # ── Morning: receive deliveries ──────────────────────────────────────
        delivery = int(inbound.pop(day_idx, 0))
        opening_stock = on_hand + delivery

        # ── Serve customers using true demand ────────────────────────────────
        demand_today = int(true_demand[day_idx])
        sales = min(demand_today, opening_stock)
        lost = demand_today - sales
        closing_stock = opening_stock - sales

        # ── Track outcomes ───────────────────────────────────────────────────
        result.lost_sales_units += lost
        result.total_holding_cost += closing_stock * holding_cost_per_unit
        result.total_lost_sales_cost += lost * lost_sales_cost_per_unit
        if closing_stock == 0:
            result.stockout_days += 1

        # ── Replenishment decision using FORECAST (not true demand) ──────────
        pipeline = sum(inbound.values())
        current_position = closing_stock + pipeline

        # Standardized Base-Stock Policy across ALL policies (level playing field)
        # Policy A uses constant historical mean forecast; B/C/D use their respective models
        d_hat = max(float(demand_forecast[day_idx]), 0.5)
        lead_time_demand = d_hat * lead_time_days
        # Safety stock buffer based on demand variability: z = 1.645 for 95% service level SLA
        sigma_d = max(1.0, d_hat * 0.30)
        safety_stock = float(1.645 * np.sqrt(lead_time_days) * sigma_d)
        dynamic_rop = lead_time_demand + safety_stock
        target_stock = dynamic_rop + (d_hat * 4.0)  # cycle stock: 4-day coverage

        should_order = current_position <= dynamic_rop
        order_size = max(0, int(np.ceil(target_stock - current_position)))

        if should_order and order_size > 0:
            arrival_day = day_idx + lead_time_days
            if arrival_day < n_days:  # only order if delivery arrives before sim ends
                inbound[arrival_day] = inbound.get(arrival_day, 0) + order_size
                result.n_orders += 1
                result.total_ordering_cost += ordering_cost

        on_hand = closing_stock

    return result


# ── Multi-policy comparison ────────────────────────────────────────────────────

def simulate_policy(
    *,
    policy_name: str,
    demand_model: Any,
    df_test: pd.DataFrame,
    df_truth: pd.DataFrame,
    product_params: dict[str, dict],
) -> list[SimulationResult]:
    """Simulate a replenishment policy across all products on the test set.

    Parameters
    ----------
    policy_name:
        Label for this policy (e.g. ``"censoring_aware"``).
    demand_model:
        Fitted model with a ``.predict(df)`` method.
        Pass ``None`` to use a constant forecast (Policy A: no forecasting).
    df_test:
        Test-set DataFrame with features built (output of build_features).
    df_truth:
        Ground-truth DataFrame — used to get true_demand for the test period.
    product_params:
        {product_name: {"initial_stock": int, "reorder_point": int,
                         "order_qty": int, "lead_time_days": int}}

    Returns
    -------
    List of SimulationResult, one per product.
    """
    results = []
    products = sorted(df_test["product"].unique())

    for product in products:
        prod_test = df_test[df_test["product"] == product].sort_values("date")
        prod_truth = df_truth[df_truth["product"] == product].sort_values("date")

        # Align truth to test dates
        test_dates = set(prod_test["date"].astype(str))
        prod_truth = prod_truth[prod_truth["date"].astype(str).isin(test_dates)]

        true_demand = prod_truth["true_demand"].values.astype(float)

        # Get demand forecast
        if demand_model is None:
            # Policy A: no forecast — use historical mean as constant
            historical_mean = df_test[df_test["product"] == product]["sales"].mean()
            demand_forecast = np.full(len(prod_test), historical_mean)
        else:
            from src.train import predict_with_model
            demand_forecast = predict_with_model(demand_model, prod_test)

        params = product_params.get(
            product,
            {"initial_stock": 30, "reorder_point": 15, "order_qty": 30, "lead_time_days": 2},
        )

        result = simulate_product(
            product_name=product,
            demand_forecast=demand_forecast,
            true_demand=true_demand,
            policy_name=policy_name,
            **params,
        )
        results.append(result)

    return results


def compare_policies(
    all_results: dict[str, list[SimulationResult]],
) -> pd.DataFrame:
    """Aggregate results across all products for each policy.

    Parameters
    ----------
    all_results:
        {policy_name: [SimulationResult, ...]}

    Returns
    -------
    pd.DataFrame with one row per policy, aggregated across all products.
    """
    rows = []
    for policy_name, results in all_results.items():
        total_days = sum(r.total_days for r in results)
        stockout_days = sum(r.stockout_days for r in results)
        rows.append({
            "policy": policy_name,
            "stockout_pct": round(stockout_days / max(total_days, 1) * 100, 2),
            "service_level": round(1 - stockout_days / max(total_days, 1), 4),
            "lost_sales_units": sum(r.lost_sales_units for r in results),
            "holding_cost": round(sum(r.total_holding_cost for r in results), 2),
            "ordering_cost": round(sum(r.total_ordering_cost for r in results), 2),
            "lost_sales_cost": round(sum(r.total_lost_sales_cost for r in results), 2),
            "total_cost": round(sum(r.total_cost for r in results), 2),
            "n_orders": sum(r.n_orders for r in results),
        })

    df = pd.DataFrame(rows).sort_values("total_cost").reset_index(drop=True)
    # Mark the best policy
    if len(df) > 0:
        df["is_best"] = df["total_cost"] == df["total_cost"].min()
    return df


def get_product_params_from_simulator(df_observed: pd.DataFrame) -> dict[str, dict]:
    """Extract per-product simulation parameters from observed data.

    Uses the observed opening_stock patterns to infer reasonable
    initial_stock, reorder_point, order_qty, and lead_time_days.

    These are approximations since the true simulator parameters are
    embedded in src/simulate.py.
    """
    params = {}
    for product, group in df_observed.groupby("product"):
        avg_sales = group["sales"].mean()
        max_stock = group["opening_stock"].max()
        params[str(product)] = {
            "initial_stock": int(max_stock * 0.5),
            "reorder_point": int(avg_sales * 3),   # 3-day buffer
            "order_qty": int(avg_sales * 7),        # 1-week supply
            "lead_time_days": 2,
        }
    return params

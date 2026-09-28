"""
src/inventory/decision.py — Deterministic Replenishment Decision Contract.

Strict domain logic for inventory replenishment decisions.
Pure Python. Zero external API calls. Zero ML dependencies. Zero framework code.

Implements the fundamental supply-chain equation:
  inventory_position = on_hand + on_order + in_transit - reserved
  target_stock       = expected_demand_during_lead_time + safety_stock
  raw_shortage       = target_stock - inventory_position
  constrained_order  = max(ceil(raw_shortage / pack_size) * pack_size, MOQ)
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any


class ActionType(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    REVIEW = "REVIEW"


@dataclass(frozen=True)
class InventoryState:
    """Immutable snapshot of SKU inventory levels and supply constraints."""
    sku: str
    on_hand: int
    on_order: int = 0
    in_transit: int = 0
    reserved: int = 0
    lead_time_days: float = 3.0
    lead_time_std: float = 0.5
    moq: int = 1
    pack_size: int = 1
    unit_cost: float = 0.0
    unit_price: float = 0.0

    def __post_init__(self) -> None:
        if self.on_hand < 0:
            raise ValueError(f"on_hand cannot be negative, got {self.on_hand}")
        if self.on_order < 0:
            raise ValueError(f"on_order cannot be negative, got {self.on_order}")
        if self.in_transit < 0:
            raise ValueError(f"in_transit cannot be negative, got {self.in_transit}")
        if self.reserved < 0:
            raise ValueError(f"reserved cannot be negative, got {self.reserved}")
        if self.moq < 1:
            raise ValueError(f"moq must be >= 1, got {self.moq}")
        if self.pack_size < 1:
            raise ValueError(f"pack_size must be >= 1, got {self.pack_size}")
        if self.lead_time_days <= 0:
            raise ValueError(f"lead_time_days must be > 0, got {self.lead_time_days}")

    @property
    def inventory_position(self) -> int:
        """Effective stock available to cover demand until replenishment arrives."""
        return self.on_hand + self.on_order + self.in_transit - self.reserved


@dataclass(frozen=True)
class DemandForecast:
    """Probabilistic demand forecast over the replenishment horizon."""
    p10: float
    p50: float
    p90: float
    daily_run_rate: float
    daily_std: float

    def __post_init__(self) -> None:
        if self.p10 < 0 or self.p50 < 0 or self.p90 < 0:
            raise ValueError("Demand quantiles cannot be negative")
        if not (self.p10 <= self.p50 <= self.p90):
            raise ValueError(f"Quantile monotonicity violated: P10={self.p10}, P50={self.p50}, P90={self.p90}")
        if self.daily_run_rate < 0:
            raise ValueError(f"daily_run_rate cannot be negative, got {self.daily_run_rate}")


@dataclass(frozen=True)
class DecisionContract:
    """
    Auditable, defensible replenishment decision.
    Every recommendation must explain its inputs, constraints, and business risk.
    """
    sku: str
    action: ActionType
    recommended_order: int
    raw_shortage: int
    inventory_position: int
    target_stock: int
    safety_stock: int
    expected_lead_time_demand: float
    runway_days: float
    stockout_probability_pct: float
    financial_exposure_cost: float
    holding_cost_penalty: float
    constraints_applied: dict[str, Any]
    reasoning_chain: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sku": self.sku,
            "action": self.action.value,
            "recommended_order": self.recommended_order,
            "raw_shortage": self.raw_shortage,
            "inventory_position": self.inventory_position,
            "target_stock": self.target_stock,
            "safety_stock": self.safety_stock,
            "expected_lead_time_demand": round(self.expected_lead_time_demand, 2),
            "runway_days": round(self.runway_days, 1),
            "stockout_probability_pct": round(self.stockout_probability_pct, 1),
            "financial_exposure_cost": round(self.financial_exposure_cost, 2),
            "holding_cost_penalty": round(self.holding_cost_penalty, 2),
            "constraints_applied": self.constraints_applied,
            "reasoning_chain": self.reasoning_chain,
        }


class DecisionEngine:
    """
    Deterministic Inventory Replenishment Decision Engine.
    Evaluates InventoryState against DemandForecast under explicit business risk.
    """

    @staticmethod
    def compute_stochastic_safety_stock(
        lead_time_mean: float,
        lead_time_std: float,
        demand_mean: float,
        demand_std: float,
        service_factor_z: float = 1.645,  # 95% cycle service level
    ) -> int:
        """
        Calculates safety stock accounting for both demand variability AND supplier delay:
        SS = z * sqrt(LT_avg * sigma_d^2 + d_avg^2 * sigma_LT^2)
        """
        variance = (lead_time_mean * (demand_std ** 2)) + ((demand_mean ** 2) * (lead_time_std ** 2))
        return max(0, int(math.ceil(service_factor_z * math.sqrt(variance))))

    @classmethod
    def evaluate(
        cls,
        state: InventoryState,
        forecast: DemandForecast,
        service_factor_z: float = 1.645,
    ) -> DecisionContract:
        """
        Transforms inventory state and demand forecast into an auditable DecisionContract.
        """
        inv_pos = state.inventory_position
        lead_time = state.lead_time_days

        # 1. Expected demand during supplier lead time
        ltd_p50 = forecast.daily_run_rate * lead_time

        # 2. Stochastic safety stock
        ss = cls.compute_stochastic_safety_stock(
            lead_time_mean=lead_time,
            lead_time_std=state.lead_time_std,
            demand_mean=forecast.daily_run_rate,
            demand_std=forecast.daily_std,
            service_factor_z=service_factor_z,
        )

        # 3. Target stock & Raw shortage
        target_stock = int(math.ceil(ltd_p50 + ss))
        raw_shortage = target_stock - inv_pos

        # 4. Runway calculation
        runway = (state.on_hand / forecast.daily_run_rate) if forecast.daily_run_rate > 0 else 999.0

        # 5. Apply supplier constraints (Pack size & MOQ)
        reasons = [
            f"Expected demand during {lead_time:.1f}d lead time is {ltd_p50:.1f} units (P50: {forecast.p50:.1f}, P90: {forecast.p90:.1f}).",
            f"Inventory position is {inv_pos} ({state.on_hand} on-hand + {state.on_order} on-order + {state.in_transit} in-transit - {state.reserved} reserved).",
            f"Safety buffer of {ss} units required for 95% service level with supplier delay variance ±{state.lead_time_std:.1f}d.",
        ]

        constraints_applied = {
            "moq": state.moq,
            "pack_size": state.pack_size,
            "raw_shortage": raw_shortage,
            "lead_time_variance": f"±{state.lead_time_std:.1f}d",
        }

        if raw_shortage <= 0:
            # Healthy or Overstocked
            if runway > 30.0:
                action = ActionType.HOLD
                reasons.append(f"Excess inventory detected: {runway:.1f} days of cover exceeds 30-day operating limit.")
            else:
                action = ActionType.HOLD
                reasons.append(f"Inventory position of {inv_pos} units satisfies target stock of {target_stock} units.")
            recommended_order = 0
            stockout_prob = max(1.0, 100.0 - (runway / lead_time) * 50.0) if runway < lead_time * 2 else 2.0
            financial_exposure = 0.0
            # Holding cost penalty for stock above target
            excess_units = max(0, inv_pos - target_stock)
            holding_cost = excess_units * (state.unit_cost or state.unit_price * 0.6) * 0.20 / 365 * 14
        else:
            # Deficit exists — Order required
            action = ActionType.BUY
            # Must satisfy both MOQ and case pack:
            base_order = max(raw_shortage, state.moq)
            final_order = int(math.ceil(base_order / state.pack_size)) * state.pack_size
            recommended_order = final_order

            if state.moq > raw_shortage:
                reasons.append(f"Raw shortage of {raw_shortage} elevated to satisfy supplier MOQ of {state.moq} units.")
            if state.pack_size > 1 and final_order > base_order:
                reasons.append(f"Order rounded up to case pack multiple of {state.pack_size} -> {final_order} units.")

            reasons.append(f"Final approved order quantity: {final_order} units.")

            stockout_prob = min(98.0, max(25.0, (1.0 - (runway / max(0.1, lead_time))) * 100.0))
            # Financial exposure: lost margin over next 7 days if no order placed
            unit_margin = (state.unit_price - state.unit_cost) if state.unit_cost > 0 else (state.unit_price * 0.40)
            potential_unfulfilled_units = max(0.0, (forecast.daily_run_rate * 7) - state.on_hand)
            financial_exposure = potential_unfulfilled_units * unit_margin
            holding_cost = 0.0

        return DecisionContract(
            sku=state.sku,
            action=action,
            recommended_order=recommended_order,
            raw_shortage=raw_shortage,
            inventory_position=inv_pos,
            target_stock=target_stock,
            safety_stock=ss,
            expected_lead_time_demand=ltd_p50,
            runway_days=runway,
            stockout_probability_pct=stockout_prob,
            financial_exposure_cost=financial_exposure,
            holding_cost_penalty=holding_cost,
            constraints_applied=constraints_applied,
            reasoning_chain=reasons,
        )

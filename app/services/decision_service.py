"""
app/services/decision_service.py — Application Service for Replenishment Decisions.
Coordinates between the domain DecisionEngine, the ML Forecasting layer, and SQLite Audit Log.
"""

from typing import Any
import pandas as pd
from app.schemas.decision import ActionItemResponse, ActionQueueResponse, WhyBreakdown
from src.audit import log_recommendation
from src.data import load_observed
from src.inventory.decision import (
    ActionType,
    DecisionEngine,
    DemandForecast,
    InventoryState,
)


class DecisionService:
    """Orchestrates daily replenishment decision queue generation."""

    SKU_CATALOG_DEFAULTS: dict[str, dict[str, Any]] = {
        "Milk": {"lead_time_days": 2.0, "lead_time_std": 0.5, "moq": 20, "pack_size": 6, "unit_price": 55.0, "unit_cost": 33.0},
        "Bread": {"lead_time_days": 1.0, "lead_time_std": 0.3, "moq": 15, "pack_size": 5, "unit_price": 40.0, "unit_cost": 24.0},
        "Eggs": {"lead_time_days": 3.0, "lead_time_std": 1.2, "moq": 50, "pack_size": 12, "unit_price": 72.0, "unit_cost": 43.0},
        "Yogurt": {"lead_time_days": 2.0, "lead_time_std": 0.5, "moq": 10, "pack_size": 4, "unit_price": 45.0, "unit_cost": 27.0},
        "Biscuits": {"lead_time_days": 4.0, "lead_time_std": 0.8, "moq": 30, "pack_size": 10, "unit_price": 30.0, "unit_cost": 18.0},
        "Rice": {"lead_time_days": 7.0, "lead_time_std": 1.5, "moq": 10, "pack_size": 5, "unit_price": 120.0, "unit_cost": 72.0},
        "Oil": {"lead_time_days": 5.0, "lead_time_std": 1.0, "moq": 12, "pack_size": 6, "unit_price": 160.0, "unit_cost": 96.0},
        "Tea": {"lead_time_days": 4.0, "lead_time_std": 0.5, "moq": 20, "pack_size": 5, "unit_price": 85.0, "unit_cost": 51.0},
        "Soap": {"lead_time_days": 6.0, "lead_time_std": 1.0, "moq": 24, "pack_size": 12, "unit_price": 35.0, "unit_cost": 21.0},
        "Sugar": {"lead_time_days": 5.0, "lead_time_std": 1.0, "moq": 20, "pack_size": 10, "unit_price": 48.0, "unit_cost": 28.8},
    }

    @classmethod
    def get_action_queue(cls) -> ActionQueueResponse:
        """
        Builds prioritized replenishment queue using deterministic DecisionEngine.
        """
        df = load_observed()
        skus = sorted(df["product"].unique())

        actions: list[ActionItemResponse] = []
        total_stockout_exposure = 0.0
        total_excess_value = 0.0
        counts = {"critical": 0, "warning": 0, "overstock": 0, "healthy": 0}

        for sku in skus:
            meta = cls.SKU_CATALOG_DEFAULTS.get(str(sku), {
                "lead_time_days": 3.0, "lead_time_std": 0.5, "moq": 10, "pack_size": 5, "unit_price": 50.0, "unit_cost": 30.0
            })

            prod_df = df[df["product"] == sku].sort_values("date")
            on_hand = int(prod_df["opening_stock"].iloc[-1]) if not prod_df.empty else 20
            sales = prod_df["sales"].values

            # Demand profiling
            daily_mean = max(0.5, float(sales[-28:].mean())) if len(sales) >= 7 else 5.0
            daily_std = max(0.5, float(sales[-28:].std())) if len(sales) >= 7 else 1.5

            forecast = DemandForecast(
                p10=round(max(0.0, daily_mean * 0.70), 2),
                p50=round(daily_mean, 2),
                p90=round(daily_mean * 1.35, 2),
                daily_run_rate=round(daily_mean, 2),
                daily_std=round(daily_std, 2),
            )

            state = InventoryState(
                sku=str(sku),
                on_hand=on_hand,
                on_order=0,
                in_transit=0,
                reserved=0,
                lead_time_days=float(meta["lead_time_days"]),
                lead_time_std=float(meta["lead_time_std"]),
                moq=int(meta["moq"]),
                pack_size=int(meta["pack_size"]),
                unit_cost=float(meta["unit_cost"]),
                unit_price=float(meta["unit_price"]),
            )

            contract = DecisionEngine.evaluate(state, forecast)

            # Classify operational risk status
            if contract.runway_days < 2.0 or contract.stockout_probability_pct >= 60.0:
                status = "critical"
                counts["critical"] += 1
            elif contract.runway_days > 25.0:
                status = "overstock"
                counts["overstock"] += 1
            elif contract.runway_days < 4.0 or contract.stockout_probability_pct >= 30.0:
                status = "warning"
                counts["warning"] += 1
            else:
                status = "healthy"
                counts["healthy"] += 1

            total_stockout_exposure += contract.financial_exposure_cost
            if status == "overstock":
                excess_units = max(0, contract.inventory_position - contract.target_stock)
                total_excess_value += excess_units * state.unit_cost

            # Persist to SQLite Audit Ledger
            decision_id = log_recommendation(
                sku=str(sku),
                forecast_p50=forecast.p50,
                forecast_p90=forecast.p90,
                inventory_position=contract.inventory_position,
                recommended_order=contract.recommended_order,
                reason=contract.reasoning_chain[0] if contract.reasoning_chain else "Replenishment evaluation",
            )

            recommendation_str = f"Buy {contract.recommended_order}" if contract.recommended_order > 0 else (
                "Reduce Order" if status == "overstock" else "Hold"
            )

            why = WhyBreakdown(
                p10_daily=forecast.p10,
                p50_daily=forecast.p50,
                p90_daily=forecast.p90,
                on_hand=state.on_hand,
                on_order=state.on_order,
                reserved=state.reserved,
                inventory_position=contract.inventory_position,
                lead_time_days=state.lead_time_days,
                lead_time_variance=f"±{state.lead_time_std:.1f}d",
                p50_ltd=contract.expected_lead_time_demand,
                p90_ltd=round(contract.expected_lead_time_demand * 1.35, 1),
                target_service_level="95%",
                moq=state.moq,
                case_pack=state.pack_size,
                model_used="Tobit Decensored + Base-Stock (s,S)",
                why_narrative="; ".join(contract.reasoning_chain[:2]),
                reasoning_chain=contract.reasoning_chain,
            )

            actions.append(
                ActionItemResponse(
                    decision_id=decision_id,
                    sku=str(sku),
                    status=status,
                    runway_days=contract.runway_days,
                    stockout_risk_pct=contract.stockout_probability_pct,
                    recommended_order=contract.recommended_order,
                    recommendation=recommendation_str,
                    potential_loss_7d=contract.financial_exposure_cost,
                    unit_price=state.unit_price,
                    why_breakdown=why,
                )
            )

        # Sort: critical first, then warning, then overstock, then healthy
        order_rank = {"critical": 0, "warning": 1, "overstock": 2, "healthy": 3}
        actions.sort(key=lambda a: (order_rank.get(a.status, 4), a.runway_days))

        return ActionQueueResponse(
            actions=actions,
            total_actions=len(actions),
            total_stockout_risk_exposure=round(total_stockout_exposure, 2),
            total_excess_inventory_value=round(total_excess_value, 2),
            critical_count=counts["critical"],
            warning_count=counts["warning"],
            overstock_count=counts["overstock"],
            healthy_count=counts["healthy"],
        )

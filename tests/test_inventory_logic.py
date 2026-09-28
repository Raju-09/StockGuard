"""tests/test_inventory_logic.py — MOQ, case-pack, and inventory constraint tests."""

import pytest
from src.predict import _compute_recommended_order


class TestMOQEnforcement:
    """Minimum Order Quantity must be enforced when order > 0."""

    def test_moq_enforced_when_order_positive(self):
        """If raw order is 5 but MOQ is 50, final order should be 50."""
        order, _, _ = _compute_recommended_order(
            forecast_demand=3.0,
            current_inventory=0,
            lead_time_days=2,
            moq=50,
            case_pack_size=1,
        )
        assert order >= 50, f"MOQ violated: order={order}, MOQ=50"

    def test_moq_not_applied_when_zero_order(self):
        """If inventory is huge, order should be 0 regardless of MOQ."""
        order, _, _ = _compute_recommended_order(
            forecast_demand=3.0,
            current_inventory=10000,
            lead_time_days=2,
            moq=50,
            case_pack_size=1,
        )
        assert order == 0, f"Should be 0 when stock is high, got {order}"

    def test_default_moq_is_1(self):
        """Default MOQ=1 should not change existing behavior."""
        order_default, _, _ = _compute_recommended_order(
            forecast_demand=10.0,
            current_inventory=5,
            lead_time_days=2,
        )
        order_explicit, _, _ = _compute_recommended_order(
            forecast_demand=10.0,
            current_inventory=5,
            lead_time_days=2,
            moq=1,
            case_pack_size=1,
        )
        assert order_default == order_explicit


class TestCasePackSize:
    """Orders must be rounded up to case pack multiples."""

    def test_case_pack_rounding(self):
        """If case_pack_size=6, order should be a multiple of 6."""
        order, _, _ = _compute_recommended_order(
            forecast_demand=10.0,
            current_inventory=0,
            lead_time_days=2,
            moq=1,
            case_pack_size=6,
        )
        assert order > 0
        assert order % 6 == 0, f"Order {order} is not a multiple of case pack 6"

    def test_case_pack_and_moq_together(self):
        """Both constraints applied: round to pack, then enforce MOQ."""
        order, _, _ = _compute_recommended_order(
            forecast_demand=2.0,
            current_inventory=0,
            lead_time_days=1,
            moq=24,
            case_pack_size=6,
        )
        assert order >= 24, f"MOQ violated: order={order}"
        assert order % 6 == 0, f"Case pack violated: order={order}"

    def test_order_non_negative(self):
        """Order is never negative regardless of constraints."""
        for inv in [0, 10, 100, 1000]:
            order, _, _ = _compute_recommended_order(
                forecast_demand=5.0,
                current_inventory=inv,
                lead_time_days=2,
                moq=10,
                case_pack_size=6,
            )
            assert order >= 0, f"Negative order: {order}"

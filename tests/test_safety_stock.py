import pytest
from src.inventory.safety_stock import SafetyStockCalculator, StockParameters

def test_safety_stock_constant_lead_time():
    params = StockParameters(
        avg_daily_demand=100.0,
        std_daily_demand=15.0,
        avg_lead_time_days=4.0,
        std_lead_time_days=0.0,
        service_level=0.95
    )
    ss = SafetyStockCalculator.calculate_safety_stock(params)
    assert ss == pytest.approx(1.65 * 30.0, rel=1e-2)

def test_reorder_point_calculation():
    params = StockParameters(
        avg_daily_demand=50.0,
        std_daily_demand=10.0,
        avg_lead_time_days=5.0,
        std_lead_time_days=1.0,
        service_level=0.95
    )
    rop = SafetyStockCalculator.calculate_reorder_point(params)
    assert rop > (50.0 * 5.0)

def test_z_score_boundary():
    assert SafetyStockCalculator.get_z_score(0.95) == 1.65
    assert SafetyStockCalculator.get_z_score(0.99) == 2.33

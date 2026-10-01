from src.cost_evaluator import InventoryCostEvaluator

def test_cost_calculation():
    cost = InventoryCostEvaluator.calculate_total_costs(2.0, 50.0, 10.0, 5)
    assert cost == 150.0

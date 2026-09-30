class InventoryCostEvaluator:
    @staticmethod
    def calculate_total_costs(holding_cost_per_unit: float, avg_inventory: float, stockout_penalty_per_unit: float, lost_units: int) -> float:
        holding_total = holding_cost_per_unit * avg_inventory
        stockout_total = stockout_penalty_per_unit * lost_units
        return round(holding_total + stockout_total, 2)

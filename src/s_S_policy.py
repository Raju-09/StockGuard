from dataclasses import dataclass

@dataclass
class PeriodicPolicyParams:
    reorder_level_s: float
    order_up_to_level_S: float

    def should_order(self, current_inventory: float) -> bool:
        return current_inventory <= self.reorder_level_s

    def compute_order_quantity(self, current_inventory: float) -> float:
        if self.should_order(current_inventory):
            return self.order_up_to_level_S - current_inventory
        return 0.0

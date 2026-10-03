from typing import Dict, Any

class ReplenishmentAPIRouter:
    @staticmethod
    def get_recommendation(sku_id: str, current_stock: float, reorder_point: float, batch_size: float) -> Dict[str, Any]:
        needs_order = current_stock <= reorder_point
        return {
            "sku_id": sku_id,
            "current_stock": current_stock,
            "reorder_point": reorder_point,
            "order_recommended": needs_order,
            "recommended_quantity": batch_size if needs_order else 0.0
        }

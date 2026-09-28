from typing import List, Dict, Any

class InventorySnapshotLoader:
    @staticmethod
    def validate_record(record: Dict[str, Any]) -> bool:
        required = ["sku_id", "timestamp", "stock_level", "warehouse_id"]
        return all(k in record for k in required) and record["stock_level"] >= 0

from src.snapshot_loader import InventorySnapshotLoader

def test_snapshot_loader_valid():
    rec = {"sku_id": "SKU_01", "timestamp": "2026-09-28", "stock_level": 45, "warehouse_id": "WH_A"}
    assert InventorySnapshotLoader.validate_record(rec) is True

def test_snapshot_loader_invalid():
    rec = {"sku_id": "SKU_01", "stock_level": -5}
    assert InventorySnapshotLoader.validate_record(rec) is False

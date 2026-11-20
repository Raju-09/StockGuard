from src.monitoring_metrics import InventoryMetricsCollector

def test_metrics_counter_increment():
    collector = InventoryMetricsCollector()
    collector.increment("stockouts_prevented", 5)
    assert collector._counters["stockouts_prevented"] == 5

def test_metrics_prometheus_export():
    collector = InventoryMetricsCollector()
    collector.increment("orders_placed", 12)
    collector.set_gauge("inventory_turnover_ratio", 5.1)
    text = collector.export_prometheus_format()
    assert "stockguard_orders_placed 12" in text
    assert "stockguard_inventory_turnover_ratio 5.1" in text

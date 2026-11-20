import time
from typing import Dict, Any

class InventoryMetricsCollector:
    def __init__(self):
        self._counters: Dict[str, int] = {
            "stockouts_prevented": 0,
            "orders_placed": 0,
            "audit_anomalies_detected": 0
        }
        self._gauges: Dict[str, float] = {
            "average_service_level": 0.95,
            "inventory_turnover_ratio": 4.2
        }

    def increment(self, metric_name: str, count: int = 1) -> None:
        if metric_name in self._counters:
            self._counters[metric_name] += count
        else:
            self._counters[metric_name] = count

    def set_gauge(self, metric_name: str, value: float) -> None:
        self._gauges[metric_name] = value

    def export_prometheus_format(self) -> str:
        lines = []
        for k, v in self._counters.items():
            lines.append(f"# TYPE stockguard_{k} counter")
            lines.append(f"stockguard_{k} {v}")
        for k, v in self._gauges.items():
            lines.append(f"# TYPE stockguard_{k} gauge")
            lines.append(f"stockguard_{k} {v}")
        return "\n".join(lines) + "\n"

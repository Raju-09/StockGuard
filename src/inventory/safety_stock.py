from dataclasses import dataclass
import math
from typing import Dict, Optional

@dataclass
class StockParameters:
    avg_daily_demand: float
    std_daily_demand: float
    avg_lead_time_days: float
    std_lead_time_days: float
    service_level: float = 0.95  # 95% default

class SafetyStockCalculator:
    # Standard normal distribution Z-scores
    Z_TABLE = {
        0.80: 0.84,
        0.85: 1.04,
        0.90: 1.28,
        0.95: 1.65,
        0.98: 2.05,
        0.99: 2.33,
        0.999: 3.09
    }

    @classmethod
    def get_z_score(cls, service_level: float) -> float:
        closest = min(cls.Z_TABLE.keys(), key=lambda k: abs(k - service_level))
        return cls.Z_TABLE[closest]

    @classmethod
    def calculate_safety_stock(cls, params: StockParameters) -> float:
        z = cls.get_z_score(params.service_level)
        variance_demand = (params.std_daily_demand ** 2) * params.avg_lead_time_days
        variance_lead_time = (params.std_lead_time_days ** 2) * (params.avg_daily_demand ** 2)
        combined_std = math.sqrt(variance_demand + variance_lead_time)
        return round(z * combined_std, 2)

    @classmethod
    def calculate_reorder_point(cls, params: StockParameters) -> float:
        lead_time_demand = params.avg_daily_demand * params.avg_lead_time_days
        safety_stock = cls.calculate_safety_stock(params)
        return round(lead_time_demand + safety_stock, 2)

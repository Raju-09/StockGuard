from typing import Dict

class CategoryPriorEstimator:
    CATEGORY_DEFAULTS = {
        "electronics": 25.0,
        "apparel": 120.0,
        "groceries": 450.0,
        "home": 40.0
    }

    @classmethod
    def get_initial_daily_demand(cls, category: str) -> float:
        return cls.CATEGORY_DEFAULTS.get(category.lower(), 50.0)

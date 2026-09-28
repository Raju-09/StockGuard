import math
from typing import List

class RollingDemandFeatureExtractor:
    @staticmethod
    def rolling_std(series: List[float], window: int = 7) -> List[float]:
        res = []
        for i in range(len(series)):
            sub = series[max(0, i - window + 1): i + 1]
            if len(sub) < 2:
                res.append(0.0)
            else:
                mean = sum(sub) / len(sub)
                var = sum((x - mean) ** 2 for x in sub) / (len(sub) - 1)
                res.append(round(math.sqrt(var), 3))
        return res

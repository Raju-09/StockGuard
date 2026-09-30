import random

class StochasticLeadTimeSampler:
    @staticmethod
    def sample_lead_time_days(mean_days: float, cv: float = 0.2) -> int:
        # Generate positive bounded stochastic lead time
        variance = (mean_days * cv) ** 2
        std = variance ** 0.5
        val = random.gauss(mean_days, std)
        return max(1, int(round(val)))

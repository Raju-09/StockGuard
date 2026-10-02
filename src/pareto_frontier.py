from typing import List, Tuple

class ParetoFrontierEvaluator:
    @staticmethod
    def find_pareto_optimal(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        # Points are (holding_cost, 1 - service_level) where both should be minimized
        sorted_points = sorted(points)
        frontier = []
        min_y = float("inf")
        for x, y in sorted_points:
            if y < min_y:
                frontier.append((x, y))
                min_y = y
        return frontier

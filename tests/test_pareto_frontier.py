from src.pareto_frontier import ParetoFrontierEvaluator

def test_pareto_optimal():
    candidates = [(10, 0.1), (20, 0.05), (15, 0.2)]
    optimal = ParetoFrontierEvaluator.find_pareto_optimal(candidates)
    assert (10, 0.1) in optimal
    assert (20, 0.05) in optimal
    assert (15, 0.2) not in optimal

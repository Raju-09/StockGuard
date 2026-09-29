from src.category_prior import CategoryPriorEstimator

def test_category_prior():
    assert CategoryPriorEstimator.get_initial_daily_demand("groceries") == 450.0
    assert CategoryPriorEstimator.get_initial_daily_demand("unknown_category") == 50.0

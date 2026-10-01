from src.s_S_policy import PeriodicPolicyParams

def test_periodic_policy():
    policy = PeriodicPolicyParams(reorder_level_s=20.0, order_up_to_level_S=100.0)
    assert policy.should_order(15.0) is True
    assert policy.compute_order_quantity(15.0) == 85.0
    assert policy.should_order(25.0) is False

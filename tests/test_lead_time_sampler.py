from src.lead_time_sampler import StochasticLeadTimeSampler

def test_sample_lead_time():
    sample = StochasticLeadTimeSampler.sample_lead_time_days(mean_days=5.0)
    assert sample >= 1

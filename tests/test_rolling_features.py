from src.rolling_features import RollingDemandFeatureExtractor

def test_rolling_std_constant():
    data = [10.0] * 10
    stds = RollingDemandFeatureExtractor.rolling_std(data, window=5)
    assert stds[-1] == 0.0

def test_rolling_std_fluctuating():
    data = [10.0, 20.0, 10.0, 20.0, 10.0]
    stds = RollingDemandFeatureExtractor.rolling_std(data, window=3)
    assert stds[-1] > 0.0

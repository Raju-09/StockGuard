"""
tests/test_train.py — Unit tests for temporal split, models, and registry.
"""

from src.data import load_observed
from src.features import build_features
from src.train import (
    CensoringAwareModel,
    MovingAverageForecast,
    NaiveForecast,
    load_model,
    save_model,
    temporal_split,
)


def test_temporal_split_chronological():
    df = load_observed()
    train, val, test = temporal_split(df, train_frac=0.70, val_frac=0.10)
    assert len(train) + len(val) + len(test) == len(df)
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()


def test_naive_and_ma_models():
    df = build_features(load_observed())
    train, _, test = temporal_split(df, 0.7, 0.1)

    naive = NaiveForecast().fit(train)
    preds = naive.predict(test)
    assert len(preds) == len(test)

    ma7 = MovingAverageForecast(7).fit(train)
    preds7 = ma7.predict(test)
    assert len(preds7) == len(test)


def test_censoring_aware_model_fit_predict():
    df = build_features(load_observed())
    train, val, test = temporal_split(df, 0.7, 0.1)

    model = CensoringAwareModel()
    model.fit(train, val)
    assert model.n_imputed > 0

    preds = model.predict(test)
    assert len(preds) == len(test)
    assert (preds >= 0).all()


def test_model_registry_save_and_load(tmp_path):
    df = build_features(load_observed())
    train, _, _ = temporal_split(df, 0.7, 0.1)

    model = NaiveForecast().fit(train)
    path = save_model(model, "test_naive", {"mae": 1.23}, models_dir=tmp_path)
    assert path.exists()

    loaded = load_model("test_naive", models_dir=tmp_path)
    assert isinstance(loaded, NaiveForecast)

"""
# ruff: noqa: E402
scripts/train_all.py — Orchestrate the full training pipeline.

Run this once to train all models and produce the business simulator results:

    python scripts/train_all.py

What it does:
    1. Load and validate the simulated dataset.
    2. Build features (lag, rolling, calendar).
    3. Temporal split: 70% train / 10% val / 20% test.
    4. Train and evaluate all demand-forecasting models:
           naive, ma7, ma14, linear, random_forest,
           gradient_boost, censoring_aware
    5. Train and evaluate the stockout classifier.
    6. Print a model comparison table.
    7. Run the 4-policy business simulator on the test set.
    8. Save all models to models/ with version numbers.
    9. Save evaluation results to models/results.json.
    10. Save simulation results to models/simulation_results.json.

No arguments needed. All paths default to the project root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src/ is importable regardless of working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_observed, load_with_ground_truth, summary_stats
from src.evaluate import (
    compare_models,
    evaluation_report,
    format_report,
    stockout_classifier_metrics,
)
from src.features import (
    add_stockout_future_label,
    build_features,
)
from src.simulator import (
    compare_policies,
    get_product_params_from_simulator,
    simulate_policy,
)
from src.train import (
    DEFAULT_MODELS_DIR,
    CensoringAwareModel,
    MovingAverageForecast,
    NaiveForecast,
    StockoutClassifier,
    build_model,
    predict_with_model,
    save_model,
    temporal_split,
)

RESULTS_PATH = DEFAULT_MODELS_DIR / "results.json"
SIM_RESULTS_PATH = DEFAULT_MODELS_DIR / "simulation_results.json"


def _banner(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def _section(text: str) -> None:
    print(f"\n-- {text} {'-' * (56 - len(text))}")


def main() -> None:
    _banner("StockGuard — Full Training Pipeline")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    _section("Loading data")
    df_observed = load_observed()
    df_truth = load_with_ground_truth()
    print(f"  Observed : {df_observed.shape}")
    print(f"  Truth    : {df_truth.shape}")
    print(f"  Products : {sorted(df_observed['product'].unique())}")
    print(f"  Date range: {df_observed['date'].min().date()} -> {df_observed['date'].max().date()}")

    # Summary stats
    stats = summary_stats(df_truth)
    print("\n  Per-product stockout summary:")
    print(stats[["product", "avg_daily_sales", "stockout_days", "stockout_pct"]].to_string(index=False))

    # ── 2. Build features ─────────────────────────────────────────────────────
    _section("Building features")
    df_featured_obs = build_features(df_observed)
    # Merge is_censored and true_demand from truth for evaluation only
    eval_cols = ["date", "product", "true_demand", "is_censored", "lost_sales"]
    df_featured = df_featured_obs.merge(
        df_truth[eval_cols],
        on=["date", "product"],
        how="left",
    )
    print(f"  Feature columns: {[c for c in df_featured.columns if c not in df_truth.columns][:5]}...")
    print(f"  Rows with lag_1 NaN (first day/product): {df_featured['lag_1'].isna().sum()}")

    # ── 3. Temporal split ─────────────────────────────────────────────────────
    _section("Temporal split (70/10/20)")
    train, val, test = temporal_split(df_featured, train_frac=0.70, val_frac=0.10)
    print(f"  Train : {len(train)} rows | dates {train['date'].min().date()} -> {train['date'].max().date()}")
    print(f"  Val   : {len(val)} rows | dates {val['date'].min().date()} -> {val['date'].max().date()}")
    print(f"  Test  : {len(test)} rows | dates {test['date'].min().date()} -> {test['date'].max().date()}")

    # Verify no date leakage between splits
    assert not (set(train["date"]) & set(val["date"])), "LEAKAGE: train/val date overlap!"
    assert not (set(train["date"]) & set(test["date"])), "LEAKAGE: train/test date overlap!"
    assert not (set(val["date"]) & set(test["date"])), "LEAKAGE: val/test date overlap!"
    print("  [OK] No date overlap between splits.")

    # Verify temporal ordering
    assert train["date"].max() < val["date"].min(), "LEAKAGE: train extends past val start!"
    assert val["date"].max() < test["date"].min(), "LEAKAGE: val extends past test start!"
    print("  [OK] Temporal ordering verified.")

    # ── 4. Train demand models ────────────────────────────────────────────────
    _section("Training demand forecasting models")
    model_names = [
        "naive", "ma7", "ma14",
        "linear", "random_forest", "gradient_boost",
        "censoring_aware",
    ]

    trained_models: dict[str, object] = {}
    all_reports: dict[str, dict] = {}

    for name in model_names:
        print(f"\n  [{name}]")
        model = build_model(name)

        # Fit
        if isinstance(model, (NaiveForecast, MovingAverageForecast)):
            model.fit(train)
        elif isinstance(model, CensoringAwareModel):
            train_with_stockout = train  # stockout_flag is already in train
            model.fit(train_with_stockout, val)
            print(f"    Imputed demand on {model.n_imputed} censored training rows.")
        else:
            model.fit(train, val)

        # Predict on test set
        test_preds = predict_with_model(model, test)

        # Evaluate against TRUE demand (not sales — that would hide censoring effect)
        test_with_preds = test.copy()
        test_with_preds["predicted_demand"] = test_preds

        report = evaluation_report(
            test_with_preds,
            actual_col="true_demand",
            predicted_col="predicted_demand",
            censored_col="is_censored",
        )
        all_reports[name] = report

        print(format_report(report))

        # Save model
        path = save_model(
            model,
            model_name=name,
            metrics={k: round(v, 4) if isinstance(v, float) else v
                     for k, v in report.items()},
        )
        print(f"    Saved -> {path.name}")
        trained_models[name] = model

    # ── 5. Model comparison table ─────────────────────────────────────────────
    _section("Model comparison (test set, sorted by MAE)")
    comparison = compare_models(all_reports, primary_metric="mae_all")
    display_cols = [
        "model", "n_all", "mae_all", "mae_censored", "mae_uncensored",
        "wape_all", "wape_censored",
    ]
    display_cols = [c for c in display_cols if c in comparison.columns]
    print(comparison[display_cols].to_string(index=False))

    # ── 6. Train stockout classifier ──────────────────────────────────────────
    _section("Training stockout classifier")
    # Add future stockout label (uses shift to look forward — safe for training)
    train_with_label = add_stockout_future_label(train, horizon=3)
    val_with_label = add_stockout_future_label(val, horizon=3)
    test_with_label = add_stockout_future_label(test, horizon=3)

    clf = StockoutClassifier(horizon=3)
    clf.fit(train_with_label, val_with_label)

    # Evaluate on test
    clf_preds = clf.predict(test_with_label)
    clf_probs = clf.predict_proba(test_with_label)
    clf_metrics = stockout_classifier_metrics(
        test_with_label["future_stockout_3d"].values,
        clf_preds,
        y_prob=clf_probs,
    )
    print(f"  Precision: {clf_metrics['precision']:.3f}")
    print(f"  Recall   : {clf_metrics['recall']:.3f}")
    print(f"  F1       : {clf_metrics['f1']:.3f}")
    if "pr_auc" in clf_metrics:
        print(f"  PR-AUC   : {clf_metrics['pr_auc']:.3f}")

    save_model(clf, "stockout_classifier", metrics=clf_metrics)
    print("  Saved -> stockout_classifier_v1.joblib")

    # ── 7. Business simulator ─────────────────────────────────────────────────
    _section("Running business simulator (test set)")
    product_params = get_product_params_from_simulator(df_observed)

    # We need test rows from df_featured_obs (no ground truth cols) for feature dispatch
    test_obs_only = test[
        [c for c in test.columns
         if c not in ["true_demand", "is_censored", "lost_sales", "demand_mean"]]
    ]

    policies: dict[str, object | None] = {
        "A_no_forecast": None,
        "B_naive": trained_models["naive"],
        "C_gradient_boost": trained_models["gradient_boost"],
        "D_censoring_aware": trained_models["censoring_aware"],
    }

    all_sim_results = {}
    for policy_name, model in policies.items():
        results = simulate_policy(
            policy_name=policy_name,
            demand_model=model,
            df_test=test_obs_only,
            df_truth=df_truth,
            product_params=product_params,
        )
        all_sim_results[policy_name] = results

    sim_comparison = compare_policies(all_sim_results)
    print("\n  Business simulation results:")
    print(
        sim_comparison[
            ["policy", "stockout_pct", "service_level",
             "lost_sales_units", "holding_cost", "lost_sales_cost", "total_cost"]
        ].to_string(index=False)
    )

    # ── 8. Save results ────────────────────────────────────────────────────────
    _section("Saving results")
    DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Evaluation results
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {k: {kk: float(vv) if isinstance(vv, float) else vv
                 for kk, vv in v.items()}
             for k, v in all_reports.items()},
            f, indent=2,
        )
    print(f"  Evaluation results -> {RESULTS_PATH.name}")

    # Simulation results
    sim_records = sim_comparison.to_dict(orient="records")
    with open(SIM_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(sim_records, f, indent=2, default=str)
    print(f"  Simulation results -> {SIM_RESULTS_PATH.name}")

    _banner("Training complete. All models saved to models/")


if __name__ == "__main__":
    main()

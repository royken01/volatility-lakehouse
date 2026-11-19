"""Train and evaluate the volatility models: one MLflow run per model, using the walk-forward of
``walk_forward.py``.  The target and lag columns follow the settings (``--post-window-seconds``,
``--lags`` or the config file), so the same command trains a 30-second or a 5-minute model.

Start the MLflow server first (``vol-lakehouse mlflow``).  ``vol-lakehouse train`` equals
``python -m lakehouse.modelling.train``::

    python -m lakehouse.modelling.train --model rf                  # variants A, C0, C for the last 4 years
    python -m lakehouse.modelling.train --model all --register --publish
    python -m lakehouse.modelling.train --model arimax --params '{"ar_order": 3, "ma_order": 3}'
    python -m lakehouse.modelling.train --model xgb --params '{"grid": {"max_depth": [3, 4]}}'
    python -m lakehouse.modelling.train --model rf --feature-set base_seq
    python -m lakehouse.modelling.train --model rf --news-features news_sent_ewma news_count_24h
    python -m lakehouse.modelling.train --model rf --no-news        # tier A only, scored on all events
    python -m lakehouse.modelling.train --model rf --target-scale raw   # model y itself, not ln y
    python -m lakehouse.modelling.train --model rf --test-years 2021 2022 2023 2024
    python -m lakehouse.modelling.train --model rf --snapshot-id 1234567890   # reproduce an old run

Parameters, metrics, plots and the refit models are logged to MLflow.  Predictions and metrics are
also staged as parquet in ``data/results_staging`` for ``publish_results.py`` (``--publish``
publishes right away). MLflow metrics: ``<variant>_<MAE|WMAPE|MSE>`` (pooled),
``fold_<variant>_<metric>`` (per test year); the unprefixed ones belong to the report variant (C
with news, otherwise A).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression

from lakehouse.feature_columns import NEWS_SENTIMENT_COLUMNS, SEQUENCE_COLUMNS, TARGET_COLUMN
from lakehouse.infrastructure.mlflow_config import REGISTERED_MODEL_PREFIX, init_mlflow
from lakehouse.infrastructure.spark_session import GOLD_SCHEMA
from lakehouse.modelling.feature_sets import (
    BENCHMARK_FEATURE_SETS, DEFAULT_FEATURE_SET, FEATURE_SETS, resolve_feature_set)
from lakehouse.modelling.gold_data import (
    DEFAULT_TEST_YEAR_COUNT, SOURCE_TABLE, default_test_years, load_gold_events, prepare_features)
from lakehouse.modelling.losses import TARGET_SCALES, floor_value, smearing_factor, to_model_target
from lakehouse.modelling.ml_models import (
    build_model, feature_importance, final_estimator, make_news_correction, tuned_settings)
from lakehouse.modelling.model_params import BENCHMARK_MODELS, DEFAULT_PARAMS, MODEL_NAMES, SEED, SEQUENCE_MODELS
from lakehouse.modelling.walk_forward import METRIC_NAMES, frame_metrics, two_stage_walk_forward, variant_comparison
from lakehouse.paths import PACKAGE_DIR, RESULTS_STAGING_DIR
from lakehouse.plot_style import BLUE, ORANGE, apply_default_style, clean_axes, save_figure
from lakehouse.settings import get_settings

WINDOW_LABEL = f"{get_settings().post_window_seconds} s post-event"

# Files copied into every logged model so it can be loaded outside the repository (relative to the package folder).
FILES_SHIPPED_WITH_MODELS = ["__init__.py", "feature_columns.py", "settings.py", "paths.py", "modelling/__init__.py",
                             "modelling/ml_models.py", "modelling/model_params.py", "modelling/lstm_model.py", "modelling/sarimax_model.py",
                             "modelling/losses.py"]


def git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PACKAGE_DIR,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def bundle_code_for_mlflow(destination: Path) -> str:
    """Copy the modules a logged model needs into ``destination/lakehouse/...`` and return that
    folder.
    """
    package_folder = destination / "lakehouse"
    for relative_path in FILES_SHIPPED_WITH_MODELS:
        (package_folder / relative_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PACKAGE_DIR / relative_path, package_folder / relative_path)
    return str(package_folder)


def log_model_to_mlflow(mlflow, model, name: str = "model", **keyword_arguments):
    """Log a fitted pipeline"""
    with tempfile.TemporaryDirectory() as temporary_directory:
        keyword_arguments.setdefault("code_paths", [bundle_code_for_mlflow(Path(temporary_directory))])
        if type(final_estimator(model)).__name__ == "LSTMRegressor":
            return mlflow.sklearn.log_model(sk_model=model, name=name, serialization_format="cloudpickle",
                                            **keyword_arguments)
        try:
            import skops.io as skops_io

            trusted_types = skops_io.get_untrusted_types(data=skops_io.dumps(model))
            return mlflow.sklearn.log_model(sk_model=model, name=name, serialization_format="skops",
                                            skops_trusted_types=trusted_types, **keyword_arguments)
        except Exception as error:
            print(f"  skops serialisation not possible ({type(error).__name__}); using cloudpickle")
            return mlflow.sklearn.log_model(sk_model=model, name=name, serialization_format="cloudpickle",
                                            **keyword_arguments)


def variant_feature_set_label(feature_set_label: str, variant: str) -> str:
    """Label stored with published results"""
    return feature_set_label if variant == "A" else f"{feature_set_label}|{variant}"


def plot_forecast_vs_realized(predictions: pd.DataFrame, path: Path, title: str) -> None:
    apply_default_style()
    figure, axis = plt.subplots(figsize=(6, 6))
    for (event_type, group), color in zip(predictions.groupby("event_type"), (BLUE, ORANGE, "grey")):
        axis.scatter(group["y_true"], group["y_pred"], s=10, alpha=0.6, color=color, label=event_type)
    limit = float(np.nanmax([predictions["y_true"].max(), predictions["y_pred"].max()])) * 1.05
    axis.plot([0, limit], [0, limit], "k--", lw=1, label="perfect forecast")
    axis.set(xlim=(0, limit), ylim=(min(0, float(predictions["y_pred"].min())), limit),
             xlabel=f"Realized vol ({WINDOW_LABEL})", ylabel="Forecast vol", title=title)
    clean_axes(axis, grid="y")
    axis.legend()
    save_figure(figure, path, dpi=150)


def plot_forecast_over_time(predictions: pd.DataFrame, path: Path, title: str) -> None:
    apply_default_style()
    figure, axis = plt.subplots(figsize=(11, 4))
    ordered = predictions.sort_values("ts_utc")
    axis.plot(ordered["ts_utc"], ordered["y_true"], lw=0.8, color=BLUE, label="realized")
    axis.plot(ordered["ts_utc"], ordered["y_pred"], lw=0.8, color=ORANGE, label="forecast")
    axis.set(ylabel=f"Vol ({WINDOW_LABEL})", title=title)
    clean_axes(axis, grid="y")
    axis.legend()
    save_figure(figure, path, dpi=150)


def plot_importance(importance: pd.DataFrame, path: Path, title: str) -> None:
    apply_default_style()
    figure, axis = plt.subplots(figsize=(7, 0.35 * len(importance) + 1))
    ordered = importance.sort_values("value")
    axis.barh(ordered["feature"], ordered["value"], color=BLUE)
    axis.set(xlabel=importance["kind"].iloc[0], title=title)
    clean_axes(axis, grid="x")
    save_figure(figure, path, dpi=150)


def run_model(mlflow, model_name: str, gold_events: pd.DataFrame, snapshot_id: int | None, test_years: list[int],
              parameter_overrides: dict | None = None, *, feature_set: str = DEFAULT_FEATURE_SET,
              register: bool = False, run_tags: dict | None = None, stage_results: bool = True,
              log_model: bool = True, nested: bool = False, verbose: bool = True,
              news_features: list[str] | None = None, jobs: int = 1,
              target_scale: str = "log") -> dict:
    """Walk-forward evaluation of one model inside one MLflow run.

        news_features: stage-2 columns.  None = all news columns, [] = tier A only.  Benchmarks
        never get a news stage.
        target_scale: "log" (default) or "raw" (label '...+raw'); see losses.py.
        jobs: processes for the tier A blocks (1 = sequential, -1 = all cores).

        Returns the results dict that select_features.py reads.
    """
    from mlflow.models import infer_signature

    if model_name in BENCHMARK_FEATURE_SETS:
        feature_set = BENCHMARK_FEATURE_SETS[model_name]
    sequence_columns = list(SEQUENCE_COLUMNS) if model_name in SEQUENCE_MODELS else []   # lags for LSTM / ARIMAX
    feature_set_label, feature_columns = resolve_feature_set(feature_set, sequence_columns)
    if target_scale == "raw":
        feature_set_label += "+raw"
    raw_target = target_scale == "raw"
    if model_name in BENCHMARK_MODELS:
        news_features = []
    elif news_features is None:
        news_features = list(NEWS_SENTIMENT_COLUMNS)
    report_variant = "C" if news_features else "A"

    events = prepare_features(gold_events, feature_columns, news_features)
    overrides = dict(parameter_overrides or {})
    parameters = {**DEFAULT_PARAMS[model_name], **overrides}
    trained_at = datetime.now(timezone.utc)
    run_name = f"{model_name}__{feature_set_label}{'+news' if news_features else ''}__{trained_at:%Y%m%d_%H%M%S}"
    if verbose:
        print(f"\n=== {run_name} ===")

    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        run_id = run.info.run_id
        mlflow.set_tags({"model_name": model_name, "feature_set": feature_set_label, "report_variant": report_variant,
                         "git_sha": git_commit_hash() or "unknown", **(run_tags or {})})
        mlflow.log_params({f"model.{name}": value for name, value in parameters.items()})
        mlflow.log_params({f"framework.{name}": value for name, value in get_settings().describe().items()})
        mlflow.log_params({
            "model": model_name, "feature_set": feature_set_label,
            "source_table": f"{GOLD_SCHEMA}.{SOURCE_TABLE}", "gold_snapshot_id": snapshot_id, "target": TARGET_COLUMN,
            "target_scale": target_scale,
            "target_transform": "none (raw realized volatility)" if raw_target
                                else "ln(max(y, floor)), floor = 0.5 * smallest positive training y",
            "bias_correction": "none (raw scale; forecasts floored at 0)" if raw_target
                               else "Duan smearing of the volatility forecast: mean exp(e)",
            "primary_loss": "MSE on volatility (also MAE, WMAPE)",
            "n_features": len(feature_columns), "news_features": json.dumps(news_features),
            "stage2": ("ridge on tier A out-of-sample " + ("errors y - y_hat" if raw_target else "log errors"))
                      if news_features else "none",
            "report_variant": report_variant, "test_years": json.dumps(test_years), "n_rows": len(events),
            "data_start": events["ts_utc"].min().isoformat(), "data_end": events["ts_utc"].max().isoformat(),
            "n_invalid_target_dropped": gold_events.attrs.get("n_invalid_target_dropped", "unknown"), "seed": SEED,
            "zero_volatility_handling": "floored at 0.5 * smallest positive training y"})
        mlflow.log_text(json.dumps({"tier_a": feature_columns, "news": news_features}, indent=2), "features.json")

        result = two_stage_walk_forward(events, feature_columns, model_name, overrides, test_years, news_features,
                                        verbose, jobs, target_scale)
        variants, events = result.variants, result.events
        predictions = variants[report_variant].predictions
        fold_metrics = variants[report_variant].fold_metrics
        pooled_by_variant = {name: frame_metrics(variant.predictions) for name, variant in variants.items()}
        pooled_metrics = pooled_by_variant[report_variant]

        mlflow.log_metrics({"target_zero_share": float((predictions["y_true"] == 0).mean()),
                            "n_scored_events": len(predictions)})
        for name, variant in variants.items():
            mlflow.log_metrics({f"{name}_{metric}": value for metric, value in pooled_by_variant[name].items()})
            for row in variant.fold_metrics.itertuples():
                mlflow.log_metrics({f"fold_{name}_{metric}": getattr(row, metric) for metric in METRIC_NAMES},
                                   step=int(row.test_year))
        mlflow.log_metrics(pooled_metrics)
        for event_type, group in predictions.groupby("event_type"):
            mlflow.log_metrics({f"{metric}_{event_type}": value for metric, value in frame_metrics(group).items()})
        comparison = variant_comparison(variants)
        for row in comparison.itertuples():
            tag = row.comparison.replace(" vs ", "_vs_")
            mlflow.log_metrics({f"{tag}_{row.loss}_pct": row.pct_change})
        if verbose:
            print("  pooled (volatility scale):")
            for name, variant in variants.items():
                pooled = pooled_by_variant[name]
                print(f"    {name:<2} n={len(variant.predictions):>4}  MSE={pooled['MSE']:.3e}  "
                      f"MAE={pooled['MAE']:.5f}  WMAPE={pooled['WMAPE']:.3f}")
            if not comparison.empty:
                print(comparison[comparison["loss"] == "mse"].round(3).to_string(index=False))

        fit_mask = (events["year"] <= int(max(test_years))).to_numpy()
        fit_features = events.loc[fit_mask, feature_columns]
        fit_target = events.loc[fit_mask, TARGET_COLUMN].to_numpy(float)
        floor = floor_value(fit_target)
        fit_model_target = to_model_target(fit_target, floor, target_scale)
        final_model = build_model(model_name, overrides)
        final_model.fit(fit_features, fit_model_target)
        fit_errors = fit_model_target - np.asarray(final_model.predict(fit_features), float)
        deployment = {"target_scale": target_scale, "floor_value": None if raw_target else floor,
                      "smearing_factor_A": smearing_factor(fit_errors, target_scale),
                      "refit_until_year": int(max(test_years)),
                      "model_output": TARGET_COLUMN if raw_target else f"ln({TARGET_COLUMN})",
                      "n_fit_events": int(len(fit_target))}
        estimator = final_estimator(final_model)
        importance = feature_importance(final_model, feature_columns)
        if hasattr(estimator, "history_"):   # LSTM: training curve of the refit
            mlflow.log_metric("refit_epochs_trained", estimator.n_epochs_trained_)
            for record in estimator.history_:
                mlflow.log_metrics({f"refit_{key}": value for key, value in record.items() if key != "epoch"},
                                   step=record["epoch"])
        refit_settings = tuned_settings(final_model)
        if refit_settings:
            mlflow.log_params({f"refit.{name}": value for name, value in refit_settings.items()})

        coefficient_table, coefficient_file = None, None
        if hasattr(estimator, "coefficient_table"):   # ARIMAX: exact-ML coefficient table
            coefficient_table, coefficient_file = estimator.coefficient_table(), "arimax_coefficients.csv"
            mlflow.log_params({"arima_order": f"({parameters['ar_order']},0,{parameters['ma_order']})"})
            mlflow.log_metrics(estimator.summary_metrics())
        elif isinstance(estimator, LinearRegression):   # har_ols: plain OLS on ln y
            term_names = final_model.named_steps["impute"].get_feature_names_out(list(fit_features.columns))
            coefficient_table = pd.DataFrame({"term": ["intercept", *term_names],
                                              "coef": [float(estimator.intercept_), *map(float, estimator.coef_)]})
            coefficient_file = "ols_coefficients.csv"
        if coefficient_table is not None and verbose:
            print(coefficient_table.round(4).to_string(index=False))

        final_correction = None
        if news_features:   # stage 2 refit: every news-era out-of-sample error up to the last test year
            news_values = events[news_features].to_numpy(float)
            pool_mask = (result.news_available & (events["ts_utc"] >= result.news_start).to_numpy() & fit_mask
                         & np.isfinite(result.tier_a_errors))
            pool_errors = result.tier_a_errors[pool_mask]
            final_correction = make_news_correction().fit(news_values[pool_mask], pool_errors)
            ridge, scaler = final_correction.named_steps["ridge"], final_correction.named_steps["scale"]
            deployment.update({
                "stage2_pool_events": int(pool_mask.sum()), "ridge_alpha": float(ridge.alpha_),
                "ridge_intercept": float(ridge.intercept_),
                "smearing_factor_C": smearing_factor(pool_errors - final_correction.predict(news_values[pool_mask]),
                                                     target_scale),
                "ridge_beta_standardised": dict(zip(news_features, map(float, ridge.coef_))),
                "news_mean": dict(zip(news_features, map(float, scaler.mean_))),
                "news_scale": dict(zip(news_features, map(float, scaler.scale_)))})
            if verbose:
                print(f"  stage 2 (ridge alpha {ridge.alpha_:.3g}) on {int(pool_mask.sum())} out-of-sample errors")

        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_dir = Path(temporary_directory)
            predictions.to_csv(artifact_dir / "predictions.csv", index=False)
            pd.concat([v.predictions.assign(variant=n) for n, v in variants.items()]).to_csv(
                artifact_dir / "predictions_all_variants.csv", index=False)
            pd.concat([v.fold_metrics.assign(variant=n) for n, v in variants.items()]).to_csv(
                artifact_dir / "fold_metrics_all_variants.csv", index=False)
            result.smearing_table.to_csv(artifact_dir / "smearing.csv", index=False)
            (artifact_dir / "deployment.json").write_text(json.dumps(deployment, indent=2))
            if not result.hyperparameters.empty:
                result.hyperparameters.to_csv(artifact_dir / "hyperparameters.csv", index=False)
            if not result.news_coefficients.empty:
                result.news_coefficients.to_csv(artifact_dir / "news_correction_coefficients.csv", index=False)
            if not comparison.empty:
                comparison.to_csv(artifact_dir / "variant_comparison.csv", index=False)
            title = f"{model_name} [{feature_set_label}|{report_variant}]"
            plot_forecast_vs_realized(predictions, artifact_dir / f"pred_vs_actual_{model_name}.png", f"{title}: out-of-sample")
            plot_forecast_over_time(predictions, artifact_dir / f"pred_over_time_{model_name}.png", f"{title}: walk-forward")
            if importance is not None:
                importance.to_csv(artifact_dir / "feature_importance.csv", index=False)
                plot_importance(importance, artifact_dir / f"feature_importance_{model_name}.png", f"{title}: feature importance")
            if coefficient_table is not None:
                coefficient_table.to_csv(artifact_dir / coefficient_file, index=False)
            mlflow.log_artifacts(str(artifact_dir), artifact_path="evaluation")

        if log_model:
            registered_name = f"{REGISTERED_MODEL_PREFIX}_{model_name}"
            model_info = log_model_to_mlflow(
                mlflow, final_model, name="model",
                signature=infer_signature(fit_features, final_model.predict(fit_features)),
                input_example=fit_features.head(5), registered_model_name=registered_name if register else None)
            if final_correction is not None:
                log_model_to_mlflow(mlflow, final_correction, name="news_correction")
            if register and getattr(model_info, "registered_model_version", None):
                # The newest version becomes the "challenger"; promote one to "champion" after comparing runs.
                mlflow.MlflowClient().set_registered_model_alias(registered_name, "challenger",
                                                                 model_info.registered_model_version)

        if stage_results:
            RESULTS_STAGING_DIR.mkdir(parents=True, exist_ok=True)
            for name, variant in variants.items():
                common = {"run_id": run_id, "model_name": model_name,
                          "feature_set": variant_feature_set_label(feature_set_label, name),
                          "gold_snapshot_id": snapshot_id, "trained_at_utc": trained_at}
                pl.from_pandas(variant.predictions.assign(**common)).write_parquet(
                    RESULTS_STAGING_DIR / f"{run_id}__{name}_predictions.parquet")
                overall = pd.DataFrame([{"fold": "overall", "test_year": None, "n_train": None,
                                         "n_test": len(variant.predictions), **pooled_by_variant[name]}])
                metrics_table = pd.concat([variant.fold_metrics, overall], ignore_index=True).astype(
                    {"test_year": "Int64", "n_train": "Int64", "n_test": "Int64"})
                pl.from_pandas(metrics_table.assign(**common)).write_parquet(
                    RESULTS_STAGING_DIR / f"{run_id}__{name}_metrics.parquet")

        if verbose:
            print(f"  logged run {run_id}")
        return {"run_id": run_id, "feature_set": feature_set_label, "features": feature_columns,
                "news_features": news_features, "predictions": predictions, "fold_metrics": fold_metrics,
                "pooled_metrics": pooled_metrics, "variants": variants, "pooled_by_variant": pooled_by_variant,
                "n_events": len(predictions)}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="all", choices=MODEL_NAMES + ["all"], help="model to train (default: all)")
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET, choices=[n for n in FEATURE_SETS if n != "base_news"],
                        help=f"tier A feature set (default: {DEFAULT_FEATURE_SET}); benchmarks ignore it")
    parser.add_argument("--news-features", nargs="+", default=None, metavar="COLUMN",
                        help=f"stage-2 news columns (default: {NEWS_SENTIMENT_COLUMNS})")
    parser.add_argument("--no-news", action="store_true", help="tier A only, scored on every event of the test years")
    parser.add_argument("--target-scale", choices=list(TARGET_SCALES), default="log",
                        help="model ln(volatility) (default, smeared back) or the raw volatility (label '+raw')")
    parser.add_argument("--jobs", type=int, default=1, help="processes for the walk-forward blocks (-1 = all cores)")
    parser.add_argument("--params", default="{}", help="JSON dict of hyperparameter overrides, e.g. '{\"ar_order\": 2}'")
    parser.add_argument("--test-years", type=int, nargs="+",
                        help=f"walk-forward test years (default: last {DEFAULT_TEST_YEAR_COUNT} years in the data)")
    parser.add_argument("--snapshot-id", type=int, default=None, help="read this Iceberg snapshot of the gold table")
    parser.add_argument("--register", action="store_true", help="register the refit model in the MLflow registry")
    parser.add_argument("--publish", action="store_true", help="append results to Iceberg right away")
    parser.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE",
                        help="extra MLflow run tag, repeatable (the Airflow DAG passes airflow_run_id)")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    parameter_overrides = json.loads(arguments.params)
    if parameter_overrides and arguments.model == "all":
        raise SystemExit("--params only makes sense with a single --model")
    if arguments.no_news and arguments.news_features:
        raise SystemExit("--no-news and --news-features are mutually exclusive")
    run_tags = dict(tag.split("=", 1) for tag in arguments.tag if "=" in tag)
    mlflow = init_mlflow()
    gold_events, snapshot_id = load_gold_events(arguments.snapshot_id)
    test_years = arguments.test_years or default_test_years(gold_events)
    print(f"Loaded {len(gold_events):,} events from {GOLD_SCHEMA}.{SOURCE_TABLE} (snapshot {snapshot_id}); "
          f"test years {test_years}")
    for model_name in (MODEL_NAMES if arguments.model == "all" else [arguments.model]):
        run_model(mlflow, model_name, gold_events, snapshot_id, test_years, parameter_overrides,
                  feature_set=arguments.feature_set, register=arguments.register,
                  news_features=[] if arguments.no_news else arguments.news_features, jobs=arguments.jobs,
                  run_tags=run_tags, target_scale=arguments.target_scale)
    if arguments.publish:
        from lakehouse.medallion.serving.publish_results import main as publish_results

        publish_results()


if __name__ == "__main__":
    main()

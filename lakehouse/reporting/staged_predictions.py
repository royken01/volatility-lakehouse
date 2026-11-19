"""Loading, selecting and aligning the out-of-sample predictions used by the results figures

Selection keeps one feature set (the one most models share by default), the
benchmarks with their fixed sets and variant A only, runs fitted to
ln(volatility) only, and the latest run of every (model, variant).

``prepare_scored_events`` keeps the events every model and variant was
scored on and adds ``mse``, ``year`` and ``quarter``. ``y_true``, ``y_pred``
and ``mse`` are on the log scale (the ``*_volatility`` columns are for
descriptive figures only). Runs staged before the log columns existed are
completed from ``ln(max(y_pred, floor))`` with a note; retrain for exact log
forecasts. Quarters with fewer than ``--min-quarter-events`` events are
listed in ``df.attrs['small_quarters']``.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from lakehouse.modelling.feature_sets import BENCHMARK_FEATURE_SETS
from lakehouse.modelling.losses import FLOOR_FRACTION, squared_error
from lakehouse.modelling.model_params import BENCHMARK_MODELS, MODEL_NAMES
from lakehouse.paths import RESULTS_PUBLISHED_DIR, RESULTS_STAGING_DIR

VARIANT_NAMES = ("A", "C0", "C")

MODEL_ORDER = ["har_ols", "arimax", "rf", "xgb", "lstm"]
MODEL_DISPLAY_NAMES = {
    "naive_mean": "Naive Mean", "har_ols": "HAR-RV", "arimax": "ARIMAX",
    "rf": "Random Forest", "xgb": "XGBoost", "lstm": "LSTM",
}


def display_name(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model)


def feature_set_root(label: str) -> str:
    """'base+seq10' -> 'base', 'base+raw' -> 'base'."""
    return re.sub(r"\+seq\d+$", "", re.sub(r"\+raw$", "", str(label)))


def drop_unknown_models(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep only models in MODEL_NAMES"""
    unknown = ~predictions["model"].isin(MODEL_NAMES)
    if unknown.any():
        print(
            f"  note: ignoring {int(unknown.sum())} staged rows of removed models "
            f"{sorted(predictions.loc[unknown, 'model'].unique())}"
        )
    return predictions[~unknown]


def drop_outdated_benchmarks(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep only benchmark runs made with the benchmark's current feature set."""
    expected = predictions["model"].map(BENCHMARK_FEATURE_SETS)
    outdated = expected.notna() & (predictions["base_fs"] != expected)
    if outdated.any():
        old = sorted(set(zip(
            predictions.loc[outdated, "model"],
            predictions.loc[outdated, "base_fs"],
        )))
        print(
            f"  note: ignoring {int(outdated.sum())} staged rows of outdated "
            f"benchmark runs {old} (retrain: python -m "
            f"lakehouse.modelling.train --model <model>)"
        )
    return predictions[~outdated]


def load_staged_predictions(directories: list[Path]) -> pd.DataFrame:
    """All *_predictions.parquet files, with base_fs, variant, raw and model columns added."""
    files = sorted(
        file
        for directory in directories
        for file in Path(directory).glob("*_predictions.parquet")
    )
    if not files:
        raise SystemExit(
            f"No *_predictions.parquet in {[str(d) for d in directories]} "
            f"- run train.py first"
        )
    predictions = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    parts = predictions["feature_set"].astype(str).str.partition("|")
    predictions["base_fs"] = parts[0].map(feature_set_root)
    predictions["variant"] = parts[2].replace("", "A")
    predictions["raw"] = parts[0].str.endswith("+raw")
    renamed = predictions.rename(columns={"model_name": "model"})
    return drop_outdated_benchmarks(drop_unknown_models(renamed))


def load_mlflow_predictions(run_ids: list[str]) -> pd.DataFrame:
    """Predictions of MLflow runs (all variants), in the layout of load_staged_predictions"""
    from lakehouse.infrastructure.mlflow_config import init_mlflow

    mlflow = init_mlflow()
    frames = []
    for run_id in run_ids:
        parameters = mlflow.get_run(run_id).data.params
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="evaluation/predictions_all_variants.csv",
        )
        frame = pd.read_csv(path)
        frame["model"] = parameters["model"]
        frame["base_fs"] = feature_set_root(parameters["feature_set"])
        frame["run_id"] = run_id
        frame["raw"] = parameters["feature_set"].endswith("+raw")
        frame["trained_at_utc"] = pd.Timestamp.now(tz="UTC")
        frames.append(frame)
    return drop_outdated_benchmarks(
        drop_unknown_models(pd.concat(frames, ignore_index=True))
    )


def select_runs(predictions: pd.DataFrame, feature_set: str | None,
                models: list[str] | None) -> pd.DataFrame:
    """One feature set, the latest run of every (model, variant)."""
    predictions = predictions[~predictions["raw"]]
    if predictions.empty:
        raise SystemExit(
            "No runs fitted to ln(volatility) - train with "
            "python -m lakehouse.modelling.train"
        )
    if models:
        predictions = predictions[predictions["model"].isin(models)]
    if feature_set is None:
        feature_set = (
            predictions.groupby("base_fs")["model"].nunique().sort_values().index[-1]
        )
        print(f"  feature set: {feature_set} (the one with most models; "
              f"--feature-set to change)")
    is_benchmark = predictions["model"].isin(BENCHMARK_MODELS)
    predictions = pd.concat([
        predictions[(predictions["base_fs"] == feature_set) & ~is_benchmark],
        predictions[is_benchmark],
    ])

    stale = predictions["model"].isin(BENCHMARK_MODELS) & (
        predictions["variant"] != "A"
    )
    if stale.any():
        print(
            f"  note: {sorted(predictions[stale]['model'].unique())} are "
            f"benchmarks - only their variant A is used"
        )
        predictions = predictions[~stale]
    if "trained_at_utc" in predictions:
        latest = predictions.groupby(["model", "variant"])["trained_at_utc"].transform("max")
        predictions = predictions[predictions["trained_at_utc"] == latest]
    return predictions.copy()


# The legacy note below is printed once per run.
LEGACY_NOTE_PRINTED: list[bool] = []


def add_log_scale(scored: pd.DataFrame) -> pd.DataFrame:
    """Add ``y_true_log`` / ``y_pred_log`` and completing missing rows from the volatility scale"""
    for column in ("floor_value", "log_y_true", "log_y_pred"):
        if column not in scored:
            scored[column] = np.nan
    fallback_floor = FLOOR_FRACTION * float(
        scored.loc[scored["y_true"] > 0, "y_true"].min()
    )
    floor = scored["floor_value"].fillna(fallback_floor)
    scored["y_true_log"] = scored["log_y_true"].fillna(
        np.log(np.maximum(scored["y_true"], floor))
    )
    scored["y_pred_log"] = scored["log_y_pred"].fillna(
        np.log(np.maximum(scored["y_pred"], floor))
    )
    if scored["log_y_pred"].isna().any() and not LEGACY_NOTE_PRINTED:
        LEGACY_NOTE_PRINTED.append(True)
        print(
            "  note: some staged runs predate the log-scale columns - their "
            "log forecasts are ln(y_pred), which still contains the smearing "
            "shift (retrain to get the exact ones)"
        )
    return scored.drop(columns=["log_y_true", "log_y_pred"])


def prepare_scored_events(predictions: pd.DataFrame, min_quarter_events: int,
                          drop_zero_events: bool = False,
                          exclude_years: list[int] | None = None) -> pd.DataFrame:
    """Align every model and variant on the same events by adding mse, year and quarter"""
    scored = predictions.copy()
    scored["ts_utc"] = pd.to_datetime(scored["ts_utc"], utc=True)
    if drop_zero_events:
        zero_times = set(scored.loc[scored["y_true"] <= 0, "ts_utc"])
        print(
            f"  note: {len(zero_times)} zero-volatility events are not scored "
            f"(--drop-zero-events)"
        )
        scored = scored[~scored["ts_utc"].isin(zero_times)]
    scored["year"] = scored["ts_utc"].dt.year
    if exclude_years:
        dropped = int(
            scored["year"].isin(exclude_years).sum()
            // max(scored.groupby(["model", "variant"]).ngroups, 1)
        )
        print(
            f"  note: test years {sorted(exclude_years)} are not scored "
            f"(--exclude-years; ~{dropped} events per series)"
        )
        scored = scored[~scored["year"].isin(exclude_years)]
    scored["quarter"] = (
        scored["year"].astype(str) + "Q" + scored["ts_utc"].dt.quarter.astype(str)
    )
    scored["y_true_volatility"] = scored["y_true"]
    scored["y_pred_volatility"] = scored["y_pred"]
    scored = add_log_scale(scored)
    scored["mse"] = squared_error(
        scored["y_true_log"].to_numpy(float),
        scored["y_pred_log"].to_numpy(float),
    )
    scored["y_true"] = scored["y_true_log"]
    scored["y_pred"] = scored["y_pred_log"]

    event_sets = scored.groupby(["model", "variant"])["ts_utc"].apply(set)
    common_events = set.intersection(*event_sets.tolist())
    dropped_per_series = len(scored) // len(event_sets) - len(common_events)
    if dropped_per_series > 0:
        print(
            f"  note: restricting to the {len(common_events)} events every "
            f"model and variant was scored on (~{dropped_per_series} per "
            f"series dropped)"
        )
    scored = scored[scored["ts_utc"].isin(common_events)]
    events_per_quarter = scored.groupby("quarter")["ts_utc"].nunique()
    small = events_per_quarter[events_per_quarter < min_quarter_events]
    if len(small):
        print(
            f"  note: quarters with fewer than {min_quarter_events} events are "
            f"left out of the quarterly figures: {dict(small)}"
        )
    scored.attrs["small_quarters"] = list(small.index)
    return scored


def headline_rows(scored: pd.DataFrame, variant: str = "C") -> pd.DataFrame:
    """Each model's headline rows: ``variant`` where present, else variant A."""
    variants_of_model = scored.groupby("model")["variant"].agg(set)
    chosen = {
        model: (variant if variant in available else "A")
        for model, available in variants_of_model.items()
    }
    return scored[scored["variant"] == scored["model"].map(chosen)]


def model_order(scored: pd.DataFrame, variant: str = "C") -> list[str]:
    """Models best first by the pooled MSE of their headline variant."""
    return list(
        headline_rows(scored, variant)
        .groupby("model")["mse"]
        .mean()
        .sort_values()
        .index
    )


def news_models(scored: pd.DataFrame, order: list[str]) -> list[str]:
    """Models with a news stage (variants C0 and C), in ``order``."""
    return [
        model for model in order
        if {"C0", "C"} <= set(scored.loc[scored["model"] == model, "variant"])
    ]


def pick_reference(scored: pd.DataFrame, order: list[str],
                   wanted: str | None = None) -> str:
    """The reference model that equals 0 % in the figures."""
    if wanted and wanted in set(scored["model"]):
        return wanted
    for model in ("har_ols", "naive_mean"):
        if model in set(scored["model"]):
            return model
    return order[0]


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """The run-selection options of results_figures.py."""
    parser.add_argument(
        "--staging", type=Path, nargs="+",
        default=[RESULTS_STAGING_DIR, RESULTS_PUBLISHED_DIR],
        help="folders with train.py's *_predictions.parquet files",
    )
    parser.add_argument(
        "--runs", nargs="+",
        help="MLflow run ids instead of the staging folders",
    )
    parser.add_argument(
        "--feature-set", default=None,
        help="tier A feature set label (default: the one with most models)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, help="only these models",
    )
    parser.add_argument(
        "--variant", choices=list(VARIANT_NAMES), default="C",
        help="headline variant of the models with a news stage "
             "(benchmarks always use A)",
    )
    parser.add_argument(
        "--reference", default=None,
        help="reference model (default har_ols, else naive_mean)",
    )
    parser.add_argument(
        "--drop-zero-events", action="store_true",
        help="do not score zero-volatility events",
    )
    parser.add_argument(
        "--exclude-years", type=int, nargs="+", default=None, metavar="YEAR",
        help="do not score these test years (robustness check, e.g. "
             "--exclude-years 2026)",
    )
    parser.add_argument(
        "--min-quarter-events", type=int, default=8,
        help="quarters with fewer events are left out of the quarterly figures",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="default: <outputs>/<name>[_no_zero_events][_excl_<years>]",
    )


def load_scored_events(arguments: argparse.Namespace) -> pd.DataFrame:
    """Predictions selected and aligned according to ``add_selection_arguments``."""
    raw = (
        load_mlflow_predictions(arguments.runs)
        if arguments.runs
        else load_staged_predictions(arguments.staging)
    )
    selected = select_runs(raw, arguments.feature_set, arguments.models)
    return prepare_scored_events(
        selected, arguments.min_quarter_events, arguments.drop_zero_events,
        getattr(arguments, "exclude_years", None),
    )


def output_directory(arguments: argparse.Namespace, default_directory: Path) -> Path:
    """``--out-dir``, or the default folder suffixed with the robustness options used."""
    if arguments.out_dir:
        return arguments.out_dir
    suffix = ""
    if arguments.drop_zero_events:
        suffix += "_no_zero_events"
    if arguments.exclude_years:
        years = "_".join(str(y) for y in sorted(arguments.exclude_years))
        suffix += f"_excl_{years}"
    return default_directory.with_name(default_directory.name + suffix)
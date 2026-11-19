"""How accurately can the post-release volatility be forecast for different window lengths?

The same study (events, prices, news, models, walk-forward) is run once per post-event window, e.g.
30 s, 1 minute and 5 minutes, and the out-of-sample accuracy is compared::

    vol-lakehouse --config configs/crude_oil_thesis.toml window-study --windows 30 60 300
    vol-lakehouse window-study --windows 30 60 300 --models naive_mean har_ols rf xgb --source raw \\
        --raw-dir ~/data/raw --scored-headlines ~/data/news/headlines_sentiment.csv
    vol-lakehouse window-study report            # only redraw the figures from the saved predictions

Steps (each resumable: finished pieces are cached in ``outputs/window_study/``):

    prepare   silver-level inputs once, for the longest window: events in UTC, 1-minute bars and the
              ticks around every release.  ``--source lakehouse`` reads the silver Iceberg tables;
              ``--source raw`` builds them locally from the raw files with the same functions (no
              Spark or cloud needed, so the study runs on a laptop).
    run       per window, in its own process with ``--post-window-seconds W`` (so every module uses the
              W-specific target and lag columns): gold features, then the two-stage walk-forward of
              every model over the last ``--test-years`` years.
    report    accuracy per window and model on the same releases, and three figures.

Accuracy is measured on the log scale, the scale the models are fitted on:

    R2 vs naive   1 - MSE(model) / MSE(naive mean per release type): the share of the release-to-release
                  variation in ln(volatility) the model explains beyond "every release of this type is
                  average"
    vs HAR-RV     % change in MSE against the HAR-RV benchmark (negative = more accurate)
    rank corr.    Spearman correlation of forecast and realized volatility (are volatile releases
                  forecast as volatile?)

Figures: ``fig_window_accuracy`` (R2 per window and model), ``fig_window_calibration`` (realized vs
forecast volatility in forecast deciles, per window) and ``fig_window_scatter`` (every release).
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from lakehouse.paths import NEWS_DATA_DIR, RAW_DATA_DIR, WINDOW_STUDY_OUTPUT_DIR
from lakehouse.plot_style import (AXIS, BLUE, CATEGORY_COLORS, GRID, INK, MUTED_INK, SURFACE, apply_default_style,
                                  clean_axes, save_figure)
from lakehouse.settings import get_settings

DEFAULT_WINDOWS = [30, 60, 300]
DEFAULT_MODELS = ["naive_mean", "har_ols", "rf", "xgb"]
MODEL_LABELS = {"naive_mean": "Naive mean", "har_ols": "HAR-RV", "arimax": "ARIMAX", "rf": "Random forest",
                "xgb": "XGBoost", "lstm": "LSTM"}
CACHE_DIR = WINDOW_STUDY_OUTPUT_DIR / "cache"
DECILES = 10


def window_label(seconds: int) -> str:
    return f"{seconds} s" if seconds < 60 else f"{seconds // 60} min" if seconds % 60 == 0 else f"{seconds} s"


# --- prepare: silver-level inputs -----------------------------------------------------------------------
def read_events_raw(raw_dir: Path) -> pl.DataFrame:
    from lakehouse.medallion.silver.time_conversion import drop_bad_events, events_to_utc

    settings = get_settings()
    files = sorted(glob.glob(str(raw_dir / settings.events_source)))
    if not files:
        raise SystemExit(f"no event calendar matches {raw_dir / settings.events_source}")
    events = pl.concat([pl.read_csv(f, infer_schema_length=0) for f in files], how="diagonal_relaxed")
    numeric = [c for c in ("actual", "forecast", "previous", "surprise") if c in events.columns]
    events = events_to_utc(events.with_columns(pl.col(numeric).cast(pl.Float64, strict=False)),
                           "release_datetime", "ts_utc")
    if {"actual", "forecast", "surprise"} <= set(events.columns):
        events = events.with_columns(pl.coalesce("surprise", pl.col("actual") - pl.col("forecast")).alias("surprise"))
    return drop_bad_events(events)


def read_minute_bars_raw(raw_dir: Path) -> pl.DataFrame:
    from lakehouse.medallion.silver.price_processing import build_silver_minute_bars
    from lakehouse.medallion.silver.time_conversion import prices_to_utc

    files = sorted(glob.glob(str(raw_dir / get_settings().prices_m1_source)))
    bars = pl.concat([prices_to_utc(pl.read_parquet(f, columns=["datetime", "close"]), "datetime", "ts_utc")
                      for f in files])
    return build_silver_minute_bars(bars.unique("ts_utc").sort("ts_utc"))


def read_ticks_raw(raw_dir: Path, events: pl.DataFrame, window_seconds: int) -> pl.DataFrame:
    """Ticks within +/- window_seconds of a release, file by file and batch by batch (cached per file)."""
    import pyarrow.parquet as parquet

    from lakehouse.medallion.silver.silver_transform import tick_window_filter

    keep = tick_window_filter(events, window_seconds)
    parts = []
    for file in sorted(glob.glob(str(raw_dir / get_settings().prices_tick_source))):
        cached = CACHE_DIR / f"ticks_{window_seconds}s_{Path(file).stem}.parquet"
        if not cached.exists():
            reader = parquet.ParquetFile(file)
            columns = [c for c in ("datetime", "bid", "ask", "price") if c in reader.schema_arrow.names]
            kept = [keep(pl.from_arrow(batch)) for batch in reader.iter_batches(batch_size=1_000_000, columns=columns)]
            pl.concat(kept).write_parquet(cached)
            print(f"  {Path(file).name}: {sum(len(k) for k in kept):,} ticks near releases")
        parts.append(pl.read_parquet(cached))
    return pl.concat(parts).sort("ts_utc")


def read_scored_headlines(path: Path | None) -> pl.DataFrame | None:
    if path is None or not path.exists():
        return None
    headlines = pl.read_csv(path, columns=["seen_utc", "uid", "sentiment_score"], infer_schema_length=0)
    return (headlines
            .with_columns(pl.col("seen_utc").str.to_datetime(time_zone="UTC", strict=False),
                          pl.col("sentiment_score").cast(pl.Float64, strict=False))
            .drop_nulls().unique("uid"))


def prepare(source: str, raw_dir: Path, scored_headlines: Path | None, longest_window: int) -> None:
    """Write events / minute bars / ticks / scored headlines to the cache (once)."""
    from lakehouse.medallion.silver.price_processing import build_silver_ticks

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tick_window = max(get_settings().tick_window_seconds, longest_window)
    targets = {name: CACHE_DIR / f"{name}.parquet" for name in ("events", "minute_bars", "ticks", "headlines")}
    if all(path.exists() for name, path in targets.items() if name != "headlines"):
        print(f"inputs already prepared in {CACHE_DIR}")
        return
    if source == "lakehouse":
        from lakehouse.infrastructure.iceberg_io import load_iceberg_table
        from lakehouse.infrastructure.spark_session import SILVER_SCHEMA
        from lakehouse.medallion.silver.silver_transform import load_optional_table

        events = load_iceberg_table(SILVER_SCHEMA, "events")
        minute_bars = load_iceberg_table(SILVER_SCHEMA, "prices_m1")
        ticks = load_iceberg_table(SILVER_SCHEMA, "prices_tick")
        headlines = load_optional_table(SILVER_SCHEMA, get_settings().scored_headlines_table)
    else:
        print(f"building the inputs from the raw files in {raw_dir}")
        events = read_events_raw(raw_dir)
        minute_bars = read_minute_bars_raw(raw_dir)
        ticks = build_silver_ticks(read_ticks_raw(raw_dir, events, tick_window), minute_bars, events,
                                   window_seconds=tick_window)
        headlines = read_scored_headlines(scored_headlines)
    events.write_parquet(targets["events"])
    minute_bars.write_parquet(targets["minute_bars"])
    ticks.write_parquet(targets["ticks"])
    if headlines is not None:
        headlines.select("seen_utc", "uid", "sentiment_score").write_parquet(targets["headlines"])
    print(f"prepared {len(events):,} events, {len(minute_bars):,} minute bars, {len(ticks):,} tick rows, "
          f"{0 if headlines is None else len(headlines):,} scored headlines")


# --- run: one window ----------------------------------------------------------------------------------------
def run_window(window: int, models: list[str], test_year_count: int, news: bool) -> None:
    """Gold features and walk-forward predictions for the window of THIS process's settings."""
    from lakehouse.feature_columns import NEWS_SENTIMENT_COLUMNS, TARGET_COLUMN
    from lakehouse.medallion.gold.event_features import build_features
    from lakehouse.modelling.feature_sets import model_feature_columns
    from lakehouse.modelling.gold_data import default_test_years, model_frame, prepare_features
    from lakehouse.modelling.model_params import BENCHMARK_MODELS
    from lakehouse.modelling.walk_forward import two_stage_walk_forward

    if get_settings().post_window_seconds != window:
        raise SystemExit("run the window step through `window-study` (it sets --post-window-seconds)")
    out_dir = WINDOW_STUDY_OUTPUT_DIR / f"window_{window}s"
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_file = out_dir / "gold_events_features.parquet"
    if not gold_file.exists():
        headlines_file = CACHE_DIR / "headlines.parquet"
        gold = build_features(pl.read_parquet(CACHE_DIR / "minute_bars.parquet"),
                              pl.read_parquet(CACHE_DIR / "ticks.parquet"),
                              pl.read_parquet(CACHE_DIR / "events.parquet"),
                              scored_headlines=pl.read_parquet(headlines_file) if headlines_file.exists() else None)
        gold.write_parquet(gold_file)
    events = model_frame(pl.read_parquet(gold_file).to_pandas())
    test_years = default_test_years(events, test_year_count)
    print(f"\n=== {window_label(window)} window: {TARGET_COLUMN}, {len(events):,} releases, test years {test_years}")

    for model_name in models:
        target_file = out_dir / f"predictions_{model_name}.parquet"
        if target_file.exists():
            print(f"  {model_name}: cached")
            continue
        news_columns = list(NEWS_SENTIMENT_COLUMNS) if news and model_name not in BENCHMARK_MODELS \
            and all(c in events.columns and events[c].notna().any() for c in NEWS_SENTIMENT_COLUMNS) else []
        features = model_feature_columns(model_name)
        prepared = prepare_features(events, features, news_columns)
        result = two_stage_walk_forward(prepared, features, model_name, None, test_years, news_columns, verbose=False)
        variant = "C" if "C" in result.variants else "A"
        predictions = result.variants[variant].predictions.assign(model_name=model_name, variant=variant,
                                                                  window_seconds=window)
        predictions.to_parquet(target_file)
        error = predictions["log_y_true"] - predictions["log_y_pred"]
        print(f"  {model_name:<10} variant {variant}: {len(predictions):,} releases, MSE(ln y) {np.mean(error ** 2):.3f}")


# --- report ---------------------------------------------------------------------------------------------------
def load_predictions() -> pd.DataFrame:
    files = sorted(glob.glob(str(WINDOW_STUDY_OUTPUT_DIR / "window_*s" / "predictions_*.parquet")))
    if not files:
        raise SystemExit(f"no predictions in {WINDOW_STUDY_OUTPUT_DIR}; run the study first")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def common_releases(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per window, only the releases every model forecast (so all models are scored on the same set)."""
    frames = []
    for _, group in predictions.groupby("window_seconds"):
        counts = group.groupby("ts_utc")["model_name"].nunique()
        shared = counts[counts == group["model_name"].nunique()].index
        frames.append(group[group["ts_utc"].isin(shared)])
    return pd.concat(frames, ignore_index=True)


def accuracy_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (window, model), group in predictions.groupby(["window_seconds", "model_name"]):
        error = group["log_y_true"] - group["log_y_pred"]
        rows.append({"window_seconds": window, "model_name": model, "variant": group["variant"].iloc[0],
                     "n_releases": len(group), "mse_log": float(np.mean(error ** 2)),
                     "mae_vol": float(np.mean(np.abs(group["y_true"] - group["y_pred"]))),
                     "rank_corr": float(stats.spearmanr(group["y_pred"], group["y_true"]).statistic),
                     "mean_realized_vol": float(group["y_true"].mean())})
    table = pd.DataFrame(rows)
    for reference, column in (("naive_mean", "r2_vs_naive"), ("har_ols", "pct_mse_vs_har")):
        base = table[table["model_name"] == reference].set_index("window_seconds")["mse_log"]
        ratio = table["mse_log"] / table["window_seconds"].map(base)
        table[column] = (1 - ratio) if column == "r2_vs_naive" else 100 * (ratio - 1)
    return table.sort_values(["window_seconds", "mse_log"]).reset_index(drop=True)


def cell(table: pd.DataFrame, window: int, model: str, column: str):
    return table.loc[(table["window_seconds"] == window) & (table["model_name"] == model), column].iloc[0]


def model_colors(models) -> dict:
    ordered = [m for m in MODEL_LABELS if m in set(models)] + sorted(set(models) - set(MODEL_LABELS))
    palette = ["#b4b2a9", BLUE] + [c for c in CATEGORY_COLORS if c != BLUE]
    return {model: palette[i % len(palette)] for i, model in enumerate(ordered)}


def plot_accuracy(table: pd.DataFrame, path: Path, asset: str) -> None:
    windows = sorted(table["window_seconds"].unique())
    models = [m for m in MODEL_LABELS if m in set(table["model_name"]) and m != "naive_mean"]
    colors = model_colors(table["model_name"])
    width = 0.8 / len(models)
    figure, axis = plt.subplots(figsize=(10, 5.6))
    for index, model in enumerate(models):
        values = [cell(table, w, model, "r2_vs_naive") for w in windows]
        positions = np.arange(len(windows)) + (index - (len(models) - 1) / 2) * width
        bars = axis.bar(positions, np.asarray(values, float) * 100, width * 0.92, color=colors[model],
                        label=MODEL_LABELS.get(model, model), zorder=3)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"{value * 100:.0f}%",
                      ha="center", va="bottom", fontsize=10, color=INK)
    axis.axhline(0, color=AXIS, lw=1)
    axis.set_xticks(np.arange(len(windows)), [f"{window_label(w)} after the release" for w in windows], fontsize=11)
    axis.set_ylabel("Variation explained beyond the naive forecast (%)", fontsize=11)
    clean_axes(axis, grid="y")
    axis.legend(frameon=False, ncols=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=10)
    figure.suptitle(f"{asset}: how predictable is the volatility after a scheduled release?",
                    x=0.01, ha="left", fontsize=14, fontweight="bold")
    axis.set_title("Out-of-sample R² of ln(realized volatility) vs. a per-release-type average, walk-forward test years",
                   loc="left", fontsize=10, color=MUTED_INK)
    figure.tight_layout()
    save_figure(figure, path, dpi=200)


def best_models(table: pd.DataFrame) -> dict:
    return {window: group.sort_values("mse_log")["model_name"].iloc[0]
            for window, group in table.groupby("window_seconds")}


def plot_calibration(predictions: pd.DataFrame, table: pd.DataFrame, path: Path, asset: str) -> None:
    best = best_models(table)
    windows = sorted(best)
    figure, axes = plt.subplots(1, len(windows), figsize=(4.2 * len(windows), 4.6), squeeze=False)
    for axis, window in zip(axes[0], windows):
        group = predictions[(predictions["window_seconds"] == window) & (predictions["model_name"] == best[window])]
        decile = pd.qcut(group["y_pred"].rank(method="first"), DECILES, labels=False)
        binned = group.groupby(decile).agg(forecast=("y_pred", "mean"), realized=("y_true", "mean"))
        limit = float(max(binned.max())) * 1.1
        axis.plot([0, limit], [0, limit], color=AXIS, lw=1, ls="--", zorder=1, label="perfect forecast")
        axis.plot(binned["forecast"], binned["realized"], color=BLUE, lw=2, marker="o", ms=6, zorder=3,
                  label=MODEL_LABELS.get(best[window], best[window]))
        row = {c: cell(table, window, best[window], c) for c in ("rank_corr", "n_releases")}
        axis.set(xlim=(0, limit), ylim=(0, limit), title=f"{window_label(window)} window",
                 xlabel="Forecast volatility (%)")
        axis.text(0.04, 0.96, f"rank correlation {row['rank_corr']:.2f}\n{row['n_releases']} releases",
                  transform=axis.transAxes, va="top", fontsize=9, color=MUTED_INK)
        clean_axes(axis, grid="both")
        axis.legend(frameon=False, loc="lower right", fontsize=9)
    axes[0][0].set_ylabel("Realized volatility (%)")
    figure.suptitle(f"{asset}: forecasts sorted into deciles vs. what happened (best model per window)",
                    x=0.01, ha="left", fontsize=13, fontweight="bold")
    figure.tight_layout()
    save_figure(figure, path, dpi=200)


def plot_scatter(predictions: pd.DataFrame, table: pd.DataFrame, path: Path, asset: str) -> None:
    best = best_models(table)
    windows = sorted(best)
    figure, axes = plt.subplots(1, len(windows), figsize=(4.2 * len(windows), 4.4), squeeze=False)
    for axis, window in zip(axes[0], windows):
        group = predictions[(predictions["window_seconds"] == window) & (predictions["model_name"] == best[window])]
        values = group[["y_pred", "y_true"]].clip(lower=group["floor_value"].min(), axis=0)
        low, high = float(values.min().min()) * 0.8, float(values.max().max()) * 1.25
        axis.plot([low, high], [low, high], color=AXIS, lw=1, ls="--", zorder=1)
        axis.scatter(values["y_pred"], values["y_true"], s=12, alpha=0.5, color=BLUE, edgecolor=SURFACE, lw=0.3,
                     zorder=3)
        axis.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high),
                 title=f"{window_label(window)}: {MODEL_LABELS.get(best[window], best[window])}",
                 xlabel="Forecast volatility (%)")
        axis.grid(True, which="major", color=GRID, lw=0.6)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0][0].set_ylabel("Realized volatility (%)")
    figure.suptitle(f"{asset}: every out-of-sample release, forecast vs. realized", x=0.01, ha="left",
                    fontsize=13, fontweight="bold")
    figure.tight_layout()
    save_figure(figure, path, dpi=200)


def report() -> pd.DataFrame:
    settings = get_settings()
    apply_default_style()
    predictions = common_releases(load_predictions())
    table = accuracy_table(predictions)
    WINDOW_STUDY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(WINDOW_STUDY_OUTPUT_DIR / "window_accuracy.csv", index=False)
    plot_accuracy(table, WINDOW_STUDY_OUTPUT_DIR / "fig_window_accuracy.png", settings.asset)
    plot_calibration(predictions, table, WINDOW_STUDY_OUTPUT_DIR / "fig_window_calibration.png", settings.asset)
    plot_scatter(predictions, table, WINDOW_STUDY_OUTPUT_DIR / "fig_window_scatter.png", settings.asset)
    shown = table.assign(window=table["window_seconds"].map(window_label))[
        ["window", "model_name", "variant", "n_releases", "r2_vs_naive", "pct_mse_vs_har", "rank_corr",
         "mse_log", "mae_vol", "mean_realized_vol"]]
    print(shown.round(3).to_string(index=False))
    print(f"\nFigures and window_accuracy.csv in {WINDOW_STUDY_OUTPUT_DIR}")
    return table


# --- command line ---------------------------------------------------------------------------------------------
def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vol-lakehouse window-study", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", nargs="?", default="all", choices=["all", "prepare", "run", "report"])
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS, help="post-event windows in seconds")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--test-years", type=int, default=4, help="number of final calendar years tested")
    parser.add_argument("--no-news", action="store_true", help="tier A only (no news correction)")
    parser.add_argument("--source", choices=["lakehouse", "raw"], default="lakehouse")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR, help="raw files for --source raw")
    parser.add_argument("--scored-headlines", type=Path, default=None,
                        help="scored headline CSV for --source raw (default: newest *sentiment*.csv in data/news)")
    parser.add_argument("--window", type=int, help=argparse.SUPPRESS)   # the per-window child process
    options = parser.parse_args(arguments)

    if options.step == "run" and options.window:
        run_window(options.window, options.models, options.test_years, not options.no_news)
        return 0
    if options.step in ("all", "prepare"):
        scored = options.scored_headlines or next(iter(sorted(glob.glob(str(NEWS_DATA_DIR / "*sentiment*.csv")))), None)
        prepare(options.source, options.raw_dir.expanduser(), Path(scored).expanduser() if scored else None,
                max(options.windows))
    if options.step in ("all", "run"):
        for window in options.windows:
            environment = {**os.environ, "LAKEHOUSE_POST_WINDOW_SECONDS": str(window)}
            command = [sys.executable, "-m", "lakehouse", "window-study", "run", "--window", str(window),
                       "--test-years", str(options.test_years), "--models", *options.models]
            subprocess.run(command + (["--no-news"] if options.no_news else []), check=True, env=environment)
    if options.step in ("all", "report"):
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

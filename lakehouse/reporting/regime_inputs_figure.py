"""Why does HAR-RV break down in some quarters?  Two figures over the test quarters.

    python -m lakehouse.reporting.regime_inputs_figure

regime_forecasts.png: mean post-release volatility per quarter next to the mean forecast of
HAR-RV and the other tier A models (variant A).

regime_inputs.png: the model inputs averaged per quarter.  Top, HAR-RV: rv_d, rv_w and rv_m of the
sessions before the release.  Bottom, ARIMAX / LSTM: the post-release volatility after the previous
releases of the same type (left out with a note when gold cannot be reached).

Uses the event selection of the other results figures (``add_selection_arguments``).  Writes both
figures and ``regime_inputs_quarterly.csv`` to ``lakehouse/outputs/results_figures/``.
"""
from __future__ import annotations

import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from lakehouse.feature_columns import SEQUENCE_COLUMNS
from lakehouse.medallion.gold.event_features import daily_har_table, trading_date
from lakehouse.medallion.silver.price_processing import build_silver_minute_bars
from lakehouse.medallion.silver.time_conversion import prices_to_utc
from lakehouse.settings import get_settings
from lakehouse.paths import RAW_DATA_DIR, RESULTS_FIGURES_OUTPUT_DIR
from lakehouse.plot_style import BLUE, INK, ORANGE, SURFACE, apply_default_style, clean_axes, save_figure
from lakehouse.reporting.results_figures import MODEL_COLOR, MODEL_MARKER
from lakehouse.reporting.staged_predictions import add_selection_arguments, display_name, load_scored_events

MODELS = ["har_ols", "arimax", "lstm", "rf", "xgb"]
HAR_SHADES = {"rv_d": "#9cc3f0", "rv_w": "#5b9be3", "rv_m": BLUE}       # one hue (HAR-RV's), light = short horizon
HAR_NAMES = {"rv_d": "Daily  rv_d", "rv_w": "Weekly  rv_w", "rv_m": "Monthly  rv_m"}
LAG_SHADES = {"lag1": "#f5b08f", "lag_mean": ORANGE}


def quarterly_forecasts(scored: pd.DataFrame) -> pd.DataFrame:
    """Quarter x (realised, model means, model log biases) for variant A, small quarters left out."""
    rows = scored[(scored["variant"] == "A") & scored["model"].isin(MODELS)]
    rows = rows[~rows["quarter"].isin(scored.attrs.get("small_quarters", []))]
    realised = rows[rows["model"] == "har_ols"].groupby("quarter").agg(n=("ts_utc", "size"), realised=("y_true_volatility", "mean"))
    forecast = rows.pivot_table(index="quarter", columns="model", values="y_pred_volatility", aggfunc="mean")
    bias = (rows.assign(error=rows["y_true_log"] - rows["y_pred_log"])
            .pivot_table(index="quarter", columns="model", values="error", aggfunc="mean").add_suffix("_log_bias"))
    return realised.join(forecast).join(bias)


def har_terms_at_events(event_times: pd.Series) -> pd.DataFrame:
    """rv_d, rv_w, rv_m (%) that gold attaches to each event, computed from the raw 1-minute bars
    with the gold functions."""
    files = sorted(glob.glob(str(RAW_DATA_DIR / get_settings().prices_m1_source)))
    bars = pl.concat([pl.read_parquet(file) for file in files], how="diagonal_relaxed")
    minute_bars = build_silver_minute_bars(prices_to_utc(bars, "datetime", "ts_utc"))
    har_table = daily_har_table(minute_bars).rename({"trading_date": "har_date"})
    events = pl.DataFrame({"ts_utc": pd.to_datetime(event_times, utc=True).dt.tz_convert("UTC")}).sort("ts_utc")
    events = events.with_columns((trading_date("ts_utc", "UTC") - pl.duration(days=1)).alias("har_date"))
    return events.join_asof(har_table, on="har_date", strategy="backward").drop("har_date").to_pandas()


def lag_inputs(event_times: pd.Series) -> pd.DataFrame | None:
    """lag1 and the mean of the last ten same-type releases, from gold; None if gold is not
    reachable."""
    try:
        from lakehouse.modelling.gold_data import load_gold_events

        gold, _ = load_gold_events()
    except Exception as error:   # no catalog credentials / network
        print(f"  note: gold not reachable ({type(error).__name__}); the lag panel is left out")
        return None
    gold = gold[["ts_utc", *SEQUENCE_COLUMNS]].copy()
    gold["lag1"], gold["lag_mean"] = gold[SEQUENCE_COLUMNS[0]], gold[SEQUENCE_COLUMNS].mean(axis=1)
    return gold[["ts_utc", "lag1", "lag_mean"]]


def quarterly_inputs(scored: pd.DataFrame) -> pd.DataFrame:
    """Mean of every model input per quarter, over the events that were scored (one row per event)."""
    events = scored[(scored["variant"] == "A") & (scored["model"] == "har_ols")][["ts_utc", "quarter"]].drop_duplicates("ts_utc")
    events = events[~events["quarter"].isin(scored.attrs.get("small_quarters", []))]
    har = har_terms_at_events(events["ts_utc"])
    table = events.merge(har, on="ts_utc", how="left")
    lags = lag_inputs(events["ts_utc"])
    if lags is not None:
        table = table.merge(lags, on="ts_utc", how="left")
    columns = [c for c in ("rv_d", "rv_w", "rv_m", "lag1", "lag_mean") if c in table]
    missing = int(table[columns].isna().any(axis=1).sum())
    if missing:
        print(f"  note: {missing} of {len(table)} scored events have no value for every input (left out of the means)")
    return table.groupby("quarter")[columns].mean()


def quarter_axis(axis, quarters) -> np.ndarray:
    x = np.arange(len(quarters))
    axis.set_xticks(x, [q[2:4] + "\nQ" + q[-1] if q.endswith("Q1") else "Q" + q[-1] for q in quarters])
    return x


def figure_forecasts(table: pd.DataFrame, path, dpi: int) -> None:
    quarters = list(table.index)
    figure, axis = plt.subplots(figsize=(10.5, 4.6))
    x = quarter_axis(axis, quarters)
    axis.plot(x, table["realised"], color=INK, lw=2.8, marker="o", markersize=6, markeredgecolor=SURFACE,
              markeredgewidth=1.2, zorder=5, label="Realised")
    for model in MODELS:
        emphasised = model == "har_ols"
        axis.plot(x, table[model], color=MODEL_COLOR[model], lw=2.2 if emphasised else 1.3, alpha=1 if emphasised else 0.85,
                  marker=MODEL_MARKER[model], markersize=5.5 if emphasised else 4, markeredgecolor=SURFACE,
                  markeredgewidth=1, zorder=4 if emphasised else 3, label=display_name(model))
    axis.set_ylabel(f"Mean {get_settings().post_window_seconds} s volatility after release (%)")
    axis.set_title("Realised post-release volatility and mean forecasts, by quarter")
    axis.set_ylim(0, None)
    axis.legend(loc="upper left", ncol=3, fontsize=9)
    clean_axes(axis, grid="y")
    figure.tight_layout()
    save_figure(figure, path, dpi)


def figure_inputs(table: pd.DataFrame, path, dpi: int) -> None:
    quarters = list(table.index)
    with_lags = "lag_mean" in table
    figure, axes = plt.subplots(2 if with_lags else 1, 1, figsize=(10.5, 7.2 if with_lags else 4.6), sharex=True, squeeze=False)
    top = axes[0, 0]
    x = quarter_axis(top, quarters)
    for name in ("rv_d", "rv_w", "rv_m"):
        top.plot(x, table[name], color=HAR_SHADES[name], lw=2.4 if name == "rv_m" else 1.7, marker="o",
                 markersize=5 if name == "rv_m" else 4, markeredgecolor=SURFACE, markeredgewidth=1, label=HAR_NAMES[name])
    top.set_ylabel("Realised volatility of Brent (%)")
    top.set_title("Long-term market volatility of the sessions before each release (mean per quarter)")
    top.set_ylim(0, None)
    top.legend(loc="upper left", fontsize=9)
    clean_axes(top, grid="y")
    if with_lags:
        bottom = axes[1, 0]
        quarter_axis(bottom, quarters)
        bottom.plot(x, table["lag_mean"], color=LAG_SHADES["lag_mean"], lw=2.4, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1, label="Mean of the last 10 releases of the same type")
        bottom.plot(x, table["lag1"], color=LAG_SHADES["lag1"], lw=1.7, marker="o", markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=1, label="Previous release of the same type")
        bottom.set_ylabel(f"{get_settings().post_window_seconds} s volatility after release (%)")
        bottom.set_title("Short-term market volatility after the previous same-type releases (mean per quarter)")
        bottom.set_ylim(0, bottom.get_ylim()[1] * 1.45)   # head-room so the legend sits above the lines
        bottom.legend(loc="upper left", fontsize=9)
        clean_axes(bottom, grid="y")
    figure.tight_layout()
    save_figure(figure, path, dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selection_arguments(parser)
    parser.add_argument("--dpi", type=int, default=200)
    arguments = parser.parse_args()
    apply_default_style()
    scored = load_scored_events(arguments)
    forecasts = quarterly_forecasts(scored)
    inputs = quarterly_inputs(scored).reindex(forecasts.index)
    directory = RESULTS_FIGURES_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    combined = forecasts.join(inputs)
    combined.round(4).to_csv(directory / "regime_inputs_quarterly.csv")
    print(combined.round(3).to_string())
    figure_forecasts(forecasts, directory / "regime_forecasts.png", arguments.dpi)
    figure_inputs(inputs, directory / "regime_inputs.png", arguments.dpi)


if __name__ == "__main__":
    main()

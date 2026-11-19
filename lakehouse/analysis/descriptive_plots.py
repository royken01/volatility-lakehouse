"""Descriptive figures from gold.events_features.

    vol-lakehouse descriptive distribution       # target histograms
    vol-lakehouse descriptive event-volatility   # volatility per release (--start YYYY-MM-DD)
    vol-lakehouse descriptive all
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from lakehouse.feature_columns import TARGET_COLUMN
from lakehouse.modelling.gold_data import load_gold_events
from lakehouse.modelling.losses import floor_value, to_log
from lakehouse.paths import DESCRIPTIVE_OUTPUT_DIR, RESULTS_FIGURES_OUTPUT_DIR
from lakehouse.plot_style import BLUE, SURFACE, apply_default_style, clean_axes, event_type_colors, save_figure
from lakehouse.settings import get_settings

EVENT_VOLATILITY_START = "2023-01-01"


def make_distribution_plot() -> None:
    events, _ = load_gold_events()
    target = events[TARGET_COLUMN].to_numpy()
    log_target = to_log(target, floor_value(target))
    apply_default_style()
    figure, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].hist(target, bins=80, color=BLUE)
    axes[0].set_title(r"Realized volatility $y_k$")
    axes[1].hist(log_target, bins=80, color=BLUE)
    axes[1].set_title(r"Log realized volatility $\ln y_k$")
    for axis in axes:
        clean_axes(axis, grid="y")
    figure.tight_layout()
    save_figure(figure, DESCRIPTIVE_OUTPUT_DIR / "target_distribution.png", dpi=200)
    for name, values in (("y", target), ("ln y", log_target)):
        print(f"{name:5s} skewness={stats.skew(values):6.2f}  "
              f"zeros={np.mean(values == 0):.1%}")


def make_event_volatility_plot(start: str = EVENT_VOLATILITY_START) -> None:
    settings = get_settings()
    events, _ = load_gold_events()
    events = events[events["ts_utc"] >= start]
    apply_default_style()
    figure, axis = plt.subplots(figsize=(10.5, 4.3))
    colors = event_type_colors()
    for event_type, color in colors.items():
        subset = events[events["event_type"] == event_type]
        times = subset["ts_utc"].dt.tz_convert(None)
        axis.plot(times, subset[TARGET_COLUMN], color=color, lw=0.9, alpha=0.5, zorder=1)
        axis.scatter(times, subset[TARGET_COLUMN], s=20, color=color, edgecolor=SURFACE,
                     linewidth=0.7, label=event_type, zorder=3)
    axis.set_ylim(bottom=0)
    axis.set_ylabel(f"{settings.post_window_seconds} s realized volatility after the release (%)")
    clean_axes(axis, grid="y")
    axis.legend(loc="upper right")
    figure.suptitle(f"{settings.asset}: {settings.post_window_seconds} s realized volatility after each "
                    f"{' / '.join(colors)} release",
                    x=0.01, ha="left", fontsize=11, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(figure, RESULTS_FIGURES_OUTPUT_DIR / "fig_event_vol.png", dpi=200)


FIGURES = {
    "distribution": make_distribution_plot,
    "event-volatility": make_event_volatility_plot,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=[*FIGURES, "all"], help="which figure to make")
    parser.add_argument("--start", default=EVENT_VOLATILITY_START, help="first date of the event-volatility plot")
    args = parser.parse_args()
    names = FIGURES if args.figure == "all" else [args.figure]
    for name in names:
        FIGURES[name](start=args.start) if name == "event-volatility" else FIGURES[name]()


if __name__ == "__main__":
    main()
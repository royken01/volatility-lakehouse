"""Volatility around the scheduled releases, from the silver tables.

The tick log returns around each release are cut into fixed-width buckets ([k, k + width) seconds
relative to t), with ``abs_log_return`` (sum of |returns|) and ``realized_vol`` (sqrt of the summed
squares) per bucket, in %. Empty buckets count as 0; releases with fewer than ``--min-ticks`` ticks
are skipped.  Bucket values are averaged over releases, combined and per event type.

Figures (``lakehouse/outputs/surprise_volatility/``, png and pdf): ``returns_releases`` (mean |log
returns| per 5-second window) and ``volatility_minutes_releases`` (realized volatility per window
from ``--minutes-before`` to ``--minutes-after``).  Tables: ``profile_short_window.csv`` and
``profile_minutes.csv``.  The plotted range is clamped to the +-``TICK_WINDOW_SECONDS`` that
silver.prices_tick holds.

    vol-lakehouse surprise-volatility
    vol-lakehouse surprise-volatility --minutes-before 3 --minutes-after 5
    vol-lakehouse surprise-volatility --event-types NFP
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

from lakehouse.analysis.surprise_reaction import (
    EVENT_TYPE_COLORS, EVENT_TYPES, event_type_of, load_silver_frames, prepare_tick_returns,
)
from lakehouse.medallion.silver.price_processing import TICK_WINDOW_SECONDS
from lakehouse.paths import SURPRISE_VOLATILITY_OUTPUT_DIR
from lakehouse.plot_style import AXIS, BLUE, RED, apply_default_style, clean_axes, save_figure

DEFAULT_MIN_TICKS = 5
SILVER_WINDOW_SECONDS = TICK_WINDOW_SECONDS


def release_times(events: pd.DataFrame, event_types) -> pd.DataFrame:
    """Release time (UTC) and configured type per release; other event names and missing times are
    dropped."""
    releases = events[["ts_utc", "event"]].copy()
    releases["event_type"] = releases["event"].map(event_type_of)
    releases["ts_utc"] = pd.to_datetime(releases["ts_utc"], utc=True, errors="coerce")
    releases = releases.dropna(subset=["ts_utc", "event_type"])
    releases = releases[releases["event_type"].isin(event_types)]
    return releases.drop(columns="event").drop_duplicates().sort_values("ts_utc").reset_index(drop=True)


def clamp_to_silver_window(requested_seconds: int, label: str, silver_seconds: int) -> int:
    """Silver has no ticks beyond +/-``silver_seconds``; clamp so missing data is not plotted as zero."""
    if requested_seconds > silver_seconds:
        print(f"note: {label} limited to {silver_seconds} s - silver.prices_tick only holds "
              f"+/-{silver_seconds} s around each release (requested {requested_seconds} s)")
        return silver_seconds
    return requested_seconds


def bucketed_event_profile(returns: pd.DataFrame, releases: pd.DataFrame,
                           seconds_before: int, seconds_after: int,
                           bucket_seconds: int, min_ticks: int) -> pd.DataFrame:
    """One row per (release, bucket): ``abs_log_return`` and ``realized_vol`` in %; empty buckets
    are 0."""
    index = returns.index
    buckets = np.arange(-seconds_before, seconds_after, bucket_seconds)
    frames = []
    for release_number, (release_time, event_type) in enumerate(
            zip(releases["ts_utc"], releases["event_type"])):
        start = index.searchsorted(release_time - pd.Timedelta(seconds=seconds_before),
                                   side="left")
        stop = index.searchsorted(release_time + pd.Timedelta(seconds=seconds_after),
                                  side="left")
        if stop - start < min_ticks:
            continue
        window = returns.iloc[start:stop]
        offsets = ((window.index - release_time) / pd.Timedelta(seconds=bucket_seconds)).to_numpy()
        bucket_of_tick = np.floor(offsets).astype(int) * bucket_seconds
        absolute_sum = (window["ret"].abs().groupby(bucket_of_tick)
                        .sum().reindex(buckets, fill_value=0.0))
        squared_sum = (window["squared_ret"].groupby(bucket_of_tick)
                       .sum().reindex(buckets, fill_value=0.0))
        frames.append(pd.DataFrame({
            "release": release_number,
            "event_type": event_type,
            "bucket": buckets,
            "abs_log_return": absolute_sum.to_numpy(),
            "realized_vol": np.sqrt(squared_sum.to_numpy()),
        }))
    if not frames:
        raise SystemExit("No release has enough ticks around it - check silver.prices_tick and the window")
    return pd.concat(frames, ignore_index=True)


def summarise_profile(profile: pd.DataFrame) -> pd.DataFrame:
    """Per type (and 'all') and bucket: release count, mean |log return| and mean RV."""
    stacked = pd.concat([profile, profile.assign(event_type="all")], ignore_index=True)
    summary = stacked.groupby(["event_type", "bucket"]).agg(
        n_releases=("release", "nunique"),
        mean_abs_log_return=("abs_log_return", "mean"),
        mean_realized_vol=("realized_vol", "mean"),
    )
    return summary.reset_index()


def panel_types(summary: pd.DataFrame, event_types) -> list[str]:
    """Requested types that have data (the combined 'all' rows are not a panel)."""
    return [t for t in event_types if (summary["event_type"] == t).any()]


def plot_absolute_returns(summary: pd.DataFrame, event_types, bucket_seconds: int, path) -> None:
    """Bars of mean aggregated |log returns| per window over all releases combined."""
    rows = summary[summary["event_type"] == "all"]
    types = " and ".join(panel_types(summary, event_types))
    figure, axis = plt.subplots(figsize=(8.0, 4.4))
    axis.bar(rows["bucket"] + bucket_seconds / 2, rows["mean_abs_log_return"],
             width=bucket_seconds * 0.9, color=BLUE, alpha=0.7, zorder=3)
    axis.axvline(0, color=RED, ls="--", lw=1.2, zorder=4)
    axis.set_title(f"{types} releases combined (n = {int(rows['n_releases'].max())})")
    axis.set_xlabel("Seconds relative to release")
    axis.set_ylabel("Mean aggregated |log returns| %")
    clean_axes(axis, grid="y")
    figure.suptitle("Absolute Log Returns Surrounding Event Releases", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(figure, path, also_pdf=True)


def plot_volatility_minutes(summary: pd.DataFrame, event_types, bucket_seconds: int, path) -> None:
    """Mean realized volatility per window around the release, one panel per type with a shared
    y-axis; x in minutes at window centres."""
    panels = panel_types(summary, event_types)
    figure, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.4),
                                sharey=True, squeeze=False)
    for axis, panel in zip(axes[0], panels):
        rows = summary[summary["event_type"] == panel]
        minutes = (rows["bucket"] + bucket_seconds / 2) / 60
        axis.plot(minutes, rows["mean_realized_vol"],
                  color=EVENT_TYPE_COLORS.get(panel, BLUE), lw=1.8, marker="o",
                  markersize=3.5, zorder=3)
        axis.axvline(0, color=RED, ls="--", lw=1.2, zorder=4)
        axis.axhline(0, color=AXIS, lw=0.8)
        axis.set_title(f"{panel} releases (n = {int(rows['n_releases'].max())})")
        axis.set_xlabel("Minutes relative to release")
        axis.xaxis.set_major_locator(MultipleLocator(max(1, int(np.ceil(np.ptp(minutes) / 10)))))
        clean_axes(axis, grid="y")
    axes[0, 0].set_ylabel("Mean RV %")
    figure.suptitle("Volatility in the minutes before and after the releases", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(figure, path, also_pdf=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--event-types", nargs="+", default=list(EVENT_TYPES),
                        choices=list(EVENT_TYPES))
    parser.add_argument("--short-window-minutes", type=float, default=10,
                        help="returns plot: minutes before and after (default 10)")
    parser.add_argument("--short-bucket-seconds", type=int, default=5,
                        help="returns plot: window width in seconds (default 5)")
    parser.add_argument("--minutes-before", type=int, default=10,
                        help="volatility plot: minutes before the release (default 10)")
    parser.add_argument("--minutes-after", type=int, default=10,
                        help="volatility plot: minutes after the release (default 10)")
    parser.add_argument("--minute-bucket-seconds", type=int, default=30,
                        help="volatility plot: window width in seconds (default 30)")
    parser.add_argument("--silver-window-seconds", type=int, default=SILVER_WINDOW_SECONDS,
                        help="half-width of the tick windows stored in silver")
    parser.add_argument("--min-ticks", type=int, default=DEFAULT_MIN_TICKS,
                        help="releases with fewer ticks in the window are skipped")
    args = parser.parse_args()

    apply_default_style(font_size=9)
    raw_events, raw_ticks = load_silver_frames()
    releases = release_times(raw_events, args.event_types)
    returns = prepare_tick_returns(raw_ticks)
    print(f"{len(releases)} releases ({releases['event_type'].value_counts().to_dict()})")

    silver_seconds = args.silver_window_seconds
    short_seconds = clamp_to_silver_window(int(args.short_window_minutes * 60),
                                           "returns plot window", silver_seconds)
    seconds_before = clamp_to_silver_window(args.minutes_before * 60, "minutes before",
                                            silver_seconds)
    seconds_after = clamp_to_silver_window(args.minutes_after * 60, "minutes after",
                                           silver_seconds)
    short_profile = bucketed_event_profile(returns, releases, short_seconds, short_seconds,
                                           args.short_bucket_seconds, args.min_ticks)
    minute_profile = bucketed_event_profile(returns, releases, seconds_before, seconds_after,
                                            args.minute_bucket_seconds, args.min_ticks)
    short_summary = summarise_profile(short_profile)
    minute_summary = summarise_profile(minute_profile)

    output_dir = SURPRISE_VOLATILITY_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_absolute_returns(short_summary, args.event_types, args.short_bucket_seconds,
                          output_dir / "returns_releases.png")
    plot_volatility_minutes(minute_summary, args.event_types, args.minute_bucket_seconds,
                            output_dir / "volatility_minutes_releases.png")
    short_summary.to_csv(output_dir / "profile_short_window.csv", index=False)
    minute_summary.to_csv(output_dir / "profile_minutes.csv", index=False)

    used = minute_profile.groupby("event_type")["release"].nunique().to_dict()
    print(f"releases with enough ticks in the {seconds_before}+{seconds_after} second window: {used}")
    before = (minute_summary[minute_summary["bucket"] < 0]
              .groupby("event_type")["mean_realized_vol"].mean())
    after = (minute_summary[minute_summary["bucket"] >= 0]
             .groupby("event_type")["mean_realized_vol"].mean())
    print("mean realized volatility per window, before -> after the release (%):")
    for event_type in before.index:
        print(f"  {event_type:<4} {before[event_type]:.4f} -> {after[event_type]:.4f}")
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
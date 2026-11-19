"""How the market reacts to release surprises (actual - forecast), split by release type.

Figures: ``reaction_by_event_type`` (post-release volatility within surprise bins),
``event_time_profiles_by_event_type`` (average volatility per second around the release, negative
vs positive surprises) and ``event_type_profile_comparison`` (the types' profiles on one axis).
Tables: ``post_event_reactions.csv``, ``dispersion_summary.csv`` and ``reaction_tests.csv``
(Spearman tests and a Mann-Whitney test between the first two types).

Conventions match the gold features: tick rows only, ``post_rv`` over [t, t + W] equals the gold
target (W = ``windows.post_event_seconds``), and surprises are z-scored within each release type
unless ``--pooled-z`` is given.  Events with fewer than ``MIN_PROFILE_TICKS`` ticks are skipped
from the profiles.

    vol-lakehouse surprise-reaction
    vol-lakehouse surprise-reaction --event-types NFP
    vol-lakehouse surprise-reaction --pooled-z --threshold 0.5

Outputs go to ``lakehouse/outputs/surprise_reaction/``.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from lakehouse.infrastructure.iceberg_io import load_iceberg_table
from lakehouse.infrastructure.spark_session import SILVER_SCHEMA
from lakehouse.paths import SURPRISE_REACTION_OUTPUT_DIR
from lakehouse.plot_style import (
    AXIS, BLUE, INK, RED, SURFACE, apply_default_style, clean_axes, event_type_colors,
)
from lakehouse.settings import get_settings

_settings = get_settings()
EVENT_TYPES = tuple(t.label for t in _settings.resolved_event_types)
EVENT_TYPE_COLORS = event_type_colors()
MIN_PROFILE_TICKS = 5


def event_type_of(event_name) -> str | None:
    """Configured label of a calendar event name (None for events that are not a configured type)."""
    event_type = _settings.event_type_of(event_name)
    return event_type.label if event_type else None


# Data preparation

def load_silver_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """silver.events and the tick returns of silver.prices_tick (fallback bars removed)."""
    import polars as pl

    events = load_iceberg_table(SILVER_SCHEMA, "events").to_pandas()
    ticks = load_iceberg_table(SILVER_SCHEMA, "prices_tick")
    total_rows = len(ticks)
    if "is_tick" in ticks.columns:
        ticks = ticks.filter(pl.col("is_tick").fill_null(False))
    elif "bid" in ticks.columns:  # fallback bars have no bid / ask
        ticks = ticks.filter(pl.col("bid").is_not_null())
    print(f"silver.prices_tick: {len(ticks):,} tick rows kept of {total_rows:,} "
          f"(1-minute fallback bars dropped)")
    return events, ticks.select(["ts_utc", "log_ret"]).to_pandas()


def prepare_events(events: pd.DataFrame, event_types, z_score_by_type: bool) -> pd.DataFrame:
    """Events with ts_utc, surprise (actual - forecast if absent), event_type and the z-scored
    surprise (per type unless ``z_score_by_type`` is False)."""
    prepared = events.copy()
    prepared["event_type"] = prepared["event"].map(event_type_of).fillna("other")
    if "surprise" not in prepared.columns:
        prepared["surprise"] = (
            pd.to_numeric(prepared["actual"], errors="coerce")
            - pd.to_numeric(prepared["forecast"], errors="coerce")
        )
    prepared["ts_utc"] = pd.to_datetime(prepared["ts_utc"], utc=True, errors="coerce")
    prepared["surprise"] = pd.to_numeric(prepared["surprise"], errors="coerce")
    prepared = prepared.dropna(subset=["ts_utc", "surprise"])
    prepared = (prepared[prepared["event_type"].isin(event_types)]
                .sort_values("ts_utc").reset_index(drop=True))
    if z_score_by_type:
        group = prepared.groupby("event_type")["surprise"]
        mean, deviation = group.transform("mean"), group.transform("std")
    else:
        mean = prepared["surprise"].mean()
        deviation = prepared["surprise"].std(ddof=1)
    prepared["surprise_z"] = (prepared["surprise"] - mean) / deviation
    return prepared[["ts_utc", "surprise", "surprise_z", "event_type"]]


def prepare_tick_returns(ticks: pd.DataFrame) -> pd.DataFrame:
    """Tick log returns indexed by UTC time with their squares; duplicate timestamps are kept."""
    returns = ticks[["ts_utc", "log_ret"]].copy()
    returns["ts_utc"] = pd.to_datetime(returns["ts_utc"], utc=True, errors="coerce")
    returns["ret"] = pd.to_numeric(returns["log_ret"], errors="coerce")
    returns = (returns.dropna(subset=["ts_utc", "ret"])
               .sort_values("ts_utc", kind="mergesort")
               .set_index("ts_utc"))
    returns["squared_ret"] = returns["ret"] ** 2
    return returns[["ret", "squared_ret"]]


# Computations

def post_event_volatility(returns: pd.DataFrame, events: pd.DataFrame,
                          end_second: int = 30) -> pd.DataFrame:
    """Add ``post_rv`` = sqrt(sum of squared tick returns in [t, t + end_second]) per event; NaN
    without a tick in the window."""
    index = returns.index
    cumulative = np.concatenate([[0.0], np.cumsum(returns["squared_ret"].to_numpy())])
    start = index.searchsorted(events["ts_utc"], side="left")
    stop = index.searchsorted(events["ts_utc"] + pd.Timedelta(seconds=end_second), side="right")
    volatility = np.where(
        stop > start,
        np.sqrt(np.maximum(cumulative[stop] - cumulative[start], 0.0)),
        np.nan,
    )
    return events.assign(post_rv=volatility)


def event_time_profile(returns: pd.DataFrame, events: pd.DataFrame,
                       seconds_before: int, seconds_after: int) -> pd.Series:
    """Average volatility per second relative to the release (bucket k covers [k, k+1) s)."""
    index = returns.index
    offsets = np.arange(-seconds_before, seconds_after)
    per_event = []
    for release_time in events["ts_utc"]:
        start = index.searchsorted(release_time - pd.Timedelta(seconds=seconds_before),
                                   side="left")
        stop = index.searchsorted(release_time + pd.Timedelta(seconds=seconds_after),
                                  side="left")
        if stop - start < MIN_PROFILE_TICKS:
            continue
        window = returns.iloc[start:stop]
        second = np.floor((window.index - release_time) / pd.Timedelta(seconds=1)).astype(int)
        squared_by_second = (
            window["squared_ret"].groupby(second).sum().reindex(offsets, fill_value=0.0)
        )
        per_event.append(np.sqrt(squared_by_second))
    return pd.concat(per_event, axis=1).mean(axis=1) if per_event else pd.Series(dtype=float)


def symmetric_bin_edges(max_abs: float = 3.0, step: float = 0.5) -> np.ndarray:
    """[-max_abs, ..., -step, 0, step, ..., max_abs]."""
    count = int(np.round(max_abs / step))
    return np.arange(-count * step, (count + 1) * step + 1e-12, step)


# Plots

def plot_dispersion_by_surprise(reactions: pd.DataFrame, axis, title: str,
                                bin_step: float, min_per_bin: int) -> None:
    """Mean (dot) and 10-90% band per surprise bin; blue positive, red negative."""
    data = reactions[["surprise_z", "post_rv"]].dropna().copy()
    data["bin"] = pd.cut(data["surprise_z"], bins=symmetric_bin_edges(3.0, bin_step),
                         include_lowest=True)
    grouped = data.groupby("bin", observed=True)["post_rv"]
    table = (
        grouped.agg(["mean", "count"])
        .join(grouped.quantile(0.1).rename("low"))
        .join(grouped.quantile(0.9).rename("high"))
    )
    table = table[table["count"] >= min_per_bin].reset_index()
    table["center"] = (
        table["bin"].astype(object).map(lambda b: (float(b.left) + float(b.right)) / 2).astype(float)
    )
    for row in table.itertuples():
        color = BLUE if row.center > 0 else RED
        axis.vlines(row.center, row.low, row.high, color=color, linewidth=1.6, alpha=0.9)
        axis.hlines([row.low, row.high], row.center - 0.06, row.center + 0.06,
                    color=color, linewidth=1.6)
        axis.scatter(row.center, row.mean, color=color, s=26, zorder=3,
                     edgecolor=SURFACE, linewidth=1.0)
    axis.axvline(0, color=AXIS, lw=1)
    axis.set(xlabel="Surprise (z-score bins)",
             ylabel=f"Post-event realized volatility (0-{_settings.post_window_seconds} s)", title=title)
    clean_axes(axis, grid="y")


def plot_profile_by_sign(returns, reactions, axis, title: str, threshold: float,
                         before: int, after: int) -> None:
    """Average per-second volatility around the release for negative vs positive surprises."""
    strong = reactions[reactions["surprise_z"].abs() >= threshold]
    panels = [
        ("Negative surprise", strong[strong["surprise_z"] < 0], RED),
        ("Positive surprise", strong[strong["surprise_z"] > 0], BLUE),
    ]
    for label, subset, color in panels:
        if subset.empty:
            continue
        profile = event_time_profile(returns, subset, before, after)
        axis.plot(profile.index.to_numpy(), profile.to_numpy(), color=color, lw=1.6,
                  label=f"{label} (n={len(subset)})")
    axis.axvline(0, color=AXIS, lw=1)
    axis.set(xlabel="Seconds relative to release", ylabel="Mean rv per second", title=title)
    clean_axes(axis, grid="y")
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=8)


def plot_type_comparison(returns, reactions, axis, event_types, before: int, after: int) -> None:
    """The release types on one axis: average profile of all releases of each type."""
    for event_type in event_types:
        subset = reactions[reactions["event_type"] == event_type]
        if subset.empty:
            continue
        profile = event_time_profile(returns, subset, before, after)
        axis.plot(profile.index.to_numpy(), profile.to_numpy(),
                  color=EVENT_TYPE_COLORS.get(event_type, INK), lw=1.8,
                  label=f"{event_type} (n={len(subset)})")
    axis.axvline(0, color=AXIS, lw=1)
    axis.set(xlabel="Seconds relative to release", ylabel="Mean rv per second",
             title=" vs ".join(event_types) + ": average event-time volatility")
    clean_axes(axis, grid="y")
    axis.legend(fontsize=9)


# Tables

def summarise_reaction(reactions: pd.DataFrame) -> pd.DataFrame:
    """Per type and sign (all / negative / positive): n, mean, median, std, 10% and 90%."""
    rows = []
    for event_type, group in reactions.groupby("event_type"):
        subsets = [
            ("all", group),
            ("negative", group[group["surprise_z"] < 0]),
            ("positive", group[group["surprise_z"] > 0]),
        ]
        for sign, subset in subsets:
            values = subset["post_rv"].dropna()
            rows.append({
                "event_type": event_type,
                "surprise_sign": sign,
                "n": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(ddof=1),
                "q10": values.quantile(0.1),
                "q90": values.quantile(0.9),
            })
    return pd.DataFrame(rows)


def reaction_statistics(reactions: pd.DataFrame) -> pd.DataFrame:
    """Descriptive Spearman correlations with |surprise| and signed surprise, plus a level test
    between the first two release types."""
    rows = []
    for event_type, group in reactions.groupby("event_type"):
        data = group[["post_rv", "surprise_z"]].dropna()
        row = {"comparison": event_type, "n": len(data)}
        if len(data) >= 10:
            rho_abs, p_abs = stats.spearmanr(data["surprise_z"].abs(), data["post_rv"])
            rho_signed, p_signed = stats.spearmanr(data["surprise_z"], data["post_rv"])
            row["spearman_abs_surprise"] = rho_abs
            row["p_abs_surprise"] = p_abs
            row["spearman_signed_surprise"] = rho_signed
            row["p_signed_surprise"] = p_signed
        rows.append(row)
    present = [t for t in EVENT_TYPES if (reactions["event_type"] == t).any()]
    if len(present) >= 2:
        first_type, second_type = present[:2]
        first = reactions.loc[reactions["event_type"] == first_type, "post_rv"].dropna()
        second = reactions.loc[reactions["event_type"] == second_type, "post_rv"].dropna()
        if len(first) >= 10 and len(second) >= 10:
            _, p_value = stats.mannwhitneyu(first, second, alternative="two-sided")
            rows.append({
                "comparison": f"{first_type} vs {second_type} (Mann-Whitney U on reaction level)",
                "n": len(first) + len(second),
                f"median_{first_type}": first.median(),
                f"median_{second_type}": second.median(),
                "p_level_diff": p_value,
            })
    return pd.DataFrame(rows)


def save_with_pdf(figure, stem) -> None:
    for extension, options in (("png", {"dpi": 200}), ("pdf", {})):
        figure.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight", **options)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--event-types", nargs="+", default=list(EVENT_TYPES),
                        choices=list(EVENT_TYPES))
    parser.add_argument("--pooled-z", action="store_true",
                        help="z-score surprises across all types, not within each")
    parser.add_argument("--post-end", type=int, default=_settings.post_window_seconds,
                        help=f"reaction window end in seconds (default {_settings.post_window_seconds} = "
                             f"{_settings.target_column})")
    parser.add_argument("--pre", type=int, default=30, help="profile: seconds before the release")
    parser.add_argument("--post", type=int, default=60, help="profile: seconds after the release")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="profile by sign: only |z| >= threshold")
    parser.add_argument("--bin-step", type=float, default=0.5,
                        help="surprise bin width in z units")
    parser.add_argument("--min-per-bin", type=int, default=8,
                        help="bins with fewer events are not drawn")
    args = parser.parse_args()

    apply_default_style(font_size=9)
    raw_events, raw_ticks = load_silver_frames()
    events = prepare_events(raw_events, args.event_types, z_score_by_type=not args.pooled_z)
    returns = prepare_tick_returns(raw_ticks)
    reactions = post_event_volatility(returns, events, args.post_end)
    present = [t for t in args.event_types if (reactions["event_type"] == t).any()]

    output_dir = SURPRISE_REACTION_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, len(present), figsize=(7 * len(present), 4.4),
                                sharey=True, squeeze=False)
    for axis, event_type in zip(axes[0], present):
        subset = reactions[reactions["event_type"] == event_type]
        title = (f"{event_type}: volatility within surprise bins "
                 f"(n={int(subset['post_rv'].notna().sum())})")
        plot_dispersion_by_surprise(subset, axis, title, args.bin_step, args.min_per_bin)
    figure.tight_layout()
    save_with_pdf(figure, output_dir / "reaction_by_event_type")

    figure, axes = plt.subplots(1, len(present), figsize=(7 * len(present), 4.4),
                                sharey=True, squeeze=False)
    for axis, event_type in zip(axes[0], present):
        plot_profile_by_sign(
            returns, reactions[reactions["event_type"] == event_type], axis,
            f"{event_type}: event-time volatility (|z| >= {args.threshold})",
            args.threshold, args.pre, args.post,
        )
    figure.tight_layout()
    save_with_pdf(figure, output_dir / "event_time_profiles_by_event_type")

    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    plot_type_comparison(returns, reactions, axis, present, args.pre, args.post)
    figure.tight_layout()
    save_with_pdf(figure, output_dir / "event_type_profile_comparison")

    summary = summarise_reaction(reactions)
    tests = reaction_statistics(reactions)
    reactions.to_csv(output_dir / "post_event_reactions.csv", index=False)
    summary.to_csv(output_dir / "dispersion_summary.csv", index=False)
    tests.to_csv(output_dir / "reaction_tests.csv", index=False)

    print(f"{len(reactions):,} events "
          f"({reactions['event_type'].value_counts().to_dict()}); "
          f"{int(reactions['post_rv'].isna().sum())} without ticks in the "
          f"{args.post_end} s window")
    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n" + tests.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
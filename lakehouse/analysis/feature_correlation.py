"""Correlation of candidate explanatory variables with the target, on the table the models train
on.

Sample: news-era events (from ``NEWS_ERA_START``) up to ``--until-year`` (default 2022, before the
test years) with ALL candidates observed, so every correlation uses the same events.  Per candidate
and event group (all events and each configured release type): Spearman correlation with the target,
its p-value and the Benjamini-Hochberg q-value.

Outputs in ``lakehouse/outputs/correlation/``: feature_target_correlations.csv,
feature_feature_spearman.csv, fig_spearman_vs_target and fig_feature_collinearity (png / pdf),
run_metadata.json (snapshot id, sample).

    python -m lakehouse.analysis.feature_correlation
    python -m lakehouse.analysis.feature_correlation --until-year 2026
    python -m lakehouse.analysis.feature_correlation --snapshot-id 123456789
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from statsmodels.stats.multitest import multipletests

from lakehouse.feature_columns import (
    HAR_COLUMNS, LEAKAGE_COLUMNS, NEWS_SENTIMENT_COLUMNS, RETURN_SEQUENCE_PREFIX,
    SEQUENCE_PREFIX, TARGET_COLUMN,
)
from lakehouse.modelling.feature_sets import EVENT_DUMMIES, LAST_HOUR
from lakehouse.modelling.gold_data import SOURCE_TABLE, load_gold_events
from lakehouse.paths import CORRELATION_OUTPUT_DIR
from lakehouse.settings import get_settings
from lakehouse.plot_style import (
    AXIS, BLUE, GRID, INK, MUTED_INK, NEUTRAL, RED, SECONDARY_INK, SURFACE,
    apply_default_style, save_figure,
)

NEWS_ERA_START = "2019-09-01"  # GDELT stores headlines from late September 2019
DEFAULT_UNTIL_YEAR = 2026
MINIMUM_GROUP_SIZE = 30
_settings = get_settings()
EVENT_GROUPS = ["all"] + ([t.label for t in _settings.resolved_event_types]
                          if len(_settings.resolved_event_types) > 1 else [])
NON_FEATURE_COLUMNS = ["ts_utc", "year", "event_type"]
# the reference type's dummy is implied by the others
NOT_SCREENED_COLUMNS = ["pre_ret_h"] + [d for d in _settings.all_event_dummies if d not in EVENT_DUMMIES]

# Feature families in plotting order; anything else is shown as "Other".
FAMILIES: dict[str, list[str]] = {
    "HAR-RV": HAR_COLUMNS,
    "Last hour": LAST_HOUR,
    "Consensus": ["forecast"],
    "News sentiment": NEWS_SENTIMENT_COLUMNS,
    "Event type": EVENT_DUMMIES,
}
OTHER_FAMILY = "Other"
DIVERGING_COLORMAP = LinearSegmentedColormap.from_list("blue_red", [RED, NEUTRAL, BLUE])


def family_of(column: str) -> str:
    return next((f for f, cols in FAMILIES.items() if column in cols), OTHER_FAMILY)


def news_era_sample(events: pd.DataFrame, until_year: int) -> tuple[pd.DataFrame, list[str]]:
    """News-era events up to ``until_year`` with every candidate observed; the lag columns are not
    screened."""
    lag_columns = [c for c in events.columns if c.startswith((SEQUENCE_PREFIX, RETURN_SEQUENCE_PREFIX))]
    events = events.drop(columns=lag_columns + NOT_SCREENED_COLUMNS, errors="ignore")
    window = events[
        (events["ts_utc"] >= pd.Timestamp(NEWS_ERA_START, tz="UTC"))
        & (events["year"] <= until_year)
    ]
    skipped = set(LEAKAGE_COLUMNS) | set(NON_FEATURE_COLUMNS)
    candidates = [
        c for c in window.columns
        if c not in skipped and pd.api.types.is_numeric_dtype(window[c])
    ]
    sample = (window.replace([np.inf, -np.inf], np.nan)
              .dropna(subset=candidates).reset_index(drop=True))
    constant = [c for c in candidates if sample[c].nunique() <= 1]
    if constant:
        print(f"  left out (constant in this sample): {constant}")
    candidates = [c for c in candidates if c not in constant]
    family_order = list(FAMILIES) + [OTHER_FAMILY]
    candidates = sorted(
        candidates,
        key=lambda c: (family_order.index(family_of(c)), candidates.index(c)),
    )
    return sample, candidates


def correlate_with_target(events: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Spearman correlation of each feature with the target, plus its p- and BH q-value."""
    rows = []
    for feature in features:
        if events[feature].nunique() > 1:
            rho, p_value = stats.spearmanr(events[feature], events[TARGET_COLUMN])
        else:
            rho, p_value = np.nan, np.nan
        rows.append({"feature": feature, "family": family_of(feature),
                     "spearman": rho, "p_spearman": p_value})
    table = pd.DataFrame(rows)
    testable = table["p_spearman"].notna()
    table["q_spearman"] = np.nan
    if testable.any():
        table.loc[testable, "q_spearman"] = multipletests(
            table.loc[testable, "p_spearman"], method="fdr_bh"
        )[1]
    return table


def correlation_table(events: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Target correlations for all events and each release type."""
    parts = []
    for group in EVENT_GROUPS:
        subset = events if group == "all" else events[events["event_type"] == group]
        if len(subset) >= MINIMUM_GROUP_SIZE:
            table = correlate_with_target(subset, features)
            table.insert(0, "group", group)
            table.insert(1, "n_events", len(subset))
            parts.append(table)
    return pd.concat(parts, ignore_index=True)


def plot_spearman_bars(table: pd.DataFrame, path: Path, subtitle: str) -> None:
    """Ranked Spearman correlations grouped by family (blue positive, red negative)."""
    apply_default_style(font_size=9)
    table = table[table["spearman"].notna()]
    rows: list[tuple[str, object]] = []
    for family in list(FAMILIES) + [OTHER_FAMILY]:
        members = table[table["family"] == family].sort_values("spearman", key=np.abs,
                                                               ascending=False)
        if not members.empty:
            rows.append(("heading", family))
            rows.extend(("feature", row) for _, row in members.iterrows())

    figure, axis = plt.subplots(figsize=(7.4, 0.27 * len(rows) + 1.6))
    tick_positions, tick_labels = [], []
    for position, (kind, item) in enumerate(rows):
        y = len(rows) - position
        if kind == "heading":
            axis.text(-0.012, y, item, transform=axis.get_yaxis_transform(),
                      ha="right", va="center", fontsize=8.5, fontweight="bold", color=INK)
            continue
        color = BLUE if item["spearman"] >= 0 else RED
        axis.barh(y, item["spearman"], height=0.62, color=color, edgecolor=color, zorder=3)
        tick_positions.append(y)
        tick_labels.append(item["feature"])
    axis.set_yticks(tick_positions)
    axis.set_yticklabels(tick_labels, fontsize=8)
    axis.tick_params(axis="y", length=0)
    axis.set_ylim(0.3, len(rows) + 0.8)
    axis.axvline(0, color=AXIS, lw=1, zorder=2)
    axis.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    axis.set_axisbelow(True)
    axis.set_xlabel(f"Spearman correlation with {TARGET_COLUMN}")
    axis.spines["left"].set_visible(False)
    axis.set_title("Explanatory variables vs target", loc="left", pad=22)
    axis.text(0, 1.012, subtitle, transform=axis.transAxes, fontsize=8, color=MUTED_INK,
              va="bottom")
    save_figure(figure, path, dpi=220, also_pdf=True)


def plot_collinearity(matrix: pd.DataFrame, path: Path, subtitle: str) -> None:
    """Spearman heatmap between candidates; family boundaries are white lines."""
    apply_default_style(font_size=9)
    size = len(matrix)
    side = max(6.0, 0.24 * size + 2.2)
    figure, axis = plt.subplots(figsize=(side, side * 0.92))
    image = axis.imshow(matrix.values, cmap=DIVERGING_COLORMAP, vmin=-1, vmax=1,
                        interpolation="nearest")
    axis.set_xticks(range(size))
    axis.set_yticks(range(size))
    axis.set_xticklabels(matrix.columns, rotation=90, fontsize=6.5)
    axis.set_yticklabels(matrix.index, fontsize=6.5)
    axis.tick_params(length=0)
    families = [family_of(c) for c in matrix.columns]
    for boundary in range(1, size):
        if families[boundary] != families[boundary - 1]:
            axis.axhline(boundary - 0.5, color=SURFACE, lw=1.6)
            axis.axvline(boundary - 0.5, color=SURFACE, lw=1.6)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("Spearman rho", color=SECONDARY_INK)
    colorbar.outline.set_visible(False)
    axis.set_title("Collinearity between candidate variables", loc="left", pad=22)
    axis.text(0, 1.012, subtitle, transform=axis.transAxes, fontsize=8, color=MUTED_INK,
              va="bottom")
    save_figure(figure, path, dpi=220, also_pdf=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot-id", type=int, default=None,
                        help="older Iceberg snapshot of gold (default: latest)")
    parser.add_argument("--until-year", type=int, default=DEFAULT_UNTIL_YEAR,
                        help=f"last year included (default {DEFAULT_UNTIL_YEAR})")
    parser.add_argument("--out-dir", type=Path, default=CORRELATION_OUTPUT_DIR)
    args = parser.parse_args()

    events, snapshot_id = load_gold_events(args.snapshot_id)
    sample, features = news_era_sample(events, args.until_year)
    first, last = sample["ts_utc"].min(), sample["ts_utc"].max()
    print(f"gold.{SOURCE_TABLE} snapshot {snapshot_id}: {len(sample):,} news-era events "
          f"({first:%Y-%m-%d} to {last:%Y-%m-%d}), {len(features)} candidates vs {TARGET_COLUMN}")

    table = correlation_table(sample, features)
    matrix = sample[features].corr(method="spearman")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "feature_target_correlations.csv", index=False)
    matrix.to_csv(args.out_dir / "feature_feature_spearman.csv")

    subtitle = f"n = {len(sample):,} events ({first:%Y-%m} to {last:%Y-%m}), same events for every variable."
    all_events_table = table[table["group"] == "all"]
    plot_spearman_bars(all_events_table, args.out_dir / "fig_spearman_vs_target.png", subtitle)
    plot_collinearity(matrix, args.out_dir / "fig_feature_collinearity.png", subtitle)

    metadata = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gold_table": f"gold.{SOURCE_TABLE}",
        "gold_snapshot_id": snapshot_id,
        "target": TARGET_COLUMN,
        "news_era_start": NEWS_ERA_START,
        "until_year": args.until_year,
        "events_used": len(sample),
        "first_event": first.isoformat(),
        "last_event": last.isoformat(),
        "events_by_type": sample["event_type"].value_counts().to_dict(),
        "candidates": features,
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    ranked = all_events_table.dropna(subset=["spearman"]).sort_values(
        "spearman", key=np.abs, ascending=False
    )
    print("\nCorrelations with the target (all events):")
    print(ranked[["feature", "spearman", "q_spearman"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"\nSaved outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
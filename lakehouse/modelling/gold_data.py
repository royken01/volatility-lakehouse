"""Loading ``gold.events_features`` as a model-ready pandas frame: the one place this is done."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lakehouse.feature_columns import LEAKAGE_COLUMNS, NEWS_SENTIMENT_COLUMNS, TARGET_COLUMN
from lakehouse.infrastructure.iceberg_io import load_iceberg_table_with_snapshot
from lakehouse.infrastructure.spark_session import GOLD_SCHEMA
from lakehouse.settings import get_settings

SOURCE_TABLE = "events_features"
DEFAULT_TEST_YEAR_COUNT = 4   # the last 4 calendar years of data are the walk-forward test years


def invalid_target_mask(events: pd.DataFrame) -> pd.Series:
    """True where the target is not a genuine post-event volatility (no tick in the window, or a
    minute-bar fallback)."""
    invalid = pd.to_numeric(events["n_post_ticks"], errors="coerce").eq(0)
    if "fallback_bars" in events.columns:
        invalid |= pd.to_numeric(events["fallback_bars"], errors="coerce").eq(1)
    return invalid


def load_gold_events(snapshot_id: int | None = None, drop_invalid_target: bool = True):
    """Read gold and can optionally use older Iceberg snapshot id
    """
    gold_table, snapshot_id = load_iceberg_table_with_snapshot(GOLD_SCHEMA, SOURCE_TABLE, snapshot_id)
    return model_frame(gold_table.to_pandas(), drop_invalid_target), snapshot_id


def model_frame(events: pd.DataFrame, drop_invalid_target: bool = True) -> pd.DataFrame:
    """A gold table (pandas) -> the time-sorted modelling frame with ``year`` and ``event_type``,
    without events whose target is invalid."""
    if TARGET_COLUMN not in events.columns:
        built_with = sorted(c for c in events.columns if c.startswith("realized_vol_"))
        raise KeyError(f"{GOLD_SCHEMA}.{SOURCE_TABLE} has {built_with}, not {TARGET_COLUMN}: gold was built with "
                       f"another post-event window. Use the same --post-window-seconds / config as for gold, "
                       f"or rebuild gold (vol-lakehouse gold).")
    events["ts_utc"] = pd.to_datetime(events["ts_utc"], utc=True)
    events[TARGET_COLUMN] = pd.to_numeric(events[TARGET_COLUMN], errors="coerce").astype("float64")
    events = events.dropna(subset=[TARGET_COLUMN]).sort_values("ts_utc").reset_index(drop=True)
    events["year"] = events["ts_utc"].dt.year
    if "event_type" not in events.columns:   # gold tables built before the event_type column existed
        types = get_settings().resolved_event_types
        present = [t for t in types if t.dummy in events.columns]
        events["event_type"] = np.select([events[t.dummy] == 1 for t in present], [t.label for t in present],
                                         default="other") if present else types[0].label

    invalid = invalid_target_mask(events) if drop_invalid_target else pd.Series(False, index=events.index)
    if invalid.any():
        print(f"  dropped {int(invalid.sum()):,} of {len(events):,} events with an invalid target: "
              f"{events.loc[invalid, 'event_type'].value_counts().to_dict()}")
    events.attrs["n_invalid_target_dropped"] = int(invalid.sum())
    return events.loc[~invalid].reset_index(drop=True)


def prepare_features(events: pd.DataFrame, feature_columns: list[str], news_columns: list[str]) -> pd.DataFrame:
    """Validate the requested columns and give them one numeric dtype

        Raises error on a post-event column (leakage), a news column used as a tier A feature, or a
        missing column. This is the leakage prevented by construction as I refered to in the appendix of the thesis
    """
    requested = list(dict.fromkeys(feature_columns + news_columns))
    leaking = [column for column in requested if column in LEAKAGE_COLUMNS]
    if leaking:
        raise ValueError(f"Post-event columns cannot be features: {leaking}")
    news_in_tier_a = [column for column in feature_columns if column in NEWS_SENTIMENT_COLUMNS]
    if news_in_tier_a:
        raise ValueError(f"News columns {news_in_tier_a} exist only from late 2019 and cannot be tier A "
                         f"features; pass them as --news-features")
    missing = [column for column in requested if column not in events.columns]
    if missing:
        raise KeyError(f"{GOLD_SCHEMA}.{SOURCE_TABLE} is missing {missing}; rebuild gold: "
                       f"python -m lakehouse.medallion.gold.gold_features")
    prepared = events.copy()
    prepared[requested] = prepared[requested].apply(pd.to_numeric, errors="coerce").astype("float64")
    return prepared.sort_values("ts_utc").reset_index(drop=True)


def default_test_years(events: pd.DataFrame, count: int = DEFAULT_TEST_YEAR_COUNT) -> list[int]:
    """The last ``count`` calendar years in the data."""
    return [int(year) for year in sorted(events["year"].unique())[-count:]]


def default_validation_years(events: pd.DataFrame, test_year_count: int = DEFAULT_TEST_YEAR_COUNT) -> list[int]:
    """Every year before the test years (used by feature selection so the test years stay untouched)."""
    return [int(year) for year in sorted(events["year"].unique())[:-test_year_count]]

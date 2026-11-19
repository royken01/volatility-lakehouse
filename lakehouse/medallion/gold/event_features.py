"""Gold feature engineering: one row per scheduled release in ``gold.events_features``.

Everything is measured strictly before the release, except the target and the post-event
bookkeeping columns (``feature_columns.LEAKAGE_COLUMNS``).  ``<W>`` is the post-event window of
the settings (``windows.post_event_seconds``, e.g. ``30s``), ``N`` the number of lags:

    event_type, forecast, <one 0/1 dummy per event type>
        Release type label (from ``[events] types``), consensus forecast and type dummies.
    pre_vol_h
        Realized volatility of the ``pre_event_minutes`` before the release (1-minute bars).
    rv_d, rv_w, rv_m
        HAR-RV terms (Corsi 2009) of the last complete trading session.
    realized_vol_<W>
        Target: realized volatility in [release, release + W], from real tick rows only (never
        the minute fallback bars).
    pre_vol_<W>_lag1..N
        Target of the previous 1..N releases of the same event type.
    news_*
        Pre-event news sentiment features (see ``news_features.py``).

Trading sessions roll over at ``session_open_hour`` in ``session_timezone`` and are labelled by
the date they close on.
"""
import inspect

import numpy as np
import polars as pl

from lakehouse.feature_columns import (
    RETURN_SUM_COLUMN,
    SEQUENCE_COLUMNS,
    SEQUENCE_LENGTH,
    SEQUENCE_PREFIX,
    TARGET_COLUMN,
)
from lakehouse.medallion.gold.news_features import pre_event_news_features
from lakehouse.settings import get_settings

_settings = get_settings()
SESSION_TIMEZONE = _settings.session_timezone
SESSION_OPEN_HOUR = _settings.session_open_hour

RV_SAMPLING_MINUTES = _settings.rv_sampling_minutes
MIN_SESSION_COVERAGE = _settings.min_session_coverage

HAR_WEEK_SESSIONS = _settings.har_week_sessions
HAR_MONTH_SESSIONS = _settings.har_month_sessions
POST_WINDOW_SECONDS = _settings.post_window_seconds
PRE_WINDOW_MINUTES = _settings.pre_window_minutes


def rolling_mean_full_window(column: str, window: int) -> pl.Expr:
    """Rolling mean that stays null until ``window`` values exist.

    Polars renamed the argument (``min_periods`` -> ``min_samples``) in 1.x,
    so the right name is picked at import time.
    """
    parameters = inspect.signature(pl.Expr.rolling_mean).parameters
    arg_name = "min_samples" if "min_samples" in parameters else "min_periods"
    return pl.col(column).rolling_mean(window, **{arg_name: window})


def realized_volatility(returns: np.ndarray) -> float:
    """sqrt of the sum of squared returns."""
    return float(np.sqrt(np.sum(returns ** 2)))


def trading_date(timestamp_column: str, time_zone: str | None) -> pl.Expr:
    """Session label of a timestamp.

    The timestamp is moved to New York local time and shifted so the 18:00
    open becomes midnight, which makes the label the closing date.
    """
    if time_zone:
        utc_time = pl.col(timestamp_column)
    else:
        utc_time = pl.col(timestamp_column).dt.replace_time_zone("UTC")
    local_time = (
        utc_time.dt.convert_time_zone(SESSION_TIMEZONE).dt.replace_time_zone(None)
    )
    return (local_time + pl.duration(hours=24 - SESSION_OPEN_HOUR)).dt.date()


def daily_har_table(minute_bars: pl.DataFrame) -> pl.DataFrame:
    """One row per complete trading session with rv_d, rv_w and rv_m.

    rv_d is the square root of the summed squared 5-minute returns. Sessions
    that cover fewer than ``MIN_SESSION_COVERAGE`` of the median bar count are
    treated as holidays or half days and dropped, so a partial session does
    not leak into a HAR term.
    """
    time_zone = minute_bars.schema["ts_utc"].time_zone
    labelled = minute_bars.select(
        "ts_utc",
        "log_ret",
        trading_date("ts_utc", time_zone).alias("trading_date"),
        pl.col("ts_utc")
        .dt.truncate(f"{RV_SAMPLING_MINUTES}m")
        .alias("bucket"),
    )
    bucket_returns = labelled.group_by("trading_date", "bucket").agg(
        pl.col("log_ret").sum().alias("bucket_return")
    )
    sessions = (
        bucket_returns.group_by("trading_date")
        .agg((pl.col("bucket_return") ** 2).sum().sqrt().alias("rv_d"))
        .join(
            labelled.group_by("trading_date").agg(pl.len().alias("bar_count")),
            on="trading_date",
        )
    )
    minimum_bars = MIN_SESSION_COVERAGE * sessions["bar_count"].median()
    return (
        sessions
        .filter(pl.col("bar_count") >= minimum_bars)
        .sort("trading_date")
        .with_columns(
            rolling_mean_full_window("rv_d", HAR_WEEK_SESSIONS).alias("rv_w"),
            rolling_mean_full_window("rv_d", HAR_MONTH_SESSIONS).alias("rv_m"),
        )
        .select("trading_date", "rv_d", "rv_w", "rv_m")
    )


def label_event_types(events: pl.DataFrame) -> pl.DataFrame:
    """Add ``event_type`` (the configured label) and keep only the configured release types.

    With no ``[events] types`` in the settings every row is one pooled type ``all``.
    """
    types = _settings.resolved_event_types
    if len(types) == 1 and types[0].name == "*":
        names = events["event"].unique().to_list() if "event" in events.columns else []
        if len(names) > 1:
            print(f"  NOTE: {len(names)} different events are pooled as one type; list them under "
                  f"[events] types in the config to get per-type lags and dummies: {sorted(map(str, names))[:10]}")
        return events.with_columns(pl.lit(types[0].label).alias("event_type"))
    mapping = {t.name: t.label for t in types}
    labelled = events.with_columns(
        pl.col("event").replace_strict(mapping, default=None, return_dtype=pl.Utf8).alias("event_type")
    )
    unused = labelled.filter(pl.col("event_type").is_null())
    if len(unused):
        print(f"  {len(unused):,} calendar rows are not a configured event type and are left out "
              f"(e.g. {unused['event'].unique().to_list()[:5]})")
    return labelled.filter(pl.col("event_type").is_not_null())


def build_features(minute_bars: pl.DataFrame, tick_prices: pl.DataFrame,
                   events: pl.DataFrame,
                   scored_headlines: pl.DataFrame | None = None) -> pl.DataFrame:
    """The gold table: one row per event with every feature above.

    HAR terms come from the last complete session strictly before the event
    (a backward ``join_asof``). The pre-event and target windows are cut from
    the sorted return arrays. Events are dropped when they lack
    ``har_month_sessions`` prior sessions, when no tick follows the release, or
    when fewer than ``lags`` earlier same-type releases exist.
    """
    minute_bars = minute_bars.sort("ts_utc")
    tick_prices = tick_prices.sort("ts_utc")
    # silver.prices_tick also holds 1-minute fallback bars (no bid): keep
    # only real ticks.
    if "is_tick" in tick_prices.columns:
        tick_prices = tick_prices.filter(pl.col("is_tick").fill_null(False))
    elif "bid" in tick_prices.columns:
        tick_prices = tick_prices.filter(pl.col("bid").is_not_null())

    events = label_event_types(events)
    event_time_zone = events.schema["ts_utc"].time_zone
    events = events.sort("ts_utc").with_columns(
        trading_date("ts_utc", event_time_zone).alias("trading_date")
    )

    # Backward asof join on (session date - 1 day). That gives the strictly
    # earlier session, which is what ``allow_exact_matches=False`` would do
    # on dates, but that argument is missing from older Polars versions.
    har_table = daily_har_table(minute_bars).rename({"trading_date": "har_date"})
    events = (
        events.with_columns(
            (pl.col("trading_date") - pl.duration(days=1)).alias("har_date")
        )
        .join_asof(har_table, on="har_date", strategy="backward")
        .drop("har_date")
    )

    minute_timestamps = minute_bars["ts_utc"].to_numpy()
    minute_returns = minute_bars["log_ret"].to_numpy()
    tick_timestamps = tick_prices["ts_utc"].to_numpy()
    tick_returns = tick_prices["log_ret"].to_numpy()
    event_timestamps = events["ts_utc"].to_numpy()

    pre_window_end = np.searchsorted(
        minute_timestamps, event_timestamps, side="left"
    )
    pre_window_start = np.searchsorted(
        minute_timestamps,
        event_timestamps - np.timedelta64(PRE_WINDOW_MINUTES, "m"),
        side="left",
    )
    post_window_start = np.searchsorted(
        tick_timestamps, event_timestamps, side="left"
    )
    post_window_end = np.searchsorted(
        tick_timestamps,
        event_timestamps + np.timedelta64(POST_WINDOW_SECONDS, "s"),
        side="right",
    )

    event_types = _settings.resolved_event_types
    rows = []
    skipped_without_ticks = []
    for event_index, event in enumerate(events.iter_rows(named=True)):
        # A full month of history is needed for rv_m.
        if event["rv_m"] is None:
            continue
        post_tick_count = (
            post_window_end[event_index] - post_window_start[event_index]
        )
        if post_tick_count == 0:
            skipped_without_ticks.append(event["ts_utc"])
            continue
        pre_returns = minute_returns[
            pre_window_start[event_index]:pre_window_end[event_index]
        ]
        post_returns = tick_returns[
            post_window_start[event_index]:post_window_end[event_index]
        ]
        rows.append({
            "ts_utc": event["ts_utc"],
            "event_type": event["event_type"],
            "forecast": event.get("forecast"),
            **{t.dummy: int(event["event_type"] == t.label) for t in event_types},
            "pre_vol_h": round(realized_volatility(pre_returns), 4),
            "rv_d": round(event["rv_d"], 6),
            "rv_w": round(event["rv_w"], 6),
            "rv_m": round(event["rv_m"], 6),
            RETURN_SUM_COLUMN: float(post_returns.sum()),
            TARGET_COLUMN: round(realized_volatility(post_returns), 4),
            "n_post_ticks": post_tick_count,
        })

    if skipped_without_ticks:
        print(
            f"WARNING: {len(skipped_without_ticks)} events skipped, "
            f"no tick in the {POST_WINDOW_SECONDS} s after the release "
            f"(tick feed gap or only fallback bars): "
            f"{[str(t) for t in skipped_without_ticks]}"
        )

    lagged_outcomes = [
        pl.col(TARGET_COLUMN)
        .shift(lag)
        .over("event_type")
        .alias(f"{SEQUENCE_PREFIX}{lag}")
        for lag in range(1, SEQUENCE_LENGTH + 1)
    ]
    features = (
        pl.DataFrame(rows)
        .with_columns(lagged_outcomes)
        .drop_nulls(SEQUENCE_COLUMNS)
    )

    if scored_headlines is not None:
        news = pre_event_news_features(features["ts_utc"], scored_headlines)
        features = pl.concat([features, news], how="horizontal")
    return features.sort("ts_utc")
"""Gold pre-event news features from sentiment-scored GDELT headlines.

Pure Polars / NumPy. Input is the scored headline table of silver (``[news] scored_table``)
(``seen_utc`` plus ``sentiment_score``). Only headlines with ``seen_utc``
strictly before the event are used:

    news_sent_ewma       exponentially weighted mean score
    news_sent_abs_ewma   the same for |score|
    news_count_24h       number of scored headlines in the 24 hours
                         before the event

Weights decay as ``0.5 ** (age_hours / NEWS_HALF_LIFE_HOURS)`` over the last
``NEWS_LOOKBACK_TRADING_DAYS`` weekdays. A feature is null when the
look-back window is not fully covered by collected headlines (GDELT stores
headlines from September 2019) or when the window is empty. The half-life,
look-back and count window come from the ``[news]`` settings.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from lakehouse.feature_columns import NEWS_SENTIMENT_COLUMNS
from lakehouse.settings import get_settings

NEWS_HALF_LIFE_HOURS = get_settings().news_half_life_hours
NEWS_LOOKBACK_TRADING_DAYS = get_settings().news_lookback_trading_days
NEWS_COUNT_WINDOW_HOURS = get_settings().news_count_window_hours


def to_naive_utc_microseconds(timestamps: pl.Series) -> np.ndarray:
    """Any Datetime series (naive = UTC, or tz-aware) to ``datetime64[us]`` in UTC."""
    if timestamps.dtype.time_zone is not None:
        timestamps = (
            timestamps.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )
    return (
        timestamps.dt.cast_time_unit("us").to_numpy().astype("datetime64[us]")
    )


def lookback_start(event_timestamps: np.ndarray,
                   lookback_days: int) -> np.ndarray:
    """Midnight of the day ``lookback_days`` weekdays before each event.

    Exchange holidays are not excluded; the feature does not need that level
    of precision.
    """
    event_days = event_timestamps.astype("datetime64[D]")
    return np.busday_offset(
        event_days, -lookback_days, roll="forward"
    ).astype("datetime64[us]")


def exponentially_weighted_mean(event_timestamps: np.ndarray,
                                window_starts: np.ndarray,
                                headline_timestamps: np.ndarray,
                                values: np.ndarray,
                                coverage_start: np.datetime64) -> np.ndarray:
    """Per event: mean of ``values`` over headlines in [start, event).

    Weights decay with age. The result is NaN when the window starts before
    ``coverage_start`` or contains no headline.
    """
    decay_rate = np.log(2.0) / NEWS_HALF_LIFE_HOURS
    first_in_window = np.searchsorted(
        headline_timestamps, window_starts, side="left"
    )
    first_after_event = np.searchsorted(
        headline_timestamps, event_timestamps, side="left"
    )
    result = np.full(len(event_timestamps), np.nan)
    for event_index in range(len(event_timestamps)):
        first = first_in_window[event_index]
        last = first_after_event[event_index]
        if window_starts[event_index] < coverage_start or last <= first:
            continue
        age_hours = (
            event_timestamps[event_index] - headline_timestamps[first:last]
        ) / np.timedelta64(1, "h")
        weights = np.exp(-decay_rate * age_hours)
        result[event_index] = float(
            np.sum(weights * values[first:last]) / np.sum(weights)
        )
    return result


def pre_event_news_features(event_timestamps: pl.Series,
                            scored_headlines: pl.DataFrame | None) -> pl.DataFrame:
    """One row per event (same order as ``event_timestamps``) with ``NEWS_SENTIMENT_COLUMNS``.

    Passing ``None`` for ``scored_headlines`` leaves every column null.
    """
    events = to_naive_utc_microseconds(event_timestamps)
    window_starts = lookback_start(events, NEWS_LOOKBACK_TRADING_DAYS)
    columns = {name: np.full(len(events), np.nan) for name in NEWS_SENTIMENT_COLUMNS}

    if scored_headlines is not None and len(scored_headlines):
        scored = (
            scored_headlines
            .select("seen_utc", "sentiment_score")
            .drop_nulls()
            .sort("seen_utc")
        )
        headline_timestamps = to_naive_utc_microseconds(scored["seen_utc"])
        scores = scored["sentiment_score"].to_numpy().astype(float)
        coverage_start = (
            headline_timestamps[0]
            .astype("datetime64[D]")
            .astype("datetime64[us]")
        )

        columns["news_sent_ewma"] = exponentially_weighted_mean(
            events, window_starts, headline_timestamps, scores, coverage_start
        )
        columns["news_sent_abs_ewma"] = exponentially_weighted_mean(
            events,
            window_starts,
            headline_timestamps,
            np.abs(scores),
            coverage_start,
        )

        count_window_start = events - np.timedelta64(NEWS_COUNT_WINDOW_HOURS, "h")
        headline_count = (
            np.searchsorted(headline_timestamps, events, side="left")
            - np.searchsorted(headline_timestamps, count_window_start, side="left")
        )
        columns["news_count_24h"] = np.where(
            count_window_start >= coverage_start, headline_count, np.nan
        )

    news = pl.DataFrame(columns).with_columns(
        pl.col(NEWS_SENTIMENT_COLUMNS).fill_nan(None)
    )
    return news.with_columns(
        pl.col(["news_sent_ewma", "news_sent_abs_ewma"]).round(6),
        pl.col("news_count_24h").cast(pl.Int32),
    )
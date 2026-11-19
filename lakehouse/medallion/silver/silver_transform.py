"""Silver layer: cleaned, deduplicated, UTC-aligned tables.

    python -m lakehouse.medallion.silver.silver_transform

Reads bronze through Polars + PyIceberg, writes silver through Spark:

    silver.events             event calendar with the corrected ``ts_utc``
    silver.prices_m1          1-minute bars on a gap-free grid plus log returns
    silver.prices_tick        ticks (and minute-bar fallbacks) within
                              +/- ``windows.tick_window_seconds`` of every event
    silver.<scored_table>     sentiment score per GDELT headline (``[news] scored_table``;
                              appended, scoring is incremental; skipped when news is disabled)

Events are converted first because the tick windows are cut with the event
times.
"""
import numpy as np
import polars as pl

from lakehouse.infrastructure.iceberg_io import (
    load_iceberg_table,
    load_iceberg_table_filtered,
    write_iceberg_table,
)
from lakehouse.infrastructure.spark_session import (
    BRONZE_SCHEMA,
    SILVER_SCHEMA,
    ensure_schemas,
    get_spark,
)
from lakehouse.medallion.silver.gdelt_cleaning import clean_gdelt, score_new_headlines
from lakehouse.medallion.silver.price_processing import (
    TICK_WINDOW_SECONDS,
    build_silver_minute_bars,
    build_silver_ticks,
)
from lakehouse.medallion.silver.time_conversion import (
    drop_bad_events,
    events_to_utc,
    prices_to_utc,
)
from lakehouse.settings import get_settings


def clean_events() -> pl.DataFrame:
    """bronze.events with ``release_datetime`` converted to ``ts_utc``.

    Invalid releases are dropped.
    """
    converted = events_to_utc(
        load_iceberg_table(BRONZE_SCHEMA, "events"),
        "release_datetime",
        "ts_utc",
    )
    if {"actual", "forecast", "surprise"} <= set(converted.columns):
        converted = converted.with_columns(
            pl.coalesce(pl.col("surprise"), pl.col("actual") - pl.col("forecast")).alias("surprise")
        )
    return drop_bad_events(converted)


def event_days(events: pl.DataFrame, padding_days: int = 1) -> list:
    """Sorted unique event dates, padded by +/- ``padding_days``.

    Used to read only the tick days that matter.
    """
    days = (
        events.with_columns(pl.col("ts_utc").dt.date().alias("day"))
        .select(
            pl.date_ranges(
                pl.col("day") - pl.duration(days=padding_days),
                pl.col("day") + pl.duration(days=padding_days),
                interval="1d",
            ).alias("padded_date")
        )
        .explode("padded_date")
        .unique()
        .sort("padded_date")
    )
    return days["padded_date"].to_list()


def clean_prices(bronze_table: str) -> pl.DataFrame:
    """Whole bronze price table -> UTC timestamps.

    Fine for the 1-minute bars; ticks go through ``clean_tick_windows``.
    """
    return prices_to_utc(
        load_iceberg_table(BRONZE_SCHEMA, bronze_table),
        "datetime",
        "ts_utc",
    )


def tick_window_filter(events: pl.DataFrame, window_seconds: int = TICK_WINDOW_SECONDS):
    """A function that reduces one batch of raw ticks to the UTC ticks within +/- ``window_seconds``
    of an event (day pre-filter, timestamp conversion, exact window filter)."""
    days = event_days(events, padding_days=1)
    event_ms = np.sort(events["ts_utc"].dt.epoch("ms").to_numpy())
    window_ms = window_seconds * 1000

    def keep(batch: pl.DataFrame) -> pl.DataFrame:
        # Cheap pre-filter on the local calendar day (+/- 1 day of padding covers any clock offset).
        if isinstance(batch.schema["datetime"], pl.Datetime):
            day = pl.col("datetime").dt.date()
        else:
            day = pl.coalesce(
                pl.col("datetime").cast(pl.Utf8).str.slice(0, 8).str.strptime(pl.Date, "%Y%m%d", strict=False),
                pl.col("datetime").cast(pl.Utf8).str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            )
        batch = batch.filter(day.is_in(days))
        if batch.is_empty():
            return batch.with_columns(
                pl.lit(None, dtype=pl.Datetime("ms", "UTC")).alias("ts_utc")
            ).drop("datetime")
        batch = prices_to_utc(batch, "datetime", "ts_utc")
        tick_ms = batch["ts_utc"].dt.epoch("ms").to_numpy()
        position = np.searchsorted(event_ms, tick_ms)
        previous_gap = np.abs(
            tick_ms - event_ms[np.clip(position - 1, 0, len(event_ms) - 1)]
        )
        next_gap = np.abs(
            event_ms[np.clip(position, 0, len(event_ms) - 1)] - tick_ms
        )
        return batch.filter(
            pl.Series(np.minimum(previous_gap, next_gap) <= window_ms)
        )

    return keep


def clean_tick_windows(events: pl.DataFrame,
                       window_seconds: int = TICK_WINDOW_SECONDS) -> pl.DataFrame:
    """Bronze ticks -> UTC timestamps, keeping only the ticks near an event.

    The raw tick history is larger than memory, so it is read one Arrow
    batch at a time and reduced per batch before the next batch is fetched.
    """
    keep = tick_window_filter(events, window_seconds)
    print("Reading bronze.prices_tick batch by batch "
          "(only the ticks around the events are kept)")
    ticks = load_iceberg_table_filtered(BRONZE_SCHEMA, "prices_tick", keep)
    print(f"  kept {len(ticks):,} ticks within "
          f"+/-{window_seconds} s of {len(events):,} events")
    return ticks.sort("ts_utc")


def load_optional_table(schema: str, table_name: str) -> pl.DataFrame | None:
    """Read a table, or None if it does not exist yet (first run)."""
    try:
        return load_iceberg_table(schema, table_name)
    except Exception as error:  # NoSuchTableError, or an empty namespace
        print(f"  ({schema}.{table_name} not found - "
              f"{type(error).__name__}; treating as empty)")
        return None


def main():
    settings = get_settings()
    spark = get_spark("silver-transform")
    ensure_schemas(spark)

    events = clean_events()
    minute_bars = build_silver_minute_bars(clean_prices("prices_m1"))
    ticks = build_silver_ticks(clean_tick_windows(events), minute_bars, events)

    # Gold reads silver.events, so it is rewritten every run: the corrected
    # ts_utc must reach gold.
    tables = {"events": events, "prices_m1": minute_bars, "prices_tick": ticks}
    for table_name, table in tables.items():
        write_iceberg_table(spark, table, SILVER_SCHEMA, table_name)
        print(f"Wrote {SILVER_SCHEMA}.{table_name} ({len(table):,} rows)")

    if settings.news_enabled:
        scored_table = settings.scored_headlines_table
        headlines = clean_gdelt(load_iceberg_table(BRONZE_SCHEMA, "gdelt_headlines"))
        new_scores = score_new_headlines(
            headlines,
            load_optional_table(SILVER_SCHEMA, scored_table),
        )
        if len(new_scores):
            write_iceberg_table(
                spark, new_scores, SILVER_SCHEMA,
                scored_table, mode="append",
            )
        print(f"Appended {len(new_scores):,} new row(s) to "
              f"{SILVER_SCHEMA}.{scored_table} ({settings.sentiment_model})")

    spark.stop()


if __name__ == "__main__":
    main()
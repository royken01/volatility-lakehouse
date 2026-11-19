"""UTC timestamps for the event calendar and the price series.

Every source has its own local clock, described by a list of ``ClockRule`` in the settings
(``[events] clock`` and ``[prices] clock``): a time zone per period, optionally with a fixed hour
offset.  A single rule ``"UTC"`` (the default) means the source already stamps in UTC.  Known calendar
errors are fixed with ``time_corrections`` and ``copies`` (see ``configs/crude_oil_thesis.toml``).
Both conversion functions return a new frame sorted by the UTC column, with the naive column dropped.
"""
from datetime import datetime

import polars as pl

from lakehouse.settings import ClockRule, Settings, get_settings

UTC = "UTC"
TZ_OPTIONS = dict(ambiguous="earliest", non_existent="null")


def local_to_utc(wall: pl.Expr, rules: tuple[ClockRule, ...]) -> pl.Expr:
    """Naive wall-clock stamps -> UTC under the clock rules (the first rule whose period contains the stamp)."""
    def convert(rule: ClockRule) -> pl.Expr:
        shifted = wall + pl.duration(seconds=round(rule.offset_hours * 3600)) if rule.offset_hours else wall
        return shifted.dt.replace_time_zone(rule.timezone, **TZ_OPTIONS).dt.convert_time_zone(UTC)

    expression = convert(rules[-1])
    for rule in reversed(rules[:-1]):
        switch = datetime(rule.until.year, rule.until.month, rule.until.day)
        expression = pl.when(wall < pl.lit(switch)).then(convert(rule)).otherwise(expression)
    return expression


def parse_timestamps(column: str, formats, time_unit: str = "us") -> pl.Expr:
    """String stamps in any of ``formats`` (first match wins) -> naive Datetime."""
    return pl.coalesce([
        pl.col(column).cast(pl.Utf8).str.strptime(pl.Datetime(time_unit), fmt, strict=False)
        .dt.replace_time_zone(None)
        for fmt in formats
    ])


def events_to_utc(events: pl.DataFrame, naive_column: str, utc_column: str,
                  settings: Settings | None = None) -> pl.DataFrame:
    """Convert the naive ``release_datetime`` strings to UTC and apply the configured stamp corrections."""
    settings = settings or get_settings()
    stamps = events.schema[naive_column]
    if isinstance(stamps, pl.Datetime):
        wall = pl.col(naive_column).dt.replace_time_zone(None)
    else:
        wall = pl.col(naive_column).str.strptime(pl.Datetime("us"), settings.event_timestamp_format)
    for correction in settings.time_corrections:
        late = (pl.col("event") == correction.event) & (wall.dt.minute() == correction.minute)
        wall = pl.when(late).then(wall + pl.duration(minutes=correction.shift_minutes)).otherwise(wall)
    return (
        events
        .with_columns(local_to_utc(wall, settings.event_clock).alias(utc_column))
        .drop(naive_column)
        .sort(utc_column)
    )


def drop_bad_events(events: pl.DataFrame, utc_column: str = "ts_utc",
                    settings: Settings | None = None) -> pl.DataFrame:
    """Drop calendar rows that are not real releases.

    Two cases are dropped: a row without a forecast within ``duplicate_window_minutes`` of another
    row of the same event, and (for the configured ``copies``) a row that repeats the actual,
    forecast and surprise of another event within a day. Dropped rows are reported.
    """
    settings = settings or get_settings()
    window = pl.duration(minutes=settings.duplicate_window_minutes)
    before = pl.col(utc_column).diff().over("event")
    after = (-pl.col(utc_column).diff(-1)).over("event")
    has_forecast = pl.col("forecast").is_not_null() if "forecast" in events.columns else pl.lit(True)
    duplicate = ~has_forecast & ((before.abs() <= window) | (after.abs() <= window))
    flagged = [events.sort(utc_column).filter(duplicate).select(utc_column, "event")]

    for copy_event, source_event in settings.copied_events:
        source_rows = (
            events
            .filter((pl.col("event") == source_event) & has_forecast)
            .select("actual", "forecast", "surprise", pl.col(utc_column).alias("source_ts"))
        )
        flagged.append(
            events
            .filter(pl.col("event") == copy_event)
            .join(source_rows, on=["actual", "forecast", "surprise"], how="inner")
            .filter((pl.col(utc_column) - pl.col("source_ts")).abs() <= pl.duration(days=1))
            .select(utc_column, "event")
        )

    bad = pl.concat(flagged).unique()
    if len(bad):
        dropped = [(row[0].strftime("%Y-%m-%d %H:%M"), row[1]) for row in bad.sort(utc_column).iter_rows()]
        print(f"  dropped {len(bad)} invalid event row(s): {dropped}")
    return events.join(bad, on=[utc_column, "event"], how="anti").sort(utc_column)


def prices_to_utc(prices: pl.DataFrame, naive_column: str, utc_column: str,
                  settings: Settings | None = None) -> pl.DataFrame:
    """Convert a price frame's timestamp column to UTC.

    String stamps are parsed with the configured ``timestamp_formats`` (HistData's
    ``YYYYmmdd HHMMSSmmm`` and ISO formats by default); Datetime columns are used as they are.
    """
    settings = settings or get_settings()
    if isinstance(prices.schema[naive_column], pl.Datetime):
        wall = pl.col(naive_column).dt.replace_time_zone(None).dt.cast_time_unit("ms")
    else:
        wall = parse_timestamps(naive_column, settings.price_timestamp_formats, "ms")
    return (
        prices
        .with_columns(local_to_utc(wall, settings.price_clock).alias(utc_column))
        .filter(pl.col(utc_column).is_not_null())
        .drop(naive_column)
        .sort(utc_column)
    )

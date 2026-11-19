"""Gold layer: the per-event feature table ``gold.events_features``.

    python -m lakehouse.medallion.gold.gold_features

Reads ``silver.events``, ``silver.prices_m1``, ``silver.prices_tick`` and the
scored headlines (de-duplicated on ``uid``; skipped with ``[news] enabled = false``),
builds the features in ``event_features.py`` and writes them with Spark.
"""
from lakehouse.infrastructure.iceberg_io import load_iceberg_table, write_iceberg_table
from lakehouse.infrastructure.spark_session import (
    GOLD_SCHEMA,
    SILVER_SCHEMA,
    ensure_schemas,
    get_spark,
)
from lakehouse.medallion.gold.event_features import build_features
from lakehouse.medallion.silver.silver_transform import load_optional_table
from lakehouse.settings import get_settings


def main():
    settings = get_settings()
    print(f"Gold for {settings.asset}: target {settings.target_column}, {settings.lags} lags, "
          f"event types {settings.describe()['event_types']}")
    spark = get_spark("gold-features")
    ensure_schemas(spark)

    events = load_iceberg_table(SILVER_SCHEMA, "events")
    minute_bars = load_iceberg_table(SILVER_SCHEMA, "prices_m1")
    tick_prices = load_iceberg_table(SILVER_SCHEMA, "prices_tick")
    scored_headlines = None
    if settings.news_enabled:
        scored_headlines = load_optional_table(SILVER_SCHEMA, settings.scored_headlines_table)
        scored_headlines = scored_headlines.unique("uid") if scored_headlines is not None else None

    features = build_features(
        minute_bars, tick_prices, events, scored_headlines=scored_headlines
    )
    write_iceberg_table(spark, features, GOLD_SCHEMA, "events_features")
    print(f"Wrote {GOLD_SCHEMA}.events_features ({len(features):,} rows)")

    spark.stop()


if __name__ == "__main__":
    main()
"""Bronze layer: raw files in ``lakehouse/data/raw`` -> Iceberg tables.

    vol-lakehouse bronze
    vol-lakehouse bronze prices_m1

Bronze is "as landed": every source is a full reload (``createOrReplace``)
with two metadata columns, ``_source_file`` and ``_loaded_at``. No cleaning,
no derived columns.  The file patterns come from ``[sources]`` in the settings.

Expected inputs (any asset, any release):

    events       CSV: release_datetime (local time, see ``[events] clock``), event,
                 and optionally importance, actual, forecast, previous, surprise
                 (``vol-lakehouse calendar`` writes this format)
    headlines    CSV from ``vol-lakehouse gdelt``: seen_utc, uid, domain, title, tone, score, url
    prices_tick  Parquet: datetime, bid, ask (or price)
    prices_m1    Parquet: datetime, close (open/high/low optional)

``release_datetime`` lands as the raw STRING, because Spark would otherwise
parse it in the session time zone (silver converts it to UTC). Every GDELT
column lands as STRING with the escape character set to ``"``, because
titles use RFC-4180 quoting.
"""
import argparse
import glob
from datetime import datetime, timezone

from pyspark.sql.functions import col, input_file_name, lit

from lakehouse.infrastructure.spark_session import BRONZE_SCHEMA, ensure_schemas, get_spark
from lakehouse.paths import RAW_DATA_DIR
from lakehouse.settings import get_settings

_settings = get_settings()
# name -> (file pattern, kind). ``kind`` selects the reader in ``load_frame``.
SOURCES = {
    "events": (RAW_DATA_DIR / _settings.events_source, "events_csv"),
    "gdelt_headlines": (RAW_DATA_DIR / _settings.headlines_source, "gdelt_csv"),
    "prices_tick": (RAW_DATA_DIR / _settings.prices_tick_source, "parquet"),
    # Rarely changes, so not reloaded by default; silver reads it, so run it
    # once by name.
    "prices_m1": (RAW_DATA_DIR / _settings.prices_m1_source, "parquet"),
}
DEFAULT_SOURCES = ("events", "gdelt_headlines", "prices_tick") if _settings.news_enabled \
    else ("events", "prices_tick")

EVENT_TEXT_COLUMNS = ("release_datetime", "event", "importance")
EVENT_NUMBER_COLUMNS = ("actual", "forecast", "previous", "surprise")
GDELT_FIELDS = (
    "seen_utc STRING, uid STRING, domain STRING, title STRING, "
    "tone STRING, score STRING, url STRING"
)


def matching_files(path) -> list[str]:
    parts = sorted(glob.glob(str(path)))
    if not parts:
        raise FileNotFoundError(f"No files match {path} (set the pattern under [sources] in the config)")
    return parts


def load_events(spark, files):
    """The calendar CSV(s) in the canonical column set; missing optional columns become null."""
    frame = spark.read.option("header", "true").option("inferSchema", "false").csv(files)
    missing = [c for c in ("release_datetime", "event") if c not in frame.columns]
    if missing:
        raise KeyError(f"event calendar needs the columns {missing}; found {frame.columns}")
    text = [col(c).cast("string").alias(c) if c in frame.columns else lit(None).cast("string").alias(c)
            for c in EVENT_TEXT_COLUMNS]
    numbers = [col(c).cast("double").alias(c) if c in frame.columns else lit(None).cast("double").alias(c)
               for c in EVENT_NUMBER_COLUMNS]
    return frame.select(*text, *numbers)


def load_frame(spark, path, kind):
    """Spark DataFrame for one source file pattern."""
    files = matching_files(path)
    if kind == "events_csv":
        return load_events(spark, files)
    if kind == "gdelt_csv":
        return (
            spark.read.option("header", "true")
            .option("quote", '"')
            .option("escape", '"')
            .option("mode", "PERMISSIVE")
            .schema(GDELT_FIELDS)
            .csv(files)
        )
    return spark.read.option("mergeSchema", "true").parquet(*files)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sources",
        nargs="*",
        help=f"sources to (re)load, from {list(SOURCES)} "
             f"(default: {' '.join(DEFAULT_SOURCES)})",
    )
    args = parser.parse_args()
    bad = [s for s in args.sources if s not in SOURCES]
    if bad:
        parser.error(f"unknown source(s) {bad}; choose from {list(SOURCES)}")
    chosen = args.sources or DEFAULT_SOURCES

    spark = get_spark("bronze-ingest")
    ensure_schemas(spark)
    stamp = datetime.now(timezone.utc).isoformat()

    for name in chosen:
        path, kind = SOURCES[name]
        frame = (
            load_frame(spark, path, kind)
            .withColumn("_source_file", input_file_name())
            .withColumn("_loaded_at", lit(stamp))
        )
        target = f"{BRONZE_SCHEMA}.{name}"
        frame.writeTo(target).using("iceberg").createOrReplace()
        print(f"{path} -> {target} ({spark.table(target).count():,} rows)")

    spark.stop()


if __name__ == "__main__":
    main()
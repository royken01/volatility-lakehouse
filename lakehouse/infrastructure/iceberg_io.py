"""Shared Iceberg read / write helpers.

Reads use Polars + PyIceberg (no JVM); writes go through one thin Spark job.  ``schema`` arguments
accept a bare namespace ("silver") or the Spark-qualified name ("crude_oil.silver"): reads use the
last segment only.
"""
from typing import Callable

import polars as pl
from pyiceberg.catalog import load_catalog

from lakehouse.infrastructure.catalog_config import (
    R2_ACCESS_KEY_ID, R2_CATALOG_TOKEN, R2_CATALOG_URI, R2_CATALOG_WAREHOUSE, R2_ENDPOINT, R2_SECRET_ACCESS_KEY)

STORAGE_OPTIONS = {
    "aws_access_key_id": R2_ACCESS_KEY_ID,
    "aws_secret_access_key": R2_SECRET_ACCESS_KEY,
    "aws_endpoint_url": R2_ENDPOINT,
    "aws_region": "auto",
    "aws_virtual_hosted_style_request": "false",
}


def load_rest_catalog(extra_properties: dict | None = None):
    """PyIceberg handle on the R2 Data Catalog (``extra_properties`` are added to the catalog settings)."""
    return load_catalog("r2", **{"type": "rest", "uri": R2_CATALOG_URI, "warehouse": R2_CATALOG_WAREHOUSE,
                                 "token": R2_CATALOG_TOKEN, **(extra_properties or {})})


def load_iceberg_table(schema: str, table_name: str) -> pl.DataFrame:
    """Read a whole bronze / silver / gold table into a Polars DataFrame."""
    dataframe, _ = load_iceberg_table_with_snapshot(schema, table_name)
    return dataframe


def load_catalog_with_s3_credentials():
    """Catalog handle whose PyIceberg file reads use the R2 keys directly (PyIceberg reads the data files itself)."""
    return load_rest_catalog({"s3.endpoint": R2_ENDPOINT, "s3.access-key-id": R2_ACCESS_KEY_ID,
                              "s3.secret-access-key": R2_SECRET_ACCESS_KEY, "s3.region": "auto",
                              "s3.force-virtual-addressing": "false"})


def filter_in_batches(table, keep: Callable[[pl.DataFrame], pl.DataFrame]) -> pl.DataFrame:
    """Read a PyIceberg table one Arrow batch at a time, apply ``keep`` to each batch and stack the
    results."""
    reader = table.scan().to_arrow_batch_reader()
    kept = [keep(pl.from_arrow(batch)) for batch in reader]
    kept = [frame for frame in kept if len(frame)]
    return pl.concat(kept) if kept else keep(pl.from_arrow(reader.schema.empty_table()))


def load_iceberg_table_filtered(schema: str, table_name: str, keep: Callable[[pl.DataFrame], pl.DataFrame]) -> pl.DataFrame:
    """Current snapshot of a table, reduced batch by batch with ``keep``; for tables too large for
    memory."""
    table = load_catalog_with_s3_credentials().load_table(f"{schema.split('.')[-1]}.{table_name}")
    return filter_in_batches(table, keep)


def read_snapshot(table, snapshot_id: int | None, requested: bool) -> pl.DataFrame:
    """Read one snapshot of a PyIceberg table into Polars.

        The current snapshot uses a plain ``pl.scan_iceberg``; an older one uses time travel,
        through PyIceberg
        when this Polars version has no ``snapshot_id`` argument.
    """
    if not requested:
        return pl.scan_iceberg(table, storage_options=STORAGE_OPTIONS).collect()
    try:
        return pl.scan_iceberg(table, snapshot_id=snapshot_id, storage_options=STORAGE_OPTIONS).collect()
    except TypeError:   # this Polars version has no snapshot_id argument
        catalog = load_catalog_with_s3_credentials()
        old_table = catalog.load_table(table.name())
        return pl.from_arrow(old_table.scan(snapshot_id=snapshot_id).to_arrow())


def load_iceberg_table_with_snapshot(schema: str, table_name: str, snapshot_id: int | None = None):
    """Read a table and also return the Iceberg snapshot id that was read: ``(polars.DataFrame,
    snapshot_id)``.

        Pass ``snapshot_id`` to read an older version, which is how a logged run is reproduced
        exactly.
    """
    namespace = schema.split(".")[-1]
    table = load_rest_catalog().load_table(f"{namespace}.{table_name}")
    current_snapshot = table.current_snapshot()
    current_snapshot_id = current_snapshot.snapshot_id if current_snapshot is not None else None
    requested = snapshot_id is not None and snapshot_id != current_snapshot_id
    dataframe = read_snapshot(table, snapshot_id, requested)
    return dataframe, (snapshot_id if snapshot_id is not None else current_snapshot_id)


def write_iceberg_table(spark, polars_dataframe: pl.DataFrame, schema: str, table_name: str,
                        mode: str = "replace") -> None:
    """Write a Polars DataFrame to an Iceberg table through Spark.

        mode="replace" overwrites the table; mode="append" adds rows and creates the table if
        needed.  The frame
        goes through a temporary parquet file; nanosecond timestamps are cast to microseconds, and
        appends enable
        ``merge-schema``.
    """
    if mode not in ("replace", "append"):
        raise ValueError(f"mode must be 'replace' or 'append', got {mode!r}")

    nanosecond_columns = [name for name, dtype in polars_dataframe.schema.items()
                          if isinstance(dtype, pl.Datetime) and dtype.time_unit == "ns"]
    if nanosecond_columns:
        polars_dataframe = polars_dataframe.with_columns(
            [pl.col(name).dt.cast_time_unit("us") for name in nanosecond_columns])

    full_table_name = f"{schema}.{table_name}"
    temporary_parquet = f"/tmp/{schema.replace('.', '_')}_{table_name}_staging.parquet"
    polars_dataframe.write_parquet(temporary_parquet)
    spark_dataframe = spark.read.parquet(temporary_parquet)

    if mode == "append" and spark.catalog.tableExists(full_table_name):
        spark.sql(f"ALTER TABLE {full_table_name} SET TBLPROPERTIES ('write.spark.accept-any-schema'='true')")
        spark_dataframe.writeTo(full_table_name).option("merge-schema", "true").append()
    else:
        spark_dataframe.writeTo(full_table_name).using("iceberg").createOrReplace()

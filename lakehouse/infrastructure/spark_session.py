"""SparkSession wired to Apache Iceberg and the Cloudflare R2 Data Catalog (an Iceberg REST
catalog).

Spark is only used for WRITING Iceberg tables; all reads go through Polars + PyIceberg.  Needs R2
Data Catalog enabled on the bucket and ``lakehouse/.env`` with the catalog URI, warehouse and an R2
token.  The medallion layers are the namespaces ``<catalog>.bronze``, ``.silver`` and ``.gold``.
"""
import os
import sys

from pyspark.sql import SparkSession

from lakehouse.infrastructure.catalog_config import (
    CATALOG_NAME, R2_CATALOG_TOKEN, R2_CATALOG_URI, R2_CATALOG_WAREHOUSE,
)

# Spark workers must use the same Python interpreter as this process.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

ICEBERG_VERSION = "1.6.1"
SPARK_ICEBERG_PACKAGE = f"org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:{ICEBERG_VERSION}"
ICEBERG_AWS_BUNDLE = f"org.apache.iceberg:iceberg-aws-bundle:{ICEBERG_VERSION}"

BRONZE_SCHEMA = f"{CATALOG_NAME}.bronze"
SILVER_SCHEMA = f"{CATALOG_NAME}.silver"
GOLD_SCHEMA = f"{CATALOG_NAME}.gold"


def get_spark(app_name: str = "volatility-lakehouse") -> SparkSession:
    """Create (or reuse) the session; its time zone is UTC so no timestamp is silently shifted."""
    catalog = f"spark.sql.catalog.{CATALOG_NAME}"
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", "4g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.jars.packages", f"{SPARK_ICEBERG_PACKAGE},{ICEBERG_AWS_BUNDLE}")
        .config("spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config(catalog, "org.apache.iceberg.spark.SparkCatalog")
        .config(f"{catalog}.type", "rest")
        .config(f"{catalog}.uri", R2_CATALOG_URI)
        .config(f"{catalog}.warehouse", R2_CATALOG_WAREHOUSE)
        .config(f"{catalog}.token", R2_CATALOG_TOKEN)
        .config(f"{catalog}.header.X-Iceberg-Access-Delegation", "vended-credentials")
        .config(f"{catalog}.s3.remote-signing-enabled", "false")
        .config("spark.sql.defaultCatalog", CATALOG_NAME)
    )
    return builder.getOrCreate()


def ensure_schemas(spark: SparkSession) -> None:
    """Create the bronze / silver / gold namespaces if they do not exist yet."""
    for schema in (BRONZE_SCHEMA, SILVER_SCHEMA, GOLD_SCHEMA):
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {schema}")
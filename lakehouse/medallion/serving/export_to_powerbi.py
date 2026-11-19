"""Export the data behind the Power BI report to local parquet files

    vol-lakehouse powerbi
    vol-lakehouse powerbi --skip-gold
    vol-lakehouse powerbi --skip-scored

Every table is read with Polars and PyIceberg

"""
import argparse
from pathlib import Path

from lakehouse.paths import POWERBI_EXPORT_DIR
from lakehouse.reporting.staged_predictions import (
    MODEL_ORDER,
    add_selection_arguments,
    display_name,
    headline_rows,
    load_scored_events,
)

GOLD_TABLES = [
    "events_features",
    "model_predictions",
    "model_metrics",
]

SCORED_COLUMNS = [
    "ts_utc", "event_type", "year", "quarter", "small_quarter",
    "model", "model_name", "model_order", "variant", "is_headline",
    "base_fs", "feature_set", "run_id", "gold_snapshot_id",
    "y_true_volatility", "y_pred_volatility", "y_true_log", "y_pred_log", "mse_log",
]


def export_gold_tables(directory: Path) -> None:
    from pyiceberg.exceptions import NoSuchTableError

    from lakehouse.infrastructure.iceberg_io import load_iceberg_table
    from lakehouse.infrastructure.spark_session import GOLD_SCHEMA

    for table_name in GOLD_TABLES:
        try:
            table = load_iceberg_table(GOLD_SCHEMA, table_name)
        except NoSuchTableError:
            print(f"Skipping {GOLD_SCHEMA}.{table_name} (does not exist yet)")
            continue
        output_path = directory / f"{table_name}.parquet"
        table.write_parquet(output_path)
        print(f"Exported {GOLD_SCHEMA}.{table_name} -> {output_path} ({len(table):,} rows)")


def export_scored_events(arguments: argparse.Namespace, directory: Path) -> None:
    scored = load_scored_events(arguments)
    export = scored.copy()
    export["ts_utc"] = export["ts_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    export["mse_log"] = export["mse"]
    export["small_quarter"] = export["quarter"].isin(scored.attrs.get("small_quarters", []))
    export["model_name"] = export["model"].map(display_name)
    order = {
        model: rank
        for rank, model in enumerate([*MODEL_ORDER, "naive_mean"], start=1)
    }
    export["model_order"] = (
        export["model"].map(order).fillna(len(order) + 1).astype(int)
    )
    export["is_headline"] = export.index.isin(
        headline_rows(export, arguments.variant).index
    )
    export = (
        export[SCORED_COLUMNS]
        .sort_values(["ts_utc", "model_order", "variant"])
        .reset_index(drop=True)
    )
    output_path = directory / "scored_events.parquet"
    export.to_parquet(output_path, index=False)
    variants = export.groupby(["model", "variant"]).ngroups
    print(
        f"Exported scored events -> {output_path} "
        f"({len(export):,} rows: {export['ts_utc'].nunique()} events "
        f"x {variants} model variants)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_selection_arguments(parser)
    parser.add_argument("--skip-gold", action="store_true",
                        help="do not export the gold tables (needs the catalog credentials)")
    parser.add_argument("--skip-scored", action="store_true",
                        help="do not export scored_events.parquet")
    arguments = parser.parse_args()
    directory = arguments.out_dir or POWERBI_EXPORT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    if not arguments.skip_gold:
        export_gold_tables(directory)
    if not arguments.skip_scored:
        export_scored_events(arguments, directory)


if __name__ == "__main__":
    main()
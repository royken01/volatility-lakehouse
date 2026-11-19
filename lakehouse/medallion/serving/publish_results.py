"""Publish staged model results to Iceberg.

    vol-lakehouse publish --validate-only
    vol-lakehouse publish --replace
    vol-lakehouse publish --validate-only
    vol-lakehouse publish --validate-only --archive-staged

--replace makes both tables exactly the staged runs instead of appending

Validation runs first and stops the job before Spark starts on: a missing
column or non-finite prediction, more than one gold_snapshot_id,
duplicate rows, more than one run per, a benchmark with
an outdated feature set, an unknown model, or metrics and predictions that
disagree.
"""
import argparse
import shutil
from datetime import datetime

import polars as pl

from lakehouse.modelling.feature_sets import BENCHMARK_FEATURE_SETS
from lakehouse.modelling.model_params import MODEL_NAMES, NEWS_MODEL_NAMES
from lakehouse.paths import RESULTS_PUBLISHED_DIR, RESULTS_STAGING_DIR
from lakehouse.reporting.staged_predictions import feature_set_root

TABLE_FILE_PATTERNS = {
    "model_predictions": "*_predictions.parquet",
    "model_metrics": "*_metrics.parquet",
}
PREDICTION_COLUMNS = [
    "ts_utc", "event_type", "y_true", "y_pred", "run_id", "model_name",
    "feature_set", "gold_snapshot_id",
]
METRIC_COLUMNS = [
    "fold", "run_id", "model_name", "feature_set", "gold_snapshot_id",
]
LOG_COLUMNS = ["floor_value", "log_y_true", "log_y_pred"]


def split_label(label: str) -> tuple[str, str]:
    """'<set>|C0' -> ('<set>', 'C0'); a label without '|' is variant A."""
    base, _, variant = str(label).partition("|")
    return base, variant or "A"


def short_ids(run_ids, limit: int = 4) -> str:
    """'cd6cb24f, 00ae051a, ... (+12 more)'.

    The first 8 characters of each run id are unique enough to find the
    files.
    """
    ids = [str(run_id)[:8] for run_id in run_ids]
    return (
        ", ".join(ids[:limit])
        + (f", ... (+{len(ids) - limit} more)" if len(ids) > limit else "")
    )


def validate_staged(predictions: pl.DataFrame | None,
                    metrics: pl.DataFrame | None) -> tuple[list[str], list[str]]:
    """Check the staged tables. Returns (errors, warnings).

    The publish only goes ahead when there are no errors.
    """
    errors: list[str] = []
    warnings: list[str] = []
    checks = (
        ("predictions", predictions, PREDICTION_COLUMNS),
        ("metrics", metrics, METRIC_COLUMNS),
    )
    for name, frame, required in checks:
        if frame is None:
            errors.append(f"no staged {name} files")
            continue
        missing = [column for column in required if column not in frame.columns]
        if missing:
            errors.append(f"{name}: required column(s) missing: {missing}")
    if errors:
        return errors, warnings

    bad = predictions.filter(
        pl.col("y_pred").is_null()
        | ~pl.col("y_pred").is_finite()
        | pl.col("y_true").is_null()
        | ~pl.col("y_true").is_finite()
    )
    if len(bad):
        errors.append(
            f"predictions: {len(bad)} rows with a null / non-finite y_true or y_pred "
            f"(runs: {short_ids(sorted(bad['run_id'].unique().to_list()))})"
        )

    snapshots = sorted(
        set(predictions["gold_snapshot_id"].to_list())
        | set(metrics["gold_snapshot_id"].to_list()),
        key=str,
    )
    if len(snapshots) > 1:
        per_snapshot = (
            predictions.group_by("gold_snapshot_id")
            .agg(pl.col("run_id").n_unique().alias("runs"))
            .sort("runs")
        )
        errors.append(
            f"staged runs come from {len(snapshots)} different gold snapshots "
            f"(snapshot: number of runs) {dict(per_snapshot.iter_rows())} - "
            f"retrain everything on the current gold, or archive the old files"
        )

    duplicated = (
        predictions.group_by(["run_id", "feature_set", "ts_utc"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if len(duplicated):
        errors.append(
            f"predictions: {len(duplicated)} (run_id, feature_set, ts_utc) "
            f"combinations occur more than once"
        )

    runs = (
        predictions.group_by(["model_name", "feature_set"])
        .agg(pl.col("run_id").unique().sort().alias("runs"))
    )
    for model, label, run_ids in runs.iter_rows():
        if len(run_ids) > 1:
            errors.append(
                f"{model} / {label}: {len(run_ids)} run_ids staged "
                f"({short_ids(run_ids)}) - archive the old ones"
            )

    removed = sorted(set(predictions["model_name"].unique().to_list()) - set(MODEL_NAMES))
    if removed:
        errors.append(
            f"runs of models that no longer exist {removed} are staged - archive them"
        )

    for model, label in predictions.select("model_name", "feature_set").unique().iter_rows():
        base = feature_set_root(split_label(label)[0])
        if model in BENCHMARK_FEATURE_SETS and base != BENCHMARK_FEATURE_SETS[model]:
            errors.append(
                f"{model} was staged with feature set '{label}', "
                f"expected '{BENCHMARK_FEATURE_SETS[model]}' "
                f"(an outdated benchmark run - retrain it)"
            )

    repeated = (
        metrics.group_by(["run_id", "feature_set", "fold"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if len(repeated):
        errors.append(
            f"metrics: {len(repeated)} (run_id, feature_set, fold) "
            f"combinations occur more than once"
        )

    in_predictions = set(predictions.select("run_id", "feature_set").unique().iter_rows())
    in_metrics = set(metrics.select("run_id", "feature_set").unique().iter_rows())
    if in_predictions != in_metrics:
        errors.append(
            f"predictions and metrics describe different runs: "
            f"only in predictions {sorted(in_predictions - in_metrics)}, "
            f"only in metrics {sorted(in_metrics - in_predictions)}"
        )

    present = set(predictions["model_name"].unique().to_list())
    absent = [model for model in MODEL_NAMES if model not in present]
    if absent:
        warnings.append(f"models without any staged run: {absent}")

    variants = {
        (model, split_label(label)[1])
        for model, label in predictions.select("model_name", "feature_set").unique().iter_rows()
    }
    for model in NEWS_MODEL_NAMES:
        if model not in present:
            continue
        gap = [v for v in ("A", "C0", "C") if (model, v) not in variants]
        if gap:
            warnings.append(f"{model}: variant(s) {gap} missing")

    odd = sorted(
        {label for label in predictions["feature_set"].unique().to_list() if "+raw" in label}
    )
    if odd:
        preview = f"{odd[:4]}{' ...' if len(odd) > 4 else ''}"
        warnings.append(
            f"labels of alternative runs (+raw) staged: {preview} - "
            f"the figures and tables ignore them"
        )

    counts = (
        predictions.group_by(["model_name", "feature_set"])
        .agg(pl.col("ts_utc").n_unique().alias("events"))
        .sort("model_name", "feature_set")
    )
    if counts["events"].n_unique() > 1:
        sizes = (
            counts.group_by("events")
            .agg(pl.col("model_name").unique().sort())
            .sort("events")
            .to_dicts()
        )
        warnings.append(
            f"series are scored on different numbers of events {sizes} - "
            f"the reports restrict to the events all series share "
            f"(benchmark-only runs have a few more events than news-era runs)"
        )

    no_log = [column for column in LOG_COLUMNS if column not in predictions.columns]
    if no_log:
        warnings.append(
            f"no log-scale columns {no_log} in the staged predictions - "
            f"the reports rebuild the log forecasts from y_pred "
            f"(retrain to store them)"
        )
    else:
        legacy = (
            predictions
            .filter(pl.col("log_y_pred").is_null())["run_id"]
            .unique()
            .to_list()
        )
        if legacy:
            warnings.append(
                f"{len(legacy)} runs without log-scale forecasts "
                f"(trained before those columns existed): "
                f"{short_ids(sorted(legacy))}"
            )

    return errors, warnings


def read_staged() -> tuple[dict[str, list], dict[str, pl.DataFrame | None]]:
    files = {
        table: sorted(RESULTS_STAGING_DIR.glob(pattern))
        for table, pattern in TABLE_FILE_PATTERNS.items()
    }
    frames = {
        table: (
            pl.concat([pl.read_parquet(file) for file in table_files],
                      how="diagonal_relaxed")
            if table_files
            else None
        )
        for table, table_files in files.items()
    }
    return files, frames


def archive_staged() -> None:
    """Move every staged file to ``data/results_staging_archive/<timestamp>/``.

    A full run calls this before it trains, so an earlier unpublished run
    cannot mix with the new one.
    """
    files = [
        file
        for pattern in TABLE_FILE_PATTERNS.values()
        for file in sorted(RESULTS_STAGING_DIR.glob(pattern))
    ]
    if not files:
        print(f"No staged files in {RESULTS_STAGING_DIR}")
        return
    destination = (
        RESULTS_STAGING_DIR.parent
        / "results_staging_archive"
        / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )
    destination.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.move(str(file), destination / file.name)
    print(f"Archived {len(files)} staged file(s) to {destination}")


def main(replace: bool = False, validate_only: bool = False) -> None:
    files, frames = read_staged()
    if not any(files.values()):
        raise SystemExit(
            f"Nothing to publish in {RESULTS_STAGING_DIR} - train first "
            f"(or restore the files from data/results_staging_archive/)."
        )

    errors, warnings = validate_staged(frames["model_predictions"], frames["model_metrics"])
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR:   {error}")
    if errors:
        raise SystemExit(
            f"Validation failed ({len(errors)} error(s)); nothing was published."
        )

    predictions = frames["model_predictions"]
    print(
        f"Validation passed: {len(predictions):,} prediction rows, "
        f"{predictions.select('run_id', 'feature_set').n_unique()} model variants, "
        f"gold snapshot {predictions['gold_snapshot_id'].unique().to_list()}"
    )
    if validate_only:
        return

    from lakehouse.infrastructure.iceberg_io import write_iceberg_table
    from lakehouse.infrastructure.spark_session import (
        GOLD_SCHEMA, ensure_schemas, get_spark,
    )

    spark = get_spark("publish-model-results")
    ensure_schemas(spark)
    RESULTS_PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)

    mode = "replace" if replace else "append"
    for table_name, table_files in files.items():
        write_iceberg_table(spark, frames[table_name], GOLD_SCHEMA, table_name, mode=mode)
        verb = "Replaced the contents of" if replace else "Appended to"
        print(
            f"{verb} {GOLD_SCHEMA}.{table_name}: "
            f"{len(frames[table_name]):,} rows from {len(table_files)} file(s)"
        )
        for file in table_files:
            shutil.move(str(file), RESULTS_PUBLISHED_DIR / file.name)

    spark.stop()


def cli() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="overwrite model_predictions and model_metrics with exactly the "
             "staged runs (default: append)",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="run the checks and stop (no Spark, nothing written or moved)",
    )
    parser.add_argument(
        "--archive-staged", action="store_true",
        help="move all staged files to data/results_staging_archive/ and stop "
             "(start of a full run)",
    )
    arguments = parser.parse_args()
    if arguments.archive_staged:
        archive_staged()
        return
    main(replace=arguments.replace, validate_only=arguments.validate_only)


if __name__ == "__main__":
    cli()
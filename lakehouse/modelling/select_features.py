"""Decide whether candidate news variables belong in the context-aware model

For one tier A model (default rf) and each candidate group, the two-stage walk-forward of
``walk_forward.py`` runs on VALIDATION years (by default every year before the final test years of
train.py) and compares C (tier A + ridge on the candidate) with C0 (tier A + constant) on exactly
the same events.  Choose the loss the verdict is based on (``--primary-loss``) before looking at
results.

Result table
    n_events          events C and C0 were scored on
    C0_MSE, MSE       pooled loss of the placebo / the candidate model
    dMSE, dMAE        change vs C0 (negative = candidate better), absolute and in % (dMSE_pct, ...)
    dMSE_vs_A_pct     change vs tier A alone
    folds_won         validation years in which the primary loss improved
    corr_max          highest |correlation| between the candidate and a tier A feature (flags near
                      duplicates)
    verdict           keep: lower primary loss in EVERY validation year
                      weak: lower primary loss and better in at least half of the years
                      drop: otherwise

Everything is logged as one parent MLflow run (tag study=feature_selection) with a child run per
candidate::

    python -m lakehouse.modelling.select_features
    python -m lakehouse.modelling.select_features --model xgb --primary-loss mae
    python -m lakehouse.modelling.select_features --candidates news_sent_ewma news_sent_abs_ewma,news_count_24h
"""
from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from lakehouse.feature_columns import NEWS_SENTIMENT_COLUMNS
from lakehouse.infrastructure.mlflow_config import init_mlflow
from lakehouse.modelling.feature_sets import FEATURE_SETS
from lakehouse.modelling.gold_data import (
    DEFAULT_TEST_YEAR_COUNT, default_test_years, default_validation_years, load_gold_events)
from lakehouse.modelling.model_params import NEWS_MODEL_NAMES
from lakehouse.modelling.train import run_model

DEFAULT_CANDIDATES = list(NEWS_SENTIMENT_COLUMNS)
PRIMARY_LOSSES = {"mse": ("mse", "MSE"), "mae": ("ae", "MAE")}


def parse_candidates(items: list[str] | None) -> dict[str, list[str]]:
    """'a,b' means the group of columns a and b are tested together"""
    return {item: item.split(",") for item in (items or DEFAULT_CANDIDATES)}


def assert_same_events(candidate_predictions: pd.DataFrame, placebo_predictions: pd.DataFrame, name: str) -> None:
    """A candidate and its placebo must be scored on identical events, or the d* columns are meaningless."""
    candidate_times = set(pd.to_datetime(candidate_predictions["ts_utc"], utc=True))
    placebo_times = set(pd.to_datetime(placebo_predictions["ts_utc"], utc=True))
    if candidate_times != placebo_times:
        raise RuntimeError(f"candidate {name!r} and its placebo were scored on different events "
                           f"({len(candidate_times)} vs {len(placebo_times)}) - not like-for-like")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="rf", choices=NEWS_MODEL_NAMES, help="tier A model (benchmarks have no news stage)")
    parser.add_argument("--base", default="base", choices=list(FEATURE_SETS), help="tier A feature set (default: base)")
    parser.add_argument("--val-years", type=int, nargs="+", default=None,
                        help=f"validation years (default: every year before the last {DEFAULT_TEST_YEAR_COUNT})")
    parser.add_argument("--primary-loss", choices=list(PRIMARY_LOSSES), default="mse",
                        help="loss the keep/weak/drop verdict is based on (default: mse)")
    parser.add_argument("--candidates", nargs="+", help=f"candidate columns, comma-joined for a group (default: {DEFAULT_CANDIDATES})")
    parser.add_argument("--snapshot-id", type=int, default=None)
    arguments = parser.parse_args()

    _, primary_label = PRIMARY_LOSSES[arguments.primary_loss]
    candidates = parse_candidates(arguments.candidates)
    mlflow = init_mlflow()
    gold_events, snapshot_id = load_gold_events(arguments.snapshot_id)
    test_years = default_test_years(gold_events)
    validation_years = arguments.val_years or default_validation_years(gold_events)
    overlap = sorted(set(validation_years) & set(test_years))
    if overlap:
        print(f"WARNING: validation years {overlap} are also test years {test_years}; selecting features on "
              f"them makes the later out-of-sample result optimistic.")
    started = datetime.now(timezone.utc)
    study_id = f"feature_selection_{started:%Y%m%d_%H%M%S}"
    tags = {"study": "feature_selection", "study_id": study_id}
    print(f"{len(gold_events):,} events (snapshot {snapshot_id}); tier A={arguments.model} [{arguments.base}], "
          f"validation years {validation_years[0]}-{validation_years[-1]}, test years {test_years}, "
          f"primary loss {primary_label}, {len(candidates)} candidate group(s)")
    quiet = dict(stage_results=False, log_model=False, nested=True, verbose=False)

    with mlflow.start_run(run_name=f"{study_id}__{arguments.model}") as parent_run:
        mlflow.set_tags({**tags, "model_name": arguments.model})
        mlflow.log_params({"model": arguments.model, "base_feature_set": arguments.base,
                           "val_years": json.dumps(validation_years), "test_years_reserved": json.dumps(test_years),
                           "gold_snapshot_id": snapshot_id, "primary_loss": primary_label,
                           "n_candidates": len(candidates),
                           "n_invalid_target_dropped": gold_events.attrs.get("n_invalid_target_dropped", "unknown"),
                           "comparison": "context-aware C vs placebo C0 on identical events"})

        # Reference only: tier A alone on every event of the validation years.
        reference = run_model(mlflow, arguments.model, gold_events, snapshot_id, validation_years, news_features=[],
                              feature_set=arguments.base, run_tags={**tags, "role": "tier_a_all_events"}, **quiet)
        reference_metrics = reference["pooled_metrics"]
        print(f"reference tier A on ALL events [{reference['feature_set']}]: n={reference['n_events']:,}  "
              f"MSE={reference_metrics['MSE']:.3e}  MAE={reference_metrics['MAE']:.5f}")

        rows = []
        for name, columns in candidates.items():
            try:
                outcome = run_model(mlflow, arguments.model, gold_events, snapshot_id, validation_years,
                                    news_features=columns, feature_set=arguments.base,
                                    run_tags={**tags, "role": "candidate"}, **quiet)
            except (RuntimeError, KeyError) as error:   # e.g. no validation year has a large enough stage-2 pool
                print(f"  {name:<32} skipped: {error}")
                continue
            variant_c, variant_c0 = outcome["variants"]["C"], outcome["variants"]["C0"]
            assert_same_events(variant_c.predictions, variant_c0.predictions, name)

            folds_c = variant_c.fold_metrics.set_index("test_year")
            folds_c0 = variant_c0.fold_metrics.set_index("test_year")
            shared_years = folds_c.index.intersection(folds_c0.index)
            folds_won = int((folds_c.loc[shared_years, primary_label] < folds_c0.loc[shared_years, primary_label]).sum())

            # near-duplicate check: strongest |correlation| of the candidate with any tier A feature
            tier_a_columns = [column for column in outcome["features"] if column not in columns]
            complete = gold_events[columns].apply(pd.to_numeric, errors="coerce")
            correlations = (gold_events.loc[np.isfinite(complete).all(axis=1), columns + tier_a_columns]
                            .apply(pd.to_numeric, errors="coerce").corr().loc[columns, tier_a_columns].abs())
            strongest_correlation = float(correlations.max().max())
            strongest_partner = correlations.max(axis=0).idxmax()

            pooled = outcome["pooled_by_variant"]
            row = {"candidate": name, "n_cols": len(columns), "n_events": len(variant_c.predictions)}
            for metric in ("MSE", "MAE"):
                row.update({f"C0_{metric}": pooled["C0"][metric], metric: pooled["C"][metric],
                            f"d{metric}": pooled["C"][metric] - pooled["C0"][metric],
                            f"d{metric}_pct": 100 * (pooled["C"][metric] - pooled["C0"][metric]) / pooled["C0"][metric]})
            row["dMSE_vs_A_pct"] = 100 * (pooled["C"]["MSE"] - pooled["A"]["MSE"]) / pooled["A"]["MSE"]
            row["folds_won"] = f"{folds_won}/{len(shared_years)}"
            row.update({"corr_max": strongest_correlation, "corr_with": strongest_partner, "run_id": outcome["run_id"]})
            change = row[f"d{primary_label}"]
            if change < 0 and folds_won == len(shared_years):
                row["verdict"] = "keep"
            elif change < 0 and folds_won >= len(shared_years) / 2:
                row["verdict"] = "weak"
            else:
                row["verdict"] = "drop"
            rows.append(row)
            print(f"  {name:<32} n={row['n_events']:>5}  dMSE={row['dMSE_pct']:+.2f}%  dMAE={row['dMAE_pct']:+.2f}%  "
                  f"folds_won({primary_label})={row['folds_won']}  "
                  f"max|corr|={strongest_correlation:.2f} ({strongest_partner})  -> {row['verdict']}")

        if not rows:
            raise SystemExit("No candidate could be evaluated (see messages above).")
        table = pd.DataFrame(rows).sort_values(f"d{primary_label}").reset_index(drop=True)
        mlflow.log_metrics({"tier_a_all_events_MSE": reference_metrics["MSE"],
                            "tier_a_all_events_MAE": reference_metrics["MAE"],
                            "n_keep": int((table["verdict"] == "keep").sum()),
                            "n_weak": int((table["verdict"] == "weak").sum())})
        with tempfile.TemporaryDirectory() as temporary_directory:
            table_path = Path(temporary_directory) / "feature_selection.csv"
            table.to_csv(table_path, index=False)
            mlflow.log_artifact(str(table_path))

        print("\n" + table.drop(columns=["run_id"]).to_string(index=False, float_format=lambda value: f"{value:.4g}"))
        print(f"\nParent run {parent_run.info.run_id} (study_id={study_id}). Verdict uses {primary_label} "
              f"and the number of validation years won. Negative d* = the context-aware model improves on its "
              f"placebo C0, both scored on the same events.")


if __name__ == "__main__":
    main()

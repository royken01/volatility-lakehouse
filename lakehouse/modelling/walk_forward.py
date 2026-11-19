"""Two-stage, yearly expanding walk-forward evaluation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lakehouse.feature_columns import TARGET_COLUMN
from lakehouse.modelling.forecast_comparison import compare_predictions
from lakehouse.modelling.losses import (
    check_target_scale, floor_value, smearing_factor, to_log, to_model_target, volatility_forecast)
from lakehouse.modelling.ml_models import fit_predict_block, make_news_correction

MIN_TRAIN_ROWS = 50
MIN_POOL_EVENTS = 30
VARIANTS = ("A", "C0", "C")
METRIC_NAMES = ("MAE", "WMAPE", "MSE")


def compute_metrics(realized, forecast) -> dict[str, float]:
    """MAE, WMAPE and MSE of a volatility forecast."""
    realized, forecast = np.asarray(realized, dtype=float), np.asarray(forecast, dtype=float)
    total_realized = np.abs(realized).sum()
    absolute_errors = np.abs(realized - forecast)
    return {"MAE": float(absolute_errors.mean()),
            "WMAPE": float(absolute_errors.sum() / total_realized) if total_realized > 0 else float("nan"),
            "MSE": float(((realized - forecast) ** 2).mean())}


def frame_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """``compute_metrics`` of a prediction frame with columns y_true and y_pred."""
    return compute_metrics(predictions["y_true"], predictions["y_pred"])


def variant_comparison(variants: dict) -> pd.DataFrame:
    """Mean losses of the variants on their common events and the % change between them
    """
    comparisons = []
    for first, second in (("C", "C0"), ("C", "A"), ("C0", "A")):
        if first in variants and second in variants:
            table = compare_predictions(variants[first].predictions, variants[second].predictions)
            table.insert(0, "comparison", f"{first} vs {second}")
            comparisons.append(table)
    return pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()


@dataclass
class VariantResult:
    """Out-of-sample forecasts of one variant: one row per scored event, and one metrics row per test year."""
    predictions: pd.DataFrame   # ts_utc, event_type, test_year, y_true, y_pred, floor_value, log_y_true = ln(max(y, floor)) and log_y_pred = forecast of ln y
    fold_metrics: pd.DataFrame  # fold, test_year, n_train, n_test, MAE, WMAPE, MSE


@dataclass
class WalkForwardResult:
    variants: dict                 # {"A": VariantResult, ["C0", "C"]}
    news_coefficients: pd.DataFrame  # one row per test year: ridge penalty, intercept, standardised betas
    hyperparameters: pd.DataFrame    # tuned settings of every tier A block (rf / xgb)
    smearing_table: pd.DataFrame     # floor and smearing factor of every block / variant
    tier_a_errors: np.ndarray    # out-of-sample error per event: ln y - ln y_hat (raw scale: y - y_hat); NaN where not predicted
    news_available: np.ndarray       # True where all news features are present
    news_start: pd.Timestamp | None  # first event with news
    events: pd.DataFrame             # the time-sorted input


def start_of_year(year: int) -> pd.Timestamp:
    return pd.Timestamp(year=int(year), month=1, day=1, tz="UTC")


def plan_tier_a_blocks(event_times: pd.Series, event_years: np.ndarray, target: np.ndarray, block_starts: list,
                       last_test_year: int, verbose: bool, target_scale: str = "log") -> list[dict]:
    """One dict per usable block without start, training and test masks, floor and transformed training
    target.
    """
    blocks = []
    for position, block_start in enumerate(block_starts):
        next_start = block_starts[position + 1] if position + 1 < len(block_starts) else None
        year_end = start_of_year(block_start.year + 1)
        block_end = year_end if next_start is None else min(next_start, year_end)
        train_mask = (event_times < block_start).to_numpy()
        test_mask = ((event_times >= block_start) & (event_times < block_end)).to_numpy() & (event_years <= last_test_year)
        if not test_mask.any() or train_mask.sum() < MIN_TRAIN_ROWS:
            if verbose:
                print(f"  skip block {block_start:%Y-%m-%d}: {int(train_mask.sum())} train events, "
                      f"{int(test_mask.sum())} test events")
            continue
        floor = floor_value(target[train_mask])
        blocks.append({"start": block_start, "train_mask": train_mask, "test_mask": test_mask, "floor": floor,
                       "train_model_target": to_model_target(target[train_mask], floor, target_scale)})
    return blocks


def two_stage_walk_forward(events: pd.DataFrame, feature_columns: list[str], model_name: str,
                           overrides: dict | None, test_years: list[int], news_columns: list[str] | None = None,
                           verbose: bool = True, jobs: int = 1, target_scale: str = "log") -> WalkForwardResult:
    """Run stage 1 and, with ``news_columns``, stage 2 for every test year
    """
    check_target_scale(target_scale)
    events = events.sort_values("ts_utc").reset_index(drop=True)
    news_columns = list(news_columns or [])
    event_times = events["ts_utc"]
    target = events[TARGET_COLUMN].to_numpy(float)
    event_years = events["year"].to_numpy()
    event_types = events["event_type"].to_numpy()
    features = events[feature_columns]
    event_count = len(events)
    test_years = sorted({int(year) for year in test_years})
    last_test_year = test_years[-1]

    scorable = np.ones(event_count, bool)
    if news_columns:
        news_values = events[news_columns].to_numpy(float)
        news_available = np.isfinite(news_values).all(axis=1)
        if not news_available.any():
            raise RuntimeError(f"No event has all news features {news_columns}")
        scorable &= news_available
        news_start = event_times[news_available].iloc[0]
        block_starts = [news_start] + [start_of_year(year) for year in range(news_start.year + 1, last_test_year + 1)]
        if verbose:
            print(f"  news features {news_columns} available from {news_start:%Y-%m-%d} "
                  f"({int(news_available.sum()):,} of {event_count:,} events)")
    else:
        news_values, news_available, news_start = None, np.ones(event_count, bool), None
        block_starts = [start_of_year(year) for year in test_years]

    tier_a_forecast = np.full(event_count, np.nan)
    tier_a_smearing = np.full(event_count, np.nan)
    tier_a_floor = np.full(event_count, np.nan)
    realized_model_target = np.full(event_count, np.nan)
    training_size_by_year: dict[int, int] = {}
    hyperparameter_rows, smearing_rows = [], []

    blocks = plan_tier_a_blocks(event_times, event_years, target, block_starts, last_test_year, verbose, target_scale)
    stage_started = time.time()
    if jobs != 1 and len(blocks) > 1:
        from joblib import Parallel, delayed

        if verbose:
            print(f"  fitting {len(blocks)} tier A blocks in parallel (jobs={jobs})")
        fits = Parallel(n_jobs=jobs)(
            delayed(fit_predict_block)(model_name, overrides, features[block["train_mask"]], block["train_model_target"],
                                       features[block["test_mask"]]) for block in blocks)
    else:
        fits = (fit_predict_block(model_name, overrides, features[block["train_mask"]], block["train_model_target"],
                                  features[block["test_mask"]]) for block in blocks)

    for block, (train_forecast, test_forecast, tuned, seconds) in zip(blocks, fits):
        test_mask, train_count = block["test_mask"], int(block["train_mask"].sum())
        in_sample_errors = block["train_model_target"] - train_forecast
        tier_a_forecast[test_mask] = test_forecast
        tier_a_smearing[test_mask] = smearing_factor(in_sample_errors, target_scale)
        tier_a_floor[test_mask] = block["floor"]
        realized_model_target[test_mask] = to_model_target(target[test_mask], block["floor"], target_scale)
        training_size_by_year[block["start"].year] = train_count
        smearing_rows.append({"block_start": f"{block['start']:%Y-%m-%d}", "variant": "A", "n_fit": train_count,
                              "floor_value": block["floor"] if target_scale == "log" else float("nan"),
                              "smearing_factor": tier_a_smearing[test_mask][0]})
        if tuned is not None:
            hyperparameter_rows.append({"block_start": f"{block['start']:%Y-%m-%d}", "n_train": train_count, **tuned})
        if verbose:
            print(f"  tier A block {block['start']:%Y-%m-%d}: fit on {train_count:>5} events, "
                  f"predict {int(test_mask.sum()):>4}  [{seconds:.1f}s]" + (f"  tuned {tuned}" if tuned else ""))
    if verbose and blocks:
        print(f"  tier A stage: {time.time() - stage_started:.1f}s wall for {len(blocks)} blocks")

    tier_a_errors = realized_model_target - tier_a_forecast

    variant_names = VARIANTS if news_columns else ("A",)
    prediction_frames: dict[str, list[pd.DataFrame]] = {name: [] for name in variant_names}
    fold_rows: dict[str, list[dict]] = {name: [] for name in variant_names}
    coefficient_rows = []

    def score(variant: str, year: int, test_mask: np.ndarray, model_forecast: np.ndarray, smearing: float | np.ndarray):
        floor = tier_a_floor[test_mask]
        log_forecast = model_forecast if target_scale == "log" else np.log(np.maximum(model_forecast, floor))
        predictions = pd.DataFrame({
            "ts_utc": event_times[test_mask].to_numpy(), "event_type": event_types[test_mask], "test_year": year,
            "y_true": target[test_mask], "y_pred": volatility_forecast(model_forecast, smearing, target_scale),
            "floor_value": floor, "log_y_true": to_log(target[test_mask], floor), "log_y_pred": log_forecast})
        prediction_frames[variant].append(predictions)
        fold_rows[variant].append({"fold": str(year), "test_year": year, "n_train": training_size_by_year.get(year, 0),
                                   "n_test": int(test_mask.sum()), **frame_metrics(predictions)})

    for year in test_years:
        test_mask = (event_years == year) & scorable & np.isfinite(tier_a_forecast)
        if not test_mask.any():
            if verbose:
                print(f"  skip {year}: no scorable events")
            continue
        if not news_columns:
            score("A", year, test_mask, tier_a_forecast[test_mask], tier_a_smearing[test_mask])
        else:
            year_start = start_of_year(year)
            pool_mask = (news_available & (event_times >= news_start).to_numpy()
                         & (event_times < year_start).to_numpy() & np.isfinite(tier_a_errors))
            if pool_mask.sum() < MIN_POOL_EVENTS:
                if verbose:
                    print(f"  skip {year}: only {int(pool_mask.sum())} news-era out-of-sample errors "
                          f"before {year_start:%Y-%m-%d} (need {MIN_POOL_EVENTS})")
                continue
            pool_errors = tier_a_errors[pool_mask]
            correction = make_news_correction().fit(news_values[pool_mask], pool_errors)
            pool_mean_error = float(pool_errors.mean())                          # C0: constant only
            smearing_c0 = smearing_factor(pool_errors - pool_mean_error, target_scale)
            smearing_c = smearing_factor(pool_errors - correction.predict(news_values[pool_mask]), target_scale)
            score("A", year, test_mask, tier_a_forecast[test_mask], tier_a_smearing[test_mask])
            score("C0", year, test_mask, tier_a_forecast[test_mask] + pool_mean_error, smearing_c0)
            score("C", year, test_mask, tier_a_forecast[test_mask] + correction.predict(news_values[test_mask]),
                  smearing_c)
            ridge = correction.named_steps["ridge"]
            coefficient_rows.append({
                "test_year": year, "pool_events": int(pool_mask.sum()), "alpha": float(ridge.alpha_),
                "intercept": float(ridge.intercept_), "pool_mean_error": pool_mean_error,
                **{f"beta_{column}": float(beta) for column, beta in zip(news_columns, ridge.coef_)}})
            smearing_rows += [
                {"block_start": f"{year}-01-01", "variant": "C0", "n_fit": int(pool_mask.sum()), "smearing_factor": smearing_c0},
                {"block_start": f"{year}-01-01", "variant": "C", "n_fit": int(pool_mask.sum()), "smearing_factor": smearing_c}]
            if verbose:
                print(f"  {year}: pool {int(pool_mask.sum()):>4} errors, ridge alpha {ridge.alpha_:.3g}, "
                      f"scored {int(test_mask.sum()):>4} events")
        if verbose:
            print("     " + "  ".join(f"{name}: MSE={fold_rows[name][-1]['MSE']:.3e} MAE={fold_rows[name][-1]['MAE']:.4f}"
                                      for name in variant_names
                                      if fold_rows[name] and fold_rows[name][-1]["test_year"] == year))

    if not prediction_frames["A"]:
        raise RuntimeError("No usable walk-forward folds - check --test-years against the data range "
                           "(with news features, test years must come after the first year of news)")
    variants = {name: VariantResult(pd.concat(prediction_frames[name], ignore_index=True), pd.DataFrame(fold_rows[name]))
                for name in variant_names if prediction_frames[name]}
    return WalkForwardResult(
        variants=variants, news_coefficients=pd.DataFrame(coefficient_rows),
        hyperparameters=pd.DataFrame(hyperparameter_rows), smearing_table=pd.DataFrame(smearing_rows),
        tier_a_errors=tier_a_errors, news_available=news_available, news_start=news_start, events=events)

"""Comparison of two forecasts on the SAME events, shared by select_features, training and
reporting.

Compared by mean loss and by the % change in it (negative = the first forecast is better).
``compare_with_tests`` (reporting tables only) adds a 95 % moving-block bootstrap interval for that
change.

Per-event losses (volatility scale, see losses.py): ``mse`` = (y - y_pred) ** 2 and ``ae`` = |y -
y_pred|.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lakehouse.modelling.losses import squared_error

LOSS_NAMES = ("mse", "ae")


def loss_series(realized, forecast, loss_name: str) -> np.ndarray:
    """Per-event loss ('mse' or 'ae') of a volatility forecast."""
    if loss_name == "mse":
        return squared_error(realized, forecast)
    if loss_name == "ae":
        return np.abs(np.asarray(realized, dtype=float) - np.asarray(forecast, dtype=float))
    raise ValueError(f"loss must be one of {LOSS_NAMES}")


def compare_losses(losses_a, losses_b) -> dict:
    """Mean losses of A and B on the same events and A's % change relative to B."""
    losses_a, losses_b = np.asarray(losses_a, dtype=float), np.asarray(losses_b, dtype=float)
    both_finite = np.isfinite(losses_a) & np.isfinite(losses_b)
    losses_a, losses_b = losses_a[both_finite], losses_b[both_finite]
    if not len(losses_a):
        return {"n": 0, "mean_loss_a": np.nan, "mean_loss_b": np.nan, "pct_change": np.nan}
    return {"n": int(len(losses_a)), "mean_loss_a": float(losses_a.mean()), "mean_loss_b": float(losses_b.mean()),
            "pct_change": float(100 * (losses_a.mean() / losses_b.mean() - 1)) if losses_b.mean() else np.nan}


def matched_events(predictions_a: pd.DataFrame, predictions_b: pd.DataFrame, event_type: str | None = None) -> pd.DataFrame:
    """Events both forecasts predicted, in time order, with y_true, y_pred_a and y_pred_b (and event_type)."""
    def prepare(predictions, wanted_columns):
        kept = predictions[[column for column in wanted_columns if column in predictions]].copy()
        kept["ts_utc"] = pd.to_datetime(kept["ts_utc"], utc=True)
        return kept

    first = prepare(predictions_a, ["ts_utc", "event_type", "y_true", "y_pred"])
    second = prepare(predictions_b, ["ts_utc", "y_pred"])
    matched = first.merge(second, on="ts_utc", suffixes=("_a", "_b")).sort_values("ts_utc")
    if event_type:
        matched = matched[matched["event_type"] == event_type]
    return matched


def bootstrap_interval(losses_a, losses_b, block_length: int = 8, draws: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95 % interval of A's % change in mean loss relative to B (percentile method, circular moving
    blocks).
    """
    losses_a, losses_b = np.asarray(losses_a, dtype=float), np.asarray(losses_b, dtype=float)
    count = len(losses_a)
    random = np.random.default_rng(seed)
    blocks = int(np.ceil(count / block_length))
    starts = random.integers(0, count, size=(draws, blocks))
    resampled = ((starts[:, :, None] + np.arange(block_length)) % count).reshape(draws, -1)[:, :count]
    percent_change = 100 * (losses_a[resampled].mean(axis=1) / losses_b[resampled].mean(axis=1) - 1)
    low, high = np.percentile(percent_change, [2.5, 97.5])
    return float(low), float(high)


def compare_with_tests(predictions_a: pd.DataFrame, predictions_b: pd.DataFrame, loss_name: str = "mse",
                       event_type: str | None = None, block_length: int = 8, draws: int = 5000) -> dict:
    """``compare_losses`` of two prediction frames on their common events, plus the bootstrap interval (ci_low, ci_high)."""
    matched = matched_events(predictions_a, predictions_b, event_type)
    losses_a = loss_series(matched["y_true"], matched["y_pred_a"], loss_name)
    losses_b = loss_series(matched["y_true"], matched["y_pred_b"], loss_name)
    comparison = compare_losses(losses_a, losses_b)
    if comparison["n"] < 10:
        return {**comparison, "ci_low": np.nan, "ci_high": np.nan}
    finite = np.isfinite(losses_a) & np.isfinite(losses_b)
    low, high = bootstrap_interval(losses_a[finite], losses_b[finite], block_length, draws)
    return {**comparison, "ci_low": low, "ci_high": high}


def compare_predictions(predictions_a: pd.DataFrame, predictions_b: pd.DataFrame, loss_names=LOSS_NAMES,
                        event_type: str | None = None) -> pd.DataFrame:
    """Compare two prediction frames (ts_utc, event_type, y_true, y_pred) on the events in both.

        ``event_type`` (a configured release-type label) restricts the comparison to one release type.
    """
    matched = matched_events(predictions_a, predictions_b, event_type)
    rows = []
    for loss_name in loss_names:
        losses_a = loss_series(matched["y_true"], matched["y_pred_a"], loss_name)
        losses_b = loss_series(matched["y_true"], matched["y_pred_b"], loss_name)
        rows.append({"loss": loss_name, **compare_losses(losses_a, losses_b)})
    return pd.DataFrame(rows)

"""Fast checks of the pure parts of the framework: no Spark, no Iceberg, no MLflow, no network."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import polars as pl
import pytest

from pathlib import Path

from lakehouse import cli
from lakehouse.feature_columns import HAR_COLUMNS, LEAKAGE_COLUMNS, NEWS_SENTIMENT_COLUMNS, SEQUENCE_COLUMNS, TARGET_COLUMN
from lakehouse.medallion.silver.price_processing import add_log_returns, fill_grid
from lakehouse.medallion.silver.time_conversion import drop_bad_events, events_to_utc, prices_to_utc
from lakehouse.settings import load_settings

THESIS = load_settings(Path(__file__).resolve().parents[1] / "configs" / "crude_oil_thesis.toml", environment={})
from lakehouse.modelling.feature_sets import resolve_feature_set
from lakehouse.modelling.gold_data import prepare_features
from lakehouse.modelling.losses import floor_value, smearing_factor, to_model_target, volatility_forecast
from lakehouse.modelling.walk_forward import compute_metrics, two_stage_walk_forward


# --- time conversion (the thesis clocks) -------------------------------------------------------------
def utc(frame, column="ts_utc"):
    return [value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M") for value in frame[column].to_list()]


def test_events_before_2024_use_a_fixed_utc_plus_one_clock():
    events = pl.DataFrame({"release_datetime": ["2023-07-12 15:30:00", "2023-01-11 15:30:00"],
                           "event": ["Crude Oil Inventories"] * 2})
    converted = events_to_utc(events, "release_datetime", "ts_utc", THESIS)
    assert utc(converted) == ["2023-01-11 14:30", "2023-07-12 14:30"]     # 10:30 New York in winter and in summer


def test_events_from_2024_follow_oslo_time_and_late_api_stamps_move_back():
    events = pl.DataFrame({"release_datetime": ["2024-07-10 16:30:00", "2016-03-02 17:35:00"],
                           "event": ["Crude Oil Inventories", "API Weekly Crude Oil Stock"]})
    converted = events_to_utc(events, "release_datetime", "ts_utc", THESIS).sort("ts_utc")
    assert utc(converted) == ["2016-03-02 16:30", "2024-07-10 14:30"]


def test_price_clock_switches_from_us_to_eu_rules_in_2019():
    prices = pl.DataFrame({"datetime": ["20180715 090000000", "20190715 090000000", "20190115 090000"]})
    converted = prices_to_utc(prices, "datetime", "ts_utc", THESIS)
    assert utc(converted) == ["2018-07-15 13:00", "2019-01-15 14:00", "2019-07-15 13:00"]


def test_drop_invalid_events_removes_forecast_less_duplicate_stamps():
    stamps = [datetime(2022, 6, 29, 14, 25), datetime(2022, 6, 29, 14, 30), datetime(2022, 7, 6, 14, 30)]
    events = pl.DataFrame({"ts_utc": stamps, "event": ["Crude Oil Inventories"] * 3, "forecast": [None, -1.0, 0.5],
                           "actual": [1.0, 2.0, 3.0], "surprise": [None, 3.0, 2.5]})
    assert drop_bad_events(events, settings=THESIS)["ts_utc"].to_list() == stamps[1:]


# --- prices ----------------------------------------------------------------------------------------------
def test_fill_grid_fills_minutes_but_not_long_gaps():
    stamps = [datetime(2024, 1, 2, 10, 0), datetime(2024, 1, 2, 10, 3), datetime(2024, 1, 3, 10, 0)]
    prices = pl.DataFrame({"ts": stamps, "close": [100.0, 101.0, 102.0]})
    grid = fill_grid(prices, "ts", "close", step_s=60, max_gap=3600)
    assert len(grid) == 4 + 1                                   # 10:00-10:03 filled, the next day is a new block
    assert grid["close"].to_list() == [100.0, 100.0, 100.0, 101.0, 102.0]


def test_log_returns_are_in_percent_and_start_at_zero():
    returns = add_log_returns(pl.DataFrame({"ts_utc": [1, 2], "close": [100.0, 101.0]}), "close")["log_ret"].to_list()
    assert returns[0] == 0.0 and returns[1] == pytest.approx(100 * np.log(1.01))


# --- losses ---------------------------------------------------------------------------------------------
def test_floor_is_half_the_smallest_positive_training_value():
    assert floor_value([0.0, 0.4, 0.2, 1.0]) == pytest.approx(0.1)


def test_log_target_round_trip_and_raw_scale_is_left_alone():
    y = np.array([0.0, 0.2, 0.5])
    floor = floor_value(y)
    assert to_model_target(y, floor, "log") == pytest.approx(np.log([floor, 0.2, 0.5]))
    assert smearing_factor([np.log(2.0), -np.log(2.0)]) == pytest.approx(1.25)
    assert smearing_factor([1.0, 2.0], "raw") == 1.0
    assert volatility_forecast([-0.1, 0.3], 1.0, "raw").tolist() == [0.0, 0.3]


def test_metrics_on_the_volatility_scale():
    metrics = compute_metrics([1.0, 2.0], [2.0, 2.0])
    assert metrics == {"MAE": 0.5, "WMAPE": pytest.approx(1 / 3), "MSE": 0.5}


# --- feature sets ----------------------------------------------------------------------------------------
def test_sequence_columns_collapse_to_a_short_label():
    label, columns = resolve_feature_set("base", SEQUENCE_COLUMNS)
    assert label == "base+seq10" and columns[-10:] == SEQUENCE_COLUMNS


def test_post_event_columns_and_news_cannot_be_tier_a_features():
    events = pd.DataFrame({"ts_utc": pd.to_datetime(["2024-01-01"], utc=True), **{c: [1.0] for c in
                           HAR_COLUMNS + LEAKAGE_COLUMNS + NEWS_SENTIMENT_COLUMNS}})
    with pytest.raises(ValueError):
        prepare_features(events, ["rv_d", "n_post_ticks"], [])
    with pytest.raises(ValueError):
        prepare_features(events, ["rv_d", "news_count_24h"], [])


# --- walk-forward ----------------------------------------------------------------------------------------
def synthetic_events(count: int = 520, seed: int = 0) -> pd.DataFrame:
    random = np.random.default_rng(seed)
    times = pd.date_range("2018-01-03", periods=count, freq="3D", tz="UTC")
    rv = np.exp(random.normal(-1.0, 0.4, count))
    events = pd.DataFrame({"ts_utc": times, "year": times.year, "rv_d": rv, "rv_w": rv, "rv_m": rv,
                           "crude_inventories": np.arange(count) % 2, "event_type": np.where(np.arange(count) % 2, "EIA", "API")})
    events[TARGET_COLUMN] = np.round(rv * np.exp(random.normal(0, 0.3, count)) * (1 + events["crude_inventories"]), 4)
    news = random.normal(size=(count, 3))
    news[times < "2019-10-01"] = np.nan
    events[NEWS_SENTIMENT_COLUMNS] = news
    return events


@pytest.mark.parametrize("model", ["naive_mean", "har_ols"])
def test_two_stage_walk_forward_scores_the_same_events_in_every_variant(model):
    events = synthetic_events()
    features = ["rv_d", "rv_w", "rv_m", "crude_inventories"] if model == "har_ols" else ["crude_inventories"]
    result = two_stage_walk_forward(events, features, model, None, [2021, 2022], NEWS_SENTIMENT_COLUMNS, verbose=False)
    assert set(result.variants) == {"A", "C0", "C"}
    stamps = [set(variant.predictions["ts_utc"]) for variant in result.variants.values()]
    assert stamps[0] == stamps[1] == stamps[2] and len(stamps[0]) > 100
    predictions = result.variants["A"].predictions
    assert predictions["ts_utc"].dt.year.isin([2021, 2022]).all() and (predictions["y_pred"] > 0).all()


def test_walk_forward_without_news_scores_only_variant_a():
    result = two_stage_walk_forward(synthetic_events(), ["crude_inventories"], "naive_mean", None, [2021], [], verbose=False)
    assert list(result.variants) == ["A"]


# --- workspace and command line -----------------------------------------------------------------------------
def test_workspace_follows_lakehouse_home(monkeypatch, tmp_path):
    from lakehouse import paths
    monkeypatch.setenv("LAKEHOUSE_HOME", str(tmp_path))
    assert paths.find_workspace() == tmp_path.resolve()
    monkeypatch.delenv("LAKEHOUSE_HOME")
    assert paths.find_workspace() == paths.PACKAGE_DIR       # source checkout: state lives next to the code


def test_cli_lists_commands_and_rejects_unknown_ones(capsys):
    assert cli.main([]) == 0 and "vol-lakehouse" in capsys.readouterr().out
    assert cli.main(["no-such-command"]) == 2
    assert all(module.startswith("lakehouse.") for module in cli.MODULES.values())

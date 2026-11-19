"""The asset- and window-agnostic parts: settings, CLI options, gold features for any window,
the GDELT headline filter and the economic-calendar parsers.  No network, Spark or MLflow."""
import csv
import datetime as dt
import io
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from lakehouse import cli
from lakehouse.medallion.gold import event_features
from lakehouse.medallion.silver.time_conversion import events_to_utc, prices_to_utc
from lakehouse.settings import Settings, get_settings, load_settings, make_event_type, settings_from_mapping
from lakehouse.sources import economic_calendar as calendar
from lakehouse.sources.gdelt_collector import StorySelector, TopicMatcher, parse_gkg_blob, sample_stamps

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# --- settings ----------------------------------------------------------------------------------------
def test_defaults_are_generic():
    settings = get_settings()
    assert settings.target_column == "realized_vol_60s" and settings.lag_prefix == "pre_vol_60s_lag"
    assert [t.label for t in settings.resolved_event_types] == ["all"] and settings.model_event_dummies == ["is_event"]


def test_example_config_and_environment_overrides():
    settings = load_settings(CONFIGS / "example_eurusd_us_macro.toml",
                             environment={"LAKEHOUSE_POST_WINDOW_SECONDS": "120", "LAKEHOUSE_ASSET": "GBP/USD"})
    assert settings.asset == "GBP/USD" and settings.target_column == "realized_vol_120s" and settings.lags == 6
    assert settings.model_event_dummies == ["is_nfp", "is_cpi"]          # FOMC is the reference type
    assert settings.event_type_of("Nonfarm Payrolls").label == "NFP" and settings.event_type_of("GDP") is None


def test_every_config_in_the_repository_loads():
    for path in CONFIGS.glob("*.toml"):
        assert load_settings(path, environment={}).target_column.startswith("realized_vol_")


def test_the_tick_window_must_cover_the_post_event_window():
    with pytest.raises(ValueError):
        settings_from_mapping({"windows": {"post_event_seconds": 900, "tick_window_seconds": 600}}).validate()


def test_cli_framework_options_become_environment_variables(monkeypatch):
    for variable in cli.FRAMEWORK_FLAGS.values():
        monkeypatch.delenv(variable, raising=False)
    remaining = cli.apply_framework_options(["train", "--model", "rf", "--post-window-seconds", "300",
                                             "--event-type", "Nonfarm Payrolls=NFP", "--event-type", "CPI m/m=CPI*"])
    try:
        assert remaining == ["train", "--model", "rf"]
        assert os.environ["LAKEHOUSE_POST_WINDOW_SECONDS"] == "300"
        settings = get_settings()
        assert settings.target_column == "realized_vol_300s"
        assert [(t.label, t.reference) for t in settings.event_types] == [("NFP", False), ("CPI", True)]
    finally:
        for variable in cli.FRAMEWORK_FLAGS.values():
            os.environ.pop(variable, None)
        get_settings.cache_clear()


# --- clocks -----------------------------------------------------------------------------------------------
def test_generic_clocks_iso_strings_and_datetime_columns():
    settings = settings_from_mapping({"events": {"clock": "America/New_York"}, "prices": {"clock": "UTC"}})
    events = pl.DataFrame({"release_datetime": ["2024-03-08 08:30:00"], "event": ["Nonfarm Payrolls"]})
    converted = events_to_utc(events, "release_datetime", "ts_utc", settings)
    assert converted["ts_utc"][0] == datetime(2024, 3, 8, 13, 30, tzinfo=timezone.utc)
    prices = pl.DataFrame({"datetime": ["2024-03-08 13:30:00.250", "2024-03-08T13:31:00"]})
    assert prices_to_utc(prices, "datetime", "ts_utc", settings)["ts_utc"].dt.second().to_list() == [0, 0]
    typed = pl.DataFrame({"datetime": [datetime(2024, 3, 8, 13, 30)]})
    assert prices_to_utc(typed, "datetime", "ts_utc", settings)["ts_utc"][0].hour == 13


# --- gold features for any window ---------------------------------------------------------------------------
def synthetic_market(event_count: int = 30, seed: int = 1):
    """60 days of 1-minute bars, ticks every 5 s around daily 14:30 UTC releases of two types."""
    random = np.random.default_rng(seed)
    minutes = pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 3, 1), "1m", eager=True, time_zone="UTC")
    minute_bars = pl.DataFrame({"ts_utc": minutes, "log_ret": random.normal(0, 0.02, len(minutes))})
    release_days = pl.date_range(datetime(2024, 1, 31), datetime(2024, 2, 29), "1d", eager=True)[:event_count]
    releases = [datetime(d.year, d.month, d.day, 14, 30, tzinfo=timezone.utc) for d in release_days]
    events = pl.DataFrame({"ts_utc": releases, "event": ["Alpha" if i % 2 else "Beta" for i in range(len(releases))],
                           "forecast": random.normal(size=len(releases))})
    ticks = []
    for release in releases:
        stamps = [release + dt.timedelta(seconds=s) for s in range(-300, 301, 5)]
        ticks.append(pl.DataFrame({"ts_utc": stamps, "log_ret": random.normal(0, 0.05, len(stamps)),
                                   "is_tick": [True] * len(stamps)}))
    return minute_bars, pl.concat(ticks), events


def test_gold_target_follows_the_configured_window():
    minute_bars, ticks, events = synthetic_market()
    features = event_features.build_features(minute_bars, ticks, events)
    target = event_features.TARGET_COLUMN
    assert target == "realized_vol_60s" and {"event_type", "is_event", "pre_vol_60s_lag10"} <= set(features.columns)
    first = features.row(0, named=True)
    window = ticks.filter((pl.col("ts_utc") >= first["ts_utc"])
                          & (pl.col("ts_utc") <= first["ts_utc"] + dt.timedelta(seconds=60)))
    assert first[target] == pytest.approx(float(np.sqrt((window["log_ret"].to_numpy() ** 2).sum())), abs=1e-4)


def test_gold_labels_configured_event_types_and_lags_within_type(monkeypatch):
    settings = Settings(event_types=(make_event_type("Alpha", "A"), make_event_type("Beta", "B", reference=True)),
                        lags=3)
    monkeypatch.setattr(event_features, "_settings", settings)
    monkeypatch.setattr(event_features, "SEQUENCE_LENGTH", 3)
    monkeypatch.setattr(event_features, "SEQUENCE_COLUMNS", [f"pre_vol_60s_lag{i}" for i in (1, 2, 3)])
    minute_bars, ticks, events = synthetic_market()
    features = event_features.build_features(minute_bars, ticks, events.vstack(
        pl.DataFrame({"ts_utc": [datetime(2024, 2, 20, 9, 0, tzinfo=timezone.utc)], "event": ["Gamma"],
                      "forecast": [0.0]})).sort("ts_utc"))
    assert set(features["event_type"]) == {"A", "B"}                 # Gamma is not a configured type
    assert (features["is_a"] == (features["event_type"] == "A").cast(pl.Int64)).all()
    alpha = features.filter(pl.col("event_type") == "A")
    assert alpha["pre_vol_60s_lag1"][1:].to_list() == alpha["realized_vol_60s"][:-1].to_list()


# --- GDELT ----------------------------------------------------------------------------------------------------
def topic_settings(**news) -> Settings:
    return settings_from_mapping({"news": {"keywords": ["gold price", "bullion"], "ambiguous": {"gold": ["ounce", "price", "fed"]},
                                           "mask": ["gold medal"], "gdelt": {"min_score": 2, **news}}})


def test_topic_matcher_keywords_context_and_mask():
    matcher = TopicMatcher(topic_settings(domains=["reuters.com"]))
    assert matcher.is_relevant("Gold prices climb as Fed signals cuts")
    assert not matcher.is_relevant("Swimmer wins gold medal")
    assert not matcher.is_relevant("Gold Coast hotel opens")
    assert matcher.score("Bullion and gold price rally", domain="www.reuters.com") > matcher.score("Bullion rally")


def gkg_line(stamp, url, title, themes="", translated="") -> bytes:
    fields = [""] * 27
    fields[1], fields[3], fields[4], fields[7], fields[15] = stamp, "reuters.com", url, themes, "-2.5,1,3,4,0,0,100"
    fields[25], fields[26] = translated, f"<PAGE_TITLE>{title}</PAGE_TITLE>"
    return "\t".join(fields).encode()


def test_parse_gkg_blob_keeps_relevant_english_headlines_best_first():
    lines = [gkg_line("20240102101500", "https://a.com/1", "Gold price hits record as bullion demand soars"),
             gkg_line("20240102101500", "https://a.com/2", "Gold medal for local team"),
             gkg_line("20240102101500", "https://a.com/3", "Bullion steady", translated="srclc:deu"),
             gkg_line("20240102101500", "https://a.com/4", "Bullion steady")]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("20240102101500.gkg.csv", b"\n".join(lines))
    rows = parse_gkg_blob(buffer.getvalue(), TopicMatcher(topic_settings()))
    assert [row["url"] for row in rows] == ["https://a.com/1", "https://a.com/4"]
    assert rows[0]["seen_utc"] == "2024-01-02T10:15:00+00:00" and rows[0]["tone"] == -2.5


def test_story_selector_drops_near_duplicates_and_caps_per_file():
    selector = StorySelector(max_per_file=2, max_per_day=0, jaccard=0.6, window_hours=72)
    first = [{"seen_utc": "2024-01-02T10:15:00+00:00", "title": "Gold price hits record high"},
             {"seen_utc": "2024-01-02T10:15:00+00:00", "title": "Gold price hits a record high"},
             {"seen_utc": "2024-01-02T10:15:00+00:00", "title": "Central banks buy bullion"}]
    assert [row["title"] for row in selector.select(first)] == ["Gold price hits record high", "Central banks buy bullion"]
    assert sample_stamps(["20240102000000", "20240102001500", "20240102010000"], every=4) == \
        ["20240102000000", "20240102010000"]


# --- economic calendar --------------------------------------------------------------------------------------------
def test_number_parsing_of_calendar_values():
    assert calendar.parse_number("0.4%") == 0.4 and calendar.parse_number("-0.90T") == pytest.approx(-0.9e12)
    assert calendar.parse_number("250K") == 250000 and calendar.parse_number("") is None


def test_forexfactory_items_become_rows_in_the_calendar_clock():
    items = [{"title": "Non-Farm Employment Change", "country": "USD", "date": "2026-12-04T08:30:00-05:00",
              "impact": "High", "forecast": "180K", "previous": "175K"},
             {"title": "Trade Balance", "country": "JPY", "date": "2026-12-03T18:50:00-05:00", "impact": "Low"}]
    rows = calendar.forexfactory_rows(items, "UTC", countries=["usd"], impacts=["High"], rename="Nonfarm Payrolls")
    assert len(rows) == 1 and rows[0]["release_datetime"] == "2026-12-04 13:30:00"
    assert rows[0]["event"] == "Nonfarm Payrolls" and rows[0]["forecast"] == 180000 and rows[0]["actual"] is None


def test_alfred_initial_release_values_and_transforms():
    observations = [{"date": "2024-01-01", "realtime_start": "2024-02-02", "value": "100"},
                    {"date": "2024-02-01", "realtime_start": "2024-03-08", "value": "103"},
                    {"date": "2024-03-01", "realtime_start": "2024-04-05", "value": "."}]
    assert calendar.initial_release_values(observations, "diff")[dt.date(2024, 3, 8)] == (3.0, None)
    assert calendar.initial_release_values(observations, "pct")[dt.date(2024, 3, 8)][0] == pytest.approx(3.0)
    assert calendar.initial_release_values(observations)[dt.date(2024, 2, 2)] == (100.0, None)


def test_merge_into_csv_adds_new_releases_and_fills_in_actuals(tmp_path):
    path = tmp_path / "events.csv"
    scheduled = calendar.make_row(datetime(2026, 12, 4, 13, 30, tzinfo=timezone.utc), "NFP", "UTC", forecast=180.0)
    assert calendar.merge_into_csv([scheduled], path) == (1, 0)
    released = calendar.make_row(datetime(2026, 12, 4, 13, 30, tzinfo=timezone.utc), "NFP", "UTC", actual=210.0)
    assert calendar.merge_into_csv([released], path) == (0, 1)
    row = next(csv.DictReader(open(path)))
    assert row["actual"] == "210.0" and row["forecast"] == "180.0" and float(row["surprise"]) == 30.0

"""Tick versus 1-minute measurement of the post-release realized volatility (Appendix on tick data
versus 1-minute bars).

For each event (release time t0) and window W the volatility is measured twice.  The opening price
is the last one STRICTLY before t0 (NaN, never imputed, when none lies within
``OPEN_MAX_STALE_SECONDS``); no price change inside the window means volatility 0.

    rv_tick_<w>, rv_1s_<w>, n_updates_<w>   mid-quote based, windows 30 s / 1 m / 5 m
    rv_m1_<w>                               1-minute close-to-close returns, windows of at least
                                            one minute

``measurement_comparison`` reports, per window and on the same events: mean tick and minute
volatility, the ratio (share of tick volatility the minute bars capture) and the zero shares.

    python -m lakehouse.analysis.tick_vs_minute_volatility
    python -m lakehouse.analysis.tick_vs_minute_volatility --cutoff 2026-01-01

Source: silver.events, silver.prices_tick, silver.prices_m1.  Output:
``lakehouse/outputs/tick_vs_minute/measurement_comparison.csv``.
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from lakehouse.infrastructure.iceberg_io import load_iceberg_table
from lakehouse.infrastructure.spark_session import SILVER_SCHEMA
from lakehouse.paths import TICK_VS_MINUTE_OUTPUT_DIR

WINDOWS_SECONDS = {"30s": 30, "1m": 60, "5m": 300}
MINUTE_BAR_WINDOWS = ("1m", "5m")
GRID_STEP_SECONDS = 1
BAR_SECONDS = 60
OPEN_MAX_STALE_SECONDS = 60
DEFAULT_CUTOFF = "2023-01-01"  # first default test year
NANOSECONDS = 1_000_000_000


def to_nanoseconds(series: pl.Series) -> np.ndarray:
    """Datetime series (naive = UTC, or tz-aware) to int64 nanoseconds since the epoch, UTC."""
    if series.dtype.time_zone is not None:
        series = series.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return series.dt.epoch("ns").to_numpy().astype(np.int64)


def grid_volatility(times: np.ndarray, log_prices: np.ndarray, grid: np.ndarray) -> float:
    """Previous-tick realized volatility on ``grid`` (first point = t0); the price at a grid point
    is the last observation at or before it."""
    position = np.searchsorted(times, grid[1:], side="right") - 1
    path = np.concatenate(([log_prices[0]], log_prices[position]))
    returns = np.diff(path)
    return float(np.sqrt(np.sum(returns * returns)))


def tick_volatility(ticks: pl.DataFrame, event_times: pl.Series,
                    windows=WINDOWS_SECONDS) -> pl.DataFrame:
    """rv_tick_, rv_1s_ and n_updates_ columns per window (NaN -> null where no opening quote)."""
    ordered = (ticks.select("ts_utc", "bidask_mid").drop_nulls()
               .filter(pl.col("bidask_mid") > 0).sort("ts_utc"))
    tick_times = to_nanoseconds(ordered["ts_utc"])
    log_mid = np.log(ordered["bidask_mid"].to_numpy().astype(float))
    releases = to_nanoseconds(event_times)
    columns = {
        f"{measure}_{window}": np.full(len(releases), np.nan)
        for window in windows for measure in ("rv_tick", "rv_1s", "n_updates")
    }
    stale_limit = OPEN_MAX_STALE_SECONDS * NANOSECONDS
    for event_index, release in enumerate(releases):
        opening = np.searchsorted(tick_times, release, side="left") - 1
        if opening < 0 or tick_times[opening] < release - stale_limit:
            continue
        for window, window_seconds in windows.items():
            window_end = release + window_seconds * NANOSECONDS
            stop = np.searchsorted(tick_times, window_end, side="right")
            times, prices = tick_times[opening:stop], log_mid[opening:stop]
            returns = np.diff(prices)
            columns[f"rv_tick_{window}"][event_index] = float(np.sqrt(np.sum(returns * returns)))
            columns[f"n_updates_{window}"][event_index] = float(np.count_nonzero(returns))
            grid = np.arange(release, window_end + 1, GRID_STEP_SECONDS * NANOSECONDS,
                             dtype=np.int64)
            columns[f"rv_1s_{window}"][event_index] = grid_volatility(times, prices, grid)
    table = pl.DataFrame(columns).with_columns(pl.all().fill_nan(None))
    return table.with_columns([pl.col(f"n_updates_{w}").cast(pl.Int32) for w in windows])


def minute_volatility(bars: pl.DataFrame, event_times: pl.Series,
                      windows=MINUTE_BAR_WINDOWS) -> pl.DataFrame:
    """rv_m1_ columns. HistData stamps a bar with its OPEN time, so a bar ends 60 s later."""
    ordered = (bars.select("ts_utc", "close").drop_nulls()
               .filter(pl.col("close") > 0).sort("ts_utc"))
    bar_ends = to_nanoseconds(ordered["ts_utc"]) + BAR_SECONDS * NANOSECONDS
    log_close = np.log(ordered["close"].to_numpy().astype(float))
    releases = to_nanoseconds(event_times)
    columns = {f"rv_m1_{window}": np.full(len(releases), np.nan) for window in windows}
    stale_limit = OPEN_MAX_STALE_SECONDS * NANOSECONDS
    for event_index, release in enumerate(releases):
        opening = np.searchsorted(bar_ends, release, side="right") - 1
        if opening < 0 or bar_ends[opening] < release - stale_limit:
            continue
        for window in windows:
            window_end = release + WINDOWS_SECONDS[window] * NANOSECONDS
            stop = np.searchsorted(bar_ends, window_end, side="right")
            grid = np.arange(release, window_end + 1, BAR_SECONDS * NANOSECONDS,
                             dtype=np.int64)
            columns[f"rv_m1_{window}"][event_index] = grid_volatility(
                bar_ends[opening:stop], log_close[opening:stop], grid
            )
    return pl.DataFrame(columns).with_columns(pl.all().fill_nan(None))


def events_before(events: pl.DataFrame, cutoff: str) -> pl.DataFrame:
    """Events released strictly before ``cutoff`` (a date string)."""
    times = events["ts_utc"]
    if times.dtype.time_zone is not None:
        times = times.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    cutoff_ns = pl.Series([cutoff]).str.to_datetime(time_unit=times.dtype.time_unit)[0]
    return events.filter(times < cutoff_ns)


def measurement_comparison(volatilities: pl.DataFrame,
                           windows=WINDOWS_SECONDS) -> pl.DataFrame:
    """Per window, on the same events: mean tick vol, mean minute vol, their ratio and zero shares
    (null minute columns below 1 minute)."""
    columns = [f"rv_tick_{w}" for w in windows]
    columns += [f"rv_m1_{w}" for w in MINUTE_BAR_WINDOWS if f"rv_m1_{w}" in volatilities.columns]
    common_events = volatilities.select(columns).drop_nulls()
    rows = []
    for window in windows:
        tick = common_events[f"rv_tick_{window}"].to_numpy()
        row = {
            "window": window,
            "n_events": len(common_events),
            "mean_rv_tick": float(tick.mean()),
            "zero_tick": float(np.mean(tick == 0)),
        }
        minute_column = f"rv_m1_{window}"
        if minute_column in common_events.columns:
            minute = common_events[minute_column].to_numpy()
            row["mean_rv_m1"] = float(minute.mean())
            row["rv_ratio"] = float(minute.mean() / tick.mean())
            row["zero_m1"] = float(np.mean(minute == 0))
        else:
            row["mean_rv_m1"] = None
            row["rv_ratio"] = None
            row["zero_m1"] = None
        rows.append(row)
    return pl.DataFrame(rows).select(
        ["window", "n_events", "mean_rv_tick", "mean_rv_m1", "rv_ratio", "zero_tick", "zero_m1"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF,
                        help="only events before this date (default: first test year)")
    args = parser.parse_args()

    events = events_before(load_iceberg_table(SILVER_SCHEMA, "events"), args.cutoff)
    ticks = load_iceberg_table(SILVER_SCHEMA, "prices_tick")
    bars = load_iceberg_table(SILVER_SCHEMA, "prices_m1")
    volatilities = pl.concat(
        [events, tick_volatility(ticks, events["ts_utc"]), minute_volatility(bars, events["ts_utc"])],
        how="horizontal",
    )
    comparison = measurement_comparison(volatilities)
    print(f"Post-event volatility: tick mid-quotes vs 1-minute bars (events before {args.cutoff})")
    with pl.Config(tbl_cols=12, tbl_width_chars=200, float_precision=5):
        print(comparison)
    TICK_VS_MINUTE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.write_csv(TICK_VS_MINUTE_OUTPUT_DIR / "measurement_comparison.csv")


if __name__ == "__main__":
    main()
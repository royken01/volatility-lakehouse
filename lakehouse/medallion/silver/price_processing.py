"""Silver price tables: gap-free 1-minute bars with log returns, and tick windows around the events.

Pure Polars / NumPy. ``log_ret = 100 * ln(P_t / P_{t-1})``, in percent. In
the tick table the return is computed inside each event window, so the first
row of a window is 0 and no return spans two events.
"""
import numpy as np
import polars as pl

from lakehouse.settings import get_settings

# Ticks are kept within +/- this many seconds of every event (settings: windows.tick_window_seconds).
TICK_WINDOW_SECONDS = get_settings().tick_window_seconds
MIN_TICKS = get_settings().min_ticks


def fill_grid(frame: pl.DataFrame, ts_col: str, price_col: str,
              step_s: int, max_gap: int) -> pl.DataFrame:
    """Put a price series on a regular grid, forward-filling the price inside each trading block.

    A new block starts wherever two stamps are more than ``max_gap`` apart,
    so weekends, holidays and outages are not bridged.
    """
    frame = frame.sort(ts_col)
    frame = frame.with_columns(
        (pl.col(ts_col).diff().dt.total_seconds() > max_gap)
        .fill_null(False)
        .cast(pl.Int32)
        .alias("new_block")
    )
    frame = frame.with_columns(pl.col("new_block").cum_sum().alias("block"))

    bounds = frame.group_by("block").agg(
        pl.col(ts_col).min().alias("first_ts"),
        pl.col(ts_col).max().alias("last_ts"),
    )
    grid = (
        bounds
        .with_columns(
            pl.datetime_ranges(
                pl.col("first_ts"),
                pl.col("last_ts"),
                interval=f"{step_s}s",
                closed="both",
            ).alias(ts_col)
        )
        .select("block", ts_col)
        .explode(ts_col)
    )
    filled = grid.join(
        frame.drop("new_block"), on=["block", ts_col], how="left",
    ).sort(["block", ts_col])
    filled = filled.with_columns(pl.col(price_col).forward_fill().over("block"))
    return filled.drop("block").sort(ts_col)


def add_log_returns(frame: pl.DataFrame, price_col: str,
                    ts_col: str = "ts_utc") -> pl.DataFrame:
    """Add ``log_ret`` = 100 * ln(price / previous price); the first row gets 0."""
    frame = frame.sort(ts_col)
    previous = pl.col(price_col).shift(1)
    log_ret = (pl.col(price_col) / previous).log() * 100
    return frame.with_columns(log_ret.fill_null(0.0).alias("log_ret"))


def build_silver_minute_bars(bronze_minute_bars: pl.DataFrame) -> pl.DataFrame:
    """1-minute bars on a complete 60 s grid (blocks split at gaps over 5 h) plus ``log_ret``."""
    filled = fill_grid(
        bronze_minute_bars,
        ts_col="ts_utc",
        price_col="close",
        step_s=60,
        max_gap=5 * 3600,
    )
    return add_log_returns(filled, "close")


def window_tick_returns(window: pl.DataFrame) -> pl.DataFrame:
    """Bid-ask mid-quote (or the trade ``price`` when the feed has no quotes) and its log return
    for the ticks of ONE event window.

    The first tick of the window gets 0.
    """
    if {"bid", "ask"} <= set(window.columns):
        mid = (pl.col("bid") + pl.col("ask")) / 2
    elif "price" in window.columns:
        mid = pl.col("price")
    else:
        raise KeyError("tick data needs bid and ask columns, or a price column")
    with_mid = window.with_columns(mid.cast(pl.Float64).alias("bidask_mid"), pl.lit(True).alias("is_tick"))
    return add_log_returns(with_mid, "bidask_mid")


def build_silver_ticks(tick_prices: pl.DataFrame, minute_bars: pl.DataFrame,
                       events: pl.DataFrame,
                       window_seconds: int = TICK_WINDOW_SECONDS,
                       min_ticks: int = MIN_TICKS) -> pl.DataFrame:
    """Ticks (and a minute-bar fallback) within +/- ``window_seconds`` of every event.

    Events with at least ``min_ticks`` ticks keep them, with the bid-ask
    mid-quote and its log return. The others use the 1-minute bars of the
    same window (``is_tick`` false).
    """
    events = events.drop(
        [c for c in ("_loaded_at", "_source_file") if c in events.columns]
    )

    tick_prices = tick_prices.sort("ts_utc")
    minute_bars = minute_bars.sort("ts_utc")
    events = events.sort("ts_utc")

    tick_times = tick_prices["ts_utc"].to_numpy()
    bar_times = minute_bars["ts_utc"].to_numpy()
    event_times = events["ts_utc"].to_numpy()
    half = np.timedelta64(window_seconds, "s")

    tick_start = np.searchsorted(tick_times, event_times - half, side="left")
    tick_end = np.searchsorted(tick_times, event_times + half, side="right")
    bar_start = np.searchsorted(bar_times, event_times - half, side="left")
    bar_end = np.searchsorted(bar_times, event_times + half, side="right")

    kept, fallback = [], []
    for i in range(len(events)):
        if tick_end[i] - tick_start[i] >= min_ticks:
            kept.append(i)
        else:
            fallback.append(i)

    windows = [
        window_tick_returns(tick_prices[tick_start[i]:tick_end[i]])
        for i in kept
    ]
    tick_out = pl.concat(windows) if windows else tick_prices.clear()

    if kept:
        first_tick = tick_times[tick_start[kept]]
        stale = int(np.count_nonzero(first_tick >= event_times[kept]))
        print(
            f"  tick windows: {len(kept)} events, {len(fallback)} fallback events "
            f"(minute bars); {stale} tick windows have no tick before the release "
            f"(their first return is 0)"
        )

    bar_windows = [
        build_silver_minute_bars(minute_bars[bar_start[i]:bar_end[i]]).with_columns(pl.lit(False).alias("is_tick"))
        for i in fallback
        if bar_end[i] > bar_start[i]
    ]
    fallback_out = pl.concat(bar_windows) if bar_windows else minute_bars.clear()

    return pl.concat([tick_out, fallback_out], how="diagonal_relaxed")
"""Price and market-wide volatility of the asset per quarter over the test period
"""
from __future__ import annotations

import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from lakehouse.medallion.silver.time_conversion import prices_to_utc
from lakehouse.settings import get_settings
from lakehouse.paths import RAW_DATA_DIR, RESULTS_FIGURES_OUTPUT_DIR
from lakehouse.plot_style import (BLUE, INK, ORANGE, SURFACE, apply_default_style, clean_axes, save_figure)

FIRST_QUARTER, LAST_QUARTER = "2023Q1", "2026Q2"
WARM_UP_START = "2022-09-01"                       # rv_m needs 22 sessions before the first test quarter
MIN_RETURNS_PER_SESSION = 100
BAND = (0.10, 0.90)


def load_close() -> pd.Series:
    """1-minute closes in UTC from the raw bars (``[sources] prices_m1``; the last five files)."""
    files = sorted(glob.glob(str(RAW_DATA_DIR / get_settings().prices_m1_source)))
    bars = pl.concat([pl.read_parquet(file, columns=["datetime", "close"]) for file in files[-5:]])
    bars = prices_to_utc(bars, "datetime", "ts_utc")
    bars = bars.filter(pl.col("ts_utc") >= pl.lit(WARM_UP_START).str.to_datetime(time_zone="UTC"))
    return bars.to_pandas().set_index("ts_utc")["close"].sort_index().ffill()


def quarter_label(index: pd.DatetimeIndex) -> pd.Index:
    index = index.tz_convert(None) if index.tz is not None else index
    return pd.Index(index.year.astype(str) + "Q" + index.quarter.astype(str))


def market_by_quarter() -> pd.DataFrame:
    """Quarter x (price_median, price_p10, price_p90, rv_m_percent) for the test period."""
    close = load_close()
    by_quarter = close.groupby(quarter_label(close.index))
    table = pd.DataFrame({"price_median": by_quarter.median(), "price_p10": by_quarter.quantile(BAND[0]),
                          "price_p90": by_quarter.quantile(BAND[1])})

    five_minute = close.resample("5min").last().dropna()
    returns = np.log(five_minute).diff().dropna()
    session = returns.index.tz_convert("America/New_York").date
    daily = returns.pow(2).groupby(session).sum().pow(0.5)
    daily.index = pd.to_datetime(daily.index)
    counts = returns.groupby(session).size().reindex(daily.index.date).to_numpy()
    monthly = daily[counts >= MIN_RETURNS_PER_SESSION].rolling(22).mean() * 100
    table["rv_m_percent"] = monthly.groupby(quarter_label(monthly.index)).mean()
    return table.loc[FIRST_QUARTER:LAST_QUARTER]


def figure_price_and_volatility(table: pd.DataFrame, path, dpi: int) -> None:
    quarters = list(table.index)
    x = np.arange(len(quarters))
    labels = [q[2:4] + "\nQ" + q[-1] if q.endswith("Q1") else "Q" + q[-1] for q in quarters]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True)

    top.fill_between(x, table["price_p10"], table["price_p90"], color=BLUE, alpha=0.16, lw=0, label="10th-90th percentile")
    top.plot(x, table["price_median"], color=BLUE, lw=2.4, marker="o", markersize=5.5, markeredgecolor=SURFACE,
             markeredgewidth=1.2, label="Median")
    top.set_ylabel("Brent price ($/bbl)")
    top.set_title("Brent price by quarter")
    top.legend(loc="upper left", ncol=2, fontsize=9)
    clean_axes(top, grid="y")

    bottom.plot(x, table["rv_m_percent"], color=ORANGE, lw=2.4, marker="o", markersize=5.5, markeredgecolor=SURFACE,
                markeredgewidth=1.2)
    bottom.set_ylabel("rv_m, quarterly mean (%)")
    bottom.set_title("Monthly realised volatility of Brent (the long-run term in HAR-RV)")
    bottom.set_ylim(0, None)
    bottom.set_xticks(x, labels)
    clean_axes(bottom, grid="y")

    for axis, column, template in ((top, "price_median", "{:.0f}"), (bottom, "rv_m_percent", "{:.1f} %")):
        axis.annotate(template.format(table[column].iloc[-1]), (x[-1], table[column].iloc[-1]), xytext=(8, 0),
                      textcoords="offset points", va="center", fontsize=9.5, color=INK)
        axis.set_xlim(x[0] - 0.4, x[-1] + 0.9)

    figure.tight_layout()
    save_figure(figure, path, dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dpi", type=int, default=200)
    arguments = parser.parse_args()
    apply_default_style()
    table = market_by_quarter()
    directory = RESULTS_FIGURES_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(directory / "regime_quarterly.csv")
    print(table.round(2).to_string())
    figure_price_and_volatility(table, directory / "regime.png", arguments.dpi)


if __name__ == "__main__":
    main()

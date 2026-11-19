"""Diagnostics of the gold.events_features table: invalid targets (nothing ticked in the post-event window after
the release, or a window filled with 1-minute fallback bars; train.py drops these), valid zeros
(ticks but no mid-quote move; kept and scored against 0), where both fall in the train / validation
/ test periods, and how the valid and invalid targets differ.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lakehouse.feature_columns import TARGET_COLUMN
from lakehouse.modelling.gold_data import default_test_years, load_gold_events
from lakehouse.paths import EVALUATION_DIAGNOSTICS_OUTPUT_DIR

NEWS_ERA_START = "2019-09-01"  # start of the validation period
PERIODS = ["train", "validation", "test"]


def add_flags(events: pd.DataFrame, test_start: pd.Timestamp) -> pd.DataFrame:
    """Add validity flags, the valid-zero flag and the sample period to each event."""
    flagged = events.copy()
    if "fallback_bars" in flagged:
        flagged["fallback_bars"] = pd.to_numeric(flagged["fallback_bars"],
                                                 errors="coerce").fillna(0).astype(int)
    else:
        flagged["fallback_bars"] = 0
    flagged["n_post_ticks"] = pd.to_numeric(flagged["n_post_ticks"], errors="coerce")
    flagged["no_post_ticks"] = flagged["n_post_ticks"].eq(0)
    flagged["invalid_target"] = flagged["no_post_ticks"]
    flagged["valid_zero"] = ~flagged["invalid_target"] & flagged[TARGET_COLUMN].eq(0)
    flagged["period"] = np.select(
        [
            flagged["ts_utc"] < pd.Timestamp(NEWS_ERA_START, tz="UTC"),
            flagged["ts_utc"] < test_start,
        ],
        ["train", "validation"],
        default="test",
    )
    return flagged


def summarise(events: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Invalid and valid-zero counts per group, with their shares."""
    table = (
        events.groupby(by, sort=False)
        .agg(
            events=(TARGET_COLUMN, "size"),
            fallback=("fallback_bars", "sum"),
            no_post_ticks=("no_post_ticks", "sum"),
            invalid=("invalid_target", "sum"),
            valid_zero=("valid_zero", "sum"),
        )
        .reset_index()
    )
    table["valid"] = table["events"] - table["invalid"]
    table["invalid_share"] = (table["invalid"] / table["events"]).round(3)
    table["valid_zero_share"] = (
        table["valid_zero"] / table["valid"].where(table["valid"] > 0)
    ).round(3)
    return table


def percent(count, total) -> str:
    return f"{100 * count / total:.1f}%" if total else "n/a"


def _quantile_function(values: np.ndarray):
    def quantile(p: float) -> float:
        return float(np.nanquantile(values, p)) if len(values) else float("nan")

    return quantile


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot-id", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=EVALUATION_DIAGNOSTICS_OUTPUT_DIR)
    args = parser.parse_args()

    events, snapshot_id = load_gold_events(args.snapshot_id, drop_invalid_target=False)
    test_start = pd.Timestamp(year=default_test_years(events)[0], month=1, day=1, tz="UTC")
    events = add_flags(events, test_start)
    valid = events[~events["invalid_target"]]
    invalid = events[events["invalid_target"]]
    total = len(events)
    valid_zero_count = int(events["valid_zero"].sum())

    lines: list[str] = []

    def say(text: str = "") -> None:
        lines.append(text)
        print(text)

    say(f"gold.events_features snapshot {snapshot_id}: {total:,} events")
    say(f"periods: train < {NEWS_ERA_START} <= validation < {test_start:%Y-%m-%d} <= test")
    say("")
    say("1. TARGET VALIDITY")
    say(f"   fallback (1-minute bars) window : {int(events['fallback_bars'].sum()):>5}  "
        f"({percent(events['fallback_bars'].sum(), total)})")
    say(f"   no ticks in the 30s post window : {int(events['no_post_ticks'].sum()):>5}  "
        f"({percent(events['no_post_ticks'].sum(), total)})")
    say(f"   INVALID (either)                : {len(invalid):>5}  "
        f"({percent(len(invalid), total)})  -> dropped by train.py")
    say(f"   valid                           : {len(valid):>5}")
    say("")
    say(f"2. VALID EVENTS WITH ZERO VOLATILITY  (ticks recorded, mid-quote unchanged for {TARGET_COLUMN})")
    say(f"   valid zero-volatility events    : {valid_zero_count:>5}  "
        f"({percent(valid_zero_count, len(valid))} of valid events)  -> kept, scored against 0")
    say("")
    say("3. WHERE THEY FALL: BY SAMPLE PERIOD")
    by_period = summarise(events, ["period"])
    by_period["period"] = pd.Categorical(by_period["period"], PERIODS, ordered=True)
    by_period = by_period.sort_values("period")
    say(by_period.to_string(index=False))
    say("")
    by_period_and_type = summarise(events, ["period", "event_type"])
    by_period_and_type["period"] = pd.Categorical(by_period_and_type["period"], PERIODS,
                                                  ordered=True)
    say(by_period_and_type.sort_values(["period", "event_type"]).to_string(index=False))
    indexed = by_period.set_index("period")
    if {"train", "test"} <= set(indexed.index):
        say("")
        say(f"   invalid share    train {indexed.loc['train', 'invalid_share']:.1%}  "
            f"vs  test {indexed.loc['test', 'invalid_share']:.1%}")
        say(f"   valid-zero share train {indexed.loc['train', 'valid_zero_share']:.1%}  "
            f"vs  test {indexed.loc['test', 'valid_zero_share']:.1%}")
    by_year = summarise(events, ["year", "event_type"]).sort_values(["year", "event_type"])
    say("")
    say(by_year.to_string(index=False))
    say("")
    say("4. TARGET SCALE, VALID vs INVALID")
    for label, group in (("valid", valid), ("invalid", invalid)):
        target = group[TARGET_COLUMN]
        q = _quantile_function(target)
        say(f"   {label:<8} n={len(group):>5}  zeros={percent((target == 0).sum(), len(target)):>6}  "
            f"median={q(.5):.4f}  mean={target.mean():.4f}  "
            f"p95={q(.95):.4f}  max={target.max():.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    by_period.to_csv(args.out_dir / "validity_by_period.csv", index=False)
    by_year.to_csv(args.out_dir / "validity_by_year.csv", index=False)
    listing_columns = ["ts_utc", "period", "event_type", "fallback_bars", "n_post_ticks",
                       TARGET_COLUMN]
    invalid[listing_columns].to_csv(args.out_dir / "flagged_events.csv", index=False)
    events.loc[events["valid_zero"], listing_columns].to_csv(
        args.out_dir / "valid_zero_events.csv", index=False
    )
    print(f"\nWrote summary.txt and 4 csv files to {args.out_dir}")


if __name__ == "__main__":
    main()
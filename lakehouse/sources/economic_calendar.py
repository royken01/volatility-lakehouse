"""Scheduled-release calendars from free sources, written as the event CSV the pipeline reads.

Output (``[sources] events``, or ``--out``): one row per release, merged into the file and
de-duplicated on (release_datetime, event)::

    release_datetime, event, importance, actual, forecast, previous, surprise

``release_datetime`` is written in the calendar clock of the settings (the open-ended last rule of
``[events] clock``, UTC by default), so silver converts it back to UTC with the same rules as the
rest of the file.  ``event`` must match a ``name`` under ``[events] types`` (``--event`` renames rows).

Providers
---------
fred         Official release DATES of any FRED release (CPI, Employment Situation, GDP ... back to
             the 1990s and the announced future dates), free API key (``FRED_API_KEY``).  FRED has
             dates but no times, so ``--time`` / ``--timezone`` give the clock time of the release
             (most US data: 08:30 America/New_York).  With ``--series`` the ``actual`` of each
             release is the INITIAL release value from ALFRED (vintage data, what the market saw that
             day), optionally ``--transform diff|pct`` against the previous period's initial value.
             No consensus forecast is free, so ``forecast`` and ``surprise`` stay empty.
forexfactory The public weekly JSON feed behind ForexFactory's calendar (Fair Economy): every
             country's releases this week (``--week next`` for next week) with impact, consensus
             forecast and previous, no key.  No history and no actuals: use it to schedule
             forecasts of UPCOMING releases.  Unofficial feed; respect its fair-use terms.

    vol-lakehouse calendar fred --search "consumer price"            # find a release id
    vol-lakehouse calendar fred --release-id 10 --event "CPI m/m" --series CPIAUCSL --transform pct \\
        --time 08:30 --timezone America/New_York --start 2015-01-01
    vol-lakehouse calendar fred --release-id 50 --event "Nonfarm Payrolls" --series PAYEMS --transform diff \\
        --time 08:30 --timezone America/New_York --include-future
    vol-lakehouse calendar forexfactory --country USD --impact High --week next
    vol-lakehouse calendar forexfactory --country USD --title "Crude Oil Inventories" --dry-run

Any other source (a paid vendor export, an Investing.com download) works too: save it in the
column layout above.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
from pathlib import Path
from zoneinfo import ZoneInfo

from lakehouse.paths import ENVIRONMENT_FILE, RAW_DATA_DIR
from lakehouse.settings import Settings, get_settings

FIELDS = ["release_datetime", "event", "importance", "actual", "forecast", "previous", "surprise"]
STAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
FRED_BASE = "https://api.stlouisfed.org/fred"
FOREXFACTORY_FEEDS = {"this": "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                      "next": "https://nfs.faireconomy.media/ff_calendar_nextweek.json"}
SUFFIX_SCALE = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def require_requests():
    try:
        import requests
    except ImportError as error:
        raise SystemExit("the calendar fetchers need `requests` (pip install requests)") from error
    return requests


def calendar_timezone(settings: Settings) -> str:
    """The clock the event file is written in: the open-ended (current) rule of ``[events] clock``."""
    rule = settings.event_clock[-1]
    if rule.offset_hours:
        raise SystemExit("the event clock has an hour offset; pass --out-timezone explicitly")
    return rule.timezone


def default_output(settings: Settings) -> Path:
    return RAW_DATA_DIR / re.sub(r"[*?\[\]]", "", settings.events_source).replace("/.csv", "/calendar.csv")


def parse_number(text) -> float | None:
    """'0.4%' -> 0.4, '-0.90T' -> -0.9e12, '250K' -> 250000, '' -> None."""
    if text is None:
        return None
    text = str(text).strip().replace(",", "")
    if not text:
        return None
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*([KMBT%]?)", text, re.I)
    if not match:
        return None
    value, suffix = float(match.group(1)), match.group(2).upper()
    return value * SUFFIX_SCALE.get(suffix, 1.0)


def to_local_stamp(moment: dt.datetime, timezone: str) -> str:
    return moment.astimezone(ZoneInfo(timezone)).strftime(STAMP_FORMAT)


def make_row(moment: dt.datetime, event: str, timezone: str, importance: str = "", actual=None, forecast=None,
             previous=None) -> dict:
    surprise = actual - forecast if actual is not None and forecast is not None else None
    return {"release_datetime": to_local_stamp(moment, timezone), "event": event, "importance": importance,
            "actual": actual, "forecast": forecast, "previous": previous, "surprise": surprise}


# --- FRED / ALFRED ---------------------------------------------------------------------------------
def fred_key() -> str:
    from dotenv import load_dotenv

    load_dotenv(ENVIRONMENT_FILE)
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise SystemExit("set FRED_API_KEY in .env (free key: https://fred.stlouisfed.org/docs/api/api_key.html)")
    return key


def fred_get(endpoint: str, **parameters) -> dict:
    requests = require_requests()
    response = requests.get(f"{FRED_BASE}/{endpoint}", timeout=60,
                            params={"api_key": fred_key(), "file_type": "json", **parameters})
    response.raise_for_status()
    return response.json()


def fred_search_releases(text: str) -> list[dict]:
    releases = fred_get("releases", limit=1000).get("releases", [])
    words = text.lower().split()
    return [r for r in releases if all(w in r.get("name", "").lower() for w in words)]


def fred_release_dates(release_id: int, include_future: bool) -> list[dt.date]:
    payload = fred_get("release/dates", release_id=release_id, sort_order="asc", limit=10000,
                       include_release_dates_with_no_data=str(include_future).lower())
    return [dt.date.fromisoformat(item["date"]) for item in payload.get("release_dates", [])]


def initial_release_values(observations: list[dict], transform: str = "level") -> dict[dt.date, tuple]:
    """ALFRED 'initial release only' observations -> {first-release date: (actual, previous)}.

    ``previous`` is the previous period's initial value; ``diff`` / ``pct`` transform both against
    the period before them (also initial values).
    """
    points = []
    for item in observations:
        try:
            points.append((dt.date.fromisoformat(item["date"]), dt.date.fromisoformat(item["realtime_start"]),
                           float(item["value"])))
        except (KeyError, ValueError):   # "." marks a missing value
            continue
    points.sort()
    level = [value for _, _, value in points]

    def transformed(index: int) -> float | None:
        if transform == "level":
            return level[index]
        if index == 0:
            return None
        if transform == "diff":
            return level[index] - level[index - 1]
        return 100.0 * (level[index] / level[index - 1] - 1.0) if level[index - 1] else None

    values: dict[dt.date, tuple] = {}
    for index, (_, released, _) in enumerate(points):
        actual = transformed(index)
        previous = transformed(index - 1) if index >= 1 else None
        values[released] = (None if actual is None else round(actual, 6),
                            None if previous is None else round(previous, 6))
    return values


def fetch_fred(release_id: int, event: str, clock_time: str, timezone: str, out_timezone: str,
               start: dt.date | None, end: dt.date | None, series: str | None, transform: str,
               include_future: bool) -> list[dict]:
    hour, minute = (int(part) for part in clock_time.split(":"))
    values = {}
    if series:
        observations = fred_get("series/observations", series_id=series, output_type=4,
                                realtime_start="1776-07-04", realtime_end="9999-12-31").get("observations", [])
        values = initial_release_values(observations, transform)
    rows = []
    for day in fred_release_dates(release_id, include_future):
        if (start and day < start) or (end and day > end):
            continue
        moment = dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=ZoneInfo(timezone))
        actual, previous = values.get(day, (None, None))
        rows.append(make_row(moment, event, out_timezone, "High", actual=actual, previous=previous))
    if series:
        matched = sum(row["actual"] is not None for row in rows)
        print(f"  {matched} of {len(rows)} release dates matched an initial {series} value")
    return rows


# --- ForexFactory weekly feed -----------------------------------------------------------------------
def forexfactory_rows(items: list[dict], out_timezone: str, countries=(), impacts=(), title: str | None = None,
                      rename: str | None = None) -> list[dict]:
    """Feed items (title, country, date, impact, forecast, previous) -> calendar rows."""
    pattern = re.compile(title, re.I) if title else None
    rows = []
    for item in items:
        if countries and item.get("country", "").upper() not in {c.upper() for c in countries}:
            continue
        if impacts and item.get("impact", "").lower() not in {i.lower() for i in impacts}:
            continue
        if pattern and not pattern.search(item.get("title", "")):
            continue
        try:
            moment = dt.datetime.fromisoformat(item["date"])
        except (KeyError, ValueError):
            continue
        if moment.tzinfo is None or moment.time() == dt.time(0, 0) and item.get("impact") == "Holiday":
            continue   # all-day items have no release time
        rows.append(make_row(moment, rename or item.get("title", ""), out_timezone, item.get("impact", ""),
                             actual=parse_number(item.get("actual")), forecast=parse_number(item.get("forecast")),
                             previous=parse_number(item.get("previous"))))
    return rows


def fetch_forexfactory(week: str, out_timezone: str, countries, impacts, title, rename) -> list[dict]:
    requests = require_requests()
    response = requests.get(FOREXFACTORY_FEEDS[week], timeout=60,
                            headers={"User-Agent": "volatility-lakehouse/1.0 (research)"})
    response.raise_for_status()
    return forexfactory_rows(response.json(), out_timezone, countries, impacts, title, rename)


# --- writing -----------------------------------------------------------------------------------------
def merge_into_csv(rows: list[dict], path: Path) -> tuple[int, int]:
    """Merge rows into the event CSV; a new row replaces an old one with the same (time, event), so a
    later fetch fills in actuals of releases that were scheduled earlier.  Returns (new, updated)."""
    existing: dict[tuple, dict] = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing[(row.get("release_datetime"), row.get("event"))] = row
    added = updated = 0
    for row in rows:
        key = (row["release_datetime"], row["event"])
        if key in existing:
            old = existing[key]
            merged = {field: row[field] if row.get(field) not in (None, "") else old.get(field, "") for field in FIELDS}
            actual, forecast = parse_number(merged["actual"]), parse_number(merged["forecast"])
            if actual is not None and forecast is not None:
                merged["surprise"] = round(actual - forecast, 10)
            updated += merged != {field: old.get(field, "") for field in FIELDS}
            existing[key] = merged
        else:
            existing[key] = row
            added += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(existing):
            writer.writerow({field: "" if existing[key].get(field) is None else existing[key].get(field)
                             for field in FIELDS})
    return added, updated


def main(arguments: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="vol-lakehouse calendar", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    providers = parser.add_subparsers(dest="provider", required=True)

    fred = providers.add_parser("fred", help="official release dates (+ initial actuals) from FRED / ALFRED")
    fred.add_argument("--search", help="list FRED releases whose name contains these words, then stop")
    fred.add_argument("--release-id", type=int)
    fred.add_argument("--event", help="event name written to the file (match [events] types)")
    fred.add_argument("--time", default="08:30", help="release clock time, HH:MM (FRED has dates only)")
    fred.add_argument("--timezone", default="America/New_York", help="time zone of --time")
    fred.add_argument("--series", help="FRED series whose initial release becomes `actual` (e.g. PAYEMS)")
    fred.add_argument("--transform", choices=["level", "diff", "pct"], default="level")
    fred.add_argument("--include-future", action="store_true", help="also the announced future dates")
    fred.add_argument("--start", type=dt.date.fromisoformat)
    fred.add_argument("--end", type=dt.date.fromisoformat)

    forexfactory = providers.add_parser("forexfactory", help="this / next week from the ForexFactory feed")
    forexfactory.add_argument("--week", choices=["this", "next"], default="this")
    forexfactory.add_argument("--country", action="append", default=[], help="currency code, e.g. USD (repeatable)")
    forexfactory.add_argument("--impact", action="append", default=[], help="High / Medium / Low (repeatable)")
    forexfactory.add_argument("--title", help="regular expression on the event title")
    forexfactory.add_argument("--event", help="rename the matched events (match [events] types)")

    for provider in (fred, forexfactory):
        provider.add_argument("--out", type=Path, default=default_output(settings))
        provider.add_argument("--out-timezone", default=None,
                              help="clock of release_datetime in the file (default: the [events] clock)")
        provider.add_argument("--dry-run", action="store_true", help="print the rows instead of writing them")
    options = parser.parse_args(arguments)
    out_timezone = options.out_timezone or calendar_timezone(settings)

    if options.provider == "fred":
        if options.search:
            for release in fred_search_releases(options.search):
                print(f"{release['id']:>6}  {release['name']}")
            return 0
        if options.release_id is None or not options.event:
            parser.error("fred needs --release-id and --event (use --search to find the id)")
        rows = fetch_fred(options.release_id, options.event, options.time, options.timezone, out_timezone,
                          options.start, options.end, options.series, options.transform, options.include_future)
    else:
        rows = fetch_forexfactory(options.week, out_timezone, options.country, options.impact, options.title,
                                  options.event)

    print(f"{len(rows)} release(s); release_datetime in {out_timezone}")
    if options.dry_run:
        for row in rows[:50]:
            print("  " + ", ".join(f"{field}={row[field]}" for field in FIELDS if row[field] not in (None, "")))
        return 0
    added, updated = merge_into_csv(rows, options.out)
    print(f"{options.out}: {added} new, {updated} updated")
    if not settings.event_types:
        print("note: no [events] types configured yet; add the event names you want to model to the config")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

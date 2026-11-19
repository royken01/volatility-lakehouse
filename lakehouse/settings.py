"""Framework settings: which asset, which scheduled releases, which windows, which news.

The framework forecasts the realized volatility of ANY asset in a window of ANY length after
scheduled releases (macro data, inventory reports, central-bank decisions etc.), given three inputs:
an event calendar, prices (ticks plus 1-minute bars) and news headlines. Everything that was
specific to one study (asset, release names, the 30-second window, time-zone quirks of a data
vendor) lives here instead of in the code.

Settings are resolved once per process, lowest priority first:

    1. built-in defaults (generic: UTC clocks, 60 s post-event window, 10 lags, one pooled event type)
    2. a TOML file: ``$LAKEHOUSE_CONFIG``, otherwise ``<workspace>/framework.toml`` when it exists
       (``configs/`` holds ready-made examples, e.g. ``crude_oil_thesis.toml``)
    3. environment variables ``LAKEHOUSE_*`` (the ``vol-lakehouse`` command line sets these from its
       ``--asset``, ``--post-window-seconds`` ... flags, so every pipeline stage sees the same values)

Standard library only: this module is shipped with every logged MLflow model.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from lakehouse.paths import WORKSPACE_DIR

CONFIG_ENVIRONMENT_VARIABLE = "LAKEHOUSE_CONFIG"
DEFAULT_CONFIG_FILE = WORKSPACE_DIR / "framework.toml"

# Environment variable -> (settings attribute, parser).  The CLI flag is the attribute with dashes.
ENVIRONMENT_OVERRIDES = {
    "LAKEHOUSE_ASSET": ("asset", str),
    "LAKEHOUSE_POST_WINDOW_SECONDS": ("post_window_seconds", int),
    "LAKEHOUSE_LAGS": ("lags", int),
    "LAKEHOUSE_TICK_WINDOW_SECONDS": ("tick_window_seconds", int),
    "LAKEHOUSE_PRE_WINDOW_MINUTES": ("pre_window_minutes", int),
    "LAKEHOUSE_SENTIMENT_MODEL": ("sentiment_model", str),
    "LAKEHOUSE_NEWS_KEYWORDS": ("news_keywords", lambda text: [k.strip() for k in text.split(",") if k.strip()]),
    "LAKEHOUSE_EVENT_TYPES": ("event_types", lambda text: parse_event_type_flags(text.split("|"))),
}

HISTDATA_TIMESTAMP_FORMATS = ["%Y%m%d %H%M%S%3f", "%Y%m%d %H%M%S"]
ISO_TIMESTAMP_FORMATS = ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]


@dataclass(frozen=True)
class ClockRule:
    """The local clock of a source until ``until`` (exclusive; ``None`` = open-ended).

    ``offset_hours`` is added before the time zone is applied (a vendor whose clock runs a fixed
    number of hours behind a named zone).
    """
    timezone: str = "UTC"
    until: date | None = None
    offset_hours: float = 0.0


@dataclass(frozen=True)
class EventType:
    """One kind of scheduled release.

    ``name`` is the value in the calendar's ``event`` column, ``label`` the short name used in
    tables, metrics and plots, ``dummy`` the 0/1 column in gold.  The ``reference`` type gets no
    dummy in the models (the other dummies measure the level difference to it).
    """
    name: str
    label: str
    dummy: str
    reference: bool = False


@dataclass(frozen=True)
class TimeCorrection:
    """Shift the stamps of ``event`` that fall on ``minute`` by ``shift_minutes`` (calendar errors)."""
    event: str
    minute: int
    shift_minutes: int


@dataclass(frozen=True)
class Settings:
    asset: str = "asset"
    # windows
    post_window_seconds: int = 60
    lags: int = 10
    tick_window_seconds: int = 600
    pre_window_minutes: int = 60
    min_ticks: int = 10
    # raw source files, relative to <workspace>/data/raw (glob patterns allowed)
    events_source: str = "events/*.csv"
    headlines_source: str = "contextual/gdelt_headlines*.csv"
    prices_tick_source: str = "prices_tick/*.parquet"
    prices_m1_source: str = "prices_m1/*.parquet"
    # clocks
    price_timestamp_formats: tuple[str, ...] = tuple(HISTDATA_TIMESTAMP_FORMATS + ISO_TIMESTAMP_FORMATS)
    price_clock: tuple[ClockRule, ...] = (ClockRule(),)
    event_timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    event_clock: tuple[ClockRule, ...] = (ClockRule(),)
    # releases
    event_types: tuple[EventType, ...] = ()
    time_corrections: tuple[TimeCorrection, ...] = ()
    copied_events: tuple[tuple[str, str], ...] = ()   # (event, event it may duplicate)
    duplicate_window_minutes: int = 10
    # trading sessions for the HAR-RV terms
    session_timezone: str = "America/New_York"
    session_open_hour: int = 17
    rv_sampling_minutes: int = 5
    min_session_coverage: float = 0.5
    har_week_sessions: int = 5
    har_month_sessions: int = 22
    # news
    news_enabled: bool = True
    news_keywords: tuple[str, ...] = ()
    news_ambiguous: tuple[tuple[str, tuple[str, ...]], ...] = ()
    news_mask: tuple[str, ...] = ()
    news_boost_patterns: tuple[str, ...] = ()
    news_penalty_patterns: tuple[str, ...] = ()
    news_themes: tuple[str, ...] = ()
    news_domains: tuple[str, ...] = ()
    news_min_score: int = 3
    news_english_only: bool = True
    sentiment_model: str = "ProsusAI/finbert"
    scored_headlines_table: str = "gdelt_headlines_scored"
    news_half_life_hours: float = 72.0
    news_lookback_trading_days: int = 22
    news_count_window_hours: int = 24
    source_file: str | None = None
    extras: dict = field(default_factory=dict, compare=False, hash=False)

    # --- derived names -------------------------------------------------------------------------
    @property
    def window_tag(self) -> str:
        return f"{self.post_window_seconds}s"

    @property
    def target_column(self) -> str:
        return f"realized_vol_{self.window_tag}"

    @property
    def return_sum_column(self) -> str:
        return f"ret_sum_{self.window_tag}"

    @property
    def lag_prefix(self) -> str:
        return f"pre_vol_{self.window_tag}_lag"

    @property
    def resolved_event_types(self) -> tuple[EventType, ...]:
        """The configured types, or one pooled type when none are configured."""
        if not self.event_types:
            return (EventType(name="*", label="all", dummy="is_event", reference=True),)
        if any(t.reference for t in self.event_types):
            return self.event_types
        first, *rest = self.event_types
        return (EventType(first.name, first.label, first.dummy, True), *rest)

    @property
    def model_event_dummies(self) -> list[str]:
        """Dummies used as model features: every non-reference type, or the single type's own dummy
        (a constant column, so the naive-mean benchmark still has an intercept-only design)."""
        types = self.resolved_event_types
        if len(types) == 1:
            return [types[0].dummy]
        return [t.dummy for t in types if not t.reference]

    @property
    def all_event_dummies(self) -> list[str]:
        return [t.dummy for t in self.resolved_event_types]

    def event_type_of(self, event_name: str) -> EventType | None:
        types = self.resolved_event_types
        if len(types) == 1 and types[0].name == "*":
            return types[0]
        return next((t for t in types if t.name == event_name), None)

    def describe(self) -> dict:
        """Flat summary for logs and MLflow parameters."""
        return {"asset": self.asset, "post_window_seconds": self.post_window_seconds, "lags": self.lags,
                "tick_window_seconds": self.tick_window_seconds, "pre_window_minutes": self.pre_window_minutes,
                "event_types": ",".join(f"{t.label}{'*' if t.reference else ''}" for t in self.resolved_event_types),
                "sentiment_model": self.sentiment_model, "config_file": self.source_file or "defaults"}

    def validate(self) -> "Settings":
        if self.post_window_seconds <= 0:
            raise ValueError("post_window_seconds must be positive")
        if self.tick_window_seconds < self.post_window_seconds:
            raise ValueError(f"tick_window_seconds ({self.tick_window_seconds}) must cover the post-event window "
                             f"({self.post_window_seconds} s): silver only keeps ticks within that distance")
        if self.lags < 1:
            raise ValueError("lags must be at least 1")
        labels = [t.label for t in self.event_types]
        if len(set(labels)) != len(labels):
            raise ValueError(f"event type labels must be unique: {labels}")
        if sum(t.reference for t in self.event_types) > 1:
            raise ValueError("at most one event type can be the reference")
        return self


# --- parsing ----------------------------------------------------------------------------------
def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "event"


def make_event_type(name: str, label: str | None = None, dummy: str | None = None,
                    reference: bool = False) -> EventType:
    label = label or slug(name).upper()
    return EventType(name=name, label=label, dummy=dummy or f"is_{slug(label)}", reference=bool(reference))


def parse_event_type_flags(values) -> tuple[EventType, ...]:
    """``NAME=LABEL`` items from the command line; a trailing ``*`` marks the reference type."""
    types = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        name, _, label = value.partition("=")
        reference = label.endswith("*") or (not label and name.endswith("*"))
        types.append(make_event_type(name.rstrip("*").strip(), label.rstrip("*").strip() or None,
                                     reference=reference))
    return tuple(types)


def _clock(rules) -> tuple[ClockRule, ...]:
    if isinstance(rules, str):
        return (ClockRule(timezone=rules),)
    parsed = []
    for rule in rules or []:
        until = rule.get("until")
        if isinstance(until, str):
            until = date.fromisoformat(until)
        parsed.append(ClockRule(timezone=rule.get("timezone", "UTC"), until=until,
                                offset_hours=float(rule.get("offset_hours", 0.0))))
    if not parsed:
        return (ClockRule(),)
    if parsed[-1].until is not None:
        raise ValueError("the last clock rule must be open-ended (no 'until')")
    return tuple(parsed)


def settings_from_mapping(raw: dict, source_file: str | None = None) -> Settings:
    """Build settings from a parsed TOML document (missing keys keep their defaults)."""
    defaults = Settings()
    windows = raw.get("windows", {})
    sources = raw.get("sources", {})
    prices = raw.get("prices", {})
    events = raw.get("events", {})
    market = raw.get("market", {})
    news = raw.get("news", {})
    gdelt = news.get("gdelt", {})
    known = {"asset", "windows", "sources", "prices", "events", "market", "news"}
    return Settings(
        asset=raw.get("asset", defaults.asset),
        post_window_seconds=int(windows.get("post_event_seconds", defaults.post_window_seconds)),
        lags=int(windows.get("lags", defaults.lags)),
        tick_window_seconds=int(windows.get("tick_window_seconds", defaults.tick_window_seconds)),
        pre_window_minutes=int(windows.get("pre_event_minutes", defaults.pre_window_minutes)),
        min_ticks=int(windows.get("min_ticks", defaults.min_ticks)),
        events_source=sources.get("events", defaults.events_source),
        headlines_source=sources.get("headlines", defaults.headlines_source),
        prices_tick_source=sources.get("prices_tick", defaults.prices_tick_source),
        prices_m1_source=sources.get("prices_m1", defaults.prices_m1_source),
        price_timestamp_formats=tuple(prices.get("timestamp_formats", defaults.price_timestamp_formats)),
        price_clock=_clock(prices.get("clock", prices.get("timezone"))),
        event_timestamp_format=events.get("timestamp_format", defaults.event_timestamp_format),
        event_clock=_clock(events.get("clock", events.get("timezone"))),
        event_types=tuple(make_event_type(t["name"], t.get("label"), t.get("dummy"), t.get("reference", False))
                          for t in events.get("types", [])),
        time_corrections=tuple(TimeCorrection(c["event"], int(c["minute"]), int(c["shift_minutes"]))
                               for c in events.get("time_corrections", [])),
        copied_events=tuple((event, source) for event, source in events.get("copies", {}).items()),
        duplicate_window_minutes=int(events.get("duplicate_window_minutes", defaults.duplicate_window_minutes)),
        session_timezone=market.get("session_timezone", defaults.session_timezone),
        session_open_hour=int(market.get("session_open_hour", defaults.session_open_hour)),
        rv_sampling_minutes=int(market.get("rv_sampling_minutes", defaults.rv_sampling_minutes)),
        min_session_coverage=float(market.get("min_session_coverage", defaults.min_session_coverage)),
        har_week_sessions=int(market.get("har_week_sessions", defaults.har_week_sessions)),
        har_month_sessions=int(market.get("har_month_sessions", defaults.har_month_sessions)),
        news_enabled=bool(news.get("enabled", defaults.news_enabled)),
        news_keywords=tuple(news.get("keywords", ())),
        news_ambiguous=tuple((word, tuple(context)) for word, context in news.get("ambiguous", {}).items()),
        news_mask=tuple(news.get("mask", ())),
        news_boost_patterns=tuple(gdelt.get("boost_patterns", ())),
        news_penalty_patterns=tuple(gdelt.get("penalty_patterns", ())),
        news_themes=tuple(gdelt.get("themes", ())),
        news_domains=tuple(gdelt.get("domains", ())),
        news_min_score=int(gdelt.get("min_score", defaults.news_min_score)),
        news_english_only=bool(gdelt.get("english_only", defaults.news_english_only)),
        sentiment_model=news.get("sentiment_model", defaults.sentiment_model),
        scored_headlines_table=news.get("scored_table", defaults.scored_headlines_table),
        news_half_life_hours=float(news.get("half_life_hours", defaults.news_half_life_hours)),
        news_lookback_trading_days=int(news.get("lookback_trading_days", defaults.news_lookback_trading_days)),
        news_count_window_hours=int(news.get("count_window_hours", defaults.news_count_window_hours)),
        source_file=source_file,
        extras={key: value for key, value in raw.items() if key not in known},
    )


def config_path() -> Path | None:
    configured = os.environ.get(CONFIG_ENVIRONMENT_VARIABLE)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{CONFIG_ENVIRONMENT_VARIABLE}={configured} does not exist")
        return path.resolve()
    return DEFAULT_CONFIG_FILE if DEFAULT_CONFIG_FILE.is_file() else None


def load_settings(path: Path | None = None, environment: dict | None = None) -> Settings:
    """Settings from ``path`` (or the configured file) plus the ``LAKEHOUSE_*`` overrides."""
    environment = os.environ if environment is None else environment
    path = path or config_path()
    raw = {}
    if path is not None:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    settings = settings_from_mapping(raw, str(path) if path else None)
    overrides = {}
    for variable, (attribute, parse) in ENVIRONMENT_OVERRIDES.items():
        if environment.get(variable):
            value = parse(environment[variable])
            overrides[attribute] = tuple(value) if isinstance(value, list) else value
    if overrides:
        settings = Settings(**{**settings.__dict__, **overrides})
    return settings.validate()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The settings of this process (read once)."""
    return load_settings()

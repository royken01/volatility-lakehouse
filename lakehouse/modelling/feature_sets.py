"""The named feature sets; some are specific to certain models (benchmarks, sequence models).

    event_type   naive_mean benchmark (event-type dummies only)
    har_rv       HAR-RV benchmark (har_ols): HAR terms and event dummies
    base         default: HAR terms, event dummies, consensus forecast, last-hour volatility
    base_seq     base + the lagged same-type volatilities
    seq          the lagged volatilities only
    base_news    base + news features (arima_order_identification.py only)

The event dummies are those of the non-reference event types in the settings (``[events] types``).
"""
from __future__ import annotations

from lakehouse.feature_columns import HAR_COLUMNS, NEWS_SENTIMENT_COLUMNS, SEQUENCE_COLUMNS
from lakehouse.settings import get_settings

EVENT_DUMMIES = get_settings().model_event_dummies
LAST_HOUR = ["pre_vol_h"]

_BASE_FEATURES = HAR_COLUMNS + EVENT_DUMMIES + ["forecast"] + LAST_HOUR

FEATURE_SETS: dict[str, list[str]] = {
    "event_type": list(EVENT_DUMMIES),
    "har_rv": list(HAR_COLUMNS+EVENT_DUMMIES),
    "base": _BASE_FEATURES,
    "base_seq": _BASE_FEATURES + SEQUENCE_COLUMNS,
    "seq": list(SEQUENCE_COLUMNS),
    "base_news": _BASE_FEATURES + NEWS_SENTIMENT_COLUMNS,
}
DEFAULT_FEATURE_SET = "base"

BENCHMARK_FEATURE_SETS = {"naive_mean": "event_type", "har_ols": "har_rv"}

NO_IMPUTE_COLUMNS = NEWS_SENTIMENT_COLUMNS


def model_feature_columns(model_name: str, feature_set: str = DEFAULT_FEATURE_SET) -> list[str]:
    """Tier A columns of a model, chosen as ``train.py`` does: benchmarks use their fixed set and the
    sequence models (LSTM, ARIMAX) also get the lagged volatilities."""
    from lakehouse.modelling.model_params import SEQUENCE_MODELS

    feature_set = BENCHMARK_FEATURE_SETS.get(model_name, feature_set)
    sequence_columns = list(SEQUENCE_COLUMNS) if model_name in SEQUENCE_MODELS else []
    return resolve_feature_set(feature_set, sequence_columns)[1]


def resolve_feature_set(name: str, extra_columns: list[str] | None = None) -> tuple[str, list[str]]:
    """Return ``(label, ordered unique column list)`` for a named set plus extra columns."""
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {name!r}; choose from {list(FEATURE_SETS)}")
    columns = list(dict.fromkeys(FEATURE_SETS[name] + list(extra_columns or [])))
    added = [column for column in (extra_columns or []) if column not in FEATURE_SETS[name]]
    if all(column in added for column in SEQUENCE_COLUMNS):
        added = [column for column in added if column not in SEQUENCE_COLUMNS] + [f"seq{len(SEQUENCE_COLUMNS)}"]
    label = name if not added else name + "+" + "+".join(added)
    return label, columns

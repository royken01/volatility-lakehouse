"""Names of the model-input columns of the gold.events_features table.

The window-dependent names follow the settings (``settings.py``): with a 30-second post-event
window the target is ``realized_vol_30s`` and the lags ``pre_vol_30s_lag1..N``; with 300 s they
are ``realized_vol_300s`` and ``pre_vol_300s_lag1..N``.
"""
from lakehouse.settings import get_settings

_settings = get_settings()

SEQUENCE_LENGTH = _settings.lags
SEQUENCE_PREFIX = _settings.lag_prefix
RETURN_SEQUENCE_PREFIX = f"pre_ret_{_settings.window_tag}_lag"
SEQUENCE_COLUMNS = [f"{SEQUENCE_PREFIX}{lag}" for lag in range(1, SEQUENCE_LENGTH + 1)]

NEWS_SENTIMENT_COLUMNS = ["news_sent_ewma", "news_sent_abs_ewma", "news_count_24h"]

HAR_COLUMNS = ["rv_d", "rv_w", "rv_m"]

TARGET_COLUMN = _settings.target_column
RETURN_SUM_COLUMN = _settings.return_sum_column
# the post-event information that may never be used as a model input
LEAKAGE_COLUMNS = [RETURN_SUM_COLUMN, TARGET_COLUMN, "n_post_ticks", "fallback_bars"]

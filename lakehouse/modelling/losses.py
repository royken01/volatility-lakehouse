"""Target transform, bias correction and loss of the thesis
"""
from __future__ import annotations

import numpy as np

FLOOR_FRACTION = 0.5
TARGET_SCALES = ("log", "raw")


def floor_value(training_target) -> float:
    """We set the floor value to be half of the smallest positive value of the training targe to prevent 
    breaking normal distribution by setting value too low which is why we perform the transformation in the first place"""
    values = np.asarray(training_target, dtype=float)
    positive_values = values[np.isfinite(values) & (values > 0)]
    return float(FLOOR_FRACTION * positive_values.min()) if len(positive_values) else 1e-8


def to_log(target, floor: float) -> np.ndarray:
    """Converting target to log scale, considering 0 even if we dont have it in our training ln(max(y, floor))"""
    return np.log(np.maximum(np.asarray(target, dtype=float), floor))


def check_target_scale(target_scale: str) -> None:
    if target_scale not in TARGET_SCALES:
        raise ValueError(f"target_scale must be one of {TARGET_SCALES}, got {target_scale!r}")


def to_model_target(target, floor: float, target_scale: str = "log") -> np.ndarray:
    """What the models are fitted to is ln(max(y, floor)) on the log scale, y itself on the raw scale."""
    check_target_scale(target_scale)
    return to_log(target, floor) if target_scale == "log" else np.asarray(target, dtype=float)


def smearing_factor(residuals, target_scale: str = "log") -> float:
    """Duan smearing factor to convert log scale back to normal volatility"""
    check_target_scale(target_scale)
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if target_scale == "raw" or not len(residuals):
        return 1.0
    return float(np.mean(np.exp(residuals)))


def volatility_forecast(model_forecast, smearing: float, target_scale: str = "log") -> np.ndarray:
    """Back-transform a model forecast to the volatility scale smearing * exp(forecast)
    we report this for users who wich to see actual volatility forecasts when potentiall using on their own data"""
    check_target_scale(target_scale)
    model_forecast = np.asarray(model_forecast, dtype=float)
    if target_scale == "raw":
        return np.maximum(model_forecast, 0.0)
    return np.asarray(smearing, dtype=float) * np.exp(model_forecast)


def squared_error(realized, forecast) -> np.ndarray:
    """Per-event squared error on the volatility scale"""
    return (np.asarray(realized, dtype=float) - np.asarray(forecast, dtype=float)) ** 2

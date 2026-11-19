"""Hyperparameters and search settings of every model
"""
from __future__ import annotations

import numpy as np

SEED = 123
EMBARGO_EVENTS = 5
INNER_SPLITS = 3

RIDGE_ALPHAS = np.logspace(-2, 4, 25)
RIDGE_FOLDS = 5

# XGBoost settings that are fixed inside XGBoostEarlyStopping (early stopping on the last part of the window).
XGB_FIXED = {"gamma": 0.0, "max_rounds": 1000, "early_stopping_rounds": 50, "validation_fraction": 0.2}

# A "grid" entry is searched on the time-ordered inner folds; every other entry is fixed.
DEFAULT_PARAMS: dict[str, dict] = {
    # Mean of ln y per event type (feature set "event_type"): separates the level difference between
    # release types from real forecasting skill.
    "naive_mean": {},
    "har_ols": {},
    # Pooled event-panel SARIMAX; the orders come from analysis/arima_order_identification.py (BIC).
    "arimax": {"ar_order": 1, "ma_order": 1},
    "rf": {"n_estimators": 300, "max_features": 1 / 3,
           "grid": {"min_samples_leaf": [5, 20], "max_depth": [None, 6]}},
    "xgb": {"learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
            "grid": {"max_depth": [2, 3], "min_child_weight": [5, 20]}},
    "lstm": {"hidden_size": 64, "epochs": 30, "learning_rate": 1e-3, "batch_size": 16, "patience": 5,
             "val_fraction": 0.1, "dropout": 0.0, "weight_decay": 0.0},
}

MODEL_NAMES = list(DEFAULT_PARAMS)

# Pure benchmarks: fixed feature sets, never given news features, never recalibrated -> variant A only.
BENCHMARK_MODELS = ("naive_mean", "har_ols")
NEWS_MODEL_NAMES = [name for name in MODEL_NAMES if name not in BENCHMARK_MODELS]
SEQUENCE_MODELS = {"lstm", "arimax"}   # models that read the 10 lagged same-type volatilities

"""Tier A model zoo: scikit-learn-compatible estimators that predict ln y
"""
from __future__ import annotations

import time
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lakehouse.feature_columns import HAR_COLUMNS, SEQUENCE_COLUMNS
from lakehouse.modelling.model_params import (
    DEFAULT_PARAMS, EMBARGO_EVENTS, INNER_SPLITS, MODEL_NAMES, RIDGE_ALPHAS, RIDGE_FOLDS, SEED, XGB_FIXED)

LOG_COLUMNS = HAR_COLUMNS + ["pre_vol_h"] + list(SEQUENCE_COLUMNS)


class LogFloorColumns(BaseEstimator, TransformerMixin):
    """ln(x) of the named columns, floored at half the smallest positive TRAINING value
    for our analysis no 0 in training but floored is defined in case of extensions with other data"""

    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, features, target=None):
        features = pd.DataFrame(features)
        self.floor_by_column_ = {}
        for column in self.columns:
            if column in features.columns:
                values = pd.to_numeric(features[column], errors="coerce").to_numpy(float)
                positive_values = values[values > 0]
                self.floor_by_column_[column] = float(positive_values.min() / 2) if len(positive_values) else 1e-12
        return self

    def transform(self, features):
        features = pd.DataFrame(features).copy()
        for column, floor in self.floor_by_column_.items():
            values = pd.to_numeric(features[column], errors="coerce").to_numpy(float)
            observed = ~np.isnan(values)
            logged = values.copy()
            logged[observed] = np.log(np.maximum(values[observed], floor))
            features[column] = logged
        return features


class TimeOrderedGridSearch(BaseEstimator, RegressorMixin):
    """Grid search on time-ordered folds INSIDE the training window with the embargo defined in our thesis,
        then a refit of the best candidate on the whole window
    """

    def __init__(self, estimator_factory: Callable[..., object], grid: dict, n_splits: int = INNER_SPLITS,
                 embargo: int = EMBARGO_EVENTS, min_fit_events: int = 150):
        self.estimator_factory = estimator_factory
        self.grid = grid
        self.n_splits = n_splits
        self.embargo = embargo
        self.min_fit_events = min_fit_events

    def fit(self, features, target):
        features, target = np.asarray(features, dtype=float), np.asarray(target, dtype=float)
        candidates = list(ParameterGrid(self.grid))
        folds = [(fit_rows, validation_rows)
                 for fit_rows, validation_rows in TimeSeriesSplit(self.n_splits, gap=self.embargo).split(features)
                 if len(fit_rows) >= self.min_fit_events]
        scores = []
        for candidate in candidates:
            if not folds:
                scores.append(0.0)
                continue
            fold_errors = [np.mean((target[validation_rows] - self.estimator_factory(**candidate)
                                    .fit(features[fit_rows], target[fit_rows])
                                    .predict(features[validation_rows])) ** 2)
                           for fit_rows, validation_rows in folds]
            scores.append(float(np.mean(fold_errors)))
        self.best_params_ = candidates[int(np.argmin(scores))]
        self.cv_scores_ = dict(zip(map(str, candidates), scores))
        self.model_ = self.estimator_factory(**self.best_params_).fit(features, target)
        return self

    def predict(self, features):
        return self.model_.predict(np.asarray(features, dtype=float))


class XGBoostEarlyStopping(BaseEstimator, RegressorMixin):
    """XGBoost with early stopping on the last validation_fraction of the time-sorted window
     with embargo, the chosen number of trees is refit on the full window.
    """

    def __init__(self, max_depth: int = 3, min_child_weight: float = 10, learning_rate: float = 0.05,
                 subsample: float = 0.8, colsample_bytree: float = 0.8, reg_lambda: float = 1.0,
                 gamma: float = XGB_FIXED["gamma"], max_rounds: int = XGB_FIXED["max_rounds"],
                 early_stopping_rounds: int = XGB_FIXED["early_stopping_rounds"],
                 validation_fraction: float = XGB_FIXED["validation_fraction"], embargo: int = EMBARGO_EVENTS,
                 random_state: int = SEED):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.max_rounds = max_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.embargo = embargo
        self.random_state = random_state

    def _booster_params(self) -> dict:
        return dict(max_depth=self.max_depth, min_child_weight=self.min_child_weight,
                    learning_rate=self.learning_rate, subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda, gamma=self.gamma,
                    random_state=self.random_state, n_jobs=1)

    def fit(self, features, target):
        from xgboost import XGBRegressor

        features, target = np.asarray(features, dtype=float), np.asarray(target, dtype=float)
        event_count = len(target)
        validation_count = max(20, int(event_count * self.validation_fraction))
        fit_end = event_count - validation_count - self.embargo
        early_stopper = XGBRegressor(n_estimators=self.max_rounds, early_stopping_rounds=self.early_stopping_rounds,
                                     **self._booster_params())
        early_stopper.fit(features[:fit_end], target[:fit_end], verbose=False,
                          eval_set=[(features[event_count - validation_count:], target[event_count - validation_count:])])
        self.n_trees_ = int(early_stopper.best_iteration) + 1
        self.model_ = XGBRegressor(n_estimators=self.n_trees_, **self._booster_params()).fit(features, target)
        return self

    def predict(self, features):
        return self.model_.predict(np.asarray(features, dtype=float))


def make_news_correction() -> Pipeline:
    """Stage-2 correction g(news): news features standardised on the pool, then a ridge regression
    whose penalty is chosen by time-ordered cross-validation as explained in our thesis
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=RIDGE_ALPHAS, cv=TimeSeriesSplit(RIDGE_FOLDS, gap=EMBARGO_EVENTS))),
    ])


def split_fixed_and_grid(model_name: str, overrides: dict | None) -> tuple[dict, dict]:
    """Fixed hyperparameters and search grid of a model.
    """
    defaults = DEFAULT_PARAMS[model_name]
    overrides = dict(overrides or {})
    replacement_grid = overrides.pop("grid", None)
    fixed = {name: value for name, value in defaults.items() if name != "grid"}
    fixed.update(overrides)
    grid = dict(defaults.get("grid", {}) if replacement_grid is None else replacement_grid)
    return fixed, {name: values for name, values in grid.items() if name not in overrides}


def build_model(model_name: str, overrides: dict | None = None):
    """An unfitted estimator of ln y.  ``overrides`` (not the merged defaults) change ``DEFAULT_PARAMS``."""
    if model_name not in DEFAULT_PARAMS:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}")
    overrides = dict(overrides or {})
    log_step = ("log", LogFloorColumns(tuple(LOG_COLUMNS)))
    parameters = {**DEFAULT_PARAMS[model_name], **overrides}

    if model_name == "naive_mean":
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", LinearRegression(**parameters))])
    if model_name == "har_ols":
        return Pipeline([log_step, ("impute", SimpleImputer(strategy="median")),
                         ("model", LinearRegression(**parameters))])
    if model_name == "arimax":
        from lakehouse.modelling.sarimax_model import SARIMAXRegressor

        return Pipeline([log_step, ("model", SARIMAXRegressor(use_exogenous=True, **parameters))])
    if model_name == "rf":
        fixed, grid = split_fixed_and_grid(model_name, overrides)
        factory = partial(RandomForestRegressor, random_state=SEED, n_jobs=-1, **fixed)
        return Pipeline([log_step, ("impute", SimpleImputer(strategy="median")),
                         ("model", TimeOrderedGridSearch(factory, grid))])
    if model_name == "xgb":
        fixed, grid = split_fixed_and_grid(model_name, overrides)
        return Pipeline([log_step, ("model", TimeOrderedGridSearch(partial(XGBoostEarlyStopping, **fixed), grid))])
    if model_name == "lstm":
        from lakehouse.modelling.lstm_model import LSTMRegressor

        return Pipeline([log_step, ("model", LSTMRegressor(seed=SEED, **parameters))])
    raise ValueError(f"No builder for model {model_name!r}")


def fit_predict_block(model_name: str, overrides: dict | None, train_features, train_model_target, test_features):
    """Fit one tier A model on a training window; predict in-sample and for the test block.

        Module-level so joblib can run blocks in separate processes if desired
    """
    started = time.time()
    model = build_model(model_name, overrides)
    model.fit(train_features, np.asarray(train_model_target, float))
    return (np.asarray(model.predict(train_features), float), np.asarray(model.predict(test_features), float),
            tuned_settings(model), time.time() - started)


def final_estimator(model):
    """The last step of a pipeline or just the model if just one is specificed to be trained"""
    return model.steps[-1][1] if isinstance(model, Pipeline) else model


def tuned_settings(model) -> dict | None:
    """Hyperparameters chosen by the grid search here but obviously none if the model does not use grid search"""
    estimator = final_estimator(model)
    if not isinstance(estimator, TimeOrderedGridSearch):
        return None
    settings = {name: str(value) for name, value in estimator.best_params_.items()}
    settings["inner_cv_mse_log"] = float(min(estimator.cv_scores_.values()))
    if hasattr(estimator.model_, "n_trees_"):
        settings["n_trees"] = int(estimator.model_.n_trees_)
    return settings


def feature_importance(model, feature_names: list[str]) -> pd.DataFrame | None:
    """Impurity and gain for tree models, |coefficients| for linear models."""
    estimator = final_estimator(model)
    while isinstance(estimator, (TimeOrderedGridSearch, XGBoostEarlyStopping)):
        estimator = estimator.model_
    if hasattr(estimator, "feature_importances_"):
        values, kind = np.asarray(estimator.feature_importances_, dtype=float), "importance"
    elif hasattr(estimator, "coef_"):
        values, kind = np.abs(np.ravel(estimator.coef_)).astype(float), "abs_coef"
    else:
        return None
    if len(values) != len(feature_names):
        return None
    return (pd.DataFrame({"feature": feature_names, "value": values, "kind": kind})
            .sort_values("value", ascending=False).reset_index(drop=True))

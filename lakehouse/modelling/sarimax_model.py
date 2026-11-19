"""ARIMA / ARIMAX for the 30-second post-release volatility, estimated with statsmodels SARIMAX
which corresponds to the ARIMAX order selection in my thesis

SARIMAX takes a series so our events are stacked in time order
with MISSING_STEPS_BETWEEN_EVENTS NaN steps between them
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.statespace.sarimax import SARIMAX

from lakehouse.feature_columns import SEQUENCE_PREFIX

MISSING_STEPS_BETWEEN_EVENTS = 25


class SARIMAXRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, ar_order: int = 1, ma_order: int = 0, use_exogenous: bool = False,
                 sequence_prefix: str = SEQUENCE_PREFIX, maximum_iterations: int = 1000):
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.use_exogenous = use_exogenous
        self.sequence_prefix = sequence_prefix
        self.maximum_iterations = maximum_iterations

    def _split_columns(self, columns):
        """Lag columns oldest first"""
        lag_columns = [column for column in columns if re.fullmatch(rf"{re.escape(self.sequence_prefix)}\d+", column)]
        lag_columns.sort(key=lambda column: -int(column[len(self.sequence_prefix):]))
        return lag_columns, [column for column in columns if column not in lag_columns]

    def _stack_events(self, features: pd.DataFrame, target=None):
        """Stack the events into one series"""
        lag_columns, _ = self._split_columns(features.columns)
        event_count, lag_count = len(features), len(lag_columns)
        series = np.full((event_count, lag_count + 1 + MISSING_STEPS_BETWEEN_EVENTS), np.nan)
        series[:, :lag_count] = features[lag_columns].to_numpy(float)
        if target is not None:
            series[:, lag_count] = target
        target_positions = np.arange(event_count) * series.shape[1] + lag_count

        exogenous = np.zeros((series.size, 1 + len(self.exogenous_columns_)))
        exogenous[target_positions, 0] = 1.0                                   # the release jump D_t
        if self.exogenous_columns_:
            raw = features[self.exogenous_columns_].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            exogenous[target_positions, 1:] = (np.where(np.isnan(raw), self.medians_, raw) - self.means_) / self.scales_
        exogenous_frame = pd.DataFrame(exogenous, columns=["event_jump"] + [f"x.{c}" for c in self.exogenous_columns_])
        return pd.Series(series.ravel()), exogenous_frame, target_positions

    def fit(self, features: pd.DataFrame, target):
        """Standardise the exogenous columns and fit the stacked SARIMAX by exact ML.

                Non-lag columns are median-imputed; a column is kept only if the design matrix
                stays full rank.
        """
        self.feature_names_in_ = list(features.columns)
        _, other_columns = self._split_columns(features.columns)
        self.exogenous_columns_ = []
        if self.use_exogenous:
            other = features[other_columns].apply(pd.to_numeric, errors="coerce")
            medians = other.median().fillna(0.0)
            other = other.fillna(medians)
            for column in other_columns:
                design = np.column_stack([np.ones(len(other))] + [other[name] for name in self.exogenous_columns_ + [column]])
                if np.linalg.matrix_rank(design) == design.shape[1]:
                    self.exogenous_columns_.append(column)
            self.medians_ = medians[self.exogenous_columns_].to_numpy()
            self.means_ = other[self.exogenous_columns_].mean().to_numpy()
            standard_deviations = other[self.exogenous_columns_].std(ddof=0).to_numpy()
            self.scales_ = np.where(standard_deviations > 0, standard_deviations, 1.0)
        series, exogenous, _ = self._stack_events(features, np.asarray(target, float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.results_ = SARIMAX(series, exog=exogenous, order=(self.ar_order, 0, self.ma_order), trend="c").fit(
                method="bfgs", disp=False, maxiter=self.maximum_iterations)
        return self

    def predict(self, features):
        """One-step-ahead mean at each event's target step, with the fitted parameters applied to the new events."""
        series, exogenous, target_positions = self._stack_events(pd.DataFrame(features, columns=self.feature_names_in_))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = self.results_.apply(series, exog=exogenous)   # fitted parameters, new data
        return np.asarray(applied.get_prediction().predicted_mean)[target_positions]

    def coefficient_table(self) -> pd.DataFrame:
        """Estimate, standard error, z, p-value and 95% interval of every parameter."""
        results, interval = self.results_, self.results_.conf_int()
        return pd.DataFrame({"coef": results.params, "se": results.bse, "z": results.tvalues,
                             "p_value": results.pvalues, "ci_low": interval.iloc[:, 0],
                             "ci_high": interval.iloc[:, 1]}).rename_axis("term").reset_index()

    def summary_metrics(self) -> dict:
        results = self.results_
        return {"refit_bic": float(results.bic), "refit_aic": float(results.aic),
                "refit_sigma2": float(results.params["sigma2"]),
                "converged": float((results.mle_retvals or {}).get("converged", True))}

"""Identification of the ARIMA(X)(p, d, q) order for the 30-second volatility model (Appendix on
ARIMAX order selection).

Each event is a short series: the ten pre-release windows (pre_vol_30s_lag10 ... lag1) followed by
the release window (realized_vol_30s), read from gold.events_features.  Only events up to
``--until-year`` (default 2022) are used, so the test years never influence the order.

Steps: ADF and KPSS for d; panel PACF (HC1-robust intervals) for p; panel ACF (Fisher-z band) for
q; BIC grid over ARIMAX(p, 0, q) by exact ML with pooled Ljung-Box and AR/MA root-cancellation
diagnostics.

    python -m lakehouse.analysis.arima_order_identification
    python -m lakehouse.analysis.arima_order_identification --log-vol
    python -m lakehouse.analysis.arima_order_identification --feature-set base_news
    python -m lakehouse.analysis.arima_order_identification --mlflow

Outputs go to ``lakehouse/outputs/arima_identification/``.
"""
from __future__ import annotations

import argparse
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as statsmodels_api
from scipy import stats
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from lakehouse.feature_columns import SEQUENCE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMN
from lakehouse.infrastructure.mlflow_config import MLFLOW_EXPERIMENT
from lakehouse.modelling.feature_sets import DEFAULT_FEATURE_SET, FEATURE_SETS, NO_IMPUTE_COLUMNS
from lakehouse.modelling.gold_data import load_gold_events
from lakehouse.modelling.sarimax_model import MISSING_STEPS_BETWEEN_EVENTS, SARIMAXRegressor
from lakehouse.paths import ARIMA_OUTPUT_DIR
from lakehouse.plot_style import BLUE, apply_default_style, clean_axes, save_figure

DEFAULT_MAX_AR_ORDER = 3
LOG_EPSILON = 1e-6


def exogenous_columns_of(feature_set: str) -> list[str]:
    """Non-lag columns of a feature set, i.e. the ARIMAX regressors."""
    return [c for c in FEATURE_SETS[feature_set] if c not in SEQUENCE_COLUMNS]


def load_sequences(until_year: int, log_volatility: bool, snapshot_id: int | None,
                   exogenous_columns: list[str]):
    """Events up to ``until_year`` with complete sequences (and news columns, if used); only valid
    targets, as in train.py."""
    events, snapshot_id = load_gold_events(snapshot_id)
    required = [TARGET_COLUMN] + SEQUENCE_COLUMNS + [c for c in exogenous_columns if c in NO_IMPUTE_COLUMNS]
    keep_columns = ["ts_utc", TARGET_COLUMN] + SEQUENCE_COLUMNS + exogenous_columns
    events = events.loc[events["year"] <= until_year, list(dict.fromkeys(keep_columns))]
    events = events.dropna(subset=required).sort_values("ts_utc").reset_index(drop=True)
    if log_volatility:
        cols = [TARGET_COLUMN] + SEQUENCE_COLUMNS
        events[cols] = np.log(events[cols] + LOG_EPSILON)
    return events, snapshot_id


def pre_release_matrix(events: pd.DataFrame) -> np.ndarray:
    """(events, SEQUENCE_LENGTH) pre-release windows, oldest first."""
    return events[SEQUENCE_COLUMNS[::-1]].to_numpy(float)


# 1. Stationarity (d)

def stationarity_tests(series: pd.Series, name: str, alpha: float = 0.05) -> dict:
    values = series.dropna().to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        adf_stat, adf_p, adf_lags, _, adf_critical, _ = adfuller(values, autolag="AIC")
        kpss_stat, kpss_p, kpss_lags, _ = kpss(values, regression="c", nlags="auto")
    adf_stationary = adf_p < alpha
    kpss_stationary = kpss_p > alpha
    if adf_stationary and kpss_stationary:
        verdict = "stationary - model in levels"
    elif not adf_stationary and not kpss_stationary:
        verdict = "unit root - difference the series"
    elif adf_stationary:
        verdict = "ADF: stationary, KPSS: not stationary, likely stationary with long memory"
    else:
        verdict = "ADF cannot reject a unit root, KPSS cannot reject stationarity, inspect the plot"
    return {
        "series": name,
        "n": len(values),
        "adf_stat": adf_stat,
        "adf_p": adf_p,
        "adf_lags": adf_lags,
        "adf_crit_5%": adf_critical["5%"],
        "kpss_stat": kpss_stat,
        "kpss_p": kpss_p,
        "kpss_lags": kpss_lags,
        "verdict": verdict,
    }


def panel_pacf(events: pd.DataFrame, series_column: str, lag_columns: list[str]) -> pd.DataFrame:
    """phi_kk for k = 1..len(lag_columns): the lag-k coefficient of a pooled OLS on lags 1..k, with
    an HC1-robust 95% interval."""
    series = events[series_column].to_numpy(float)
    rows = []
    for order in range(1, len(lag_columns) + 1):
        design = statsmodels_api.add_constant(events[lag_columns[:order]].to_numpy(float))
        fit = statsmodels_api.OLS(series, design).fit(cov_type="HC1")
        coef, se = fit.params[-1], fit.bse[-1]
        rows.append({
            "lag": order,
            "value": coef,
            "ci_low": coef - 1.96 * se,
            "ci_high": coef + 1.96 * se,
            "significant": abs(coef) > 1.96 * se,
        })
    return pd.DataFrame(rows)


def panel_acf(events: pd.DataFrame, max_lag: int = SEQUENCE_LENGTH - 1) -> pd.DataFrame:
    """Correlation of x_t and x_{t-k} over pre-release positions and events; the Fisher-z band
    counts one pair per event."""
    windows = pre_release_matrix(events)
    rows = []
    for lag in range(1, max_lag + 1):
        corr = float(np.corrcoef(windows[:, lag:].ravel(), windows[:, :-lag].ravel())[0, 1])
        half_width = 1.96 / np.sqrt(max(len(events) - 3, 1))
        rows.append({
            "lag": lag,
            "value": corr,
            "ci_low": np.tanh(np.arctanh(corr) - half_width),
            "ci_high": np.tanh(np.arctanh(corr) + half_width),
            "significant": abs(corr) > np.tanh(half_width),
        })
    return pd.DataFrame(rows)


def cutoff_lag(table: pd.DataFrame) -> int:
    """Last lag of the first run of significant lags (0 if lag 1 is not significant)."""
    cutoff = 0
    for lag, significant in zip(table["lag"], table["significant"]):
        if not significant:
            break
        cutoff = int(lag)
    return cutoff


def pooled_ljung_box(residuals: np.ndarray, lags: int, estimated_parameters: int = 0) -> dict:
    """Ljung-Box on standardised residuals pooled over events.

        Autocorrelations use pairs from the same event only, so the gap between stacked events
        never enters;
        under the null Q is chi-square(lags - estimated_parameters).
    """
    residuals = np.asarray(residuals, float)
    total_squares = float(np.nansum(residuals ** 2))
    observation_count = int(np.isfinite(residuals).sum())
    statistic = 0.0
    for lag in range(1, lags + 1):
        earlier, later = residuals[:, :-lag], residuals[:, lag:]
        paired = np.isfinite(earlier) & np.isfinite(later)
        if paired.any():
            numerator = float((earlier[paired] * later[paired]).sum() / total_squares)
            statistic += numerator ** 2 / int(paired.sum())
    statistic *= observation_count * (observation_count + 2)
    degrees_of_freedom = lags - estimated_parameters
    p_value = float(stats.chi2.sf(statistic, degrees_of_freedom)) if degrees_of_freedom > 0 else float("nan")
    return {"Q": statistic, "df": degrees_of_freedom, "p_value": p_value}


def root_cancellation(results, tolerance: float) -> dict:
    """AR-MA common factors of a fitted SARIMAX: pairs of inverse AR and MA roots closer than
    ``tolerance`` (matched greedily)."""
    ar_roots = np.atleast_1d(results.arroots)
    ma_roots = np.atleast_1d(results.maroots)
    ar_inv = 1 / ar_roots if len(ar_roots) else np.array([])
    ma_inv = 1 / ma_roots if len(ma_roots) else np.array([])
    summary = {
        "ar_inv_root_max": float(np.max(np.abs(ar_inv))) if len(ar_inv) else np.nan,
        "ma_inv_root_max": float(np.max(np.abs(ma_inv))) if len(ma_inv) else np.nan,
        "min_root_distance": np.nan,
        "n_cancelling": 0,
    }
    if not len(ar_inv) or not len(ma_inv):
        return summary
    distances = np.abs(ar_inv[:, None] - ma_inv[None, :])
    summary["min_root_distance"] = float(distances.min())
    while distances.size and distances.min() < tolerance:
        row, col = np.unravel_index(np.argmin(distances), distances.shape)
        summary["n_cancelling"] += 1
        distances = np.delete(np.delete(distances, row, axis=0), col, axis=1)
    return summary


def arma_order_table(events: pd.DataFrame, max_ar_order: int, max_ma_order: int,
                     exogenous_columns: list[str], ljung_box_lags: tuple[int, ...],
                     root_tolerance: float) -> pd.DataFrame:
    """Fit ARIMAX(p, 0, q) for every p, q in range: BIC, AIC, convergence, pooled Ljung-Box, root
    checks and the largest standard error."""
    features = events[SEQUENCE_COLUMNS + exogenous_columns]
    target = events[TARGET_COLUMN].to_numpy(float)
    steps_per_event = SEQUENCE_LENGTH + 1 + MISSING_STEPS_BETWEEN_EVENTS
    rows = []
    for ar_order in range(max_ar_order + 1):
        for ma_order in range(max_ma_order + 1):
            try:
                model = SARIMAXRegressor(ar_order=ar_order, ma_order=ma_order,
                                         use_exogenous=bool(exogenous_columns)).fit(features, target)
            except ValueError as error:  # too few steps per event for this order
                rows.append({"p": ar_order, "d": 0, "q": ma_order, "error": str(error)})
                continue
            results = model.results_
            standardised = np.asarray(results.standardized_forecasts_error)[0]
            standardised = standardised.reshape(len(features), steps_per_event)
            standardised = standardised[:, :SEQUENCE_LENGTH + 1]  # drop the padding between events
            row = {
                "p": ar_order,
                "d": 0,
                "q": ma_order,
                "order": f"({ar_order},0,{ma_order})",
                "n_exog": len(model.exogenous_columns_),
                "bic": results.bic,
                "aic": results.aic,
                "sigma2": results.params["sigma2"],
                "converged": bool((results.mle_retvals or {}).get("converged", True)),
            }
            for lags in ljung_box_lags:
                test = pooled_ljung_box(standardised, lags, estimated_parameters=ar_order + ma_order)
                row[f"lb_Q_lag{lags}"] = test["Q"]
                row[f"lb_p_lag{lags}"] = test["p_value"]
            roots = root_cancellation(results, root_tolerance)
            row.update(roots)
            row["reduced_order"] = f"({ar_order - roots['n_cancelling']},0,{ma_order - roots['n_cancelling']})"
            arma_terms = [name for name in results.params.index if str(name).startswith(("ar.L", "ma.L"))]
            row["se_ar_ma_max"] = float(results.bse[arma_terms].max()) if arma_terms else np.nan
            row["p_last_ar"] = float(results.pvalues[f"ar.L{ar_order}"]) if ar_order else np.nan
            row["p_last_ma"] = float(results.pvalues[f"ma.L{ma_order}"]) if ma_order else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot_identification(pre_event_pacf: pd.DataFrame, acf: pd.DataFrame, event_count: int,
                        title_suffix: str, path) -> None:
    apply_default_style()
    figure, axes = plt.subplots(1, 2, figsize=(16, 4))
    panels = [
        (axes[0], pre_event_pacf, "PACF - pre-event: lag1 on lags 2..k+1"),
        (axes[1], acf, "ACF - pre-release sequences"),
    ]
    for axis, table, title in panels:
        axis.vlines(table["lag"], 0, table["value"], lw=2, color=BLUE)
        axis.scatter(table["lag"], table["value"], zorder=3, color=BLUE)
        axis.fill_between(
            table["lag"],
            table["ci_low"] - table["value"],
            table["ci_high"] - table["value"],
            alpha=0.2, step="mid", color=BLUE, label="95% band",
        )
        axis.axhline(0, color="k", lw=0.8)
        axis.set_xticks(table["lag"])
        axis.set_xlabel("Lag (previous releases of the same type)")
        axis.set_title(title, fontsize=9)
        clean_axes(axis, grid="y")
    axes[0].legend(fontsize=8)
    figure.suptitle(
        f"Order identification for the 30-second volatility ARIMA{title_suffix} (N = {event_count} events)"
    )
    save_figure(figure, path, dpi=150)


def _number_format(value: float) -> str:
    return f"{value:.4g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--until-year", type=int, default=2022,
                        help="last year used (default 2022: before the test years)")
    parser.add_argument("--log-vol", action="store_true",
                        help="analyse ln(vol) instead of vol (diagnostic only)")
    parser.add_argument("--p-max", type=int, default=DEFAULT_MAX_AR_ORDER,
                        help="largest AR order in the BIC grid")
    parser.add_argument("--q-max", type=int, default=3, help="largest MA order in the BIC grid")
    parser.add_argument("--lb-lags", type=int, nargs="+", default=[5, SEQUENCE_LENGTH],
                        help="Ljung-Box lags")
    parser.add_argument("--root-tol", type=float, default=0.1,
                        help="inverse AR/MA roots closer than this cancel")
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET, choices=list(FEATURE_SETS),
                        help="regressors of the ARIMAX = the non-lag columns of this set")
    parser.add_argument("--no-exog", action="store_true",
                        help="plain ARIMA grid (no exogenous regressors)")
    parser.add_argument("--snapshot-id", type=int, default=None)
    parser.add_argument("--mlflow", action="store_true", help="also log tables and plot to MLflow")
    args = parser.parse_args()

    exogenous_columns = [] if args.no_exog else exogenous_columns_of(args.feature_set)
    events, snapshot_id = load_sequences(args.until_year, args.log_vol, args.snapshot_id,
                                         exogenous_columns)
    suffix, file_tag = (" (log)", "_log") if args.log_vol else ("", "")
    print(f"{len(events):,} events up to {args.until_year} (gold snapshot {snapshot_id}){suffix}")
    if exogenous_columns:
        print(f"BIC grid model: ARIMAX, regressors of the release jump = {exogenous_columns}")
    else:
        print("BIC grid model: plain ARIMA (no exogenous regressors)")

    stationarity = pd.DataFrame([
        stationarity_tests(events[TARGET_COLUMN], TARGET_COLUMN),
        stationarity_tests(events[SEQUENCE_COLUMNS[0]], SEQUENCE_COLUMNS[0]),
    ])
    print("\n1. Stationarity across events (event-ordered series)")
    print(stationarity.to_string(index=False, float_format=_number_format))

    event_pacf = panel_pacf(events, TARGET_COLUMN, SEQUENCE_COLUMNS)
    pre_event_pacf = panel_pacf(events, SEQUENCE_COLUMNS[0], SEQUENCE_COLUMNS[1:])
    acf = panel_acf(events)
    print("\n2. PACF (event: y = realized_vol_30s)")
    print(event_pacf.to_string(index=False, float_format=_number_format))
    print("\n3. ACF (pre-release sequences)")
    print(acf.to_string(index=False, float_format=_number_format))

    grid = arma_order_table(events, args.p_max, args.q_max, exogenous_columns,
                            tuple(args.lb_lags), args.root_tol)
    fitted = grid.dropna(subset=["bic"]) if "bic" in grid else grid.iloc[0:0]
    best = fitted.loc[fitted["bic"].idxmin()] if len(fitted) else None
    model_name = "arimax" if exogenous_columns else "arima"

    print(f"\n4. BIC grid ({model_name.upper()}), d = 0 (lowest 8)")
    print(fitted.sort_values("bic").head(8)[
        ["p", "d", "q", "n_exog", "bic", "aic", "sigma2", "converged"]
    ].to_string(index=False, float_format=_number_format))

    diagnostic_columns = [c for c in ["order", "bic"]
                          + [c for c in fitted.columns if c.startswith("lb_p_lag")]
                          + ["ar_inv_root_max", "ma_inv_root_max", "min_root_distance",
                             "n_cancelling", "reduced_order", "se_ar_ma_max", "p_last_ar",
                             "p_last_ma"]
                          if c in fitted.columns]
    print("\n4b. Residual autocorrelation (pooled Ljung-Box, df = lag - p - q) "
          "and AR/MA root cancellation")
    print(fitted.sort_values(["p", "q"])[diagnostic_columns]
          .to_string(index=False, float_format=_number_format))

    print(f"\nPACF cut-off p = {cutoff_lag(event_pacf)}, ACF cut-off q = {cutoff_lag(acf)} "
          f"(a slowly decaying ACF means AR terms explain it - prefer the BIC choice for q)")
    if best is not None:
        suggested = {"ar_order": int(best.p), "ma_order": int(best.q)}
        if args.log_vol:
            print("NOTE: --log-vol is a diagnostic only; the trained SARIMAX models use the raw lag scale.")
        feature_arg = f"--feature-set {args.feature_set} " if exogenous_columns else ""
        print(
            f"BIC-best {model_name.upper()}({suggested['ar_order']},0,{suggested['ma_order']}). "
            f"Train it with:\n  python -m lakehouse.modelling.train --model {model_name} "
            f"{feature_arg}--params '{json.dumps(suggested)}'"
        )

    ARIMA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = ARIMA_OUTPUT_DIR / f"arima_identification{file_tag}.png"
    plot_identification(pre_event_pacf, acf, len(events), suffix, figure_path)

    tables = {
        f"stationarity{file_tag}.csv": stationarity,
        f"pacf_event{file_tag}.csv": event_pacf,
        f"pacf_pre_event{file_tag}.csv": pre_event_pacf,
        f"acf_pre_event{file_tag}.csv": acf,
        f"{model_name}_bic_grid{file_tag}.csv": grid,
    }
    for name, table in tables.items():
        table.to_csv(ARIMA_OUTPUT_DIR / name, index=False)
    print(f"\nSaved plot and tables to {ARIMA_OUTPUT_DIR}")

    if args.mlflow:
        from lakehouse.infrastructure.mlflow_config import init_mlflow

        mlflow = init_mlflow()
        mlflow.set_experiment(f"{MLFLOW_EXPERIMENT}_diagnostics")
        with mlflow.start_run(run_name=f"arima_identification{file_tag}"):
            mlflow.log_params({
                "until_year": args.until_year,
                "log_vol": args.log_vol,
                "bic_model": model_name,
                "feature_set": "none" if args.no_exog else args.feature_set,
                "gold_snapshot_id": snapshot_id,
                "n_events": len(events),
            })
            if best is not None:
                mlflow.log_metrics({"p_bic": int(best.p), "q_bic": int(best.q)})
            for artifact in [figure_path] + [ARIMA_OUTPUT_DIR / n for n in tables]:
                mlflow.log_artifact(str(artifact))
        print(f"Logged to MLflow experiment '{MLFLOW_EXPERIMENT}_diagnostics'")


if __name__ == "__main__":
    main()
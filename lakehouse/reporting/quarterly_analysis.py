"""Quarter-by-quarter evaluation of the walk-forward forecasts (thesis section on quarterly
performance and regimes).

The models are still refitted once a year but the error reporting is by quarter. Out-of-sample errors on
the log scale are split by calendar quarter and compared on the same events, the best model per
quarter and every model against the benchmarks and each other

    python -m lakehouse.reporting.quarterly_analysis
    python -m lakehouse.reporting.quarterly_analysis --models
    python -m lakehouse.reporting.quarterly_analysis --benchmark

"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from lakehouse.feature_columns import SEQUENCE_COLUMNS
from lakehouse.paths import QUARTERLY_OUTPUT_DIR, RESULTS_PUBLISHED_DIR, RESULTS_STAGING_DIR
from lakehouse.reporting.staged_predictions import add_log_scale, load_staged_predictions

EVENT_KEY = ["ts_utc", "event_type"]
DEFAULT_BENCHMARKS = ["naive_mean[", "har_ols["]   # prefixes for any feature-set label of these models
GOLD_COVARIATES = ["rv_d", "rv_w", "rv_m", "pre_vol_h", "forecast", "n_post_ticks", SEQUENCE_COLUMNS[0], "news_count_24h"]
OUTLIER_COUNTS = (0, 1, 2, 3)   # how many worst events are removed in the leave-out check


def load_predictions(directories: list[Path], snapshot_id: int | None) -> pd.DataFrame:
    """Staged predictions with a ``label`` column, latest run per label, optionally one gold snapshot only."""
    predictions = load_staged_predictions(directories)
    predictions = predictions.drop_duplicates(subset=["run_id"] + EVENT_KEY) 
    if snapshot_id is not None:
        predictions = predictions[predictions["gold_snapshot_id"] == snapshot_id]
    predictions["label"] = predictions["model"] + "[" + predictions["feature_set"] + "]"
    latest = predictions.groupby("label")["trained_at_utc"].transform("max")
    predictions = predictions[predictions["trained_at_utc"] == latest].copy()
    predictions["ts_utc"] = pd.to_datetime(predictions["ts_utc"], utc=True)
    predictions = add_log_scale(predictions)  
    predictions["y_true"], predictions["y_pred"] = predictions["y_true_log"], predictions["y_pred_log"]
    return predictions


def build_matched_panel(predictions: pd.DataFrame, models: list[str] | None, excluded: list[str] | None, frequency: str):
    """Wide frame of the matched sample: EVENT_KEY, y_true, quarter and one forecast column per label."""
    labels = sorted(predictions["label"].unique())
    if models:
        labels = [label for label in labels if any(label.startswith(prefix) for prefix in models)]
    if excluded:
        labels = [label for label in labels if not any(label.startswith(prefix) for prefix in excluded)]
    if len(labels) < 2:
        raise SystemExit(f"Need at least 2 models, found {labels}")
    selected = predictions[predictions["label"].isin(labels)]

    spread = selected.groupby(EVENT_KEY)["y_true"].agg(lambda values: values.max() - values.min())
    inconsistent = spread[spread > 1e-6].index
    if len(inconsistent):
        print(f"WARNING: y_true differs across runs for {len(inconsistent)} events (different gold snapshots?) - "
              f"those events are dropped. Use --snapshot-id to pin one snapshot.")
        selected = selected[~selected.set_index(EVENT_KEY).index.isin(inconsistent)]
    truth = selected.groupby(EVENT_KEY)["y_true"].first()
    wide = selected.pivot(index=EVENT_KEY, columns="label", values="y_pred")
    coverage = pd.DataFrame({"n_events": wide.notna().sum(),
                             "first": [wide[c].dropna().index.get_level_values(0).min() for c in wide],
                             "last": [wide[c].dropna().index.get_level_values(0).max() for c in wide]})
    matched = wide.dropna()
    print("Coverage per label (events with a prediction):")
    print(coverage.to_string())
    print(f"Matched sample: {len(matched):,} events predicted by all {len(labels)} labels "
          f"({len(matched) / coverage['n_events'].max():.0%} of the largest label)")
    if len(matched) < 0.5 * coverage["n_events"].max():
        smallest = coverage["n_events"].idxmin()
    panel = matched.join(truth).reset_index()
    panel["quarter"] = panel["ts_utc"].dt.tz_convert(None).dt.to_period(frequency).astype(str)
    return panel.sort_values("ts_utc").reset_index(drop=True), labels


def resolve_label(token: str, labels: list[str]) -> str | None:
    """Exact label, else the first label starting with ``token``."""
    if token in labels:
        return token
    return next((label for label in labels if label.startswith(token)), None)


def event_losses(panel: pd.DataFrame, labels: list[str], kind: str) -> pd.DataFrame:
    """Per-event squared error or absolute error of every label."""
    errors = panel[labels].sub(panel["y_true"], axis=0)
    return errors ** 2 if kind == "se" else errors.abs()


def quarterly_metrics(panel: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """MSE, MAE and bias (mean forecast minus actual) of every label in every quarter."""
    squared, absolute = event_losses(panel, labels, "se"), event_losses(panel, labels, "ae")
    errors = panel[labels].sub(panel["y_true"], axis=0)
    rows = []
    for quarter, index in panel.groupby("quarter").groups.items():
        for label in labels:
            rows.append({"quarter": quarter, "model": label, "n": len(index), "mse": squared.loc[index, label].mean(),
                         "mae": absolute.loc[index, label].mean(), "bias": errors.loc[index, label].mean()})
    return pd.DataFrame(rows)


def pairwise_ratios(panel: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Every ordered pair in every quarters' MSE / MAE ratios next
    to the pooled ones"""
    squared, absolute = event_losses(panel, labels, "se"), event_losses(panel, labels, "ae")
    pooled_squared, pooled_absolute = squared.mean(), absolute.mean()
    groups = panel.groupby("quarter").groups
    rows = []
    for model in labels:
        for reference in labels:
            if model == reference:
                continue
            for quarter, index in groups.items():
                rows.append({"quarter": quarter, "model": model, "ref": reference, "n": len(index),
                             "mse_ratio": squared.loc[index, model].mean() / squared.loc[index, reference].mean(),
                             "mae_ratio": absolute.loc[index, model].mean() / absolute.loc[index, reference].mean(),
                             "overall_mse_ratio": pooled_squared[model] / pooled_squared[reference],
                             "overall_mae_ratio": pooled_absolute[model] / pooled_absolute[reference]})
    ratios = pd.DataFrame(rows)
    ratios["reversal_mse"] = (ratios["overall_mse_ratio"] < 1) & (ratios["mse_ratio"] > 1)
    ratios["reversal_mae"] = (ratios["overall_mae_ratio"] < 1) & (ratios["mae_ratio"] > 1)
    return ratios


def attach_gold_covariates(panel: pd.DataFrame, skip_gold: bool, snapshot_id: int | None):
    """Join regime covariates from gold; returns"""
    covariates = ["y_true"]
    if skip_gold:
        return panel, covariates
    try:
        from lakehouse.modelling.gold_data import load_gold_events

        gold, _ = load_gold_events(snapshot_id)
        columns = [c for c in GOLD_COVARIATES if c in gold.columns]
        panel = panel.merge(gold[["ts_utc", "event_type"] + columns].drop_duplicates(EVENT_KEY), on=EVENT_KEY, how="left")
        covariates += [c for c in columns if panel[c].notna().mean() > 0.5]
        print(f"Regime covariates joined from gold: {covariates[1:]}")
    except Exception as error:
        print(f"NOTE: gold covariates not available {error} so using y_true-based ")
    return panel, covariates


def diagnose(panel: pd.DataFrame, model: str, reference: str, quarter: str, covariates: list[str],
             ratio_by_quarter: pd.Series):
    """Event-by-event decomposition of the MSE ratio of ``model`` vs ``reference`` in ``quarter``.

        Returns (tables, notes, event frame of the quarter): outliers, bias and calibration slope,
        release type,
        spikes, regime covariates and the relation across quarters.
    """
    in_quarter = panel["quarter"] == quarter
    events, others = panel[in_quarter].copy(), panel[~in_quarter]
    error_model, error_reference = events[model] - events["y_true"], events[reference] - events["y_true"]
    events["se_model"], events["se_ref"] = error_model ** 2, error_reference ** 2
    events["gain"] = events["se_ref"] - events["se_model"]
    mse_model, mse_reference = events["se_model"].mean(), events["se_ref"].mean()
    ratio = mse_model / mse_reference
    pooled_ratio = ((panel[model] - panel["y_true"]) ** 2).mean() / ((panel[reference] - panel["y_true"]) ** 2).mean()
    tables, notes = {}, [f"{model} vs {reference} in {quarter}: n = {len(events)}, MSE ratio = {ratio:.3f} "
                         f"({'model has the higher MSE' if ratio > 1 else 'model has the lower MSE'}); overall ratio = {pooled_ratio:.3f}"]

    worst_first = events["gain"].sort_values().index
    leave_out = pd.DataFrame([{"dropped_worst_events": k, "n": len(events) - k,
                               "mse_ratio": events.drop(index=worst_first[:k])["se_model"].mean()
                               / events.drop(index=worst_first[:k])["se_ref"].mean()} for k in OUTLIER_COUNTS])
    tables["leave_out"] = leave_out
    flips = leave_out[(leave_out["dropped_worst_events"] > 0) & (leave_out["mse_ratio"] < 1)]
    losses_only = events["gain"].clip(upper=0).abs()
    top_three_share = losses_only.sort_values(ascending=False).head(3).sum() / max(losses_only.sum(), 1e-12)
    if ratio > 1 and len(flips) and int(flips["dropped_worst_events"].min()) <= 2:
        k = int(flips["dropped_worst_events"].min())
        removed = events.loc[worst_first[:k], ["ts_utc", "event_type", "y_true", model, reference]].rename(
            columns={model: "m", reference: "r"})
        notes.append(f"OUTLIERS: removing the {k} event(s) where {model} lost most turns the ratio to "
                     f"{flips.iloc[0]['mse_ratio']:.3f} (< 1). Event(s): "
                     + "; ".join(f"{r.ts_utc:%Y-%m-%d %H:%M} {r.event_type} actual {r.y_true:.3f}, model {r.m:.3f}, ref {r.r:.3f}"
                                 for r in removed.itertuples()))
    elif ratio <= 1:
        notes.append(f"Outliers: {model} has the lower MSE this quarter (the 3 worst events carry {top_three_share:.0%} of its losses); "
                     f"pick a quarter where it loses (--explain-quarter) to diagnose a reversal.")
    else:
        survives = leave_out.iloc[3]["mse_ratio"] > 1
        notes.append(f"Outliers: the 3 worst events carry {top_three_share:.0%} of the model's total losses in the quarter; "
                     f"the reversal {'survives' if survives else 'does not survive'} dropping them "
                     f"(ratio {leave_out.iloc[3]['mse_ratio']:.3f}) - "
                     + ("broad-based, not a single-event story." if survives else "outlier-driven."))

    def calibration_slope(forecast, actual):
        return float(np.polyfit(forecast, actual, 1)[0]) if len(actual) >= 5 and np.std(forecast) > 0 else np.nan

    tables["bias"] = pd.DataFrame({
        "scope": ["quarter", "other quarters"] * 2, "model": [model, model, reference, reference],
        "bias_mean_pred_minus_actual": [error_model.mean(), (others[model] - others["y_true"]).mean(),
                                        error_reference.mean(), (others[reference] - others["y_true"]).mean()],
        "calibration_slope_actual_on_pred": [calibration_slope(events[model], events["y_true"]),
                                             calibration_slope(others[model], others["y_true"]),
                                             calibration_slope(events[reference], events["y_true"]),
                                             calibration_slope(others[reference], others["y_true"])],
        "corr_pred_actual": [np.corrcoef(events[model], events["y_true"])[0, 1], np.corrcoef(others[model], others["y_true"])[0, 1],
                             np.corrcoef(events[reference], events["y_true"])[0, 1],
                             np.corrcoef(others[reference], others["y_true"])[0, 1]]})
    bias_model, bias_reference, typical_error = error_model.mean(), error_reference.mean(), np.sqrt(mse_model)
    if abs(bias_model) > 0.35 * typical_error and abs(bias_model) > abs(bias_reference):
        notes.append(f"BIAS: {model} {'over' if bias_model > 0 else 'under'}-predicts by {abs(bias_model):.3f} on average in "
                     f"{quarter} (typical error {typical_error:.3f}; the reference's bias is {bias_reference:+.3f}); outside "
                     f"the quarter its bias is {(others[model] - others['y_true']).mean():+.3f}. Squared bias is "
                     f"{bias_model ** 2 / mse_model:.0%} of its MSE there.")
    else:
        notes.append(f"Bias: not the main driver (model {bias_model:+.3f}, reference {bias_reference:+.3f}, typical error {typical_error:.3f}).")

    by_type = events.groupby("event_type").agg(n=("gain", "size"), mse_model=("se_model", "mean"), mse_ref=("se_ref", "mean"))
    by_type["mse_ratio"] = by_type["mse_model"] / by_type["mse_ref"]
    tables["event_type"] = by_type.reset_index()
    if len(by_type) > 1 and (by_type["mse_ratio"] > 1).any() and (by_type["mse_ratio"] < 1).any():
        worst_type = by_type["mse_ratio"].idxmax()
        notes.append(f"EVENT TYPE: the loss is concentrated in {worst_type} releases (ratio {by_type.loc[worst_type, 'mse_ratio']:.2f}); "
                     f"the other type is fine.")

    threshold = panel["y_true"].quantile(0.9)
    spikes = events["y_true"] > threshold
    if spikes.any():
        notes.append(f"Spikes: {int(spikes.sum())} of {len(events)} events exceed the sample's 90th percentile of the target "
                     f"({threshold:.3f}, ln scale); they carry {events.loc[spikes, 'se_model'].sum() / events['se_model'].sum():.0%} of the "
                     f"model's and {events.loc[spikes, 'se_ref'].sum() / events['se_ref'].sum():.0%} of the reference's squared error.")

    quarter_means = panel.groupby("quarter")[covariates].mean()
    rows = []
    for covariate in covariates:
        others_means = quarter_means.drop(index=quarter)[covariate].dropna()
        if len(others_means) < 3 or others_means.std() == 0:
            continue
        rows.append({"covariate": covariate, "quarter_mean": quarter_means.loc[quarter, covariate],
                     "other_quarters_mean": others_means.mean(),
                     "z_vs_other_quarters": (quarter_means.loc[quarter, covariate] - others_means.mean()) / others_means.std()})
    regime = pd.DataFrame(rows)
    if panel["quarter"].nunique() > 3:
        quarterly_std = panel.groupby("quarter")["y_true"].std()
        other_std = quarterly_std.drop(index=quarter)
        regime = pd.concat([regime, pd.DataFrame([{
            "covariate": "y_true (within-quarter std)", "quarter_mean": quarterly_std[quarter],
            "other_quarters_mean": other_std.mean(),
            "z_vs_other_quarters": (quarterly_std[quarter] - other_std.mean()) / other_std.std() if other_std.std() > 0 else np.nan}])],
            ignore_index=True)
    tables["regime"] = regime
    if len(regime):
        unusual = regime[regime["z_vs_other_quarters"].abs() > 1.5]
        notes.append("REGIME: " + "; ".join(f"{r.covariate} is {r.z_vs_other_quarters:+.1f} SD vs the other quarters "
                                              f"({r.quarter_mean:.3g} vs {r.other_quarters_mean:.3g})" for r in unusual.itertuples())
                     + " - the quarter is unlike the training/other regimes." if len(unusual)
                     else "Regime: no covariate is more than 1.5 SD away from the other quarters.")

    if len(ratio_by_quarter) >= 6:
        rows = []
        first_type = panel["event_type"].value_counts().index[0]
        share_column = f"share_{first_type}"
        for covariate in covariates + [share_column]:
            series = (panel.assign(is_first=(panel["event_type"] == first_type).astype(float))
                      .groupby("quarter")["is_first"].mean()
                      if covariate == share_column else quarter_means[covariate])
            joined = pd.concat([np.log(ratio_by_quarter.rename("log_ratio")), series.rename("x")], axis=1).dropna()
            if len(joined) >= 6 and joined["x"].std() > 0:
                rho, p_value = stats.spearmanr(joined["x"], joined["log_ratio"])
                rows.append({"covariate": covariate, "n_quarters": len(joined),
                             "spearman_rho_with_log_mse_ratio": rho, "p_value": p_value})
        across = pd.DataFrame(rows)
        tables["cross_quarter"] = across
        clear = across[across["p_value"] < 0.10] if len(across) else across
        if len(clear):
            first = clear.iloc[0]
            notes.append("PATTERN across quarters: " + "; ".join(
                f"log MSE ratio {'rises' if r.spearman_rho_with_log_mse_ratio > 0 else 'falls'} with quarterly mean {r.covariate} "
                f"(rho {r.spearman_rho_with_log_mse_ratio:+.2f}, p {r.p_value:.2f}, {r.n_quarters} quarters)" for r in clear.itertuples())
                         + f" - the model is relatively {'worse' if first['spearman_rho_with_log_mse_ratio'] > 0 else 'better'} "
                           f"in high-{first['covariate']} quarters (descriptive, few quarters).")
        else:
            notes.append(f"Pattern across quarters: no covariate is clearly rank-correlated with relative performance "
                         f"({len(ratio_by_quarter)} quarters, p < 0.10 threshold).")
    return tables, notes, events


def markdown_table(table: pd.DataFrame, digits: int = 3) -> str:
    lines = ["| " + " | ".join(map(str, table.columns)) + " |", "|" + "|".join(["---"] * len(table.columns)) + "|"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(f"{v:.{digits}g}" if isinstance(v, (float, np.floating)) and np.isfinite(v)
                                       else ("" if isinstance(v, float) else str(v)) for v in row) + " |")
    return "\n".join(lines)


def choose_case(ratios, benchmarks, benchmark, arguments, labels):
    """(model, reference, quarter, how chosen): the most severe reversal against a benchmark unless
    the user named the case."""
    model_arg = resolve_label(arguments.explain_model, labels) if arguments.explain_model else None
    reference_arg = resolve_label(arguments.explain_ref, labels) if arguments.explain_ref else None
    if arguments.explain_model and model_arg is None:
        raise SystemExit(f"--explain-model {arguments.explain_model!r} not among {labels}")
    if arguments.explain_ref and reference_arg is None:
        raise SystemExit(f"--explain-ref {arguments.explain_ref!r} not among {labels}")
    candidates = ratios[ratios["n"] >= arguments.min_n]
    candidates = candidates[candidates["model"] == model_arg] if model_arg else candidates[~candidates["model"].isin(benchmarks)]
    if reference_arg:
        candidates = candidates[candidates["ref"] == reference_arg]
    if arguments.explain_quarter:
        candidates = candidates[candidates["quarter"] == arguments.explain_quarter]
    reversals = candidates[candidates["reversal_mse"] & (candidates["ref"].isin(benchmarks) if not reference_arg else True)]
    if not len(reversals):
        reversals = candidates[candidates["reversal_mse"]]
    how = "most severe reversal"
    if not len(reversals):
        how = "weakest quarter (no reversal found)"
        against = candidates[candidates["ref"] == (reference_arg or benchmark)]
        reversals = against if len(against) else candidates
    if not len(reversals):
        raise SystemExit("Nothing to diagnose: no (model, reference, quarter) with enough events; lower --min-n.")
    row = reversals.sort_values("mse_ratio", ascending=False).iloc[0]
    if arguments.explain_model and arguments.explain_ref and arguments.explain_quarter:
        how = "as requested"
    return row["model"], row["ref"], row["quarter"], how


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging-dir", type=Path, default=RESULTS_STAGING_DIR)
    parser.add_argument("--models", help="comma-separated label prefixes to compare, e.g. 'har_ols[har_rv,lstm[base'")
    parser.add_argument("--exclude", help="comma-separated label prefixes to leave out")
    parser.add_argument("--benchmark", default=None, help=f"primary benchmark label prefix (default {DEFAULT_BENCHMARKS[0]})")
    parser.add_argument("--freq", default="Q", help="calendar period for the split: Q (default), M, ...")
    parser.add_argument("--min-n", type=int, default=15, help="minimum events in a quarter to select it for the diagnosis")
    parser.add_argument("--snapshot-id", type=int, default=None, help="only runs trained on this gold snapshot")
    parser.add_argument("--explain-model", default=None, help="model to diagnose (default: most severe reversal vs a benchmark)")
    parser.add_argument("--explain-ref", default=None, help="reference model (default: the primary benchmark)")
    parser.add_argument("--explain-quarter", default=None, help="quarter to diagnose, e.g. 2024Q2")
    parser.add_argument("--no-gold", action="store_true", help="do not join regime covariates from the gold table")
    parser.add_argument("--out-dir", type=Path, default=QUARTERLY_OUTPUT_DIR)
    arguments = parser.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    arguments.out_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions([arguments.staging_dir, arguments.staging_dir / RESULTS_PUBLISHED_DIR.name], arguments.snapshot_id)
    split = lambda text: [item.strip() for item in text.split(",") if item.strip()] if text else None
    panel, labels = build_matched_panel(predictions, split(arguments.models), split(arguments.exclude), arguments.freq)
    if panel["quarter"].nunique() < 2:
        raise SystemExit("Only one period in the matched sample - nothing to compare across quarters.")

    requested = [arguments.benchmark] if arguments.benchmark else DEFAULT_BENCHMARKS
    benchmarks = [label for label in (resolve_label(token, labels) for token in requested) if label]
    if arguments.benchmark and not benchmarks:
        raise SystemExit(f"Benchmark {arguments.benchmark!r} not among {labels}")
    if not benchmarks:
        print("WARNING: no default benchmark (naive_mean, har_ols) among the models; using the first label.")
        benchmarks = [labels[0]]
    benchmark = benchmarks[0]
    print(f"Primary benchmark: {benchmark}; all benchmarks: {benchmarks}")

    metrics = quarterly_metrics(panel, labels)
    ratios = pairwise_ratios(panel, labels)
    metrics.to_csv(arguments.out_dir / "quarterly_metrics.csv", index=False)
    ratios.to_csv(arguments.out_dir / "quarterly_pairwise.csv", index=False)
    ratios[ratios["ref"].isin(benchmarks)].to_csv(arguments.out_dir / "quarterly_vs_benchmark.csv", index=False)
    metrics.pivot(index="quarter", columns="model", values="mse").rank(axis=1, method="min").to_csv(arguments.out_dir / "quarterly_rank.csv")
    all_reversals = ratios[ratios["reversal_mse"] | ratios["reversal_mae"]].copy()
    all_reversals["severity"] = np.log(all_reversals["mse_ratio"])
    all_reversals.sort_values("severity", ascending=False).to_csv(arguments.out_dir / "reversals.csv", index=False)

    overall = pd.DataFrame({"mse": event_losses(panel, labels, "se").mean(), "mae": event_losses(panel, labels, "ae").mean()}).sort_values("mse")
    overall["mse_ratio_vs_bench"] = overall["mse"] / overall.loc[benchmark, "mse"]
    mse_by_quarter = metrics.pivot(index="quarter", columns="model", values="mse")
    print(f"\nOverall (matched sample, n = {len(panel):,}, {panel['quarter'].nunique()} quarters):")
    print(overall.round(4).to_string())
    best_per_quarter = metrics.loc[metrics.groupby("quarter")["mse"].idxmin(), ["quarter", "model", "n", "mse"]]
    print("\nBest model per quarter (MSE):")
    print(best_per_quarter.round(4).to_string(index=False))
    reversal_columns = ["quarter", "model", "ref", "n", "mse_ratio", "overall_mse_ratio"]
    benchmark_reversals = ratios[ratios["ref"].isin(benchmarks) & ratios["reversal_mse"]].sort_values("mse_ratio", ascending=False)
    print("\nReversal quarters (better overall, worse in the quarter) vs benchmarks, MSE:")
    print(benchmark_reversals[reversal_columns].round(3).head(15).to_string(index=False) if len(benchmark_reversals) else "  none")

    model, reference, quarter, how = choose_case(ratios, benchmarks, benchmark, arguments, labels)
    print(f"\nDiagnosing the {how}: {model} vs {reference} in {quarter}")
    panel_with_covariates, covariates = attach_gold_covariates(panel, arguments.no_gold, arguments.snapshot_id)
    ratio_by_quarter = ratios[(ratios["model"] == model) & (ratios["ref"] == reference)].set_index("quarter")["mse_ratio"]
    tables, notes, quarter_events = diagnose(panel_with_covariates, model, reference, quarter, covariates, ratio_by_quarter)
    print(f"\n=== Diagnosis: {model} vs {reference}, {quarter} ===")
    for note in notes:
        print(" - " + note)
    for name, table in tables.items():
        table.to_csv(arguments.out_dir / f"diagnosis_{name}.csv", index=False)
    quarter_events.to_csv(arguments.out_dir / "diagnosis_events.csv", index=False)

    report = [f"# Quarterly evaluation ({panel['quarter'].iloc[0]} - {panel['quarter'].iloc[-1]})", "",
              f"Matched sample: {len(panel):,} events, {panel['quarter'].nunique()} periods, {len(labels)} models. Models are refit "
              f"annually (expanding window); errors are split by calendar period afterwards. Primary benchmark: `{benchmark}`.", "",
              "Overall", markdown_table(overall.reset_index().rename(columns={"index": "model"})), "",
              "MSE by quarter", markdown_table(mse_by_quarter[labels].reset_index()), "",
              "Best model per quarter (MSE)", markdown_table(best_per_quarter), "",
              "Reversals against the benchmarks (better overall, worse in the quarter)",
              markdown_table(benchmark_reversals[reversal_columns]) if len(benchmark_reversals) else "None.", "",
              f"Diagnosis: `{model}` vs `{reference}` in {quarter}", *[f"- {note}" for note in notes], "",
              "Leave-worst-events-out", markdown_table(tables["leave_out"]), "",
              "Event type", markdown_table(tables["event_type"]), "",
              "Bias and calibration", markdown_table(tables["bias"]), ""]
    if len(tables.get("regime", [])):
        report += ["Regime (quarter vs other quarters)", markdown_table(tables["regime"]), ""]
    if len(tables.get("cross_quarter", [])):
        report += ["Across quarters: relative performance vs market conditions", markdown_table(tables["cross_quarter"]), ""]
    (arguments.out_dir / "quarterly_report.md").write_text("\n".join(report))
    print(f"\nWrote tables and report to {arguments.out_dir}")


if __name__ == "__main__":
    main()

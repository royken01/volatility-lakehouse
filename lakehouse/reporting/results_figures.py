"""Figures for the results chapter, built from the out-of-sample predictions that train.py stages.

All models are scored on the SAME events with the squared error on the LOG scale (ln y against ln
y_hat, no back-transformation).  Variants: A = tier A alone, C0 = A + mean error (placebo), C = A +
news correction; the naive mean and HAR-RV are benchmarks with variant A only.  The reference model
(HAR-RV by default) is the zero line of the gain figure.

    fig2_gain_by_quarter   cumulative squared-error gain over the reference, one line per model
    fig3_mse_by_variant    pooled MSE of A, C0 and C per model
    fig4_residual_qq       normal QQ plots of the standardised residuals (volatility and log scale)

plus ``results_summary.md``, ``results_losses.csv`` and ``residual_normality.csv``.  Residuals are
those of the headline variant (``--variant``, benchmarks: A).

    python -m lakehouse.reporting.results_figures                       # latest staged run of every model
    python -m lakehouse.reporting.results_figures --feature-set base --models har_ols arimax rf xgb lstm naive_mean
    python -m lakehouse.reporting.results_figures --reference naive_mean
    python -m lakehouse.reporting.results_figures --runs <mlflow_run_id> <run_id> ...

Output: ``lakehouse/outputs/results_figures/`` (``--out-dir`` to change).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from lakehouse.modelling.model_params import BENCHMARK_MODELS
from lakehouse.paths import RESULTS_FIGURES_OUTPUT_DIR
from lakehouse.plot_style import (
    AQUA, BLUE, GREEN, MAGENTA, ORANGE, SECONDARY_INK, SURFACE, YELLOW, apply_default_style, clean_axes, save_figure)
from lakehouse.reporting.staged_predictions import (
    MODEL_ORDER, add_selection_arguments, display_name, headline_rows, load_scored_events, model_order, news_models,
    output_directory, pick_reference)

VARIANT_COLOR = {"A": BLUE, "C0": ORANGE, "C": AQUA}
VARIANT_NAME = {"A": "A: alone", "C0": "C₀:  A + mean error (recalibration only)", "C": "C:  A + news correction"}
# Fixed identity per model (never re-assigned when a subset is plotted) + a marker shape as second encoding.
MODEL_COLOR = dict(zip(MODEL_ORDER, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]))
MODEL_MARKER = dict(zip(MODEL_ORDER, ["o", "s", "^", "D", "v", "P"]))


def mean_mse(scored: pd.DataFrame, by=()) -> pd.DataFrame:
    """Mean squared error, models (and ``by`` columns) as rows, variants as columns."""
    return scored.pivot_table(index=["model", *by], columns="variant", values="mse", aggfunc="mean")


def quarterly_mse(scored: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Model x quarter mean squared error of each model's headline variant (small quarters left out)."""
    rows = headline_rows(scored, variant)
    rows = rows[~rows["quarter"].isin(scored.attrs.get("small_quarters", []))]
    return rows.pivot_table(index="model", columns="quarter", values="mse", aggfunc="mean")


def figure_gain_by_quarter(scored, order, reference, variant, path, dpi) -> None:
    """Cumulative squared-error gain over the reference model: the running sum of (reference error
    - model error) in time order."""
    rows = headline_rows(scored, variant)
    squared = rows.pivot_table(index="ts_utc", columns="model", values="mse", aggfunc="first").sort_index()
    models = [model for model in order if model != reference]
    gain = (-squared[models]).add(squared[reference], axis=0).cumsum()
    times = squared.index.tz_convert(None)
    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    axis.axhline(0, color=SECONDARY_INK, lw=1.2, zorder=1)
    for model in models:
        axis.plot(times, gain[model], color=MODEL_COLOR.get(model, "#6b6b6b"), lw=1.9, label=display_name(model), zorder=3)
    axis.set_ylabel(f"Cumulative squared-error gain over {display_name(reference)}\n")
    clean_axes(axis, grid="y")
    axis.legend(loc="upper left", ncol=2, fontsize=9)
    axis.set_title(f"Cumulative squared-error gain over {display_name(reference)}")
    figure.tight_layout()
    save_figure(figure, path, dpi)


def figure_mse_bars(scored, order, reference, path, dpi) -> None:
    """Grouped bars of the pooled MSE: one group per model, one bar per variant; the dashed line is
    the reference model's MSE."""
    variants = [v for v in ("A", "C0", "C") if v in set(scored["variant"])]
    table = mean_mse(scored).reindex(index=order, columns=variants)
    maximum = np.nanmax(table.to_numpy())
    unit = 10.0 ** np.floor(np.log10(maximum))
    exponent = int(np.log10(unit))
    bar_width = 0.26
    figure, axis = plt.subplots(figsize=(1.9 * len(order) + 2.5, 5.2))
    for row, model in enumerate(order):
        present = [v for v in variants if np.isfinite(table.loc[model, v])]
        for position, variant in enumerate(present):
            x = row + (position - (len(present) - 1) / 2) * (bar_width + 0.02)
            value = table.loc[model, variant] / unit
            axis.bar(x, value, width=bar_width, color=VARIANT_COLOR[variant], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            label = f"{value:.3f}"
            if variant != "A" and "A" in present:
                label += f"\n{100 * (table.loc[model, variant] / table.loc[model, 'A'] - 1):+.1f}%".replace("-", "−")
            axis.text(x, value + 0.012 * maximum / unit, label, ha="center", va="bottom", fontsize=7.5,
                      color=SECONDARY_INK, linespacing=1.1)
    reference_value = table.loc[reference, "A"] / unit
    axis.axhline(reference_value, color=SECONDARY_INK, lw=1, ls=(0, (4, 3)), zorder=2)
    axis.text(len(order) - 0.45, reference_value, f"{display_name(reference)} (A)", ha="right", va="bottom", fontsize=8,
              color=SECONDARY_INK)
    axis.set_xticks(range(len(order)), [display_name(m) for m in order])
    axis.set_xlim(-0.6, len(order) - 0.4)
    axis.set_ylim(0, maximum / unit * 1.18)
    axis.set_ylabel("Pooled out-of-sample MSE" + (f" (×10$^{{{exponent}}}$)" if exponent else "")
                    + ", log scale")
    clean_axes(axis, grid="y")
    figure.legend(handles=[Patch(facecolor=VARIANT_COLOR[v], label=VARIANT_NAME[v]) for v in variants],
                  loc="lower center", ncol=len(variants), bbox_to_anchor=(0.5, -0.02))
    figure.suptitle("Pooled MSE of every model and variant", fontsize=14)
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(figure, path, dpi)


SCALE_LABEL = {"raw": "Volatility scale:  y − ŷ", "log": "Log scale:  ln y − ln ŷ"}


def residual_summary(residuals: np.ndarray) -> dict:
    """Skewness, excess kurtosis and a two-sided one-sample t-test of the mean residual.

        The t-test assumes independent, roughly normal residuals, which the QQ plot and the moments
        check.
    """
    count = len(residuals)
    mean, deviation = residuals.mean(), residuals.std(ddof=0)
    standardised = (residuals - mean) / deviation
    standard_error = float(residuals.std(ddof=1) / np.sqrt(count)) if count > 1 else np.nan
    t_statistic = float(mean / standard_error) if standard_error > 0 else np.nan
    summary = {"n": count, "mean": mean, "std": deviation, "t_stat": t_statistic, "t_p": np.nan,
               "skewness": float(np.mean(standardised ** 3)), "excess_kurtosis": float(np.mean(standardised ** 4) - 3.0)}
    try:
        from scipy import stats
        if np.isfinite(t_statistic):
            summary["t_p"] = float(2 * stats.t.sf(abs(t_statistic), df=count - 1))
    except ImportError:
        pass
    return summary


def model_residuals(scored, variant, order) -> dict[str, dict[str, np.ndarray]]:
    """{model: {"raw": y - y_hat, "log": ln y - ln y_hat}} for the headline variant of each model in ``order``."""
    headline = headline_rows(scored, variant)
    rows = {model: headline[headline["model"] == model] for model in order}
    return {model: {"raw": (group["y_true_volatility"] - group["y_pred_volatility"]).to_numpy(float),
                    "log": (group["y_true_log"] - group["y_pred_log"]).to_numpy(float)} for model, group in rows.items()}


def figure_residual_qq(scored, order, variant, path, dpi) -> None:
    """Normal QQ plots (left: volatility scale, right: log scale), one line per model; residuals
    standardised, axes clipped at +-4."""
    from statistics import NormalDist
    inverse_cdf = NormalDist().inv_cdf
    residuals = model_residuals(scored, variant, order)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))
    for column, scale in enumerate(("raw", "log")):
        axis = axes[column]
        for model in order:
            values = np.sort(residuals[model][scale])
            standardised = (values - values.mean()) / values.std(ddof=0)
            positions = (np.arange(1, len(values) + 1) - 0.5) / len(values)
            theoretical = np.array([inverse_cdf(p) for p in positions])
            axis.plot(theoretical, standardised, color=MODEL_COLOR.get(model, SECONDARY_INK), lw=1.6,
                      marker=MODEL_MARKER.get(model, "o"), markevery=max(1, len(values) // 12), markersize=5,
                      markeredgecolor=SURFACE, label=display_name(model), zorder=3)
        axis.plot([-4, 4], [-4, 4], color=SECONDARY_INK, lw=1, ls=(0, (4, 3)), zorder=2)
        axis.set_xlim(-4, 4)
        axis.set_ylim(-4, 4)
        axis.set_aspect("equal")
        axis.set_xlabel("Normal quantile")
        axis.set_ylabel("Standardised residual quantile")
        axis.set_title(SCALE_LABEL[scale], fontsize=11)
        clean_axes(axis, grid="both")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(order), bbox_to_anchor=(0.5, -0.01))
    figure.suptitle("Normal QQ plot of the residuals (on the dashed line = normal)", fontsize=14)
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    save_figure(figure, path, dpi)


def write_residual_statistics(scored, order, variant, directory: Path) -> None:
    """residual_normality.csv: one row per model and scale with the moments and the t-test of the
    mean residual."""
    residuals = model_residuals(scored, variant, order)
    headline = headline_rows(scored, variant)
    rows = []
    for model in order:
        chosen = headline.loc[headline["model"] == model, "variant"].iloc[0]
        for scale in ("raw", "log"):
            rows.append({"model": model, "variant": chosen, "scale": scale, **residual_summary(residuals[model][scale])})
    table = pd.DataFrame(rows)
    table.to_csv(directory / "residual_normality.csv", index=False)
    print("\nResidual t-test (H0: mean residual = 0), skewness and excess kurtosis (both 0 for a normal distribution):")
    print(table.round({"mean": 4, "std": 4, "t_stat": 2, "t_p": 4, "skewness": 2, "excess_kurtosis": 2}).to_string(index=False))


def news_change(scored, model, against):
    """Pooled % change in MSE of C relative to ``against`` (negative = lower loss with news)."""
    rows = scored[scored["model"] == model]
    return 100 * (rows[rows["variant"] == "C"]["mse"].mean() / rows[rows["variant"] == against]["mse"].mean() - 1)


def write_summary(scored, variant, order, news_order, reference, directory: Path) -> None:
    """results_summary.md: ranking, where the overall best model loses, effect of the news step."""
    lines = ["# Results read-out (generated by results_figures.py)", "",
             f"Events scored: {scored.drop_duplicates('ts_utc').shape[0]} "
             f"({scored['ts_utc'].min():%Y-%m-%d} to {scored['ts_utc'].max():%Y-%m-%d}); every model and variant on the "
             f"same events. Lower loss = better. Models with a news stage are shown as variant {variant}; the benchmarks "
             f"({', '.join(display_name(m) for m in BENCHMARK_MODELS if m in order)}) as tier A.", "", "## MSE", ""]
    overall = headline_rows(scored, variant).groupby("model")["mse"].mean().sort_values()
    best = overall.index[0]
    lines.append("Overall ranking: " + " < ".join(f"{display_name(m)} ({100 * overall[m] / overall[reference]:.0f}%)"
                                                   for m in overall.index) + f"  (% of {display_name(reference)}).")
    quarterly = quarterly_mse(scored, variant)
    scale = float(overall[best])   # margins in % of the best model's average MSE (stable when a quarter's loss is ~0)
    margin = (quarterly.loc[best] - quarterly.loc[[m for m in quarterly.index if m != best]]) / scale * 100
    lost = sorted([(q, margin[q].idxmax(), margin[q].max()) for q in margin.columns if (margin[q] > 0).any()],
                  key=lambda item: -item[2])
    if lost:
        lines.append(f"Quarters in which the overall best ({display_name(best)}) does not have the lowest MSE, largest margins first: "
                     + "; ".join(f"{q} by {display_name(m)} ({v:.0f}% of {display_name(best)}'s average MSE)"
                                 for q, m, v in lost[:4]) + ".")
    else:
        lines.append(f"The overall best ({display_name(best)}) has the lowest MSE in every quarter.")
    excess = ((quarterly.loc[best] - quarterly.min(axis=0)) / scale * 100).sort_values(ascending=False)
    worst = [f"{q} (+{v:.0f}%)" for q, v in excess.head(3).items() if v > 0.0]
    if worst:
        lines.append("Its worst quarters relative to that quarter's best model (excess loss, % of its average): "
                     + ", ".join(worst) + ".")
    if news_order:
        lines += ["", "Effect of the news step on MSE (negative = lower loss with news):"]
        for model in news_order:
            pooled_a = news_change(scored, model, "A")
            pooled_c0 = news_change(scored, model, "C0")
            lines.append(f"- {display_name(model)}: C vs A {pooled_a:+.1f}%; C vs C₀ {pooled_c0:+.1f}%")
    lines.append("")
    (directory / "results_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def write_tidy_csv(scored, directory: Path) -> None:
    """results_losses.csv: model, variant, level (overall / year / quarter), period, n, MSE - one number per row."""
    tables = []
    for level, by in (("overall", []), ("year", ["year"]), ("quarter", ["quarter"])):
        losses = mean_mse(scored, by)
        counts = scored.drop_duplicates(["model", "variant", *by, "ts_utc"]).groupby(["model", "variant", *by]).size()
        long = losses.stack().rename("value").reset_index()
        long["level"], long["loss"] = level, "MSE"
        long["period"] = long[by[0]] if by else "all"
        long["n"] = [counts.get((row.model, row.variant, *[getattr(row, b) for b in by]), np.nan) for row in long.itertuples()]
        tables.append(long[["model", "variant", "loss", "level", "period", "n", "value"]])
    pd.concat(tables).to_csv(directory / "results_losses.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_selection_arguments(parser)
    parser.add_argument("--dpi", type=int, default=200)
    arguments = parser.parse_args()

    apply_default_style()
    scored = load_scored_events(arguments)
    order = model_order(scored, arguments.variant)
    news_order = news_models(scored, order)
    reference = pick_reference(scored, order, arguments.reference)
    print(f"{len(order)} models: {order}; with a news stage: {news_order}; reference {reference}")
    directory = output_directory(arguments, RESULTS_FIGURES_OUTPUT_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    figure_gain_by_quarter(scored, order, reference, arguments.variant, directory / "fig2_gain_by_quarter.png", arguments.dpi)
    figure_mse_bars(scored, order, reference, directory / "fig3_mse_by_variant.png", arguments.dpi)
    figure_residual_qq(scored, order, arguments.variant, directory / "fig4_residual_qq.png", arguments.dpi)
    write_residual_statistics(scored, order, arguments.variant, directory)
    write_tidy_csv(scored, directory)
    write_summary(scored, arguments.variant, order, news_order, reference, directory)
    print(f"\nFigures and tables in {directory}")


if __name__ == "__main__":
    main()

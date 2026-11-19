"""vol-lakehouse: forecast the realized volatility after scheduled releases, for any asset and window.

    These commands run the pipeline modules; ``python -m lakehouse <command>`` is the same.

Framework options (before or after the command; see ``lakehouse/settings.py``):
    --config FILE               TOML settings (asset, windows, event types, clocks, news); see configs/
    --asset NAME                label of the underlying asset
    --post-window-seconds N     target = realized volatility in [release, release + N s]
    --lags N                    number of previous same-type releases used as lags
    --tick-window-seconds N     ticks kept around each release in silver (>= post window)
    --pre-window-minutes N      length of the pre-release volatility window
    --event-type NAME=LABEL     a release type to model (repeatable; LABEL* marks the reference type)
    --news-keywords "a,b,c"     topic keywords for the GDELT headline filter
    --sentiment-model MODEL     crudebert or a Hugging Face model id (e.g. ProsusAI/finbert)
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys

COMMANDS: dict[str, dict[str, tuple[str, str]]] = {
    "data collection": {
        "calendar": ("lakehouse.sources.economic_calendar", "release calendar from FRED / ForexFactory -> events CSV"),
        "gdelt": ("lakehouse.sources.gdelt_collector", "GDELT headlines (backfill / recent days) -> headlines CSV"),
    },
    "data pipeline": {
        "bronze": ("lakehouse.medallion.bronze.bronze_ingest", "raw files -> bronze tables"),
        "silver": ("lakehouse.medallion.silver.silver_transform", "bronze -> silver (UTC, cleaning, tick windows, sentiment)"),
        "gold": ("lakehouse.medallion.gold.gold_features", "silver -> gold.events_features"),
        "score-headlines": ("lakehouse.news.score_headlines", "score a headline csv with the sentiment model"),
    },
    "models": {
        "mlflow": ("lakehouse.infrastructure.mlflow_server", "start the MLflow tracking server"),
        "train": ("lakehouse.modelling.train", "walk-forward training, staging, registry, --publish"),
        "select-features": ("lakehouse.modelling.select_features", "keep / weak / drop verdict for candidate news features"),
        "publish": ("lakehouse.medallion.serving.publish_results", "validate and publish the staged results to gold"),
        "powerbi": ("lakehouse.medallion.serving.export_to_powerbi", "parquet export for Power BI"),
    },
    "results": {
        "figures": ("lakehouse.reporting.results_figures", "results figures"),
        "window-study": ("lakehouse.analysis.window_study", "accuracy for several post-event windows (30 s, 1 min, 5 min)"),
        "quarterly": ("lakehouse.reporting.quarterly_analysis", "quarterly MSE ratios and reversal diagnosis"),
        "regime": ("lakehouse.reporting.regime_figure", "price and volatility by quarter"),
        "regime-inputs": ("lakehouse.reporting.regime_inputs_figure", "forecasts and model inputs by quarter"),
    },
    "exploratory analysis": {
        "descriptive": ("lakehouse.analysis.descriptive_plots", "target distribution, event volatility"),
        "tick-vs-minute": ("lakehouse.analysis.tick_vs_minute_volatility", "tick vs 1-minute realized volatility"),
        "correlation": ("lakehouse.analysis.feature_correlation", "Spearman correlations of the features"),
        "diagnostics": ("lakehouse.analysis.evaluation_diagnostics", "valid / invalid target events"),
        "arima-order": ("lakehouse.analysis.arima_order_identification", "Box-Jenkins order identification"),
        "surprise-reaction": ("lakehouse.analysis.surprise_reaction", "volatility reaction vs release surprise"),
        "surprise-volatility": ("lakehouse.analysis.surprise_volatility", "volatility around the release, by surprise"),
    },
}
MODULES = {command: module for group in COMMANDS.values() for command, (module, _) in group.items()}

# --flag -> environment variable read by lakehouse/settings.py (child processes inherit it)
FRAMEWORK_FLAGS = {
    "--config": "LAKEHOUSE_CONFIG",
    "--asset": "LAKEHOUSE_ASSET",
    "--post-window-seconds": "LAKEHOUSE_POST_WINDOW_SECONDS",
    "--lags": "LAKEHOUSE_LAGS",
    "--tick-window-seconds": "LAKEHOUSE_TICK_WINDOW_SECONDS",
    "--pre-window-minutes": "LAKEHOUSE_PRE_WINDOW_MINUTES",
    "--news-keywords": "LAKEHOUSE_NEWS_KEYWORDS",
    "--sentiment-model": "LAKEHOUSE_SENTIMENT_MODEL",
    "--event-type": "LAKEHOUSE_EVENT_TYPES",
}


def apply_framework_options(arguments: list[str]) -> list[str]:
    """Take the framework options out of ``arguments`` (anywhere in the list), export them as
    ``LAKEHOUSE_*`` environment variables and return the remaining arguments."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for flag in FRAMEWORK_FLAGS:
        parser.add_argument(flag, action="append" if flag == "--event-type" else "store")
    known, remaining = parser.parse_known_args(arguments)
    for flag, variable in FRAMEWORK_FLAGS.items():
        value = getattr(known, flag.lstrip("-").replace("-", "_"))
        if value is None:
            continue
        if flag == "--config":
            value = os.path.abspath(os.path.expanduser(value))
        os.environ[variable] = "|".join(value) if isinstance(value, list) else str(value)
    if any(getattr(known, flag.lstrip("-").replace("-", "_")) is not None for flag in FRAMEWORK_FLAGS):
        from lakehouse import settings

        settings.get_settings.cache_clear()
    return remaining


def print_settings() -> int:
    from lakehouse.settings import get_settings

    settings = get_settings()
    for name, value in settings.describe().items():
        print(f"  {name:<26}{value}")
    print(f"  {'target column':<26}{settings.target_column}")
    print(f"  {'lag columns':<26}{settings.lag_prefix}1..{settings.lags}")
    print(f"  {'event dummies':<26}{', '.join(settings.model_event_dummies)}")
    for event_type in settings.resolved_event_types:
        print(f"    {event_type.label:<8}{'(reference) ' if event_type.reference else ''}{event_type.name}")
    return 0


def print_help() -> None:
    print(__doc__.split("\n\n")[0], "\n")
    print("commands:")
    for group, commands in COMMANDS.items():
        print(f"  {group}")
        for command, (_, description) in commands.items():
            print(f"    {command:<20}{description}")
    print("  everything at once")
    print(f"    {'pipeline':<20}[gdelt doc] -> bronze -> silver -> gold -> train -> publish (vol-lakehouse pipeline --help)")
    print(f"    {'settings':<20}print the effective framework settings")
    print("\nOptions of a command: vol-lakehouse <command> --help")
    print(__doc__[__doc__.index("Framework options"):])


INGEST_STAGES = ("bronze", "silver", "gold")


def run_stage(*command: str) -> None:
    print(f"\n=== vol-lakehouse {' '.join(command)} ===", flush=True)
    subprocess.run([sys.executable, "-m", "lakehouse", *command], check=True)


def run_pipeline(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="vol-lakehouse pipeline", description="Run the data stages, then train every "
                                     "model and publish the results (the same steps as the Airflow DAG, without Airflow).")
    parser.add_argument("--collect-news", metavar="DAYS", type=int, default=0,
                        help="first top up the GDELT headlines with the last DAYS days (DOC API)")
    parser.add_argument("--skip-ingest", action="store_true", help="do not rebuild bronze / silver / gold")
    parser.add_argument("--skip-train", action="store_true", help="only the data stages")
    parser.add_argument("--no-register", action="store_true", help="do not register the refit models in MLflow")
    parser.add_argument("--powerbi", action="store_true", help="also export the Power BI files at the end")
    options = parser.parse_args(arguments)
    try:
        if options.collect_news:
            run_stage("gdelt", "doc", "--days", str(options.collect_news))
        if not options.skip_ingest:
            for stage in INGEST_STAGES:
                run_stage(stage)
        if not options.skip_train:
            run_stage("publish", "--archive-staged")  # results of an earlier unpublished run must not mix with this one
            run_stage("train", "--model", "all", *([] if options.no_register else ["--register"]), "--publish")
        if options.powerbi:
            run_stage("powerbi")
    except subprocess.CalledProcessError as error:
        print(f"\nPipeline stopped: {' '.join(map(str, error.cmd[3:]))} failed (exit code {error.returncode}).")
        return error.returncode
    return 0


def main(arguments: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if arguments is None else list(arguments)
    arguments = apply_framework_options(arguments)
    if not arguments or arguments[0] in ("-h", "--help", "help"):
        print_help()
        return 0
    command, rest = arguments[0], arguments[1:]
    if command == "pipeline":
        return run_pipeline(rest)
    if command == "settings":
        return print_settings()
    if command not in MODULES:
        print(f"vol-lakehouse: unknown command {command!r}\n", file=sys.stderr)
        print_help()
        return 2
    module = importlib.import_module(MODULES[command])
    sys.argv = [f"vol-lakehouse {command}", *rest]
    result = getattr(module, "cli", module.main)()
    return result if isinstance(result, int) else 0


def pipeline() -> int:
    """Console command ``vol-pipeline`` = ``vol-lakehouse pipeline``."""
    return main(["pipeline", *sys.argv[1:]])


def train() -> int:
    """Console command ``vol-train`` = ``vol-lakehouse train``."""
    return main(["train", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())

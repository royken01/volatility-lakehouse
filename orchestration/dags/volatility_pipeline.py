"""
End-to-end volatility pipeline (Apache Airflow 3) for any asset / set of scheduled releases.

The study (asset, post-event window, event types, news keywords) is the framework settings file
in LAKEHOUSE_CONFIG (see configs/); every task reads the same file.

    collect_news -> bronze_ingest -> silver_transform -> gold_features ─┐
    check_mlflow -> archive_staging ─────────────────────┤
                                                         ├─> train_naive_mean ─────┐
                                                         ├─> train_har_ols ────────┤
                                                         ├─> train_arimax ─────────┼
                                                         ├─> train_rf ─────────────┤
                                                         ├─> train_xgb ────────────┤
                                                         └─> train_lstm ───────────┘─> publish_results -> export_to_powerbi

Each task shells out to the project's own Python (LAKEHOUSE_PYTHON) and runs one command of the ``vol-lakehouse``
CLI (``python -m lakehouse <command>``, see lakehouse/cli.py), so Airflow's virtualenv never needs Spark,
torch, MLflow etc. installed - a common production pattern that keeps the orchestrator and the workloads
decoupled.  The project environment must have the package installed (``pip install -e ".[all]"``).

Environment variables (set by orchestration/start_airflow.sh):
    LAKEHOUSE_PROJECT_DIR   path to the project folder (the one with pyproject.toml and lakehouse/)
    LAKEHOUSE_PYTHON        python executable of the project environment
    LAKEHOUSE_CONFIG        framework settings file (optional; defaults to <workspace>/framework.toml)
"""
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param

PROJECT_DIR = os.environ.get("LAKEHOUSE_PROJECT_DIR", str(Path(__file__).resolve().parents[2]))
PYTHON = os.environ.get("LAKEHOUSE_PYTHON", "python3")
MODELS = ["naive_mean", "har_ols", "arimax", "rf", "xgb", "lstm"]


def run_command(command: str, arguments: str = "") -> str:
    """Shell command that runs ``vol-lakehouse <command>`` (``python -m lakehouse <command>``) from the project folder."""
    return f'cd "{PROJECT_DIR}" && "{PYTHON}" -m lakehouse {command} {arguments}'.strip()


with DAG(
    dag_id="volatility_pipeline",
    description="GDELT news -> bronze -> silver -> gold -> train/evaluate (MLflow) -> publish -> Power BI export",
    start_date=datetime(2026, 1, 1),
    # Manual trigger. For a refresh after a weekly release use a cron string, e.g. "0 20 * * 3"
    # (Wednesdays 20:00 UTC, after the EIA crude inventory report).
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "roy", "retries": 1, "retry_delay": timedelta(minutes=2)},
    params={
        "register": Param(True, type="boolean",
                          description="Register the refit models in the MLflow Model Registry"),
        "run_ingest": Param(True, type="boolean",
                            description="Re-run bronze/silver/gold before training"),
        "news_days": Param(7, type="integer", minimum=0,
                           description="Top up the GDELT headlines with the last N days first (0 = skip)"),
    },
    tags=["volatility", "lakehouse", "mlflow"],
) as dag:

    # Skips (exit code 99 -> task 'skipped') when run_ingest is unticked in the trigger form.
    skip_if_no_ingest = "{% if not params.run_ingest %}echo 'ingest skipped'; exit 99; {% endif %}"

    collect_news = BashOperator(
        task_id="collect_news",
        bash_command=(skip_if_no_ingest
                      + "{% if params.news_days == 0 %}echo 'news collection skipped'; exit 99; {% endif %}"
                      + run_command("gdelt", "doc --days {{ params.news_days }}")),
        skip_on_exit_code=99,
    )
    bronze = BashOperator(
        task_id="bronze_ingest",
        bash_command=skip_if_no_ingest + run_command("bronze"),
        skip_on_exit_code=99,
        trigger_rule="none_failed",   # runs when news collection was skipped
    )
    silver = BashOperator(
        task_id="silver_transform",
        bash_command=skip_if_no_ingest + run_command("silver"),
        skip_on_exit_code=99,
    )
    gold = BashOperator(
        task_id="gold_features",
        bash_command=skip_if_no_ingest + run_command("gold"),
        skip_on_exit_code=99,
    )

    # Fail fast if the tracking server isn't up, instead of failing every training task.
    check_mlflow = BashOperator(
        task_id="check_mlflow",
        bash_command='curl -sf "${MLFLOW_TRACKING_URI:-http://127.0.0.1:5001}/health"',
    )

    # A full run retrains every model, so staged results of an earlier run that was never published are obsolete.
    archive_staging = BashOperator(
        task_id="archive_staging",
        bash_command=run_command("publish", "--archive-staged"),
    )

    train_tasks = [
        BashOperator(
            task_id=f"train_{model_name}",
            bash_command=run_command(
                "train",
                f"--model {model_name} --tag airflow_run_id={{{{ run_id }}}} "
                "{{ '--register' if params.register else '' }}",
            ),
            # Run training even if ingest was skipped, but not if it failed.
            trigger_rule="none_failed",
        )
        for model_name in MODELS
    ]

    publish = BashOperator(
        task_id="publish_results",
        # --replace: the Iceberg tables become exactly this run's models (never a mix of gold snapshots).
        bash_command=run_command("publish", "--replace"),
        # Replacing with a partial run would drop the failed models, so publish only when every training succeeded.
        trigger_rule="all_success",
    )
    export_pbi = BashOperator(
        task_id="export_to_powerbi",
        bash_command=run_command("powerbi"),
    )

    collect_news >> bronze >> silver >> gold >> train_tasks
    check_mlflow >> archive_staging >> train_tasks
    train_tasks >> publish >> export_pbi

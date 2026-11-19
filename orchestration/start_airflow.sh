#!/usr/bin/env bash
# Starts Airflow 3 in standalone mode (scheduler + API server/UI + DAG
# processor + triggerer in one process, SQLite metadata db) for local use.
#
# Usage, with the Airflow virtualenv activated (the project environment needs `pip install -e ".[all]"`):
#   export LAKEHOUSE_PYTHON=/path/to/the/project/.venv/bin/python     # once, e.g. in ~/.zshrc
#   export LAKEHOUSE_CONFIG=/path/to/configs/your_study.toml          # optional: which asset / windows / events
#   ./orchestration/start_airflow.sh
# then open http://localhost:8080
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$HERE")"

LAKEHOUSE_PYTHON="${LAKEHOUSE_PYTHON:-${CRUDE_OIL_PYTHON:-}}"
if [[ -z "$LAKEHOUSE_PYTHON" ]]; then
  echo "Set LAKEHOUSE_PYTHON to the python of the project environment first, e.g."
  echo "  export LAKEHOUSE_PYTHON=$PROJECT_DIR/.venv/bin/python"
  exit 1
fi
export LAKEHOUSE_PYTHON

if ! (cd "$PROJECT_DIR" && "$LAKEHOUSE_PYTHON" -c "import lakehouse.cli, mlflow, pyspark, sklearn" 2>/dev/null); then
  echo "$LAKEHOUSE_PYTHON cannot import lakehouse / mlflow / pyspark / sklearn. Install the package once from $PROJECT_DIR:"
  echo "  \"$LAKEHOUSE_PYTHON\" -m pip install -e \".[all]\""
  exit 1
fi
if [[ -n "${LAKEHOUSE_CONFIG:-}" ]]; then
  export LAKEHOUSE_CONFIG="$(cd "$(dirname "$LAKEHOUSE_CONFIG")" && pwd)/$(basename "$LAKEHOUSE_CONFIG")"
  echo "Framework settings: $LAKEHOUSE_CONFIG"
fi

export AIRFLOW_HOME="${AIRFLOW_HOME:-$HOME/airflow}"
export AIRFLOW__CORE__DAGS_FOLDER="$HERE/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
# Local-only convenience: no login screen. Remove for anything shared.
export AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_ALL_ADMINS="True"
export LAKEHOUSE_PROJECT_DIR="$PROJECT_DIR"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5001}"
# macOS: avoids tasks crashing (SIGSEGV) when forked workers do proxy lookups.
export NO_PROXY="*"
export OBJC_DISABLE_INITIALIZATION_SAFETY="YES"

exec airflow standalone

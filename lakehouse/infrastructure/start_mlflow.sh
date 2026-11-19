#!/usr/bin/env bash
# Starts the MLflow tracking server (http://127.0.0.1:5001)
exec python -m lakehouse.infrastructure.mlflow_server "$@"

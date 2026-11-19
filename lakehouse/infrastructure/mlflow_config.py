"""MLflow connection settings shared by every module that logs or reads runs.

The tracking server is started with ``vol-lakehouse mlflow``
(``infrastructure/mlflow_server.py``). Port 5001 is used instead of MLflow's
default 5000 because macOS's AirPlay Receiver already listens on 5000.
"""
import os

from dotenv import load_dotenv

from lakehouse.paths import ENVIRONMENT_FILE

load_dotenv(ENVIRONMENT_FILE)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "event_volatility")

# Registered models are named f"{REGISTERED_MODEL_PREFIX}_{model_name}", e.g. event_vol_rf
# (set MLFLOW_MODEL_PREFIX per asset / study to keep their registries apart).
REGISTERED_MODEL_PREFIX = os.environ.get("MLFLOW_MODEL_PREFIX", "event_vol")


def init_mlflow():
    """Point MLflow at the tracking server and the project experiment.

    Returns the ``mlflow`` module.
    """
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    return mlflow
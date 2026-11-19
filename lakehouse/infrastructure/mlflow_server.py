"""Start the MLflow tracking server for the project (``vol-lakehouse mlflow``).

Run metadata goes to the SQLite file ``<workspace>/data/mlflow/mlflow.db``, artifacts to
``s3://$R2_BUCKET/mlflow-artifacts`` on Cloudflare R2.  The server proxies artifact transfers, so
only this process reads the R2 keys (from ``<workspace>/.env``; never printed).  Port 5001
(``MLFLOW_PORT`` to change) because macOS's AirPlay Receiver listens on 5000.
"""
from __future__ import annotations

import os
import sys

from dotenv import dotenv_values

from lakehouse.paths import ENVIRONMENT_FILE, MLFLOW_DIR

REQUIRED_VARIABLES = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET")


def main() -> None:
    settings = {**os.environ, **dotenv_values(ENVIRONMENT_FILE)}
    missing = [name for name in REQUIRED_VARIABLES if not settings.get(name)]
    if missing:
        raise SystemExit(f"Missing {missing} in the environment and in {ENVIRONMENT_FILE}")

    # R2 is S3-compatible: hand MLflow / boto3 the R2 keys and endpoint.
    # Newer boto3 versions send extra checksum headers by default; R2 only
    # wants them when required, hence the two WHEN_REQUIRED switches.
    environment = {
        **settings,
        "AWS_ACCESS_KEY_ID": settings["R2_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": settings["R2_SECRET_ACCESS_KEY"],
        "MLFLOW_S3_ENDPOINT_URL": settings["R2_ENDPOINT"],
        "AWS_DEFAULT_REGION": "auto",
        "AWS_REQUEST_CHECKSUM_CALCULATION": "when_required",
        "AWS_RESPONSE_CHECKSUM_VALIDATION": "when_required",
    }
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    port = settings.get("MLFLOW_PORT", "5001")
    print(f"MLflow server on http://127.0.0.1:{port}  "
          f"(metadata: {MLFLOW_DIR / 'mlflow.db'})", flush=True)
    os.execvpe(sys.executable, [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}",
        "--artifacts-destination", f"s3://{settings['R2_BUCKET']}/mlflow-artifacts",
        "--host", "127.0.0.1", "--port", port,
    ], environment)


if __name__ == "__main__":
    main()
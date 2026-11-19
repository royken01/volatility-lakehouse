"""Connection settings for the Iceberg catalog (Cloudflare R2 Data Catalog).

Read from ``lakehouse/.env``. Every value except ``CATALOG_NAME`` is required:
a missing variable raises ``KeyError`` at import time, which is the quickest
way to find out that the ``.env`` file is incomplete. Copy ``.env.example``
to ``.env`` and fill it in; never commit ``.env``.
"""
import os

from dotenv import load_dotenv

from lakehouse.paths import ENVIRONMENT_FILE

load_dotenv(ENVIRONMENT_FILE)

CATALOG_NAME = os.environ.get("CATALOG_NAME", "volatility_lakehouse")
R2_CATALOG_URI = os.environ["R2_CATALOG_URI"]
R2_CATALOG_WAREHOUSE = os.environ["R2_CATALOG_WAREHOUSE"]
R2_CATALOG_TOKEN = os.environ["R2_CATALOG_TOKEN"]
R2_ENDPOINT = os.environ["R2_ENDPOINT"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
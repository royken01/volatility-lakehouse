"""Every folder the lakehouse code reads from or writes to, in one place.

    PACKAGE_DIR    the code itself
    WORKSPACE_DIR  the data, outputs, ``.env`` and model weights the code works on

WORKSPACE_DIR is, in this order: the folder named by ``LAKEHOUSE_HOME``; the package folder itself
when a ``pyproject.toml`` sits next to it (source checkout, ``pip install -e .``); otherwise
``<current folder>/lakehouse``. Inside it: ``.env``, ``data/`` (raw, news, mlflow,
results_staging), ``outputs/`` and ``news/crudebert_model/``.
"""
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def find_workspace() -> Path:
    """The workspace folder (see the module docstring for the rules)."""
    configured = os.environ.get("LAKEHOUSE_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    if (PACKAGE_DIR.parent / "pyproject.toml").is_file():
        return PACKAGE_DIR
    return Path.cwd().resolve() / "lakehouse"


WORKSPACE_DIR = find_workspace()

DATA_DIR = WORKSPACE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
NEWS_DATA_DIR = DATA_DIR / "news"
MLFLOW_DIR = DATA_DIR / "mlflow"
RESULTS_STAGING_DIR = DATA_DIR / "results_staging"
RESULTS_PUBLISHED_DIR = RESULTS_STAGING_DIR / "published"
CRUDEBERT_MODEL_DIR = WORKSPACE_DIR / "news" / "crudebert_model"
ENVIRONMENT_FILE = WORKSPACE_DIR / ".env"

OUTPUTS_DIR = WORKSPACE_DIR / "outputs"
DESCRIPTIVE_OUTPUT_DIR = OUTPUTS_DIR / "descriptive"
CORRELATION_OUTPUT_DIR = OUTPUTS_DIR / "correlation"
ARIMA_OUTPUT_DIR = OUTPUTS_DIR / "arima_identification"
EVALUATION_DIAGNOSTICS_OUTPUT_DIR = OUTPUTS_DIR / "evaluation_diagnostics"
SURPRISE_REACTION_OUTPUT_DIR = OUTPUTS_DIR / "surprise_reaction"
SURPRISE_VOLATILITY_OUTPUT_DIR = OUTPUTS_DIR / "surprise_volatility"
TICK_VS_MINUTE_OUTPUT_DIR = OUTPUTS_DIR / "tick_vs_minute"
RESULTS_FIGURES_OUTPUT_DIR = OUTPUTS_DIR / "results_figures"
QUARTERLY_OUTPUT_DIR = OUTPUTS_DIR / "quarterly"
POWERBI_EXPORT_DIR = OUTPUTS_DIR / "powerbi_exports"
WINDOW_STUDY_OUTPUT_DIR = OUTPUTS_DIR / "window_study"

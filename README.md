# Volatility Lakehouse

This is an end-to-end datapipeline and ML framework designed to forecast an asset's prices movement surrounding scheduled macroeconomic announcments, utilizing open-source natural language processing models to produce sentiment scores of contextual news information as explanatory variables. This is an extention of the crude-oil specific pipeline from my master thesis which focused on modeling the short-term volatility following EIA Crude Inventory Reports. This extension was developed by actively using GenAI tools in order to implement argument handling for an easier user experience and finding a compatible stack for scraping economic calendar news and similar. The framework now allows for modeling the desired time-frame following a macro event (inventory reports, central-bank descisions, CPI, etc.) for any underlying asset (EUR/USD, Brent Crude Oil, Gold), with integrated logic for importing scheduled event data from an official economic calendar. This is done by providing the following inputs

| Input    | Description | Source |
| -------- | ------- | ------- |    
| **Event calendar**  | release times, event name, forecasts, filter for impact and country | Built-in FRED/ALFRED for historical events, ForexFactory for weekly feed        | 
| **Prices** | tick (bid/ask quotes and trade price) and 1-minute bars | provided as Parquest files (data from HistData.com was used during development)        | 
| **News headlines**    | timestamped headlines on the topic    | GDELT GKG historical backfill post 2019, DOC API for recent news | 

The framework moves these through an Apache Iceberg lakehouse, using a medallion lakehouse hierachy for data processing bronze -> silver -> gold and build pre-release features for model training; volatility terms, lagged release reactions, consensus surprise and news sentiment using selected natural language processing models. It then trains a collection of statistical, machine-learning and deep-learning models in a walk-forward evaluation.

The framework accepts a **config** .toml file containing information about desired asset and time window to perform the volatility forecasting on, as well as options for specifying NLP model for generating news sentiment scores and keywords for scraping relevant headlines from GDELT. A template of how the config files should be structured is found in 'configs/'. 'configs/example_eurusd_us_macro.toml' sets up the 5-minute post-event EUR/USD reaction to US payrolls and CPI announcments.

**Stack:** Python, Apache Spark, Polars, Apache Iceberg, Cloudflare R2, Iceberg REST catalog, Apache Airflow, MLflow, scikit-learn, XGboost, statsmodel, PyTorch, Hugging Face, Transformers, Power BI

## Highlights:

- **Configurables:** asset, post-event window size, number of lags for sequence models such as ARIMAX and LSTM, release types, source clocks, trading session and news topic, all listed in config TOML file
- **Data collection:**
News headlines retrieved in-pipeline through a GDELT collector with keyword/relevance filter. Economic calendar fetched with FRED/ALFRED for historical figures, ForexFactory for upcoming events
- **Medallion lakehouse:**
Data stored in Apache Iceberg tables on Cloudflare R2, managed using Iceberg REST catalog. Ensuring ACID transactions and allows for time travelling using snapshot id's to trace exact gold table version used for each run.
- **Hybrid computing:**
Polars is used for faster table transforming by reading Iceberg through PyIceberg, while Spark is purely used for read/write operations.
- **Clock handling:**
Considers specific time-zone rules for each data source, handles changes from daytime adjusted timestamps and similar (was a specific problem in my thesis data), as well as duplicate removal

- **Tick-level features:**
Short-term volatility estimated from bid/ask mid-quotes to minimise market microstructure noise, lagged same-type reactions.
- **News sentiment:** FinBERT or any NLP model from a Hugging Face repository. Calculates exponential weighted mean average of sentiment scores leading up to scheduled event
- **Leakage-free evaluation:** The model prevents leakage by construction by defining a set of prohibited variables to be set as features, not allowing for model training if this is the case. Models are trained using a yearly expanding-window walk-forward with time-ordered tuning and an embargo. Models are trained in a second stage by introducing news variables for the events covered by headlines, and tests wheter news improves a context-agnostic model against a placebo model with pure recalibration on the average residual error.
- **Models:** naive mean, HAR-RV, ARIMAX, LSTM, Random Forest and XGBoost
- **MLOps:** runs are tracked in MLflow, with artificats in R2 storage. One Airflow 3.0 DAG covers news collection, data ingestion, features production, training, publishing and exporting resulting data for Power BI visualisation.

## Quick start

Requires Python 3.10+ and a Java runtime (for Spark).

```bash
git clone https://github.com/royken01/volatility-lakehouse.git && cd volatility-lakehouse
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
cp lakehouse/.env.example lakehouse/.env
```

The optional extras let you install only what you need: `[lakehouse]` (Spark, PyIceberg), `[models]`
(scikit-learn, XGBoost, statsmodels, PyTorch, MLflow), `[news]` (Transformers) and `[dev]` (pytest).

### 1. Describe the study

Copy one of the files in `configs/` and set the asset, window, releases and news topic:

```toml
asset = "EUR/USD"

[windows]
post_event_seconds = 300     # target: realized volatility in the 5 minutes after each release
lags = 6                     # previous releases of the same type

[events]
clock = "America/New_York"
types = [
    { name = "Nonfarm Payrolls", label = "NFP" },
    { name = "CPI m/m", label = "CPI", reference = true },
]

[news]
sentiment_model = "ProsusAI/finbert"
keywords = ["dollar", "federal reserve", "inflation", "payrolls"]
```

Point the framework at it with `--config FILE` or `export LAKEHOUSE_CONFIG=FILE`. Any setting can also
be overridden per command (`--post-window-seconds 60`, `--event-type "GDP=GDP"`, `--asset ...`).
`vol-lakehouse settings` prints the settings that will be used.

### 2. Collect the inputs

```bash
vol-lakehouse calendar fred --search "employment situation"           # find the release id
vol-lakehouse calendar fred --release-id 50 --event "Nonfarm Payrolls" --series PAYEMS --transform diff \
    --time 08:30 --timezone America/New_York --start 2015-01-01 --include-future
vol-lakehouse calendar forexfactory --country USD --impact High --week next   # upcoming, with consensus
vol-lakehouse gdelt backfill --start 2019-09-22 --every 8 --max-per-file 3    # headline history
vol-lakehouse gdelt doc --days 7
```

Put the price files where `[sources]` points (`lakehouse/data/raw/...`). Ticks need `datetime` plus
`bid`/`ask` (or `price`); 1-minute bars need `datetime` and `close`.

### 3. Run the pipeline

```bash
vol-lakehouse bronze                               # raw files -> bronze
vol-lakehouse silver                               # bronze -> silver (+ sentiment of new headlines)
vol-lakehouse gold                                 # silver -> gold.events_features
vol-lakehouse mlflow                               # tracking server (separate terminal)
vol-lakehouse train --model all --register --publish
vol-lakehouse figures                              # resulting figures
```

Or run everything in one go with `vol-lakehouse pipeline --collect-news 7`. Use
`vol-lakehouse <command> --help` to see the options for each step.

### Run with Airflow

The DAG `volatility_pipeline` runs `collect_news → bronze_ingest → silver_transform → gold_features →
train_<model> (×6) → publish_results → export_to_powerbi`. Airflow lives in its own virtual environment and starts each task in the project environment, so the
ML dependencies stay out of the Airflow install. Requires MLflow to be running before the DAG can start.

```bash
vol-lakehouse mlflow                                           # terminal A
```

```bash
export LAKEHOUSE_PYTHON=$PWD/.venv/bin/python
export LAKEHOUSE_CONFIG=$PWD/configs/your_study.toml
./orchestration/start_airflow.sh                               # terminal B, UI on http://localhost:8080
```

## Method summary

- **Target:** realized volatility in `[release, release + W]` from mid-quote tick returns, modelled
  on the log scale.
- **Features:** strictly pre-release information: HAR-RV terms, the last hour's volatility, the
  reactions to the previous N same-type releases, the consensus forecast, type dummies and news
  sentiment.
- **Evaluation:** yearly expanding-window walk-forward. The primary loss is MSE on the log scale. Models
  are compared with a moving-block bootstrap interval for the % change in MSE against HAR-RV.
- **News effect:** each model's out-of-sample errors are recalibrated with a ridge regression on the
  news features (variant C) and compared with a mean-error placebo (C0). This separates real news
  information from simple bias correction.

## Design decisions

- **Why configuration rather than code per study?** Every study-specific detail (vendor clocks, release
  names, window length, news topic) lives in one reviewable TOML file. That makes a new asset or release
  a config change, rather than requiring any code logic to change and allows for reproducible results.
- **Why R2 + an Iceberg REST catalog?** The project started by using Azure Databricks and Data Lake Storage Gen2, hence the medallion data hiercahy, by relying on the free student credits provided by Microsoft. However, since my thesis focused on the possibilites of developing a framework using free and openly available resources, when the student credits ran out, I moved to an open source stack. Cloudflare R2 has no egress fees and a usable free tier, and Iceberg tables can be read by any engine. I also attempted to use a self-hosted Unity Catalog OSS, but this rejected external Spark writes through its Iceberg REST endpoint.
- **Why GDELT?** It is free, key-free, global, has 15-minute resolution, and stores a timestamp that
  publishers cannot edit afterwards. That timestamp is what makes the no-look-ahead guarantee
  checkable. Scraping commercial news sites would raise terms-of-service and reproducibility problems, most other vendors keep their data behind paywalls.
- **Why walk-forward rather than a random split?** Volatility is serially dependent, and news must never
  see the future. Every forecast uses only information available before the release.

## Data

Raw price data from commercial vendors is **not redistributed** in this repository. GDELT headlines are
public. FRED/ALFRED data needs a free API key. The ForexFactory feed is an unofficial public feed, so
respect its fair-use terms.

## Author

**Roy Ken Felix Duke** · MSc Data Science, University of Oslo · [LinkedIn](https://www.linkedin.com/in/roy-ken-felix-duke/)

## License

MIT (code). Data sources keep their own licences.

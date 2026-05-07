# Energy Forecasting & Anomaly Detection Platform

A local-first Python project for short-term energy load forecasting and anomaly detection from energy and weather time-series CSV files.

The project is designed for practical MLOps workflows: validated local data ingestion, feature engineering, baseline model training, anomaly scoring, evaluation, API serving, containerization, CI, and fast tests based on tiny synthetic inputs.

## Why This Matters

Short-term load forecasts help grid operators, facilities teams, and energy analysts plan capacity, investigate demand changes, and catch unusual consumption patterns before they become operational problems. A local-first setup keeps raw data and model outputs under local control while still providing a repeatable project structure.

## Key Capabilities

- Parse local energy and weather CSV files with schema and quality validation.
- Build calendar, lag, rolling, and weather features for hourly load data.
- Train Ridge and RandomForestRegressor baseline forecasting models.
- Detect unusual behavior with residual z-scores or IsolationForest.
- Evaluate forecasts and anomaly labels when labels are available.
- Save models and metrics to ignored local artifact paths.
- Serve health checks, forecasts, anomaly scoring, and batch predictions through FastAPI.
- Keep tests fast and independent from real datasets or trained model files.

## Quickstart

From the repository root in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m energy_forecasting_anomaly.training.pipeline `
  --input data/raw/energy_weather.csv `
  --models-dir models `
  --metrics-dir reports/metrics `
  --forecast-horizon 24 `
  --model ridge `
  --test-size 0.2 `
  --random-state 42
uvicorn energy_forecasting_anomaly.api.app:app --reload
```

## Common Commands

```powershell
python -m pytest
python -m energy_forecasting_anomaly.training.pipeline --input data/raw/energy_weather.csv
uvicorn energy_forecasting_anomaly.api.app:app --host 0.0.0.0 --port 8000
docker compose up --build
```

## Data Schema

The default parser expects one row per timestamp and zone:

| Column | Type | Description |
| --- | --- | --- |
| `timestamp` | datetime-compatible string | Observation timestamp. |
| `zone` | string | Load zone or meter grouping. |
| `load_mw` | numeric | Observed load in megawatts. |
| `temperature_c` | numeric | Ambient temperature in Celsius. |
| `wind_speed_mps` | numeric | Wind speed in meters per second. |
| `solar_radiation_wm2` | numeric | Solar radiation in watts per square meter. |

Rows are sorted by `zone` and `timestamp`. Missing values, invalid timestamps, non-numeric measurement values, and duplicate `(zone, timestamp)` pairs are rejected.

## Artifact Policy

Runtime outputs are written to ignored paths and are not part of the repository:

- `data/raw/`
- `data/interim/`
- `data/processed/`
- `models/`
- `reports/metrics/`
- `reports/figures/`
- `reports/artifacts/`
- `*.joblib`
- `*.pkl`

No real benchmark results are stored in this scaffold. Run training and evaluation locally to generate metrics for your own data.

## Documentation

- [Architecture](docs/architecture.md)
- [Reproducibility](docs/reproducibility.md)
- [Results](docs/results.md)

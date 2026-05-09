# Energy Forecasting & Anomaly Detection Platform

Local-first Python project for short-term energy load forecasting and anomaly detection from
energy and weather time-series CSV files. The project validates and normalizes local data,
builds time-series features, trains baseline models, evaluates chronological forecasts, compares
anomaly detectors, and serves predictions through FastAPI.

## Why This Matters

Short-term load forecasts help energy teams plan capacity, evaluate demand shifts, and identify
unusual consumption patterns. A local-first workflow keeps raw data, trained models, metrics, and
request payloads on the user's machine while still providing a repeatable MLOps-style structure.

## Key Capabilities

- Generate deterministic synthetic demo energy/weather data with optional anomaly labels.
- Normalize real local CSV exports into the canonical project schema.
- Train Ridge and RandomForestRegressor baseline forecasting models.
- Evaluate forecasts with chronological splits and rolling-origin backtesting.
- Compare residual z-score, robust residual, and IsolationForest anomaly methods.
- Tune anomaly thresholds on validation data before testing held-out rows.
- Generate sample API payload JSON from local CSV files.
- Serve health checks, forecasts, anomaly scoring, and batch predictions with FastAPI.
- Run through Docker, Docker Compose, GitHub Actions CI, and fast synthetic-data tests.

## Results Summary

These are local synthetic demo-data results from `data/raw/demo_energy_weather.csv`, not real grid
benchmark results. Full context is in [Results](docs/results.md).

| Area | Configuration | Documented result |
| --- | --- | --- |
| Forecast split | Ridge, 24-hour horizon, chronological split, 394 test rows | MAE `49.7463`, RMSE `89.4063`, MAPE `4.7236`, R2 `0.2259` |
| Rolling backtest | Ridge, 5 rolling-origin folds | mean MAE `46.1106`, mean RMSE `65.7001`, mean MAPE `4.5168`, mean R2 `0.3317` |
| Anomaly comparison | Residual z-score, robust residual, IsolationForest | Residual methods missed labeled anomalies; IsolationForest reached recall `0.7391` with many false positives |
| Anomaly tuning | Validation-selected settings, held-out test period | Tuned methods did not recover labeled test anomalies, showing non-generalization risk |

## Common Commands

Run from the repository root in Windows PowerShell after creating an environment and installing
the project dependencies.

```powershell
python -m energy_forecasting_anomaly.data.generate_demo_data `
  --output data/raw/demo_energy_weather.csv `
  --start 2025-01-01 `
  --periods 2160 `
  --freq h `
  --zone DE_DEMO `
  --random-state 42 `
  --anomaly-fraction 0.02

python -m energy_forecasting_anomaly.data.normalize_real_csv `
  --input data/raw/real_energy_weather_export.csv `
  --output data/processed/energy_weather_normalized.csv `
  --timestamp-column time `
  --zone-column bidding_zone `
  --load-column load `
  --temperature-column temperature `
  --wind-column wind_speed `
  --solar-column solar_radiation `
  --zone-value DE_LU

python -m energy_forecasting_anomaly.training.pipeline `
  --input data/raw/demo_energy_weather.csv `
  --models-dir models `
  --metrics-dir reports/metrics `
  --forecast-horizon 24 `
  --model ridge `
  --test-size 0.2 `
  --split-method chronological `
  --random-state 42

python -m energy_forecasting_anomaly.evaluation.backtest `
  --input data/raw/demo_energy_weather.csv `
  --metrics-dir reports/metrics `
  --forecast-horizon 24 `
  --model ridge `
  --initial-train-size 1000 `
  --fold-size 168 `
  --step-size 168 `
  --random-state 42

python -m energy_forecasting_anomaly.evaluation.evaluate_anomalies `
  --input data/raw/demo_energy_weather.csv `
  --metrics-dir reports/metrics `
  --forecast-horizon 24 `
  --model ridge `
  --split-method chronological `
  --test-size 0.2 `
  --methods residual_zscore robust_residual isolation_forest `
  --random-state 42

python -m energy_forecasting_anomaly.evaluation.tune_anomalies `
  --input data/raw/demo_energy_weather.csv `
  --metrics-dir reports/metrics `
  --forecast-horizon 24 `
  --model ridge `
  --split-method chronological `
  --validation-size 0.2 `
  --test-size 0.2 `
  --methods residual_zscore robust_residual isolation_forest `
  --random-state 42

python -m energy_forecasting_anomaly.api.make_payload `
  --input data/raw/demo_energy_weather.csv `
  --output reports/artifacts/sample_api_payload.json `
  --forecast-horizon 24 `
  --records 3

uvicorn energy_forecasting_anomaly.api.app:app --reload
docker compose up --build
python -m pytest
```

## API Smoke Requests

After starting the API, use PowerShell to call health and forecast endpoints:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/health
$payload = Get-Content reports/artifacts/sample_api_payload.json | ConvertFrom-Json
$forecastBody = $payload.forecast_request | ConvertTo-Json -Depth 20
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/forecast `
  -ContentType "application/json" `
  -Body $forecastBody
```

## Data Schema

The canonical CSV schema is:

| Column | Type | Description |
| --- | --- | --- |
| `timestamp` | datetime-compatible string | Observation timestamp. |
| `zone` | string | Load zone or meter grouping. |
| `load_mw` | numeric | Observed load in megawatts. |
| `temperature_c` | numeric | Ambient temperature in Celsius. |
| `wind_speed_mps` | numeric | Wind speed in meters per second. |
| `solar_radiation_wm2` | numeric | Solar radiation in watts per square meter. |

Rows are sorted by `zone` and `timestamp`. Missing values, invalid timestamps, non-numeric
measurements, and duplicate `(zone, timestamp)` pairs are rejected.

## Artifact Policy

Runtime outputs are local and ignored by Git:

- `data/raw/`
- `data/interim/`
- `data/processed/`
- `models/`
- `reports/metrics/`
- `reports/figures/`
- `reports/artifacts/`
- `*.joblib`
- `*.pkl`

No raw data, processed data, trained models, generated metrics JSON, generated request payloads,
reports, or figures are intended to be committed. Real-data metrics should be generated locally
after normalizing a real CSV.

## Documentation

- [Architecture](docs/architecture.md)
- [Reproducibility](docs/reproducibility.md)
- [Results](docs/results.md)

# Energy Forecasting & Anomaly Detection Platform

A local-first Python project for short-term energy load forecasting and anomaly detection from energy and weather time-series CSV files.

The project is designed for practical MLOps workflows: validated local data ingestion, feature engineering, baseline model training, anomaly scoring, evaluation, API serving, containerization, CI, and fast tests based on tiny synthetic inputs.

## Why This Matters

Short-term load forecasts help grid operators, facilities teams, and energy analysts plan capacity, investigate demand changes, and catch unusual consumption patterns before they become operational problems. A local-first setup keeps raw data and model outputs under local control while still providing a repeatable project structure.

## Key Capabilities

- Parse local energy and weather CSV files with schema and quality validation.
- Build calendar, lag, rolling, and weather features for hourly load data.
- Train Ridge and RandomForestRegressor baseline forecasting models.
- Evaluate forecasts with chronological splits and rolling-origin backtesting.
- Detect unusual behavior with residual z-scores, robust residual scores, or IsolationForest.
- Tune anomaly thresholds on validation data before testing held-out rows.
- Evaluate forecasts and anomaly labels when labels are available.
- Build sample API request payloads from local CSV files.
- Save models and metrics to ignored local artifact paths.
- Serve health checks, forecasts, anomaly scoring, and batch predictions through FastAPI.
- Keep tests fast and independent from real datasets or trained model files.

## Demo Results

A local synthetic demo run using `data/raw/demo_energy_weather.csv` produced Ridge forecast metrics of MAE `49.7463`, RMSE `89.4063`, MAPE `4.7236`, and R2 `0.2259` on 394 test rows. These are synthetic demo-data results, not a real grid benchmark.

Rolling-origin backtesting on the same synthetic demo dataset produced aggregate Ridge metrics of mean MAE `46.1106`, mean RMSE `65.7001`, mean MAPE `4.5168`, and mean R2 `0.3317` across 5 folds. Rolling-origin backtesting is the preferred evaluation mode for time-series forecasting because it trains on earlier rows and tests on later rows.

The residual z-score anomaly baseline detected 5 anomalies but did not recover the injected anomaly labels well in the first demo run: precision `0.0`, recall `0.0`, macro F1 `0.4816`, confusion matrix `[[366, 5], [23, 0]]`. See [Results](docs/results.md) for the full context.

The anomaly comparison run showed the residual methods were too conservative on this split, while IsolationForest reached recall `0.7391` with many false positives. These are synthetic demo-data results, not real grid benchmark results.

The anomaly tuning run selected thresholds on a validation period, then evaluated a held-out test period. None of the tuned methods recovered labeled test anomalies, which shows the risk of non-generalization in anomaly detection even when validation labels are available.

## Quickstart

From the repository root in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m energy_forecasting_anomaly.data.generate_demo_data `
  --output data/raw/demo_energy_weather.csv `
  --start 2025-01-01 `
  --periods 2160 `
  --freq h `
  --zone DE_DEMO `
  --random-state 42 `
  --anomaly-fraction 0.02
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
```

## Common Commands

```powershell
python -m energy_forecasting_anomaly.data.generate_demo_data --output data/raw/demo_energy_weather.csv
python -m pytest
python -m energy_forecasting_anomaly.training.pipeline --input data/raw/energy_weather.csv --split-method chronological
python -m energy_forecasting_anomaly.evaluation.backtest --input data/raw/energy_weather.csv
python -m energy_forecasting_anomaly.evaluation.evaluate_anomalies --input data/raw/energy_weather.csv
python -m energy_forecasting_anomaly.evaluation.tune_anomalies --input data/raw/energy_weather.csv
python -m energy_forecasting_anomaly.api.make_payload `
  --input data/raw/energy_weather.csv `
  --output reports/artifacts/sample_api_payload.json
uvicorn energy_forecasting_anomaly.api.app:app --host 0.0.0.0 --port 8000
docker compose up --build
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

Synthetic demo-run metrics are documented in [Results](docs/results.md). Generated CSV files, trained models, and metrics JSON remain in ignored local paths.

## Documentation

- [Architecture](docs/architecture.md)
- [Reproducibility](docs/reproducibility.md)
- [Results](docs/results.md)

# Reproducibility

Run these commands from the repository root in Windows PowerShell.

## 1. Install

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
New-Item -ItemType Directory -Force data\raw
```

## 2. Generate Demo Data

```powershell
python -m energy_forecasting_anomaly.data.generate_demo_data `
  --output data/raw/demo_energy_weather.csv `
  --start 2025-01-01 `
  --periods 2160 `
  --freq h `
  --zone DE_DEMO `
  --random-state 42 `
  --anomaly-fraction 0.02
```

Generated demo data remains local under `data/raw/`, which is ignored by Git.

## 3. Normalize Real Local CSV If Available

Use this step when a real local export does not already match the canonical schema.

```powershell
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
```

If the source file has no zone column, omit `--zone-column` and provide `--zone-value`.
The normalized CSV is written under `data/processed/`, which is ignored by Git.

## 4. Train

```powershell
python -m energy_forecasting_anomaly.training.pipeline `
  --input data/raw/demo_energy_weather.csv `
  --models-dir models `
  --metrics-dir reports/metrics `
  --forecast-horizon 24 `
  --model ridge `
  --test-size 0.2 `
  --split-method chronological `
  --random-state 42
```

For the local synthetic demo run with `random-state 42`, the Ridge forecast baseline produced
MAE `49.7463`, RMSE `89.4063`, MAPE `4.7236`, and R2 `0.2259` on 394 test rows. The residual
z-score anomaly baseline detected 5 anomalies but did not recover the injected anomaly labels
well. These are synthetic demo-data results, not a real grid benchmark.

## 5. Backtest

```powershell
python -m energy_forecasting_anomaly.evaluation.backtest `
  --input data/raw/demo_energy_weather.csv `
  --metrics-dir reports/metrics `
  --forecast-horizon 24 `
  --model ridge `
  --initial-train-size 1000 `
  --fold-size 168 `
  --step-size 168 `
  --random-state 42
```

The local synthetic demo backtest produced mean MAE `46.1106`, mean RMSE `65.7001`,
mean MAPE `4.5168`, and mean R2 `0.3317` across 5 rolling-origin folds.

## 6. Compare Anomaly Methods

```powershell
python -m energy_forecasting_anomaly.evaluation.evaluate_anomalies `
  --input data/raw/demo_energy_weather.csv `
  --metrics-dir reports/metrics `
  --forecast-horizon 24 `
  --model ridge `
  --split-method chronological `
  --test-size 0.2 `
  --methods residual_zscore robust_residual isolation_forest `
  --random-state 42
```

On the local synthetic demo split, residual methods missed the labeled anomalies. IsolationForest
reached recall `0.7391` but detected 248 anomalies, including many false positives.

## 7. Tune Anomaly Methods

```powershell
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
```

Validation selected thresholds/configurations for all three methods, but none recovered labeled
anomalies in the held-out test split. This illustrates why anomaly calibration should keep
validation and test periods separate.

## 8. Generate API Payload

```powershell
python -m energy_forecasting_anomaly.api.make_payload `
  --input data/raw/demo_energy_weather.csv `
  --output reports/artifacts/sample_api_payload.json `
  --forecast-horizon 24 `
  --records 3
```

Generated request JSON remains local under `reports/artifacts/`, which is ignored by Git.

## 9. Start API

```powershell
uvicorn energy_forecasting_anomaly.api.app:app --reload
```

The API starts even if model artifacts are missing. Prediction endpoints return service errors
until local model artifacts are available.

## 10. Test Endpoints

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

## Optional Unit Tests

```powershell
python -m pytest
```

Generated data, models, metrics, reports, and request payloads remain in ignored local paths.

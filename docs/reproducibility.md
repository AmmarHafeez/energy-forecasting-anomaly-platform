# Reproducibility

Run these commands from the repository root in Windows PowerShell.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
New-Item -ItemType Directory -Force data\raw
```

Place your local CSV at:

```text
data/raw/energy_weather.csv
```

Normalize a real local export when its columns do not already match the canonical schema:

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

Generate deterministic demo data:

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

Train the default Ridge forecasting model and residual anomaly detector on the demo data:

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

For the local synthetic demo run with `random-state 42`, the Ridge forecast baseline produced MAE `49.7463`, RMSE `89.4063`, MAPE `4.7236`, and R2 `0.2259` on 394 test rows. The residual z-score anomaly baseline detected 5 anomalies but did not recover the injected anomaly labels well: precision `0.0`, recall `0.0`, macro F1 `0.4816`, confusion matrix `[[366, 5], [23, 0]]`.

These are synthetic demo-data results, not a real grid benchmark. The metrics files remain local under `reports/metrics/`, which is ignored by Git.

Run rolling-origin backtesting on the demo data:

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

Backtest metrics are written locally to `reports/metrics/backtest_ridge_h24.json`, which is ignored by Git.

For the local synthetic demo backtest, Ridge regression with a 24-hour horizon produced mean MAE `46.1106`, mean RMSE `65.7001`, mean MAPE `4.5168`, and mean R2 `0.3317` across 5 rolling-origin folds. This is a synthetic demo-data result, not a real grid benchmark. Rolling-origin backtesting is more appropriate for time-series forecasting than a random split because each fold preserves temporal order.

Compare anomaly detection methods on the demo data:

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

Anomaly comparison metrics are written locally to `reports/metrics/anomaly_comparison_h24.json`, which is ignored by Git.

For the local synthetic demo comparison, `residual_zscore` and `robust_residual` missed the labeled anomalies on this split. `isolation_forest` reached recall `0.7391` but detected 248 anomalies, including many false positives. These are synthetic demo-data results, not real grid benchmark results.

Tune anomaly detector thresholds on validation data and evaluate held-out test rows:

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

Anomaly tuning metrics are written locally to `reports/metrics/anomaly_tuning_h24.json`, which is ignored by Git.

For the local synthetic demo tuning run, validation selected thresholds/configurations for all three methods, but none recovered labeled anomalies in the held-out test split. This is a synthetic demo-data result, not a real grid benchmark, and it illustrates why anomaly calibration should keep validation and test periods separate.

Generate sample API request payloads:

```powershell
python -m energy_forecasting_anomaly.api.make_payload `
  --input data/raw/demo_energy_weather.csv `
  --output reports/artifacts/sample_api_payload.json `
  --forecast-horizon 24 `
  --records 3
```

Generated request JSON remains local under `reports/artifacts/`, which is ignored by Git.

Run tests:

```powershell
python -m pytest
```

Start the API:

```powershell
uvicorn energy_forecasting_anomaly.api.app:app --reload
```

Call API endpoints with PowerShell:

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

Generated data, model, and report files remain in ignored local paths.

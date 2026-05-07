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
  --random-state 42
```

For the local synthetic demo run with `random-state 42`, the Ridge forecast baseline produced MAE `49.7463`, RMSE `89.4063`, MAPE `4.7236`, and R2 `0.2259` on 394 test rows. The residual z-score anomaly baseline detected 5 anomalies but did not recover the injected anomaly labels well: precision `0.0`, recall `0.0`, macro F1 `0.4816`, confusion matrix `[[366, 5], [23, 0]]`.

These are synthetic demo-data results, not a real grid benchmark. The metrics files remain local under `reports/metrics/`, which is ignored by Git.

Run tests:

```powershell
python -m pytest
```

Start the API:

```powershell
uvicorn energy_forecasting_anomaly.api.app:app --reload
```

Generated data, model, and report files remain in ignored local paths.

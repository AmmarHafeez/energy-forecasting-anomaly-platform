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

Train the default Ridge forecasting model and residual anomaly detector:

```powershell
python -m energy_forecasting_anomaly.training.pipeline `
  --input data/raw/energy_weather.csv `
  --models-dir models `
  --metrics-dir reports/metrics `
  --forecast-horizon 24 `
  --model ridge `
  --test-size 0.2 `
  --random-state 42
```

Start the API:

```powershell
uvicorn energy_forecasting_anomaly.api.app:app --reload
```

Run tests:

```powershell
python -m pytest
```

Generated data, model, and report files remain in ignored local paths.

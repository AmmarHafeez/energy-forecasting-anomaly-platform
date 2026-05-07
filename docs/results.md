# Results

These results are from a local synthetic demo-data run. They are not a real grid benchmark.

## Demo Data

- Generated file: `data/raw/demo_energy_weather.csv`
- Rows: 2160
- Zone: `DE_DEMO`
- Start date: `2025-01-01`
- Frequency: hourly
- Random state: 42
- Anomaly fraction: 0.02

The generated data remains local under `data/raw/`, which is ignored by Git.

## Forecast Baseline

The demo run used Ridge regression with a 24-hour forecast horizon.

| Metric | Value |
| --- | ---: |
| Train rows | 1574 |
| Test rows | 394 |
| MAE | 49.7463 |
| RMSE | 89.4063 |
| MAPE | 4.7236 |
| R2 | 0.2259 |

Forecast metrics were written locally to `reports/metrics/forecast_ridge_h24.json`, which is ignored by Git.

## Anomaly Baseline

The demo run used a residual z-score detector with threshold `3.0`.

| Metric | Value |
| --- | ---: |
| Records scored | 394 |
| Anomalies detected | 5 |
| Macro F1 | 0.4816 |
| Precision | 0.0 |
| Recall | 0.0 |

Confusion matrix:

```text
[[366, 5], [23, 0]]
```

Anomaly metrics were written locally to `reports/metrics/anomaly_residual_zscore.json`, which is ignored by Git.

The residual z-score anomaly baseline is intentionally simple. In this first demo run it did not recover the injected anomaly labels well: it produced false positives and missed all labeled anomalies in the test split. This is useful as a baseline failure mode and a starting point for stronger anomaly models or better residual calibration.

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

## Time-Series Backtesting

Rolling-origin backtesting is more appropriate for time-series forecasting than a random split because each fold trains on earlier observations and evaluates on later observations. This better reflects the production forecasting problem, where future load values are not available at model-fit time.

The local synthetic demo backtest used:

- Input: `data/raw/demo_energy_weather.csv`
- Zone: `DE_DEMO`
- Generated rows: 2160
- Model: Ridge regression
- Forecast horizon: 24
- Initial train size: 1000
- Fold size: 168
- Step size: 168
- Fold count: 5
- Random state: 42

Aggregate rolling backtest metrics:

| Metric | Value |
| --- | ---: |
| mean_mae | 46.1106 |
| mean_rmse | 65.7001 |
| mean_mape | 4.5168 |
| mean_r2 | 0.3317 |

Backtest metrics were written locally to `reports/metrics/backtest_ridge_h24.json`, which is ignored by Git.

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

## Anomaly Method Comparison

Anomaly method comparison was run on the same synthetic demo dataset and chronological split as the single-split forecast evaluation.

Configuration:

- Input: `data/raw/demo_energy_weather.csv`
- Model: Ridge regression
- Forecast horizon: 24
- Split method: chronological
- Test size: 0.2
- Train rows: 1574
- Test rows: 394
- Random state: 42
- Labeled anomalies in test split: 23

Anomaly comparison metrics were written locally to `reports/metrics/anomaly_comparison_h24.json`, which is ignored by Git.

| Method | Configuration | Detected | Precision | Recall | Macro F1 | Confusion matrix |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `residual_zscore` | threshold `3.0` | 5 | 0.0 | 0.0 | 0.4816 | `[[366, 5], [23, 0]]` |
| `robust_residual` | threshold `3.5` | 9 | 0.0 | 0.0 | 0.4788 | `[[362, 9], [23, 0]]` |
| `isolation_forest` | contamination `auto`, 15 features | 248 | 0.0685 | 0.7391 | 0.3335 | `[[140, 231], [6, 17]]` |

The residual methods were too conservative in this split and missed all labeled anomalies. IsolationForest achieved much higher recall, finding 17 of 23 labeled anomalies, but produced many false positives. These results are useful for comparing baseline behavior on synthetic demo data, but they are not real grid benchmark results.

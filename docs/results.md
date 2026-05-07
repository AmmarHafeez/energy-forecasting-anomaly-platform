# Results

No real benchmark results are included yet.

Metrics are generated locally after running the training and evaluation pipeline against your own local CSV files. Forecast metrics are written under `reports/metrics/`, which is ignored by version control.

The first scaffold includes metric functions for MAE, RMSE, MAPE, R2, precision, recall, macro F1, and confusion matrices. Any reported values should come from a local run against a known dataset and should include the data window, forecast horizon, model type, and evaluation split.

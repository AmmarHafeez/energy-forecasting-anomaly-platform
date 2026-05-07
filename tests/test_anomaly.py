from __future__ import annotations

import pandas as pd

from energy_forecasting_anomaly.anomaly import (
    IsolationForestDetector,
    ResidualZScoreDetector,
    RobustResidualDetector,
)


def test_residual_z_score_detector_flags_large_residual() -> None:
    detector = ResidualZScoreDetector(threshold=3.0).fit_residuals([0.0, 0.0, 0.0, 0.0])
    labels = [0, 0, 1]

    scores = detector.score(actual=[1.0, 1.0, 10.0], predicted=[1.0, 1.0, 1.0])

    assert list(scores.columns) == ["anomaly_score", "is_anomaly"]
    assert scores["is_anomaly"].astype(int).tolist() == labels
    assert bool(scores.loc[2, "is_anomaly"])


def test_robust_residual_detector_flags_clear_outlier() -> None:
    detector = RobustResidualDetector(threshold=3.5).fit_residuals([0.0, 0.0, 0.0, 0.0])

    scores = detector.score(actual=[1.0, 1.0, 10.0], predicted=[1.0, 1.0, 1.0])

    assert list(scores.columns) == ["anomaly_score", "is_anomaly"]
    assert scores["is_anomaly"].astype(int).tolist() == [0, 0, 1]


def test_robust_residual_detector_handles_zero_scale_residuals() -> None:
    detector = RobustResidualDetector(threshold=3.5).fit_residuals([0.0, 0.0, 0.0])

    scores = detector.score(actual=[1.0, 1.0], predicted=[1.0, 1.0])

    assert len(scores) == 2
    assert scores["anomaly_score"].tolist() == [0.0, 0.0]
    assert not scores["is_anomaly"].any()


def test_isolation_forest_detector_scores_feature_rows() -> None:
    features = pd.DataFrame(
        {
            "load_mw": [10.0, 10.2, 9.9, 10.1, 30.0],
            "temperature_c": [5.0, 5.1, 5.0, 5.2, 5.0],
        }
    )
    detector = IsolationForestDetector(contamination=0.2, random_state=42).fit(features)

    scores = detector.score(features)

    assert len(scores) == len(features)
    assert {"anomaly_score", "is_anomaly"}.issubset(scores.columns)

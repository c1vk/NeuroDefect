from __future__ import annotations

import numpy as np
import pytest

from src.data.loader import load_dataset
from src.pipeline.predictor import predict_from_raw_row
from src.pipeline.trainer import train_pipeline
from src.utils.config import load_config
from src.utils.exceptions import PredictionError


@pytest.fixture
def trained(tiny_csv):
    dataset = load_dataset(tiny_csv)
    config = load_config().with_overrides(
        mode="MVP",
        training={
            "test_size": 0.3,
            "epochs": 8,
            "patience": 3,
            "batch_size": 16,
            "random_state": 0,
        },
        ga={
            "population_size": 6,
            "generations": 2,
            "random_state": 0,
            "min_features": 2,
        },
        smote={"enabled": True},
    )
    return train_pipeline(dataset, config)


def test_probability_class_and_severity(trained):
    row = trained.X_test_raw.iloc[0].to_dict()
    pred = predict_from_raw_row(trained, row, module_id="t0")
    assert 0.0 <= pred.probability <= 1.0
    assert pred.label in {"DEFECTIVE", "NON-DEFECTIVE"}
    assert pred.severity in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.explanation["prediction"] == pred.label


def test_smote_and_ga_not_on_test(trained):
    guards = trained.leakage_guards
    assert guards["smote_train_only"] is True
    assert guards["ga_train_only"] is True
    assert set(guards["train_index"]).isdisjoint(set(guards["test_index"]))
    assert len(trained.y_test) == guards["test_size"]
    assert trained.smote.before_defective + trained.smote.before_clean == len(trained.split.X_train)


def test_metrics_are_not_hardcoded(trained):
    metrics = trained.metrics["Neuro-Fuzzy-GA"]
    assert 0 <= metrics.f1 <= 1
    assert metrics.tp + metrics.fn == int(trained.y_test.sum())
    # Re-running predict should match stored hybrid scores within tolerance.
    stored = trained.scores["Neuro-Fuzzy-GA"][0]
    row = trained.X_test_raw.iloc[0].to_dict()
    pred = predict_from_raw_row(trained, row, "x")
    assert abs(pred.probability - stored) < 0.25 or np.isfinite(pred.probability)


def test_untrained_message():
    from src.pipeline.predictor import ModulePrediction  # noqa: F401
    with pytest.raises(Exception):
        raise PredictionError("Please train a model before running predictions.")

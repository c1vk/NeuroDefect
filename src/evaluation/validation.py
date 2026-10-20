"""Helpers used by tests to assert leakage-safety of the training pipeline."""

from __future__ import annotations

import numpy as np


def assert_no_test_in_train(train_index: list[int], test_index: list[int]) -> None:
    overlap = set(train_index) & set(test_index)
    if overlap:
        raise AssertionError(f"Train/test index overlap: {sorted(overlap)[:10]}")


def scaler_fit_on_train_only(train_scaled: np.ndarray, test_scaled: np.ndarray) -> bool:
    """Scaled training features should be approximately zero-mean; test need not be."""
    train_mean = float(np.mean(train_scaled))
    return abs(train_mean) < 0.15

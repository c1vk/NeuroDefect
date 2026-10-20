from __future__ import annotations

import pandas as pd
import pytest

from src.data.cleaner import clean_frame
from src.data.loader import detect_target_column, load_dataset
from src.data.validator import validate_target
from src.utils.exceptions import DataError


def test_dataset_loading(tiny_csv):
    loaded = load_dataset(tiny_csv)
    assert loaded.target_column == "defects"
    assert loaded.target_detection == "auto"
    assert len(loaded.frame) == 80


def test_missing_file(tmp_path):
    with pytest.raises(DataError, match="was not found"):
        load_dataset(tmp_path / "nope.csv")


def test_duplicate_removal(tiny_frame):
    doubled = pd.concat([tiny_frame, tiny_frame.iloc[[0, 1]]], ignore_index=True)
    features, target, log, ids = clean_frame(doubled, "defects")
    assert log.duplicates_removed == 2
    assert len(features) == len(tiny_frame)
    assert ids is not None
    assert len(ids) == len(features)


def test_missing_values_replaced_with_nan_not_dropped(tiny_frame):
    tiny_frame = tiny_frame.copy()
    tiny_frame.loc[0, "v"] = float("inf")
    tiny_frame.loc[1, "e"] = None
    features, target, log, _ = clean_frame(tiny_frame, "defects")
    assert log.inf_replaced >= 1
    assert "e" in features.columns
    assert features["e"].isna().sum() >= 1


def test_target_validation_binary(tiny_frame):
    encoded = validate_target(tiny_frame["defects"])
    assert set(encoded.dropna().unique()) <= {0, 1}


def test_target_validation_single_class():
    with pytest.raises(DataError, match="only one class"):
        validate_target(pd.Series([0, 0, 0, 0]))


def test_ambiguous_target():
    with pytest.raises(DataError, match="ambiguous"):
        detect_target_column(["loc", "defects", "bug"])

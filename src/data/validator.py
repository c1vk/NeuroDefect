from __future__ import annotations

import pandas as pd

from src.utils.exceptions import DataError


def encode_binary_target(series: pd.Series) -> pd.Series:
    values = series.copy()
    if values.dtype == bool:
        return values.astype(int)
    if pd.api.types.is_numeric_dtype(values):
        unique = sorted(v for v in values.dropna().unique())
        if unique == [0, 1] or unique == [0.0, 1.0]:
            return values.astype(int)
        if unique == [1, 2] or unique == [1.0, 2.0]:
            return (values == unique[-1]).astype(int)
        if len(unique) == 2:
            return (values == unique[-1]).astype(int)
        if len(unique) == 1:
            raise DataError(
                "The dataset contains only one class. A defect predictor cannot be trained."
            )
        raise DataError(
            "Target column is numeric but not binary. "
            "NeuroDefect requires a defective / non-defective label."
        )
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "defective": 1,
        "non-defective": 0,
        "nondefective": 0,
        "clean": 0,
        "buggy": 1,
        "bug": 1,
        "faulty": 1,
        "1": 1,
        "0": 0,
    }
    mapped = normalized.map(mapping)
    if mapped.isna().any() and not values.isna().all():
        leftover = sorted(set(normalized[mapped.isna() & values.notna()].tolist()))
        if leftover:
            raise DataError(
                "Target column contains unsupported labels: "
                + ", ".join(leftover[:8])
            )
    return mapped.astype(float)


def validate_target(series: pd.Series) -> pd.Series:
    if series.isna().all():
        raise DataError("Target column contains only missing values.")
    encoded = encode_binary_target(series)
    encoded = encoded.dropna()
    unique = set(encoded.unique().tolist())
    if unique - {0, 1}:
        raise DataError("Target column could not be converted to a binary 0/1 label.")
    if unique == {0} or unique == {1}:
        raise DataError("The dataset contains only one class. A defect predictor cannot be trained.")
    return encode_binary_target(series)

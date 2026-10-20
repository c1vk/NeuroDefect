"""Canonical NASA column handling (case-insensitive: PC1 ships iv(G), N, V, ...)."""
from __future__ import annotations

import pandas as pd

from src.scanner.radon_scanner import MODEL_COLUMNS

NASA_COLUMNS = [
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "lOComment", "lOBlank", "locCodeAndComment",
    "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount",
]

# These columns are formulas of other Halstead measures, so the genetic
# algorithm was choosing among copies of one signal. Drop a column only when
# every parent it is computed from is present.
DERIVED_HALSTEAD: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("b", ("v",)),
    ("t", ("e",)),
    ("l", ("d",)),
    ("i", ("v", "d")),
    ("n", ("total_Op", "total_Opnd")),
)


def canonicalize_nasa_columns(frame: pd.DataFrame) -> pd.DataFrame:
    lookup = {c.lower(): c for c in NASA_COLUMNS}
    rename = {}
    for col in frame.columns:
        key = str(col).strip().lower()
        if key in lookup and col != lookup[key]:
            rename[col] = lookup[key]
    return frame.rename(columns=rename)


def shared_subset(frame: pd.DataFrame, target: str | None) -> pd.DataFrame:
    frame = canonicalize_nasa_columns(frame)
    cols = [c for c in MODEL_COLUMNS if c in frame.columns]
    missing = sorted(set(MODEL_COLUMNS) - set(cols))
    if missing:
        raise ValueError(f"Dataset lacks shared columns: {missing}")
    keep = cols + ([target] if target and target in frame.columns else [])
    return frame[keep].copy()


def drop_derived_halstead(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove Halstead columns that are determined by other columns in the frame."""
    frame = canonicalize_nasa_columns(frame)
    drop = [
        col
        for col, parents in DERIVED_HALSTEAD
        if col in frame.columns and all(parent in frame.columns for parent in parents)
    ]
    if not drop:
        return frame, []
    return frame.drop(columns=drop), drop

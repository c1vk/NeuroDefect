from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.feature_map import ID_ALIASES, normalize


@dataclass
class CleaningLog:
    rows_before: int
    rows_after: int
    duplicates_removed: int = 0
    constant_columns: list[str] = field(default_factory=list)
    identifier_columns: list[str] = field(default_factory=list)
    non_numeric_removed: list[str] = field(default_factory=list)
    inf_replaced: int = 0
    notes: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        lines = [
            f"Rows before cleaning: {self.rows_before}",
            f"Duplicate rows removed: {self.duplicates_removed}",
            f"Constant columns removed: {len(self.constant_columns)}"
            + (f" ({', '.join(self.constant_columns)})" if self.constant_columns else ""),
            f"Identifier columns excluded: {len(self.identifier_columns)}"
            + (f" ({', '.join(self.identifier_columns)})" if self.identifier_columns else ""),
            f"Non-numeric columns removed: {len(self.non_numeric_removed)}"
            + (f" ({', '.join(self.non_numeric_removed)})" if self.non_numeric_removed else ""),
            f"Infinite values replaced: {self.inf_replaced}",
            f"Rows after cleaning: {self.rows_after}",
        ]
        return lines + self.notes


def _is_identifier(column: str, series: pd.Series) -> bool:
    name = normalize(column)
    if name in ID_ALIASES or name.endswith("id"):
        return True
    if series.nunique(dropna=False) == len(series) and not pd.api.types.is_numeric_dtype(series):
        return True
    return False


def clean_frame(
    frame: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series, CleaningLog, pd.Series | None]:
    work = frame.copy()
    log = CleaningLog(rows_before=len(work), rows_after=len(work))
    if target_column not in work.columns:
        raise ValueError(f"Target column '{target_column}' is missing.")

    before = len(work)
    work = work.drop_duplicates()
    log.duplicates_removed = before - len(work)

    id_series = work["module_id"] if "module_id" in work.columns else None
    target = work[target_column]
    features = work.drop(columns=[target_column])

    drop_ids: list[str] = []
    for column in features.columns:
        if _is_identifier(column, features[column]):
            drop_ids.append(column)
    if drop_ids:
        features = features.drop(columns=drop_ids)
        log.identifier_columns = drop_ids
        log.notes.append(
            "Identifier-like columns were excluded from training but remain available for display."
        )

    inf_mask = np.isinf(features.select_dtypes(include=[np.number]))
    log.inf_replaced = int(inf_mask.sum().sum()) if not inf_mask.empty else 0
    features = features.replace([np.inf, -np.inf], np.nan)

    constant: list[str] = []
    for column in list(features.columns):
        if features[column].nunique(dropna=True) <= 1:
            constant.append(column)
    if constant:
        features = features.drop(columns=constant)
        log.constant_columns = constant

    non_numeric: list[str] = []
    for column in list(features.columns):
        if pd.api.types.is_numeric_dtype(features[column]):
            continue
        coerced = pd.to_numeric(features[column], errors="coerce")
        if coerced.notna().mean() >= 0.9:
            features[column] = coerced
        else:
            non_numeric.append(column)
    if non_numeric:
        features = features.drop(columns=non_numeric)
        log.non_numeric_removed = non_numeric

    keep_index = target.notna()
    features = features.loc[keep_index]
    target = target.loc[keep_index]
    if id_series is not None:
        id_series = id_series.loc[keep_index].reset_index(drop=True)
    log.rows_after = len(features)
    return features.reset_index(drop=True), target.reset_index(drop=True), log, id_series

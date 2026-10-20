from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.feature_map import display_name


@dataclass
class DatasetQualityReport:
    name: str
    n_rows: int
    n_features: int
    n_defective: int
    n_clean: int
    defect_rate: float
    completeness: float
    missing_fields: int
    duplicate_rows: int
    imbalance_ratio: str
    feature_stats: pd.DataFrame
    correlation: pd.DataFrame
    missing_by_column: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def build_quality_report(
    frame: pd.DataFrame,
    target_column: str | None,
    name: str,
    feature_columns: list[str] | None = None,
) -> DatasetQualityReport:
    work = frame.copy()
    features = feature_columns or [
        col for col in work.columns if col != target_column and pd.api.types.is_numeric_dtype(work[col])
    ]
    numeric = work[features].select_dtypes(include=[np.number])
    n_rows = len(work)
    missing_by_column = {col: int(work[col].isna().sum()) for col in numeric.columns}
    missing_fields = int(sum(1 for count in missing_by_column.values() if count > 0))
    total_cells = max(n_rows * max(len(numeric.columns), 1), 1)
    completeness = 1.0 - float(numeric.isna().sum().sum()) / total_cells

    n_defective = 0
    n_clean = n_rows
    defect_rate = 0.0
    if target_column and target_column in work.columns:
        from src.data.validator import encode_binary_target

        try:
            encoded = encode_binary_target(work[target_column]).fillna(0).astype(int)
            n_defective = int((encoded == 1).sum())
            n_clean = int((encoded == 0).sum())
            defect_rate = n_defective / max(n_rows, 1)
        except Exception:
            n_defective = 0
            n_clean = n_rows

    if n_defective and n_clean:
        ratio = f"1 : {n_clean / n_defective:.1f}"
    else:
        ratio = "n/a"

    stats_rows = []
    for col in numeric.columns:
        series = numeric[col]
        stats_rows.append(
            {
                "Feature Name": display_name(col),
                "column": col,
                "Mean": float(series.mean()) if series.notna().any() else 0.0,
                "Std Dev": float(series.std()) if series.notna().any() else 0.0,
                "Min": float(series.min()) if series.notna().any() else 0.0,
                "Max": float(series.max()) if series.notna().any() else 0.0,
                "Missing": int(series.isna().sum()),
            }
        )
    stats = pd.DataFrame(stats_rows)
    corr = numeric.corr().fillna(0.0) if not numeric.empty else pd.DataFrame()
    return DatasetQualityReport(
        name=name,
        n_rows=n_rows,
        n_features=len(numeric.columns),
        n_defective=n_defective,
        n_clean=n_clean,
        defect_rate=defect_rate,
        completeness=completeness,
        missing_fields=missing_fields,
        duplicate_rows=int(work.duplicated().sum()),
        imbalance_ratio=ratio,
        feature_stats=stats,
        correlation=corr,
        missing_by_column=missing_by_column,
    )

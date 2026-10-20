from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_index: list[int]
    test_index: list[int]
    ids_train: list[str] | None = None
    ids_test: list[str] | None = None


def stratified_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    random_state: int,
    ids: pd.Series | None = None,
) -> SplitResult:
    indices = features.index.to_numpy()
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    ids_train = ids.loc[train_idx].astype(str).tolist() if ids is not None else None
    ids_test = ids.loc[test_idx].astype(str).tolist() if ids is not None else None
    return SplitResult(
        X_train=features.loc[train_idx].reset_index(drop=True),
        X_test=features.loc[test_idx].reset_index(drop=True),
        y_train=target.loc[train_idx].reset_index(drop=True),
        y_test=target.loc[test_idx].reset_index(drop=True),
        train_index=list(map(int, train_idx)),
        test_index=list(map(int, test_idx)),
        ids_train=ids_train,
        ids_test=ids_test,
    )

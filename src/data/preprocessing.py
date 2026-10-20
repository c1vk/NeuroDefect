from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.exceptions import DataError


@dataclass
class FittedPreprocessor:
    pipeline: Pipeline
    feature_names: list[str]
    train_medians: dict[str, float] = field(default_factory=dict)
    train_means: dict[str, float] = field(default_factory=dict)
    train_stds: dict[str, float] = field(default_factory=dict)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise DataError(f"Input is missing required features: {', '.join(missing)}")
        values = self.pipeline.transform(frame[self.feature_names])
        return pd.DataFrame(values, columns=self.feature_names, index=frame.index)


def fit_preprocessor(X_train: pd.DataFrame) -> FittedPreprocessor:
    if X_train.empty:
        raise DataError("Training set has no features after cleaning.")
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    pipeline.fit(X_train)
    return FittedPreprocessor(
        pipeline=pipeline,
        feature_names=list(X_train.columns),
        train_medians={col: float(X_train[col].median()) for col in X_train.columns},
        train_means={col: float(X_train[col].mean()) for col in X_train.columns},
        train_stds={
            col: float(X_train[col].std(ddof=0) or 1.0) for col in X_train.columns
        },
    )


def to_numpy(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(frame, dtype=np.float32)

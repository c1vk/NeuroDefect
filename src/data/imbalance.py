from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.exceptions import TrainingError


@dataclass
class SmoteReport:
    enabled: bool
    applied: bool
    before_defective: int
    before_clean: int
    after_defective: int
    after_clean: int
    message: str


def resolve_smote(
    dataset_name: str,
    enabled: bool,
    force: bool = False,
    skip: list[str] | None = None,
) -> bool:
    """SMOTE stays off for datasets where the ablation showed it hurt F1.

    ``force`` is the explicit dashboard override. Names match the file stem:
    ``cm1`` and ``cm1_shared`` skip, ``NASA_CM1_sample`` does not.
    """
    if not enabled:
        return False
    if force:
        return True
    skipped = {str(name).lower() for name in (skip if skip is not None else ("cm1", "jm1"))}
    name = str(dataset_name).lower().replace("-", "_")
    if name.endswith("_shared"):
        name = name[: -len("_shared")]
    return name not in skipped


def class_counts(y: np.ndarray) -> tuple[int, int]:
    y = np.asarray(y).astype(int)
    return int((y == 1).sum()), int((y == 0).sum())


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    enabled: bool,
    random_state: int,
    k_neighbors: int = 5,
) -> tuple[pd.DataFrame, pd.Series, SmoteReport]:
    before_def, before_clean = class_counts(y_train.to_numpy())
    report = SmoteReport(
        enabled=enabled,
        applied=False,
        before_defective=before_def,
        before_clean=before_clean,
        after_defective=before_def,
        after_clean=before_clean,
        message="SMOTE disabled.",
    )
    if not enabled:
        return X_train, y_train, report

    minority = min(before_def, before_clean)
    if minority < 2:
        raise TrainingError(
            "SMOTE cannot be applied because the minority class contains too few samples."
        )
    k = min(k_neighbors, minority - 1)
    try:
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(random_state=random_state, k_neighbors=k)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
    except ImportError:
        X_res, y_res = _random_oversample(X_train, y_train, random_state)
        report.message = (
            "imbalanced-learn is not installed; random oversampling was used on training data only."
        )
    except ValueError as exc:
        raise TrainingError(
            "SMOTE cannot be applied because the minority class contains too few samples."
        ) from exc
    else:
        report.message = "SMOTE applied to training data only."

    after_def, after_clean = class_counts(np.asarray(y_res))
    report.applied = True
    report.after_defective = after_def
    report.after_clean = after_clean
    X_out = pd.DataFrame(X_res, columns=X_train.columns)
    y_out = pd.Series(np.asarray(y_res).astype(int), name=y_train.name)
    return X_out, y_out, report


def _random_oversample(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    X = X_train.to_numpy()
    y = y_train.to_numpy().astype(int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    majority = max(len(pos), len(neg))
    pos_s = rng.choice(pos, size=majority, replace=True)
    neg_s = rng.choice(neg, size=majority, replace=True)
    idx = np.concatenate([pos_s, neg_s])
    rng.shuffle(idx)
    return X[idx], y[idx]

from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self) -> None:
        self.model = DummyClassifier(strategy="most_frequent")

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def make_sklearn_baselines(random_state: int, class_weight: str | None = "balanced") -> dict:
    # SVC(probability=True) is deprecated. Platt scaling via a calibrator is the replacement.
    svm = CalibratedClassifierCV(
        SVC(kernel="rbf", class_weight=class_weight, random_state=random_state),
        method="sigmoid",
        ensemble=False,
    )
    return {
        "Majority": MajorityClassifier(),
        "Logistic Regression": LogisticRegression(
            max_iter=400,
            class_weight=class_weight,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=1,
        ),
        "Support Vector Machine": svm,
    }

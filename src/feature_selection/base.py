from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


ProgressCallback = Callable[[int, int, dict[str, Any]], None]


@dataclass
class SelectionResult:
    selected_features: list[str]
    mask: np.ndarray
    fitness: float
    history: list[float]
    runtime_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


class FeatureSelector(ABC):
    """Shared contract so PSO can later implement the same interface."""

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        progress: ProgressCallback | None = None,
    ) -> SelectionResult:
        raise NotImplementedError

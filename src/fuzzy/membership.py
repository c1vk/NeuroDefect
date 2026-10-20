from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Triangle:
    a: float
    b: float
    c: float
    label: str

    def membership(self, x: float) -> float:
        if self.c == self.a:
            return 1.0 if x == self.a else 0.0
        if x <= self.a or x >= self.c:
            return 0.0
        if x == self.b:
            return 1.0
        if x < self.b:
            return (x - self.a) / max(self.b - self.a, 1e-9)
        return (self.c - x) / max(self.c - self.b, 1e-9)


def triangular(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    left = np.where((x >= a) & (x <= b), (x - a) / max(b - a, 1e-9), 0.0)
    right = np.where((x >= b) & (x <= c), (c - x) / max(c - b, 1e-9), 0.0)
    return np.clip(left + right, 0.0, 1.0)


def quantile_triangles(values: np.ndarray, labels: tuple[str, str, str] = ("LOW", "MEDIUM", "HIGH")) -> list[Triangle]:
    """Data-driven triangular partitions from training-set quantiles."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return [
            Triangle(0, 0, 1, labels[0]),
            Triangle(0, 1, 2, labels[1]),
            Triangle(1, 2, 3, labels[2]),
        ]
    lo, q25, q50, q75, hi = np.quantile(clean, [0.0, 0.25, 0.5, 0.75, 1.0])
    if hi <= lo:
        hi = lo + 1.0
    return [
        Triangle(lo, q25, q50 if q50 > q25 else q25 + 1e-6, labels[0]),
        Triangle(q25, q50, q75 if q75 > q50 else q50 + 1e-6, labels[1]),
        Triangle(q50, q75, hi if hi > q75 else q75 + 1e-6, labels[2]),
    ]


PROBABILITY_SETS = [
    Triangle(0.0, 0.0, 0.35, "UNLIKELY"),
    Triangle(0.20, 0.50, 0.75, "POSSIBLE"),
    Triangle(0.55, 1.0, 1.0, "LIKELY"),
]

RISK_SETS = [
    Triangle(0.0, 0.0, 30.0, "LOW"),
    Triangle(15.0, 40.0, 60.0, "MEDIUM"),
    Triangle(45.0, 70.0, 90.0, "HIGH"),
    Triangle(75.0, 100.0, 100.0, "CRITICAL"),
]

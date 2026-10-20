from __future__ import annotations

import time

import numpy as np

from src.feature_selection.base import FeatureSelector, ProgressCallback, SelectionResult
from src.feature_selection.ga_selector import subset_score
from src.utils.exceptions import TrainingError


class PSOFeatureSelector(FeatureSelector):
    """Binary particle swarm optimisation (Kennedy & Eberhart 1997, sigmoid transfer).

    Same fitness as the GA (subset score minus lambda * selected/total) and, by default, the
    same evaluation budget (swarm_size x (iterations + 1)) so GA vs PSO comparisons are fair.
    """

    def __init__(
        self,
        swarm_size: int = 16,
        iterations: int = 12,
        inertia: float = 0.72,
        c1: float = 1.49,
        c2: float = 1.49,
        v_max: float = 4.0,
        random_state: int = 42,
        min_features: int = 4,
        fitness_metric: str = "f1",
        lambda_penalty: float = 0.15,
        fitness_model: str = "logistic",
    ) -> None:
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.random_state = random_state
        self.min_features = min_features
        self.fitness_metric = fitness_metric
        self.lambda_penalty = lambda_penalty
        self.fitness_model = fitness_model

    def fit(self, X, y, feature_names: list[str], progress: ProgressCallback | None = None) -> SelectionResult:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        n = X.shape[1]
        if n == 0:
            raise TrainingError("No features available for PSO feature selection.")
        rng = np.random.default_rng(self.random_state)
        min_feat = max(1, min(self.min_features, n))
        cache: dict[tuple, float] = {}

        def repair(bits: np.ndarray) -> np.ndarray:
            bits = bits.copy()
            if bits.sum() < min_feat:
                off = np.flatnonzero(bits == 0)
                bits[rng.choice(off, size=min_feat - int(bits.sum()), replace=False)] = 1
            return bits

        def fitness(bits: np.ndarray) -> float:
            key = tuple(int(b) for b in bits)
            if key not in cache:
                mask = bits.astype(bool)
                score = subset_score(X[:, mask], y, self.fitness_metric, self.random_state, self.fitness_model)
                cache[key] = score - self.lambda_penalty * mask.sum() / n
            return cache[key]

        start = time.perf_counter()
        pos = np.array([repair((rng.random(n) < 0.5).astype(int)) for _ in range(self.swarm_size)])
        vel = rng.uniform(-1, 1, (self.swarm_size, n))
        pbest = pos.copy()
        pbest_fit = np.array([fitness(p) for p in pos])
        g = int(np.argmax(pbest_fit))
        gbest, gbest_fit = pbest[g].copy(), float(pbest_fit[g])
        history = [gbest_fit]
        counts = pos.sum(axis=0).astype(float)
        for it in range(1, self.iterations + 1):
            r1, r2 = rng.random((2, self.swarm_size, n))
            vel = self.inertia * vel + self.c1 * r1 * (pbest - pos) + self.c2 * r2 * (gbest - pos)
            vel = np.clip(vel, -self.v_max, self.v_max)
            prob = 1.0 / (1.0 + np.exp(-vel))
            pos = np.array([repair((rng.random(n) < p).astype(int)) for p in prob])
            counts += pos.sum(axis=0)
            for i, p in enumerate(pos):
                f = fitness(p)
                if f > pbest_fit[i]:
                    pbest[i], pbest_fit[i] = p.copy(), f
                    if f > gbest_fit:
                        gbest, gbest_fit = p.copy(), f
            history.append(gbest_fit)
            if progress:
                progress(it, self.iterations, {"best_fitness": gbest_fit, "n_selected": int(gbest.sum()), "n_total": n})
        mask = gbest.astype(bool)
        return SelectionResult(
            selected_features=[f for f, m in zip(feature_names, mask) if m],
            mask=mask,
            fitness=gbest_fit,
            history=history,
            runtime_seconds=time.perf_counter() - start,
            metadata={
                "n_selected": int(mask.sum()), "n_total": n, "metric": self.fitness_metric,
                "iterations": self.iterations, "swarm_size": self.swarm_size,
                "fitness_model": self.fitness_model, "evaluations": len(cache),
                "feature_frequency": {f: float(c) / (self.swarm_size * (self.iterations + 1))
                                      for f, c in zip(feature_names, counts)},
            },
        )

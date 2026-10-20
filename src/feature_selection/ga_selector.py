from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
from deap import algorithms, base, creator, tools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from src.feature_selection.base import FeatureSelector, ProgressCallback, SelectionResult
from src.utils.exceptions import TrainingError

_CREATOR_READY = False


def _ensure_creator() -> None:
    global _CREATOR_READY
    if _CREATOR_READY:
        return
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
    _CREATOR_READY = True


class GAFeatureSelector(FeatureSelector):
    def __init__(
        self,
        population_size: int = 16,
        generations: int = 12,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.05,
        random_state: int = 42,
        min_features: int = 4,
        max_features: int | None = None,
        fitness_metric: str = "f1",
        lambda_penalty: float = 0.15,
        fitness_model: str = "logistic",  # logistic | bpn
    ) -> None:
        self.fitness_model = fitness_model
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.random_state = random_state
        self.min_features = min_features
        self.max_features = max_features
        self.fitness_metric = fitness_metric
        self.lambda_penalty = lambda_penalty

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        progress: ProgressCallback | None = None,
    ) -> SelectionResult:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        n_features = X.shape[1]
        if n_features == 0:
            raise TrainingError("No features available for genetic algorithm selection.")
        min_feat = max(1, min(self.min_features, n_features))
        max_feat = self.max_features or n_features
        max_feat = max(min_feat, min(max_feat, n_features))

        rng = random.Random(self.random_state)
        np_rng = np.random.default_rng(self.random_state)
        _ensure_creator()

        def repair(individual: list[int]) -> list[int]:
            selected = [i for i, bit in enumerate(individual) if bit]
            if len(selected) < min_feat:
                missing = [i for i in range(n_features) if individual[i] == 0]
                np_rng.shuffle(missing)
                for idx in missing[: min_feat - len(selected)]:
                    individual[idx] = 1
            selected = [i for i, bit in enumerate(individual) if bit]
            if len(selected) > max_feat:
                np_rng.shuffle(selected)
                for idx in selected[max_feat:]:
                    individual[idx] = 0
            if sum(individual) == 0:
                individual[int(np_rng.integers(0, n_features))] = 1
            return individual

        cache: dict[tuple, tuple[float]] = {}

        def evaluate(individual: list[int]) -> tuple[float]:
            repair(individual)
            mask = np.array(individual, dtype=bool)
            Xs = X[:, mask]
            key = tuple(individual)
            if key in cache:
                return cache[key]
            score = subset_score(Xs, y, self.fitness_metric, self.random_state, self.fitness_model)
            ratio = float(mask.sum()) / n_features
            cache[key] = (score - self.lambda_penalty * ratio,)
            return cache[key]

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", rng.randint, 0, 1)
        toolbox.register(
            "individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=max(self.mutation_probability, 1.0 / n_features))
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=self.population_size)
        for ind in population:
            repair(ind)

        start = time.perf_counter()
        history: list[float] = []
        feature_counts = np.zeros(n_features, dtype=np.float64)
        hof = tools.HallOfFame(1)

        invalid = [ind for ind in population if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit
        hof.update(population)
        feature_counts += np.asarray([ind for ind in population], dtype=float).sum(axis=0)
        history.append(float(hof[0].fitness.values[0]))
        if progress:
            progress(0, self.generations, _progress_payload(hof[0], n_features, history[-1]))

        for gen in range(1, self.generations + 1):
            offspring = algorithms.varAnd(
                population, toolbox, cxpb=self.crossover_probability, mutpb=1.0
            )
            for ind in offspring:
                repair(ind)
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=len(population))
            hof.update(population)
            feature_counts += np.asarray([ind for ind in population], dtype=float).sum(axis=0)
            history.append(float(hof[0].fitness.values[0]))
            if progress:
                progress(gen, self.generations, _progress_payload(hof[0], n_features, history[-1]))

        best = repair(list(hof[0]))
        mask = np.array(best, dtype=bool)
        selected = [name for name, bit in zip(feature_names, mask) if bit]
        runtime = time.perf_counter() - start
        return SelectionResult(
            selected_features=selected,
            mask=mask,
            fitness=float(hof[0].fitness.values[0]),
            history=history,
            runtime_seconds=runtime,
            metadata={
                "n_selected": int(mask.sum()),
                "n_total": n_features,
                "metric": self.fitness_metric,
                "generations": self.generations,
                "population_size": self.population_size,
                "fitness_model": self.fitness_model,
                "evaluations": len(cache),
                "feature_frequency": {
                    name: float(count) / float(self.population_size * (self.generations + 1))
                    for name, count in zip(feature_names, feature_counts)
                },
            },
        )


def _progress_payload(individual: Any, n_features: int, fitness: float) -> dict[str, Any]:
    n_sel = int(sum(individual))
    return {
        "best_fitness": fitness,
        "n_selected": n_sel,
        "n_total": n_features,
    }


def _cv_score(X: np.ndarray, y: np.ndarray, metric: str, seed: int) -> float:
    splits = min(3, max(2, int(np.min(np.bincount(y)))))
    if splits < 2:
        return 0.0
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    scores: list[float] = []
    for train_idx, val_idx in cv.split(X, y):
        model = LogisticRegression(max_iter=250, class_weight="balanced", solver="liblinear")
        try:
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
        except Exception:
            scores.append(0.0)
            continue
        if metric == "mcc":
            scores.append(float(matthews_corrcoef(y[val_idx], pred)))
        else:
            scores.append(float(f1_score(y[val_idx], pred, zero_division=0)))
    return float(np.mean(scores)) if scores else 0.0


_BPN_FIT_MAX_ROWS = 1500


def _bpn_holdout_score(X: np.ndarray, y: np.ndarray, metric: str, seed: int) -> float:
    """Fitness with a small, fast softmax BPN on one stratified train-internal holdout."""
    from sklearn.model_selection import train_test_split

    from src.models.bpn import BPNClassifier

    if len(y) > _BPN_FIT_MAX_ROWS:
        idx, _ = train_test_split(np.arange(len(y)), train_size=_BPN_FIT_MAX_ROWS, random_state=seed, stratify=y)
        X, y = X[idx], y[idx]
    try:
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    except ValueError:
        return 0.0
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    already_balanced = min(n0, n1) / max(n0, n1, 1) > 0.8
    model = BPNClassifier(hidden_sizes=(8, 4), dropout=0.0, learning_rate=0.01, epochs=15, batch_size=64,
                          patience=4, random_state=seed, tune_threshold=False,
                          balance_classes=not already_balanced)
    model.fit(Xtr, ytr, Xva, yva)
    pred = model.predict(Xva)
    if metric == "mcc":
        return float(matthews_corrcoef(yva, pred))
    return float(f1_score(yva, pred, zero_division=0))


def subset_score(X: np.ndarray, y: np.ndarray, metric: str, seed: int, model: str = "logistic") -> float:
    if model == "bpn":
        return _bpn_holdout_score(X, y, metric, seed)
    return _cv_score(X, y, metric, seed)

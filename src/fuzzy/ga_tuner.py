"""Genetic tuning of Mamdani triangular membership parameters.

Genes live in the unit interval and are decoded into ordered triangles so that
LOW / MEDIUM / HIGH (or Unlikely / Possible / Likely) stay valid. Fitness is
evaluated only on a train-internal validation split — never on the held-out test
set — using agreement between fuzzy risk and defect labels.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

from src.fuzzy.membership import Triangle
from src.fuzzy.risk_engine import FuzzyRiskEngine

ProgressCallback = Callable[[int, int, dict[str, Any]], None]

_CREATOR_READY = False


def _ensure_creator() -> None:
    global _CREATOR_READY
    if _CREATOR_READY:
        return
    if not hasattr(creator, "FuzzyFitness"):
        creator.create("FuzzyFitness", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "FuzzyIndividual"):
        creator.create("FuzzyIndividual", list, fitness=creator.FuzzyFitness)
    _CREATOR_READY = True


@dataclass
class FuzzyTuneResult:
    sets: dict[str, list[Triangle]]
    fitness: float
    history: list[float] = field(default_factory=list)
    runtime_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def decode_unit_genes(
    genes: np.ndarray,
    lo: float,
    hi: float,
    labels: tuple[str, str, str] = ("LOW", "MEDIUM", "HIGH"),
) -> list[Triangle]:
    """Map four unit-interval genes to three ordered overlapping triangles on [lo, hi]."""
    g = np.clip(np.asarray(genes, dtype=float).reshape(-1), 0.0, 1.0)
    if g.size < 4:
        g = np.pad(g, (0, 4 - g.size), constant_values=0.5)
    s = np.sort(g[:4])
    # Force a little separation so a < b < c after scaling.
    for i in range(1, 4):
        if s[i] <= s[i - 1]:
            s[i] = min(1.0, s[i - 1] + 1e-3)
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo

    def lerp(u: float) -> float:
        return float(lo + span * u)

    p1, p2, p3, p4 = (lerp(float(u)) for u in s)
    # LOW peaks early, HIGH peaks late; adjacent triangles share shoulders.
    return [
        Triangle(lo, p1, max(p2, p1 + 1e-6), labels[0]),
        Triangle(p1, p2, max(p3, p2 + 1e-6), labels[1]),
        Triangle(p2, p3, max(hi, p4, p3 + 1e-6), labels[2]),
    ]


def triangles_valid(triangles: list[Triangle]) -> bool:
    for tri in triangles:
        if not (tri.a <= tri.b <= tri.c):
            return False
        if not np.isfinite([tri.a, tri.b, tri.c]).all():
            return False
    return True


def fuzzy_agreement_metrics(
    engine: FuzzyRiskEngine,
    X_raw: pd.DataFrame,
    y: np.ndarray,
    probs: np.ndarray,
    high_labels: frozenset[str] = frozenset({"HIGH", "CRITICAL"}),
) -> dict[str, float]:
    """Agreement of fuzzy risk with binary defect labels (no model retraining)."""
    y = np.asarray(y).astype(int).reshape(-1)
    probs = np.asarray(probs, dtype=float).reshape(-1)
    scores = np.zeros(len(y), dtype=float)
    high_pred = np.zeros(len(y), dtype=int)
    for i in range(len(y)):
        row = X_raw.iloc[i].to_dict()
        risk = engine.assess(row, float(probs[i]))
        scores[i] = risk.risk_score
        high_pred[i] = int(risk.severity in high_labels)
    if len(np.unique(y)) < 2:
        corr = 0.0
    else:
        corr = float(np.corrcoef(scores, y)[0, 1]) if np.std(scores) > 1e-12 else 0.0
        if not np.isfinite(corr):
            corr = 0.0
    return {
        "risk_corr": corr,
        "high_f1": float(f1_score(y, high_pred, zero_division=0)),
        "high_mcc": float(matthews_corrcoef(y, high_pred)) if len(np.unique(y)) > 1 else 0.0,
        "high_precision": float(
            (high_pred & y).sum() / max(high_pred.sum(), 1)
        ),
        "high_recall": float((high_pred & y).sum() / max(y.sum(), 1)),
        "mean_risk": float(scores.mean()),
    }


class FuzzyMembershipGA:
    """Evolve triangular membership parameters for each fuzzy concept."""

    def __init__(
        self,
        population_size: int = 12,
        generations: int = 8,
        crossover_probability: float = 0.7,
        mutation_probability: float = 0.2,
        random_state: int = 42,
        fitness_mode: str = "hybrid",  # hybrid | high_f1 | corr
        max_val_rows: int = 600,
    ) -> None:
        self.max_val_rows = max_val_rows
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.random_state = random_state
        self.fitness_mode = fitness_mode

    def fit(
        self,
        X_raw: pd.DataFrame,
        y: np.ndarray,
        defect_probs: np.ndarray,
        risk_bounds: dict | None = None,
        progress: ProgressCallback | None = None,
    ) -> FuzzyTuneResult:
        y = np.asarray(y).astype(int).reshape(-1)
        defect_probs = np.asarray(defect_probs, dtype=float).reshape(-1)
        if len(X_raw) != len(y) or len(y) != len(defect_probs):
            raise ValueError("X_raw, y, and defect_probs must share the same length.")

        base_engine = FuzzyRiskEngine(risk_bounds=risk_bounds)
        base_engine.fit(X_raw)
        concepts = [c for c in base_engine.column_map.keys()]
        # Always include defect_prob linguistic sets in the chromosome.
        concept_order = concepts + ["defect_prob"]
        ranges: dict[str, tuple[float, float]] = {}
        for concept in concepts:
            col = base_engine.column_map[concept]
            vals = X_raw[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            lo, hi = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 1.0)
            ranges[concept] = (lo, hi)
        ranges["defect_prob"] = (0.0, 1.0)

        n_genes = 4 * len(concept_order)
        labels_metric = ("LOW", "MEDIUM", "HIGH")
        labels_prob = ("UNLIKELY", "POSSIBLE", "LIKELY")

        # Train-only inner split for fitness (no test leakage).
        try:
            idx_tr, idx_va = train_test_split(
                np.arange(len(y)),
                test_size=0.25,
                random_state=self.random_state,
                stratify=y,
            )
        except ValueError:
            rng = np.random.default_rng(self.random_state)
            perm = rng.permutation(len(y))
            cut = max(int(0.25 * len(y)), 4)
            idx_va, idx_tr = perm[:cut], perm[cut:]

        if len(idx_va) > self.max_val_rows:
            # Stratified subsample keeps fitness evaluation tractable on large sets (e.g. JM1).
            idx_va, _ = train_test_split(
                idx_va,
                train_size=self.max_val_rows,
                random_state=self.random_state,
                stratify=y[idx_va],
            )
        X_va = X_raw.iloc[idx_va].reset_index(drop=True)
        y_va = y[idx_va]
        p_va = defect_probs[idx_va]

        def decode(individual: list[float]) -> dict[str, list[Triangle]]:
            sets: dict[str, list[Triangle]] = {}
            for i, concept in enumerate(concept_order):
                chunk = individual[4 * i : 4 * i + 4]
                lo, hi = ranges[concept]
                labs = labels_prob if concept == "defect_prob" else labels_metric
                sets[concept] = decode_unit_genes(np.asarray(chunk), lo, hi, labs)
            return sets

        def evaluate(individual: list[float]) -> tuple[float]:
            sets = decode(individual)
            if not all(triangles_valid(v) for v in sets.values()):
                return (-1.0,)
            engine = FuzzyRiskEngine(risk_bounds=risk_bounds)
            engine.column_map = dict(base_engine.column_map)
            engine.sets = sets
            metrics = fuzzy_agreement_metrics(engine, X_va, y_va, p_va)
            if self.fitness_mode == "high_f1":
                score = metrics["high_f1"]
            elif self.fitness_mode == "corr":
                score = max(metrics["risk_corr"], 0.0)
            else:
                score = 0.6 * metrics["high_f1"] + 0.4 * max(metrics["risk_corr"], 0.0)
            return (float(score),)

        _ensure_creator()
        rng = random.Random(self.random_state)
        toolbox = base.Toolbox()
        toolbox.register("attr_float", rng.random)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.FuzzyIndividual,
            toolbox.attr_float,
            n=n_genes,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.4)
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.15, indpb=0.25)
        toolbox.register("select", tools.selTournament, tournsize=3)

        def clamp(ind: list[float]) -> list[float]:
            for i in range(len(ind)):
                ind[i] = float(min(1.0, max(0.0, ind[i])))
            return ind

        population = toolbox.population(n=self.population_size)
        for ind in population:
            clamp(ind)

        start = time.perf_counter()
        history: list[float] = []
        hof = tools.HallOfFame(1)

        invalid = [ind for ind in population if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit
        hof.update(population)
        history.append(float(hof[0].fitness.values[0]))
        if progress:
            progress(0, self.generations, {"best_fitness": history[-1], "n_concepts": len(concept_order)})

        for gen in range(1, self.generations + 1):
            offspring = algorithms.varAnd(
                population,
                toolbox,
                cxpb=self.crossover_probability,
                mutpb=self.mutation_probability,
            )
            for ind in offspring:
                clamp(ind)
                del ind.fitness.values
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=len(population))
            hof.update(population)
            history.append(float(hof[0].fitness.values[0]))
            if progress:
                progress(gen, self.generations, {"best_fitness": history[-1], "n_concepts": len(concept_order)})

        best_sets = decode(list(hof[0]))
        runtime = time.perf_counter() - start
        return FuzzyTuneResult(
            sets=best_sets,
            fitness=float(hof[0].fitness.values[0]),
            history=history,
            runtime_seconds=runtime,
            metadata={
                "concepts": concept_order,
                "ranges": {k: list(v) for k, v in ranges.items()},
                "fitness_mode": self.fitness_mode,
                "generations": self.generations,
                "population_size": self.population_size,
                "n_val": int(len(idx_va)),
            },
        )

from __future__ import annotations

import numpy as np
import pandas as pd


def defect_recall_at_budget(y_true, scores, budget_frac: float = 0.10) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    k = max(1, int(round(n * budget_frac)))
    order = np.argsort(-scores)[:k]
    found = int(y_true[order].sum())
    total = int(y_true.sum())
    return found / total if total else 0.0


def testing_budget_table(
    y_true,
    score_map: dict[str, np.ndarray],
    budget_frac: float = 0.10,
    random_state: int = 42,
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    rng = np.random.default_rng(random_state)
    random_scores = rng.random(len(y_true))
    rows = [
        {
            "Strategy": "Random selection",
            "Budget": f"{int(budget_frac * 100)}%",
            "Defects found (share of actual defective modules)": defect_recall_at_budget(
                y_true, random_scores, budget_frac
            ),
        }
    ]
    for name, scores in score_map.items():
        rows.append(
            {
                "Strategy": name,
                "Budget": f"{int(budget_frac * 100)}%",
                "Defects found (share of actual defective modules)": defect_recall_at_budget(
                    y_true, scores, budget_frac
                ),
            }
        )
    return pd.DataFrame(rows)

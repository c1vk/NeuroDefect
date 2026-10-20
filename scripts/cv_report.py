"""Summarise CV CSVs (reports/cv/*.csv) with mean ± std, Wilcoxon signed-rank and Friedman tests.

Writes reports/cv_results.md. Wilcoxon pairs the per-fold scores of two models on the same folds
(two-sided, zero differences dropped via zero_method='wilcox'). Friedman ranks all models per fold.
Note: folds from repeated CV overlap in training data, so p-values are optimistic (Nadeau & Bengio 2003).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "reports" / "cv"
ORDER = ["synthetic_sample", "CM1", "PC1", "JM1"]
MODELS = ["Hybrid", "BPN", "Ensemble", "LR", "RF", "SVM"]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc"]


def ms(x: pd.Series) -> str:
    return f"{x.mean():.3f} ± {x.std(ddof=0):.3f}"


def wil(a: np.ndarray, b: np.ndarray) -> str:
    d = a - b
    if np.allclose(d, 0):
        return "n/a (identical)"
    try:
        return f"{wilcoxon(a, b, zero_method='wilcox').pvalue:.4f}"
    except ValueError:
        return "n/a"


def pivot(df: pd.DataFrame, key: str, metric: str) -> pd.DataFrame:
    return df.pivot_table(index="fold", columns=key, values=metric)


def main_section(df: pd.DataFrame, name: str) -> list[str]:
    folds, reps = int(df["folds"].iloc[0]), int(df["repeats"].iloc[0])
    out = [f"### {name} ({reps}×{folds} CV, {df['fold'].nunique()} folds)", "",
           "| Model | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    cls = df[df["model"].isin(MODELS)]
    for m in MODELS:
        g = cls[cls["model"] == m]
        if len(g):
            out.append(f"| {m} | " + " | ".join(ms(g[k]) for k in METRICS) + " |")
    fz = df[df["model"].str.startswith("Fuzzy")]
    for m in fz["model"].unique():
        g = fz[fz["model"] == m]
        out.append(f"| {m} | — | {ms(g['precision'])} | {ms(g['recall'])} | {ms(g['f1'])} | — | {ms(g['mcc'])} |")
    if "ens_weight" in df:
        w = df.loc[df["model"] == "Ensemble", "ens_weight"]
        out += ["", f"Ensemble BPN weight chosen on train-only validation: mean {w.mean():.2f} (min {w.min():.1f}, max {w.max():.1f})."]
    out += ["", "| Comparison (paired over folds) | Δ mean F1 | Wilcoxon p (F1) | Δ mean MCC | Wilcoxon p (MCC) |",
            "|---|---|---|---|---|"]
    pf, pm = pivot(cls, "model", "f1"), pivot(cls, "model", "mcc")
    for a in ("Hybrid", "BPN", "Ensemble"):
        for b in ("LR", "RF", "SVM") + (("Hybrid",) if a == "Ensemble" else ()):
            if a in pf and b in pf:
                out.append(f"| {a} vs {b} | {(pf[a] - pf[b]).mean():+.3f} | {wil(pf[a].values, pf[b].values)} | "
                           f"{(pm[a] - pm[b]).mean():+.3f} | {wil(pm[a].values, pm[b].values)} |")
    cols = [m for m in MODELS if m in pf]
    for metric, pv in (("F1", pf), ("MCC", pm)):
        stat, p = friedmanchisquare(*[pv[c].values for c in cols])
        ranks = pv[cols].rank(axis=1, ascending=False).mean().sort_values()
        out.append("")
        out.append(f"Friedman ({metric}, {len(cols)} models): χ² = {stat:.2f}, p = {p:.3g}. Mean rank (1 = best): "
                   + ", ".join(f"{k} {v:.2f}" for k, v in ranks.items()))
    return out + [""]


def selector_section(df: pd.DataFrame, name: str) -> list[str]:
    folds, reps = int(df["folds"].iloc[0]), int(df["repeats"].iloc[0])
    out = [f"### {name} ({reps}×{folds} CV) — Hybrid BPN with different feature selectors", "",
           "| Selector | F1 | MCC | ROC-AUC | Recall | #features | selection s/fold |", "|---|---|---|---|---|---|---|"]
    for s in ["none", "ga_logistic", "ga_bpn", "pso"]:
        g = df[df["selector"] == s]
        if len(g):
            out.append(f"| {s} | {ms(g['f1'])} | {ms(g['mcc'])} | {ms(g['roc_auc'])} | {ms(g['recall'])} | "
                       f"{g['n_features'].mean():.1f} | {g['selection_seconds'].mean():.1f} |")
    pf, pm = pivot(df, "selector", "f1"), pivot(df, "selector", "mcc")
    out += ["", "| Comparison | Δ F1 | p (F1) | Δ MCC | p (MCC) |", "|---|---|---|---|---|"]
    for a, b in (("ga_logistic", "none"), ("ga_bpn", "ga_logistic"), ("ga_bpn", "none"), ("pso", "ga_logistic"), ("pso", "none")):
        if a in pf and b in pf:
            out.append(f"| {a} vs {b} | {(pf[a] - pf[b]).mean():+.3f} | {wil(pf[a].values, pf[b].values)} | "
                       f"{(pm[a] - pm[b]).mean():+.3f} | {wil(pm[a].values, pm[b].values)} |")
    return out + [""]


def main() -> None:
    lines = ["# Repeated stratified cross-validation results", "",
             "All preprocessing, SMOTE, feature selection, BPN early stopping, threshold tuning, ensemble weighting and "
             "fuzzy membership fitting happen inside each training fold. Metrics: mean ± std over test folds. "
             "Wilcoxon signed-rank tests pair the same folds; Friedman ranks models per fold. Folds of repeated CV "
             "share training data, so p-values are optimistic and should be read as indicative.", "",
             "- **Hybrid** = softmax BPN on GA-selected features (the app's Neuro-Fuzzy-GA classifier).",
             "- **BPN** = softmax BPN on all features. **Ensemble** = w·BPN + (1−w)·RF on GA features, w and threshold tuned on held-out real training rows.",
             "- **Fuzzy>=HIGH (x)** = derived risk tier ≥ HIGH used as the defect decision, with probabilities from x.", "",
             "## 1–2. Models and significance tests", ""]
    for d in ORDER:
        f = CV / f"main_{d}.csv"
        if f.exists():
            lines += main_section(pd.read_csv(f), d)
    lines += ["## 3–4. Feature selection: none vs GA (logistic probe) vs GA (small BPN) vs binary PSO", ""]
    for d in ORDER:
        f = CV / f"selectors_{d}.csv"
        if f.exists():
            lines += selector_section(pd.read_csv(f), d)
    (ROOT / "reports" / "cv_results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())

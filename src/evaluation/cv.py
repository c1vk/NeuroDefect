"""Leakage-safe repeated stratified cross-validation.

Every fold re-fits imputer/scaler, SMOTE, feature selection (GA / PSO), BPN early stopping,
threshold tuning, ensemble weight and fuzzy memberships on the fold's training part only;
metrics are computed once on the fold's test part.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from src.data.cleaner import clean_frame
from src.data.columns import drop_derived_halstead
from src.data.imbalance import apply_smote
from src.data.preprocessing import fit_preprocessor
from src.data.validator import validate_target
from src.evaluation.metrics import classification_metrics
from src.feature_selection.ga_selector import GAFeatureSelector
from src.feature_selection.pso_selector import PSOFeatureSelector
from src.fuzzy.ga_tuner import fuzzy_agreement_metrics
from src.fuzzy.risk_engine import FuzzyRiskEngine
from src.models.baselines import make_sklearn_baselines
from src.models.bpn import BPNClassifier
from src.models.ensemble import BPNRFEnsemble, make_rf

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc"]


@dataclass
class FoldOptions:
    selector: str = "ga_logistic"  # none | ga_logistic | ga_bpn | pso
    smote: bool = True
    models: tuple[str, ...] = ("BPN", "Hybrid", "Ensemble", "LR", "RF", "SVM")
    fuzzy: bool = True
    threshold_mode: str = "f1"
    bpn: dict = field(default_factory=lambda: dict(hidden_sizes=(64, 32), dropout=0.2, learning_rate=0.001,
                                                   optimizer="adam", epochs=80, batch_size=32, patience=12))
    ga: dict = field(default_factory=lambda: dict(population_size=16, generations=12, min_features=4,
                                                  lambda_penalty=0.15, mutation_probability=0.05))


def prepare_xy(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Label-free row/column cleaning (dedupe, id/constant columns) shared by all folds."""
    t = validate_target(frame[target])
    aligned = frame.copy()
    aligned[target] = t
    X, y, _, _ = clean_frame(aligned, target)
    X, _ = drop_derived_halstead(X)
    return X.reset_index(drop=True), validate_target(y).astype(int).to_numpy()


def _select(opts: FoldOptions, X: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[list[str], float]:
    t0 = time.perf_counter()
    names = list(X.columns)
    if opts.selector == "none":
        return names, 0.0
    if opts.selector == "pso":
        sel = PSOFeatureSelector(swarm_size=opts.ga["population_size"], iterations=opts.ga["generations"],
                                 random_state=seed, min_features=opts.ga["min_features"],
                                 lambda_penalty=opts.ga["lambda_penalty"])
    else:
        sel = GAFeatureSelector(random_state=seed, fitness_model="bpn" if opts.selector == "ga_bpn" else "logistic",
                                **opts.ga)
    res = sel.fit(X.to_numpy(), y, names)
    return res.selected_features, time.perf_counter() - t0


def fit_ensemble(X_tr_sel: pd.DataFrame, y_tr: np.ndarray, bpn_kw: dict, smote: bool, threshold_mode: str,
                 seed: int) -> BPNRFEnsemble:
    """Hold out 20% of the *real* training rows; SMOTE + fit BPN and RF on the rest; tune weight/threshold
    on the real held-out rows (tuning on SMOTE-synthetic rows over-trusts RF), then refit nothing."""
    i_tr, i_va = train_test_split(np.arange(len(y_tr)), test_size=0.2, random_state=seed, stratify=y_tr)
    Xi, yi, _ = apply_smote(X_tr_sel.iloc[i_tr].reset_index(drop=True), pd.Series(y_tr[i_tr]), enabled=smote,
                            random_state=seed)
    Xv, yv = X_tr_sel.iloc[i_va].to_numpy(dtype=np.float32), y_tr[i_va]
    bpn = BPNClassifier(**bpn_kw).fit(Xi.to_numpy(), yi.to_numpy(), Xv, yv)
    rf = make_rf(seed).fit(Xi.to_numpy(dtype=np.float32), yi.to_numpy())
    return BPNRFEnsemble.fit_weight(bpn, rf, Xv, yv, threshold_mode)


def run_fold(X_tr_raw: pd.DataFrame, y_tr: np.ndarray, X_te_raw: pd.DataFrame, y_te: np.ndarray,
             opts: FoldOptions, seed: int) -> list[dict]:
    pre = fit_preprocessor(X_tr_raw)
    X_tr = pre.transform(X_tr_raw)
    X_te = pre.transform(X_te_raw)
    X_res, y_res, _ = apply_smote(X_tr, pd.Series(y_tr), enabled=opts.smote, random_state=seed)
    y_res = y_res.to_numpy().astype(int)
    selected, sel_secs = _select(opts, X_res, y_res, seed)
    bpn_kw = dict(opts.bpn, random_state=seed, threshold_mode=opts.threshold_mode,
                  balance_classes=not opts.smote)
    rows: list[dict] = []
    probs: dict[str, np.ndarray] = {}

    def add(name: str, p: np.ndarray, thr: float, extra: dict | None = None) -> None:
        m = classification_metrics(y_te, (p >= thr).astype(int), p)
        probs[name] = p
        rows.append({"model": name, **{k: float(getattr(m, k)) for k in METRICS},
                     "n_features": len(selected) if name in {"Hybrid", "Ensemble"} else X_res.shape[1],
                     "selection_seconds": sel_secs, **(extra or {})})

    if {"Hybrid", "Ensemble"} & set(opts.models):
        Xs = X_res[selected].to_numpy(dtype=np.float32)
        i_tr, i_va = train_test_split(np.arange(len(y_res)), test_size=0.15, random_state=seed, stratify=y_res)
        hybrid = BPNClassifier(**bpn_kw).fit(Xs[i_tr], y_res[i_tr], Xs[i_va], y_res[i_va])
        Xte_s = X_te[selected].to_numpy(dtype=np.float32)
        if "Hybrid" in opts.models:
            add("Hybrid", hybrid.predict_proba(Xte_s)[:, 1], hybrid.decision_threshold_)
        if "Ensemble" in opts.models:
            ens = fit_ensemble(X_tr[selected], y_tr, bpn_kw, opts.smote, opts.threshold_mode, seed)
            add("Ensemble", ens.predict_proba(Xte_s)[:, 1], ens.decision_threshold_, {"ens_weight": ens.weight})
    if "BPN" in opts.models:
        full = BPNClassifier(**bpn_kw).fit(X_res.to_numpy(), y_res)
        add("BPN", full.predict_proba(X_te.to_numpy())[:, 1], full.decision_threshold_)
    base = make_sklearn_baselines(seed, class_weight=None if opts.smote else "balanced")
    for short, long in (("LR", "Logistic Regression"), ("RF", "Random Forest"), ("SVM", "Support Vector Machine")):
        if short in opts.models:
            est = clone(base[long]).fit(X_res, y_res)
            add(short, est.predict_proba(X_te)[:, 1], 0.5)

    if opts.fuzzy:
        engine = FuzzyRiskEngine().fit(X_tr_raw)
        for src in ("Hybrid", "Ensemble"):
            if src in probs:
                fm = fuzzy_agreement_metrics(engine, X_te_raw.reset_index(drop=True), y_te, probs[src])
                rows.append({"model": f"Fuzzy>=HIGH ({src})", "precision": fm["high_precision"],
                             "recall": fm["high_recall"], "f1": fm["high_f1"], "mcc": fm["high_mcc"],
                             "risk_corr": fm["risk_corr"]})
    return rows


def repeated_cv(frame: pd.DataFrame, target: str, opts: FoldOptions, n_splits: int = 5, n_repeats: int = 5,
                seed: int = 42, log=print) -> pd.DataFrame:
    X, y = prepare_xy(frame, target)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    out: list[dict] = []
    for k, (tr, te) in enumerate(rskf.split(X, y)):
        t0 = time.perf_counter()
        rows = run_fold(X.iloc[tr].reset_index(drop=True), y[tr], X.iloc[te].reset_index(drop=True), y[te],
                        opts, seed + k)
        for r in rows:
            r.update({"fold": k, "repeat": k // n_splits, "selector": opts.selector})
        out.extend(rows)
        if log:
            f1s = {r["model"]: round(r["f1"], 3) for r in rows if "Fuzzy" not in r["model"]}
            log(f"fold {k + 1}/{n_splits * n_repeats} {time.perf_counter() - t0:.1f}s {f1s}")
    return pd.DataFrame(out)

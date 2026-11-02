from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from src.data.cleaner import CleaningLog, clean_frame
from src.data.columns import drop_derived_halstead
from src.data.imbalance import SmoteReport, apply_smote, resolve_smote
from src.data.loader import LoadedDataset
from src.data.preprocessing import FittedPreprocessor, fit_preprocessor
from src.data.quality import DatasetQualityReport, build_quality_report
from src.data.splitting import SplitResult, stratified_split
from src.data.validator import validate_target
from src.evaluation.metrics import ClassificationMetrics, classification_metrics, roc_points
from src.explainability.explainer import ga_feature_ranks
from src.feature_selection.ga_selector import GAFeatureSelector
from src.fuzzy.ga_tuner import FuzzyMembershipGA, fuzzy_agreement_metrics
from src.fuzzy.risk_engine import FuzzyRiskEngine
from src.models.baselines import make_sklearn_baselines
from src.models.bpn import BPNClassifier, TrainHistory
from src.pipeline.artifacts import new_run_dir, save_run
from src.utils.config import AppConfig, ROOT
from src.utils.exceptions import DataError, TrainingError
from src.utils.reproducibility import set_seed

ProgressFn = Callable[[str, float, str, dict[str, Any]], None]


@dataclass
class TrainingResult:
    dataset_name: str
    target_column: str
    quality: DatasetQualityReport
    cleaning: CleaningLog
    split: SplitResult
    preprocessor: FittedPreprocessor
    smote: SmoteReport
    selected_features: list[str]
    all_features: list[str]
    ga_history: list[float]
    ga_fitness: float
    ga_runtime: float
    bpn_history: TrainHistory
    metrics: dict[str, ClassificationMetrics]
    roc: dict[str, tuple[np.ndarray, np.ndarray]]
    scores: dict[str, np.ndarray]
    y_test: np.ndarray
    X_test_raw: pd.DataFrame
    X_test_scaled: pd.DataFrame
    display_ids: list[str]
    models: dict[str, Any]
    bpn: BPNClassifier
    fuzzy: FuzzyRiskEngine
    feature_ranks: list[tuple[str, float]]
    duration: float
    seed: int
    mode: str
    run_dir: str
    logs: list[str] = field(default_factory=list)
    leakage_guards: dict[str, Any] = field(default_factory=dict)
    ga_feature_frequency: dict[str, float] = field(default_factory=dict)
    fuzzy_mode: str = "fixed_quantile"
    fuzzy_tune_history: list[float] = field(default_factory=list)
    fuzzy_tune_fitness: float = 0.0
    fuzzy_metrics_test: dict[str, float] = field(default_factory=dict)
    fuzzy_metrics_fixed_test: dict[str, float] = field(default_factory=dict)
    variant: str = "full"
    shap_background: np.ndarray | None = None
    ensemble: Any = None


def train_pipeline(
    dataset: LoadedDataset,
    config: AppConfig,
    progress: ProgressFn | None = None,
    variant: str = "full",
    with_baselines: bool = True,
) -> TrainingResult:
    if dataset.target_column is None:
        raise DataError("Target column was not found. Please select the defect label column.")

    def report(phase: str, frac: float, message: str, extra: dict | None = None) -> None:
        if progress:
            progress(phase, frac, message, extra or {})

    seed = int(config.training().get("random_state", 42))
    set_seed(seed)
    started = time.perf_counter()
    logs: list[str] = []

    ga_cfg = config.ga()
    fuzzy_cfg = config.fuzzy()
    smote_cfg = config.smote()
    use_ga = bool(ga_cfg.get("enabled", True)) and variant not in {"no_ga"}
    use_smote = resolve_smote(
        dataset.name,
        bool(smote_cfg.get("enabled", True)) and variant not in {"no_smote"},
        force=bool(smote_cfg.get("force", False)),
        skip=smote_cfg.get("skip_for"),
    )
    use_fuzzy = bool(fuzzy_cfg.get("enabled", True)) and variant not in {"no_fuzzy"}
    ga_tune_fuzzy = bool(fuzzy_cfg.get("ga_tune", False)) or variant in {"fuzzy_ga"}
    if variant == "fuzzy_fixed":
        ga_tune_fuzzy = False

    report("validate", 0.04, "Validating target column and cleaning the dataset")
    target = validate_target(dataset.frame[dataset.target_column])
    aligned = dataset.frame.copy()
    aligned[dataset.target_column] = target
    features, y, cleaning, id_series = clean_frame(aligned, dataset.target_column)
    y = validate_target(y)
    logs.extend(cleaning.lines())
    features, dropped_halstead = drop_derived_halstead(features)
    if dropped_halstead:
        logs.append(
            "Dropped derived Halstead columns before selection: " + ", ".join(dropped_halstead)
        )
    if features.shape[1] < 2:
        raise TrainingError("Fewer than two numeric features remain after cleaning.")

    quality = build_quality_report(
        dataset.frame, dataset.target_column, dataset.name, list(features.columns)
    )

    report("split", 0.10, "Creating a stratified train/test split")
    test_size = float(config.training().get("test_size", 0.3))
    split = stratified_split(
        features, y.astype(int), test_size=test_size, random_state=seed, ids=id_series
    )
    logs.append(
        f"Train/test split: {len(split.X_train)} / {len(split.X_test)} "
        f"(stratified, test_size={test_size:.0%})"
    )

    report("preprocess", 0.18, "Fitting preprocessing on training data only")
    preprocessor = fit_preprocessor(split.X_train)
    X_train = preprocessor.transform(split.X_train)
    X_test = preprocessor.transform(split.X_test)
    y_train = split.y_train.astype(int)
    y_test = split.y_test.astype(int)

    report("smote", 0.26, "Applying SMOTE to training data only" if use_smote else "Skipping SMOTE")
    X_res, y_res, smote = apply_smote(
        X_train,
        y_train,
        enabled=use_smote,
        random_state=seed,
        k_neighbors=int(smote_cfg.get("k_neighbors", 5)),
    )
    logs.append(
        f"Before SMOTE — defective: {smote.before_defective}, non-defective: {smote.before_clean}"
    )
    logs.append(
        f"After SMOTE — defective: {smote.after_defective}, non-defective: {smote.after_clean}"
    )
    logs.append("Test distribution was not resampled.")
    if use_smote:
        logs.append("Class weights are off because SMOTE already balanced the training rows.")
    else:
        if bool(smote_cfg.get("enabled", True)) and not bool(smote_cfg.get("force", False)):
            logs.append(f"SMOTE skipped for {dataset.name}. The ablation showed it hurt this dataset.")
        logs.append("Class weights stay on to handle the imbalance.")

    report("ga", 0.34, "Running genetic algorithm feature selection on training data" if use_ga else "Using all features (GA disabled)")

    def ga_progress(gen: int, total: int, payload: dict[str, Any]) -> None:
        frac = 0.34 + 0.22 * (gen / max(total, 1))
        report(
            "ga",
            frac,
            f"Generation {gen} / {total}  ·  Best fitness {payload['best_fitness']:.3f}  ·  "
            f"Features selected: {payload['n_selected']} / {payload['n_total']}",
            payload,
        )

    feature_frequency: dict[str, float] = {}
    if use_ga:
        method = str(ga_cfg.get("method", "ga")).lower()
        if method == "pso":
            from src.feature_selection.pso_selector import PSOFeatureSelector

            selector = PSOFeatureSelector(
                swarm_size=int(ga_cfg.get("population_size", 16)),
                iterations=int(ga_cfg.get("generations", 12)),
                random_state=int(ga_cfg.get("random_state", seed)),
                min_features=int(ga_cfg.get("min_features", 4)),
                fitness_metric=str(ga_cfg.get("fitness_metric", "f1")),
                lambda_penalty=float(ga_cfg.get("lambda_penalty", 0.15)),
                fitness_model=str(ga_cfg.get("fitness_model", "logistic")),
            )
        else:
            selector = GAFeatureSelector(
                fitness_model=str(ga_cfg.get("fitness_model", "logistic")),
                population_size=int(ga_cfg.get("population_size", 16)),
                generations=int(ga_cfg.get("generations", 12)),
                crossover_probability=float(ga_cfg.get("crossover_probability", 0.8)),
                mutation_probability=float(ga_cfg.get("mutation_probability", 0.05)),
                random_state=int(ga_cfg.get("random_state", seed)),
                min_features=int(ga_cfg.get("min_features", 4)),
                max_features=ga_cfg.get("max_features"),
                fitness_metric=str(ga_cfg.get("fitness_metric", "f1")),
                lambda_penalty=float(ga_cfg.get("lambda_penalty", 0.15)),
            )
        selection = selector.fit(
            X_res.to_numpy(), y_res.to_numpy(), list(X_res.columns), progress=ga_progress
        )
        selected = selection.selected_features
        ga_history = selection.history
        ga_fitness = selection.fitness
        ga_runtime = selection.runtime_seconds
        feature_frequency = dict(selection.metadata.get("feature_frequency", {}))
        probe = str(ga_cfg.get("fitness_model", "logistic"))
        probe_name = "small BPN holdout" if probe == "bpn" else "logistic feature filter"
        logs.append(
            f"GA ({probe_name}) selected {len(selected)} / {len(X_res.columns)} features "
            f"(fitness={selection.fitness:.3f}, {selection.runtime_seconds:.1f}s)"
        )
    else:
        selected = list(X_res.columns)
        ga_history = []
        ga_fitness = 0.0
        ga_runtime = 0.0
        feature_frequency = {name: 1.0 for name in selected}
        logs.append(f"GA disabled — using all {len(selected)} features.")

    X_train_sel = X_res[selected]
    X_test_sel = X_test[selected]
    X_train_all = X_res
    X_test_all = X_test

    inner_X, inner_X_val, inner_y, inner_y_val = train_test_split(
        X_train_sel,
        y_res,
        test_size=0.15,
        random_state=seed,
        stratify=y_res,
    )

    train_cfg = config.training()
    bpn_kwargs = dict(
        hidden_sizes=tuple(train_cfg.get("hidden_sizes", [64, 32])),
        dropout=float(train_cfg.get("dropout", 0.2)),
        learning_rate=float(train_cfg.get("learning_rate", 0.001)),
        optimizer=str(train_cfg.get("optimizer", "adam")),
        momentum=float(train_cfg.get("momentum", 0.9)),
        epochs=int(train_cfg.get("epochs", 80)),
        batch_size=int(train_cfg.get("batch_size", 32)),
        patience=int(train_cfg.get("patience", 12)),
        random_state=seed,
        tune_threshold=bool(train_cfg.get("tune_threshold", True)),
        threshold_mode=str(train_cfg.get("threshold_mode", "f1")),
        accuracy_floor=float(train_cfg.get("accuracy_floor", 0.65)),
        fn_cost=float(train_cfg.get("fn_cost", 2.0)),
        balance_classes=not use_smote,
    )

    report("bpn", 0.60, "Training the BPN neural network on selected features")
    bpn_history_holder: dict[str, TrainHistory] = {}

    def bpn_cb(epoch: int, total: int, history: TrainHistory) -> None:
        bpn_history_holder["h"] = history
        acc = history.train_acc[-1] if history.train_acc else 0.0
        loss = history.train_loss[-1] if history.train_loss else 0.0
        report(
            "bpn",
            0.60 + 0.16 * (epoch / max(total, 1)),
            f"Epoch {epoch} / {total}  ·  train F1 {history.train_f1[-1]:.3f}  ·  val F1 {history.val_f1[-1]:.3f}",
            {"epoch": epoch, "total": total, "acc": acc, "loss": loss, "history": history},
        )

    bpn = BPNClassifier(**bpn_kwargs)
    bpn.fit(inner_X.to_numpy(), inner_y.to_numpy(), inner_X_val.to_numpy(), inner_y_val.to_numpy(), callback=bpn_cb)
    bpn_full = BPNClassifier(**bpn_kwargs)
    bpn_full.fit(X_train_all.to_numpy(), y_res.to_numpy())

    report("baselines", 0.78, "Training baseline classifiers on the same training split")
    fitted_models: dict[str, Any] = {}
    metrics: dict[str, ClassificationMetrics] = {}
    scores: dict[str, np.ndarray] = {}
    roc: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    baselines = make_sklearn_baselines(seed, class_weight=None if use_smote else "balanced") if with_baselines else {}
    for name, model in baselines.items():
        estimator = clone(model)
        estimator.fit(X_train_all, y_res)
        proba = estimator.predict_proba(X_test_all)[:, 1]
        pred = (proba >= 0.5).astype(int)
        fitted_models[name] = estimator
        metrics[name] = classification_metrics(y_test, pred, proba)
        scores[name] = proba
        roc[name] = roc_points(y_test, proba)

    bpn_all_scores = bpn_full.predict_proba(X_test_all.to_numpy())[:, 1]
    bpn_all_thr = float(getattr(bpn_full, "decision_threshold_", 0.5))
    bpn_all_pred = (bpn_all_scores >= bpn_all_thr).astype(int)
    metrics["BPN"] = classification_metrics(y_test, bpn_all_pred, bpn_all_scores)
    scores["BPN"] = bpn_all_scores
    roc["BPN"] = roc_points(y_test, bpn_all_scores)
    fitted_models["BPN"] = bpn_full
    thr_mode = str(train_cfg.get("threshold_mode", "f1"))
    logs.append(f"BPN decision threshold ({thr_mode}): {bpn_all_thr:.3f}")

    hybrid_scores = bpn.predict_proba(X_test_sel.to_numpy())[:, 1]
    hybrid_thr = float(getattr(bpn, "decision_threshold_", 0.5))
    hybrid_pred = (hybrid_scores >= hybrid_thr).astype(int)
    metrics["Neuro-Fuzzy-GA"] = classification_metrics(y_test, hybrid_pred, hybrid_scores)
    scores["Neuro-Fuzzy-GA"] = hybrid_scores
    roc["Neuro-Fuzzy-GA"] = roc_points(y_test, hybrid_scores)
    fitted_models["Neuro-Fuzzy-GA"] = bpn
    logs.append(f"Neuro-Fuzzy-GA decision threshold ({thr_mode}): {hybrid_thr:.3f}")

    ensemble = None
    if bool(train_cfg.get("ensemble", False)):
        from src.evaluation.cv import fit_ensemble

        ens_kw = {k: v for k, v in bpn_kwargs.items() if k not in {"tune_threshold"}}
        ensemble = fit_ensemble(X_train[selected], y_train.to_numpy(), ens_kw, use_smote, thr_mode, seed)
        ens_scores = ensemble.predict_proba(X_test_sel.to_numpy())[:, 1]
        metrics["BPN+RF Ensemble"] = classification_metrics(
            y_test, (ens_scores >= ensemble.decision_threshold_).astype(int), ens_scores)
        scores["BPN+RF Ensemble"] = ens_scores
        roc["BPN+RF Ensemble"] = roc_points(y_test, ens_scores)
        hybrid_scores = ens_scores  # fuzzy layer is fed by the ensemble probability
        logs.append(f"BPN+RF ensemble: weight {ensemble.weight:.2f}, threshold {ensemble.decision_threshold_:.3f}")

    # Fuzzy engine: fixed quantile baseline, optional GA-tuned memberships (train-only).
    report("fuzzy", 0.88, "Fitting the fuzzy risk engine on training metrics")
    fuzzy_fixed = FuzzyRiskEngine(risk_bounds=fuzzy_cfg.get("risk_bounds"))
    fuzzy_fixed.fit(split.X_train)

    fuzzy_tune_history: list[float] = []
    fuzzy_tune_fitness = 0.0
    fuzzy_mode = "disabled"
    fuzzy = fuzzy_fixed
    custom_sets = None

    if use_fuzzy:
        fuzzy_mode = "fixed_quantile"
        # Train-only defect probabilities for fuzzy GA fitness / agreement.
        train_probs = None
        if ga_tune_fuzzy:
            report("fuzzy_oof", 0.89, "Computing out-of-fold BPN probabilities on training data")
            train_probs = _oof_bpn_probs(
                X_train[selected],
                y_train,
                bpn_kwargs,
                use_smote=use_smote,
                k_neighbors=int(smote_cfg.get("k_neighbors", 5)),
                n_splits=int((fuzzy_cfg.get("ga", {}) or {}).get("oof_folds", 3)),
                seed=seed,
            )
            logs.append("Fuzzy GA fitness uses out-of-fold BPN probabilities (k-fold within train).")
            report("fuzzy_ga", 0.90, "Tuning fuzzy membership triangles with GA (train-only)")
            fuzzy_ga_cfg = fuzzy_cfg.get("ga", {}) or {}

            def fuzzy_progress(gen: int, total: int, payload: dict[str, Any]) -> None:
                report(
                    "fuzzy_ga",
                    0.90 + 0.05 * (gen / max(total, 1)),
                    f"Fuzzy GA {gen}/{total} · best {payload.get('best_fitness', 0):.3f}",
                    payload,
                )

            tuner = FuzzyMembershipGA(
                population_size=int(fuzzy_ga_cfg.get("population_size", 12)),
                generations=int(fuzzy_ga_cfg.get("generations", 8)),
                crossover_probability=float(fuzzy_ga_cfg.get("crossover_probability", 0.7)),
                mutation_probability=float(fuzzy_ga_cfg.get("mutation_probability", 0.2)),
                random_state=int(fuzzy_ga_cfg.get("random_state", seed)),
                fitness_mode=str(fuzzy_ga_cfg.get("fitness_mode", "hybrid")),
            )
            tune = tuner.fit(
                split.X_train.reset_index(drop=True),
                y_train.to_numpy(),
                train_probs,
                risk_bounds=fuzzy_cfg.get("risk_bounds"),
                progress=fuzzy_progress,
            )
            custom_sets = tune.sets
            fuzzy_tune_history = tune.history
            fuzzy_tune_fitness = tune.fitness
            fuzzy_mode = "ga_tuned"
            fuzzy = FuzzyRiskEngine(risk_bounds=fuzzy_cfg.get("risk_bounds"))
            fuzzy.fit(split.X_train, custom_sets=custom_sets)
            logs.append(
                f"Fuzzy GA tuned memberships (fitness={tune.fitness:.3f}, {tune.runtime_seconds:.1f}s)"
            )
        else:
            logs.append("Fuzzy memberships from training-set quantiles (fixed).")
    else:
        logs.append("Fuzzy risk engine disabled for this variant.")

    test_probs = hybrid_scores
    fuzzy_metrics_test: dict[str, float] = {}
    fuzzy_metrics_fixed_test: dict[str, float] = {}
    if use_fuzzy:
        fuzzy_metrics_test = fuzzy_agreement_metrics(
            fuzzy, split.X_test.reset_index(drop=True), y_test.to_numpy(), test_probs
        )
        fuzzy_metrics_fixed_test = fuzzy_agreement_metrics(
            fuzzy_fixed, split.X_test.reset_index(drop=True), y_test.to_numpy(), test_probs
        )
        logs.append(
            f"Held-out fuzzy HIGH-F1 ({fuzzy_mode}): {fuzzy_metrics_test.get('high_f1', float('nan')):.3f} "
            f"| fixed: {fuzzy_metrics_fixed_test.get('high_f1', float('nan')):.3f}"
        )

    from src.explainability.shap_explainer import sample_background

    shap_background = sample_background(X_train_sel.to_numpy(), k=50, seed=seed)
    ranks = ga_feature_ranks(X_res, y_res, selected)
    duration = time.perf_counter() - started
    logs.append(f"Training duration: {duration:.1f}s")
    logs.append(f"Final training features: {len(selected)}")

    ids = split.ids_test or _display_ids(dataset.name, split.test_index)
    run_dir = new_run_dir(Path(os.environ.get("NEURODEFECT_MODELS_DIR", ROOT / "models")))
    bundle = {
        "dataset_name": dataset.name,
        "target_column": dataset.target_column,
        "selected_features": selected,
        "all_features": list(features.columns),
        "preprocessor": preprocessor,
        "fuzzy": fuzzy.to_dict(),
        "seed": seed,
        "mode": config.mode,
        "duration": duration,
        "metrics": metrics,
        "ga_history": ga_history,
        "ga_feature_frequency": feature_frequency,
        "fuzzy_mode": fuzzy_mode,
        "fuzzy_tune_history": fuzzy_tune_history,
        "models": {k: v for k, v in fitted_models.items() if k != "Neuro-Fuzzy-GA"},
        "bpn": bpn,
        "cleaning": cleaning,
        "smote": smote,
        "variant": variant,
        "shap_background": shap_background,
        "ensemble": ensemble,
        "feature_high": _feature_high(split.X_train, selected),
    }
    save_run(run_dir, bundle)

    leakage_guards = {
        "train_index": split.train_index,
        "test_index": split.test_index,
        "smote_train_only": True,
        "ga_train_only": True,
        "scaler_train_only": True,
        "fuzzy_ga_train_only": True,
        "test_size": len(split.X_test),
        "smote_grew_train": smote.after_defective + smote.after_clean
        >= smote.before_defective + smote.before_clean,
        "use_ga": use_ga,
        "use_smote": use_smote,
        "use_fuzzy": use_fuzzy,
        "fuzzy_mode": fuzzy_mode,
    }

    report("done", 1.0, "Training complete")
    return TrainingResult(
        dataset_name=dataset.name,
        target_column=dataset.target_column,
        quality=quality,
        cleaning=cleaning,
        split=split,
        preprocessor=preprocessor,
        smote=smote,
        selected_features=selected,
        all_features=list(features.columns),
        ga_history=ga_history,
        ga_fitness=ga_fitness,
        ga_runtime=ga_runtime,
        bpn_history=bpn.history_,
        metrics=metrics,
        roc=roc,
        scores=scores,
        y_test=y_test.to_numpy(),
        X_test_raw=split.X_test.reset_index(drop=True),
        X_test_scaled=X_test_sel.reset_index(drop=True),
        display_ids=ids,
        models=fitted_models,
        bpn=bpn,
        fuzzy=fuzzy,
        feature_ranks=ranks,
        duration=duration,
        seed=seed,
        mode=config.mode,
        run_dir=str(run_dir),
        logs=logs,
        leakage_guards=leakage_guards,
        ga_feature_frequency=feature_frequency,
        fuzzy_mode=fuzzy_mode,
        fuzzy_tune_history=fuzzy_tune_history,
        fuzzy_tune_fitness=fuzzy_tune_fitness,
        fuzzy_metrics_test=fuzzy_metrics_test,
        fuzzy_metrics_fixed_test=fuzzy_metrics_fixed_test,
        variant=variant,
        shap_background=shap_background,
        ensemble=ensemble,
    )


def _oof_bpn_probs(
    X_train_sel: pd.DataFrame,
    y_train: pd.Series,
    bpn_kwargs: dict,
    use_smote: bool,
    k_neighbors: int,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """Out-of-fold defect probabilities for every training row (no test data involved).

    Each fold trains a fresh BPN (with SMOTE applied only to that fold's training part)
    and predicts the held-out fold, so the fuzzy GA sees honest, unseen-row probabilities.
    """
    from sklearn.model_selection import StratifiedKFold

    y = y_train.to_numpy().astype(int)
    X = X_train_sel.reset_index(drop=True)
    probs = np.zeros(len(y), dtype=float)
    n_splits = max(2, min(n_splits, int(np.bincount(y).min())))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in folds.split(X, y):
        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        y_tr = pd.Series(y[tr_idx])
        try:
            X_tr, y_tr, _ = apply_smote(X_tr, y_tr, enabled=use_smote, random_state=seed, k_neighbors=k_neighbors)
        except TrainingError:
            pass
        model = BPNClassifier(**bpn_kwargs)
        model.fit(X_tr.to_numpy(), y_tr.to_numpy())
        probs[va_idx] = model.predict_proba(X.iloc[va_idx].to_numpy())[:, 1]
    return probs


def _feature_high(raw_train: pd.DataFrame, selected: list[str]) -> dict[str, float]:
    """95th percentile of each selected column on the pre-SMOTE training rows."""
    limits: dict[str, float] = {}
    for col in selected:
        if col not in raw_train.columns:
            continue
        series = pd.to_numeric(raw_train[col], errors="coerce").dropna()
        if series.empty:
            continue
        limits[str(col)] = float(series.quantile(0.95))
    return limits


def _display_ids(dataset_name: str, test_index: list[int]) -> list[str]:
    prefix = "CM1" if "cm1" in dataset_name.lower() else dataset_name.split("_")[0]
    return [f"{prefix}_module_{int(i):03d}" for i in test_index]

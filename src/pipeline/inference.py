"""Standalone inference from a saved run directory (used by the CLI and the scanner UI)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.columns import canonicalize_nasa_columns
from src.fuzzy.risk_engine import FuzzyRiskEngine
from src.models.bpn import BPNClassifier
from src.pipeline.artifacts import load_joblib
from src.utils.config import ROOT

DEFAULT_SHARED_MODEL = ROOT / "models" / "shared"


@dataclass
class SavedModel:
    run_dir: Path
    all_features: list[str]
    selected_features: list[str]
    preprocessor: object
    bpn: BPNClassifier
    fuzzy: FuzzyRiskEngine
    meta: dict
    shap_background: np.ndarray | None = None

    @property
    def threshold(self) -> float:
        return float(getattr(self.bpn, "decision_threshold_", 0.5))


def resolve_model_dir(model: str | Path | None) -> Path:
    if model is None:
        pointer = DEFAULT_SHARED_MODEL / "latest.json"
        if not pointer.exists():
            raise FileNotFoundError(
                "No shared-subset model found. Run `neurodefect train-shared --dataset data/raw/cm1.csv` first."
            )
        return DEFAULT_SHARED_MODEL / json.loads(pointer.read_text())["run_dir"]
    path = Path(model)
    if (path / "latest.json").exists() and not (path / "bundle.joblib").exists():
        path = path / json.loads((path / "latest.json").read_text())["run_dir"]
    if not (path / "bundle.joblib").exists():
        raise FileNotFoundError(f"No bundle.joblib in {path}")
    return path


def load_model(model: str | Path | None = None) -> SavedModel:
    run_dir = resolve_model_dir(model)
    bundle = load_joblib(run_dir / "bundle.joblib")
    bpn = BPNClassifier.from_weights(load_joblib(run_dir / "bpn_weights.joblib"))
    meta_path = run_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return SavedModel(
        run_dir=run_dir,
        all_features=list(bundle["all_features"]),
        selected_features=list(bundle["selected_features"]),
        preprocessor=bundle["preprocessor"],
        bpn=bpn,
        fuzzy=FuzzyRiskEngine.from_dict(bundle["fuzzy"]),
        meta=meta,
        shap_background=bundle.get("shap_background"),
    )


def predict_frame(model: SavedModel, frame: pd.DataFrame, id_column: str | None = None) -> pd.DataFrame:
    """Score each row. Rank is the network call, then defect probability.

    The fuzzy tier is a testing priority shown beside that call. It does not
    decide which modules are defective, and it does not outrank a defective call.
    """
    frame = canonicalize_nasa_columns(frame.copy())
    missing = [c for c in model.all_features if c not in frame.columns]
    if missing:
        raise ValueError(f"Input CSV is missing model features: {missing}")
    X = frame[model.all_features].apply(pd.to_numeric, errors="coerce")
    scaled = model.preprocessor.transform(X)[model.selected_features].to_numpy(dtype=np.float32)
    probs = model.bpn.predict_proba(scaled)[:, 1]
    if id_column is None:
        id_column = next((c for c in ("module_id", "id", "name") if c in frame.columns), None)
    ids = frame[id_column].astype(str).tolist() if id_column else [f"row_{i}" for i in range(len(frame))]
    rows = []
    for i, prob in enumerate(probs):
        raw = {k: (None if pd.isna(v) else float(v)) for k, v in X.iloc[i].items()}
        risk = model.fuzzy.assess(raw, float(prob))
        top = risk.fired_rules[0] if risk.fired_rules else None
        rows.append({
            "module_id": ids[i],
            "defect_probability": float(prob),
            "predicted_defective": bool(prob >= model.threshold),
            "risk_score": float(risk.risk_score),
            "risk_tier": risk.severity,
            "top_rule": top.rule_id if top else "",
            "top_rule_text": top.description if top else "",
            "top_rule_strength": float(top.strength) if top else 0.0,
            "rationale": top.rationale if top else "",
            "recommendation": risk.recommendation,
            "fired_rules": [(r.rule_id, round(r.strength, 3)) for r in risk.fired_rules],
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["predicted_defective", "defect_probability"], ascending=[False, False])
    out.insert(0, "rank", range(1, len(out) + 1))
    return out.reset_index(drop=True)


def scaled_selected_row(model: SavedModel, frame: pd.DataFrame, i: int) -> np.ndarray:
    frame = canonicalize_nasa_columns(frame.copy())
    X = frame[model.all_features].apply(pd.to_numeric, errors="coerce")
    return model.preprocessor.transform(X.iloc[[i]])[model.selected_features].to_numpy(dtype=np.float32)[0]


def explain_row(model: SavedModel, frame: pd.DataFrame, i: int, nsamples: int = 200):
    """SHAP attributions for row ``i`` of a metrics frame (None if shap or a background is unavailable)."""
    from src.explainability.shap_explainer import shap_values

    return shap_values(model.bpn, scaled_selected_row(model, frame, i), model.shap_background,
                       model.selected_features, nsamples=nsamples)

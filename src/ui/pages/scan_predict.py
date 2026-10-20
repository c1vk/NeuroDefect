from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src.pipeline.inference import explain_row, load_model, predict_frame
from src.scanner.radon_scanner import COLUMN_MAPPING, EXCLUDED_FROM_MODEL, UNMAPPABLE
from src.ui.components import empty_note, page_header, section


def render() -> None:
    page_header("Neurodefect / Scan & Predict", "Score Scanned Python Code")
    st.caption(
        "Upload a metrics CSV from `neurodefect scan <folder> -o metrics.csv`. "
        "The model learned NASA C and C++ modules. A high score means a Python function's "
        "measurements look like those NASA rows. It is not a measured Python bug."
    )
    try:
        model = load_model(None)
    except FileNotFoundError as exc:
        st.warning(str(exc))
        return
    except Exception as exc:
        st.error(f"The saved shared model could not be loaded: {exc}")
        return
    st.caption(
        f"Model: {model.run_dir.name} · features: {', '.join(model.selected_features)} · "
        f"decision threshold {model.threshold:.3f}"
    )
    upload = st.file_uploader("Scanner CSV", type=["csv"], key="scan_csv")
    if upload is None:
        with st.expander("Column mapping and limitations"):
            st.table(pd.DataFrame({"NASA column": list(COLUMN_MAPPING), "How it is computed": list(COLUMN_MAPPING.values())}))
            st.markdown("**Not computable:** " + "; ".join(f"`{k}` — {v}" for k, v in UNMAPPABLE.items()))
            st.markdown("**Scanned but excluded from the model:** " + "; ".join(f"`{k}` — {v}" for k, v in EXCLUDED_FROM_MODEL.items()))
        return
    frame = pd.read_csv(io.BytesIO(upload.getvalue()))
    try:
        table = predict_frame(model, frame)
    except ValueError as exc:
        st.error(str(exc))
        return
    if table.empty:
        empty_note("That CSV has no module rows to score.")
        return
    counts = table["risk_tier"].value_counts()
    cols = st.columns(4)
    for col, tier in zip(cols, ["CRITICAL", "HIGH", "MEDIUM", "LOW"]):
        col.metric(tier, int(counts.get(tier, 0)))
    with section("Ranked modules", key="scan_ranked"):
        st.dataframe(
            table[["rank", "module_id", "risk_tier", "risk_score", "defect_probability", "predicted_defective", "top_rule"]],
            hide_index=True,
            width="stretch",
        )
    st.subheader("Explanations")
    top_limit = min(25, len(table))
    top_n = st.slider("Explain top N modules", 1, top_limit, min(5, top_limit))
    for _, row in table.head(int(top_n)).iterrows():
        with st.expander(f"#{row['rank']} {row['module_id']} — {row['risk_tier']} ({row['risk_score']:.1f})"):
            st.write(f"Defect probability **{row['defect_probability']:.1%}**; top rule `{row['top_rule']}`: {row['top_rule_text']}")
            st.write(row["rationale"])
            st.write("Fired rules (strength): " + ", ".join(f"{r} ({s})" for r, s in row["fired_rules"]))
            st.info(row["recommendation"])
            _shap_block(model, frame, row)
    st.download_button("Download ranked CSV", table.to_csv(index=False), "neurodefect_predictions.csv", "text/csv")
    st.caption(
        "The list is ordered by the network's defect call, then by probability. "
        "The tier is only a testing priority. It does not decide that a function is buggy, "
        "and these ranks are not measured Python defects."
    )


def _shap_block(model, frame, row) -> None:
    from src.explainability.shap_explainer import HAS_SHAP

    if not HAS_SHAP:
        st.caption("SHAP not installed — showing fuzzy rules only.")
        return
    if model.shap_background is None:
        st.caption("Model has no SHAP background; rerun `neurodefect train-shared`.")
        return
    key = f"shap_{row['module_id']}"
    if st.checkbox("SHAP attribution", key=key):
        idx = frame.index[frame["module_id"].astype(str) == row["module_id"]] if "module_id" in frame else []
        i = int(idx[0]) if len(idx) else int(row["rank"]) - 1
        pairs = explain_row(model, frame, i)
        if pairs:
            st.dataframe(pd.DataFrame(pairs[:8], columns=["feature (scaled)", "SHAP value for P(defect)"]),
                         hide_index=True)

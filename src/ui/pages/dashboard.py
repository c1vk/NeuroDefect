from __future__ import annotations

from collections import Counter
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.pipeline.predictor import recent_predictions
from src.ui.charts import convergence_chart, donut, risk_bars
from src.ui.components import empty_note, html, html_table, kpi_card, page_header, plotly, risk_chip, section


def render() -> None:
    page_header("Neurodefect / Home", "Decision Support Dashboard")
    dataset = st.session_state.get("dataset")
    result = st.session_state.get("result")

    if result is None:
        empty_note(
            "No model is trained yet. Open Model Training and press Train Model Engine. "
            "Row counts and defect rates stay on Dataset Analysis, so this screen is not mistaken for a result."
        )
        return

    if dataset is None:
        empty_note("Load a dataset from Dataset Analysis or Model Training to begin.")
        return

    quality = result.quality
    n_rows = quality.n_rows if quality else len(dataset.frame)
    defect_rate = quality.defect_rate if quality else 0.0

    selected = f"{len(result.selected_features)}/{len(result.all_features)}"
    status = "Trained"
    status_chip = "green"
    status_meta = "Active"

    k1, k2, k3, k4 = st.columns(4, gap="small")
    k1.markdown(kpi_card("Total Modules", f"{n_rows}", dataset.name.replace("_", " "), "slate"), unsafe_allow_html=True)
    defect_meta = f"{quality.n_defective} defective" if quality else ""
    k2.markdown(
        kpi_card("Defect Rate", f"{defect_rate * 100:.1f}%", defect_meta, "amber"),
        unsafe_allow_html=True,
    )
    k3.markdown(
        kpi_card("Selected Features", selected, "GA Optimized" if result else "Not selected", "blue"),
        unsafe_allow_html=True,
    )
    k4.markdown(kpi_card("Model Status", status, status_meta, status_chip), unsafe_allow_html=True)
    st.caption(f"These counts are the file this run used ({dataset.name.replace('_', ' ')}), not a live NASA feed.")

    left, right = st.columns((1.7, 1.0), gap="small")
    with left:
        with section("Neural training convergence", key="dash_convergence", height="stretch"):
            if result and result.bpn_history.train_acc:
                fig = convergence_chart(
                    result.bpn_history.train_acc,
                    result.bpn_history.train_f1,
                    series_name="Train accuracy",
                )
            elif result and result.ga_history:
                fig = convergence_chart(result.ga_history, series_name="GA fitness")
            else:
                fig = convergence_chart()
            plotly(fig, height="stretch")

    with right:
        with st.container(height="stretch", gap="small"):
            with section("Defect Distribution", key="dash_defect"):
                if quality:
                    d1, d2 = st.columns((1.0, 1.05), gap="small", vertical_alignment="center")
                    with d1:
                        plotly(donut(quality.n_clean, quality.n_defective))
                    with d2:
                        clean_pct = (1 - quality.defect_rate) * 100
                        def_pct = quality.defect_rate * 100
                        html(
                            f"""
                            <div class="nd-dist-stats">
                              <div class="nd-dist-row">
                                <span class="nd-dist-dot clean"></span>
                                <div>
                                  <div class="nd-dist-label">Clean</div>
                                  <div class="nd-dist-value">{quality.n_clean:,}</div>
                                  <div class="nd-dist-pct">{clean_pct:.1f}%</div>
                                </div>
                              </div>
                              <div class="nd-dist-row">
                                <span class="nd-dist-dot defective"></span>
                                <div>
                                  <div class="nd-dist-label">Defective</div>
                                  <div class="nd-dist-value">{quality.n_defective:,}</div>
                                  <div class="nd-dist-pct def">{def_pct:.1f}%</div>
                                </div>
                              </div>
                            </div>
                            """
                        )
                else:
                    st.caption("Load a dataset to see class balance.")

            with section("Risk Severity Distribution", key="dash_risk", height="stretch"):
                if result:
                    counts, sampled = _cached_risk_counts(result)
                    plotly(risk_bars(counts), height="stretch")
                    st.caption(f"Derived severity on {sampled} held-out modules.")
                else:
                    st.caption("Risk severity is derived after a model is trained.")

    with section("Genetic Algorithm Diagnostics", key="dash_ga_diag"):
        _render_ga_diagnostics(result)

    with section("Recent Software Module Predictions", key="dash_predictions"):
        if result:
            recents = _cached_recent(result)
            rows = []
            for item in recents:
                color = "#ef4444" if item.probability >= 0.6 else "#0f172a"
                rows.append(
                    [
                        item.module_id,
                        item.label,
                        f"<span style='color:{color};font-weight:600'>{item.probability * 100:.1f}%</span>",
                        risk_chip(item.severity),
                        f"{item.confidence * 100:.1f}%",
                        item.inspection_date,
                    ]
                )
            html(
                html_table(
                    ["Module ID", "Network call", "Defect Probability", "Testing priority", "Confidence", "Inspection Date"],
                    rows,
                )
            )
            st.caption(
                "Network call is the defect decision. Testing priority is the fuzzy tier and does not override that call."
            )
        else:
            st.caption("Predictions appear here after the hybrid model has been trained.")


def _result_key(result) -> tuple:
    return (str(getattr(result, "run_dir", "")), str(getattr(result, "seed", "")), len(result.X_test_raw))


def _cached_risk_counts(result) -> tuple[dict[str, int], int]:
    key = _result_key(result)
    if st.session_state.get("_risk_cache_key") != key:
        st.session_state._risk_cache_key = key
        st.session_state._risk_cache = _risk_counts(result)
    return st.session_state._risk_cache


def _cached_recent(result):
    key = _result_key(result)
    if st.session_state.get("_recent_cache_key") != key:
        st.session_state._recent_cache_key = key
        st.session_state._recent_cache = recent_predictions(result, limit=5)
    return st.session_state._recent_cache


def _risk_counts(result) -> tuple[dict[str, int], int]:
    from src.pipeline.predictor import predict_module

    counts: Counter[str] = Counter()
    n = min(len(result.X_test_raw), 80)
    for i in range(n):
        pred = predict_module(result, i)
        counts[pred.severity] += 1
    return dict(counts), n


REPORT_FIGURES = Path(__file__).resolve().parents[3] / "reports" / "figures"


def _render_ga_diagnostics(result) -> None:
    """GA fitness by generation and feature-selection frequency (live run + saved report PNGs)."""
    if result and (result.ga_history or result.ga_feature_frequency):
        c1, c2 = st.columns(2, gap="small")
        with c1:
            fig = go.Figure()
            if result.ga_history:
                fig.add_scatter(y=result.ga_history, mode="lines+markers", name="Feature GA")
            if getattr(result, "fuzzy_tune_history", None):
                fig.add_scatter(y=result.fuzzy_tune_history, mode="lines+markers", name="Fuzzy membership GA")
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                              title="Best fitness by generation", xaxis_title="Generation")
            plotly(fig)
        with c2:
            freq = sorted(result.ga_feature_frequency.items(), key=lambda kv: kv[1])
            fig = go.Figure(go.Bar(x=[v for _, v in freq], y=[k for k, _ in freq], orientation="h"))
            fig.update_layout(height=max(260, 18 * len(freq)), margin=dict(l=10, r=10, t=30, b=10),
                              title="Feature selection frequency", xaxis_range=[0, 1])
            plotly(fig)
        st.caption(f"Fuzzy memberships: {getattr(result, 'fuzzy_mode', 'fixed_quantile')}")
    else:
        st.caption("Train a model to see live GA convergence and feature-selection frequency.")
    pngs = sorted(REPORT_FIGURES.glob("*.png")) if REPORT_FIGURES.exists() else []
    if pngs:
        with st.expander(f"Ablation study figures ({len(pngs)} saved PNGs in reports/figures)"):
            cols = st.columns(3)
            for i, png in enumerate(pngs):
                cols[i % 3].image(str(png), caption=png.stem, width="stretch")

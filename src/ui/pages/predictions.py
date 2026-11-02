from __future__ import annotations

import streamlit as st

from src.pipeline.predictor import predict_from_raw_row
from src.ui.charts import contribution_bars, probability_donut
from src.ui.components import empty_note, html, page_header, plotly, section
from src.utils.exceptions import NeuroDefectError


def render() -> None:
    page_header("NeuroDefect / Predictions", "Inspect one module")
    result = st.session_state.get("result")
    if result is None:
        empty_note("Please train a model before running predictions.")
        return

    with section("Module", key="pr_inspect"):
        ids = result.display_ids
        choice = st.selectbox("Select a held-out module", ids)
        idx = ids.index(choice)
        raw = result.X_test_raw.iloc[idx]
        loc_col = next((c for c in raw.index if c.lower() in {"loc", "locode"}), raw.index[0])
        cx_col = next((c for c in raw.index if c in {"v(g)", "branchCount"}), raw.index[min(1, len(raw.index) - 1)])
        c1, c2, c3 = st.columns((2.2, 1, 1), vertical_alignment="bottom")
        c1.caption("Or override LOC and complexity for this module.")
        # Key includes the module so the displayed values follow the selection.
        # A fixed key would keep the first module's numbers after the selectbox changes.
        loc_val = c2.number_input("LOC", value=float(raw[loc_col]), key=f"ovr_loc_{choice}")
        cx_val = c3.number_input("Complexity", value=float(raw[cx_col]), key=f"ovr_cx_{choice}")

    row = raw.to_dict()
    row[loc_col] = loc_val
    row[cx_col] = cx_val
    try:
        pred = predict_from_raw_row(result, row, module_id=choice)
    except NeuroDefectError as exc:
        st.error(str(exc))
        return

    with section("Network decision", key="pr_output"):
        a, b = st.columns((0.7, 1.6), gap="small", vertical_alignment="center")
        with a:
            plotly(probability_donut(pred.probability))
        with b:
            color = "#ef4444" if pred.label == "DEFECTIVE" else "#16a34a"
            bg = "#fef2f2" if pred.label == "DEFECTIVE" else "#ecfdf3"
            html(
                f"<span class='nd-badge-lg' style='background:{bg};color:{color}'>{pred.label}</span>"
                f" &nbsp; Distance from 0.5: <b>{pred.confidence*100:.1f}%</b>"
                "<p class='nd-note'>Distance from 0.5 is how far the probability sits from the default boundary. "
                "It is not a statistical confidence interval. The defective or clean call uses the tuned cutoff.</p>"
            )
            st.caption("Testing priority from the fuzzy rules. This tier does not change the network's defective or clean call.")
            active = pred.severity.lower()
            html(
                f"""
                <div class="nd-tier">
                  <div class="{'on-low' if active == 'low' else ''}">LOW</div>
                  <div class="{'on-medium' if active == 'medium' else ''}">MEDIUM</div>
                  <div class="{'on-high' if active == 'high' else ''}">HIGH</div>
                  <div class="{'on-critical' if active == 'critical' else ''}">CRITICAL</div>
                </div>
                """
            )
            top_rule = pred.risk.fired_rules[0] if pred.risk.fired_rules else None
            if top_rule:
                html(f"<p class='nd-note'><b>{top_rule.rule_id}.</b> {top_rule.description}</p>")
            else:
                st.caption("No rule fired above the minimum strength.")

    left, right = st.columns(2, gap="small")
    with left:
        with section("Feature Risk Contributions", key="pr_contrib"):
            top = pred.feature_contribs[:3]
            if top:
                plotly(contribution_bars(top))
            else:
                st.caption("No feature contributions available for this module.")
    with right:
        with section("Rule that fired", key="pr_rules"):
            rules = pred.risk.fired_rules[:3]
            if not rules:
                st.caption("No rules fired above the minimum strength threshold.")
            for rule in rules:
                html(
                    f"<div class='nd-rule'><div>{rule.description[:72]}{'…' if len(rule.description) > 72 else ''}</div>"
                    f"<span>Strength: {rule.strength:.2f}</span></div>"
                )

    with section("SHAP Attribution (defect probability)", key="pr_shap"):
        _render_shap(result, row)

    with section("Operational Action Protocol", key="pr_action"):
        if pred.label == "DEFECTIVE":
            headline = "REVIEW THIS MODULE"
            detail = (
                f"The network calls it defective ({pred.probability * 100:.0f}%). "
                f"Testing priority: {pred.severity}. {pred.risk.recommendation}"
            )
        else:
            headline = "NETWORK CALLS IT CLEAN"
            detail = (
                f"The {pred.severity} tier is a testing priority, not a second defect decision. "
                f"{pred.risk.recommendation}"
            )
        html(f"<div class='nd-alert critical'><strong>{headline}</strong><br>{detail}</div>")
        st.caption(
            "The defective or clean call comes from the network probability and its tuned cutoff. "
            "The tier is a derived testing priority, not a NASA bug-severity label."
        )


def _render_shap(result, row) -> None:
    from src.explainability.shap_explainer import HAS_SHAP, shap_values

    if not HAS_SHAP:
        st.caption("Install `shap` to see SHAP attributions (pip install shap). Fired rules above still explain the tier.")
        return
    background = getattr(result, "shap_background", None)
    if background is None:
        st.caption("This run has no SHAP background sample; retrain to enable SHAP.")
        return
    if not st.toggle("Compute SHAP values (KernelExplainer, ~1–3 s)", key="pr_shap_on"):
        return
    import pandas as pd

    scaled = result.preprocessor.transform(pd.DataFrame([row])[result.all_features])[result.selected_features]
    scorer = getattr(result, "ensemble", None) or result.bpn
    pairs = shap_values(scorer, scaled.to_numpy(dtype="float32")[0], background, result.selected_features)
    if pairs:
        plotly(contribution_bars(pairs[:6]))
        st.caption("Positive values push P(defect) up relative to the training background average; shown next to the "
                   "fired fuzzy rules so the neural and rule-based explanations can be compared.")

from __future__ import annotations

import streamlit as st

from src.data.feature_map import display_name
from src.fuzzy.membership import PROBABILITY_SETS
from src.ui.charts import membership_chart
from src.ui.components import empty_note, html, html_table, page_header, plotly, section
from src.ui.theme import GREEN, ORANGE, RED


def render() -> None:
    page_header("NeuroDefect / Explainability", "Explainable Logic Visualization")
    result = st.session_state.get("result")
    if result is None:
        empty_note("Train a model to inspect GA feature ranks, membership functions, and fired fuzzy rules.")
        return

    with section("GA Feature Priority Optimization Rank", key="ex_ranks"):
        chips = []
        for name, score in result.feature_ranks[:6]:
            chips.append(f'<span class="nd-chip slate">{display_name(name)} ({score:.2f})</span>')
        html(" ".join(chips) or "<p class='nd-empty'>No ranks available.</p>")

    left, right = st.columns(2, gap="small")
    with left:
        with section("Complexity Membership Functions", key="ex_cx"):
            triangles = result.fuzzy.sets.get("complexity")
            if triangles:
                plotly(membership_chart(triangles, [GREEN, ORANGE, RED], ["Low", "Medium", "High"]))
                lo, mid, hi = triangles
                html(
                    f"<p style='text-align:center;font-size:0.78rem'>"
                    f"<span style='color:{GREEN}'>Low ({lo.a:.0f}–{lo.c:.0f})</span> &nbsp; "
                    f"<span style='color:{ORANGE}'>Medium ({mid.a:.0f}–{mid.c:.0f})</span> &nbsp; "
                    f"<span style='color:{RED}'>High ({hi.a:.0f}+)</span></p>"
                )
            else:
                st.caption("Cyclomatic complexity is not present in this dataset, so this fuzzy input was omitted.")
    with right:
        with section("Probability Membership Functions", key="ex_prob"):
            plotly(membership_chart(PROBABILITY_SETS, [GREEN, ORANGE, RED], ["Unlikely", "Possible", "Likely"]))
            html(
                f"<p style='text-align:center;font-size:0.78rem'>"
                f"<span style='color:{GREEN}'>Unlikely (0–0.3)</span> &nbsp; "
                f"<span style='color:{ORANGE}'>Possible (0.2–0.7)</span> &nbsp; "
                f"<span style='color:{RED}'>Likely (0.6–1.0)</span></p>"
            )

    with section("Fired Rule Base and Impact Weights", key="ex_rules"):
        from src.pipeline.predictor import predict_module

        ids = result.display_ids or ["module 0"]
        choice = st.selectbox("Held-out module", ids, key="ex_module")
        sample_index = ids.index(choice) if choice in ids else 0
        sample = predict_module(result, sample_index)
        rows = []
        for rule in sample.risk.fired_rules[:8]:
            color = "#ef4444" if rule.consequent in {"HIGH", "CRITICAL"} else "#16a34a"
            cond = rule.description.replace("IF ", "").split(" THEN ")[0]
            cons = rule.description.split(" THEN ")[-1]
            rows.append(
                [
                    rule.rule_id,
                    cond,
                    cons,
                    f"<span style='color:{color};font-weight:600'>{rule.strength:.3f}</span>",
                ]
            )
        if rows:
            html(
                html_table(
                    ["Rule ID", "Antecedent Parameters (Conditions)", "Consequent Assessment", "Firing Strength"],
                    rows,
                )
            )
        else:
            st.caption("No rules fired for the first held-out module.")
        st.caption(
            f"Example explanation for {sample.module_id}. "
            "Available fuzzy inputs: " + ", ".join(result.fuzzy.available_concepts()) + ". "
            "Risk severity is derived, not a ground-truth bug-severity label."
        )

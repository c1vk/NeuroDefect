from __future__ import annotations

import pandas as pd
import streamlit as st

from src.evaluation.comparison import metrics_table
from src.evaluation.testing_budget import testing_budget_table
from src.ui.charts import grouped_metrics, roc_chart
from src.ui.components import empty_note, html, html_table, page_header, plotly, section


def _leaders(metrics: dict, attr: str) -> str:
    """Best model on one metric. Majority is left out; predicting the common class is not a result."""
    scored = [
        (name, float(getattr(item, attr)))
        for name, item in metrics.items()
        if name != "Majority"
    ]
    if not scored:
        return "n/a"
    best = max(value for _, value in scored)
    names = [name for name, value in scored if abs(value - best) < 1e-9]
    return f"{', '.join(names)} ({best:.3f})"


def render() -> None:
    page_header("Neurodefect / Model Comparison", "One held-out split")
    result = st.session_state.get("result")
    if result is None:
        empty_note(
            "Train the engine to compare Majority, Logistic Regression, BPN, and Neuro-Fuzzy-GA on one held-out split."
        )
        return

    n_test = len(result.y_test)
    n_def = int(pd.Series(result.y_test).sum())
    n_train = len(result.split.y_train)
    st.caption(
        f"{result.dataset_name} · one stratified split · seed {result.seed} · "
        f"{n_train} train rows · {n_test} test rows · {n_def} defective in the test rows. "
        "BPN and Neuro-Fuzzy-GA use a cutoff tuned on training data. The other models use 0.5."
    )
    st.caption(
        "These numbers do not name a best model. "
        f"On this split, F1 leads with {_leaders(result.metrics, 'f1')}, "
        f"recall with {_leaders(result.metrics, 'recall')}, "
        f"precision with {_leaders(result.metrics, 'precision')}, "
        f"accuracy with {_leaders(result.metrics, 'accuracy')}, "
        f"and ROC-AUC with {_leaders(result.metrics, 'roc_auc')}. "
        "A high ROC-AUC is ranking ability. It can still sit next to a weaker F1 once a cutoff is applied. "
        "Repeated cross-validation on real NASA CM1, PC1, and JM1 puts F1 near 0.31–0.45, "
        "and logistic regression, random forest, or SVM lead those tables "
        "(reports/cv_results.md). A strong score on the synthetic sample does not carry over."
    )

    with section("Defects caught if you inspect the top 10%", key="cmp_budget"):
        budget = testing_budget_table(result.y_test, result.scores, budget_frac=0.10, random_state=result.seed)
        share_col = "Defects found (share of actual defective modules)"
        budget = budget.sort_values(share_col, ascending=False)
        budget_rows = [
            [r["Strategy"], r["Budget"], f"{r[share_col] * 100:.1f}%"]
            for _, r in budget.iterrows()
        ]
        html(html_table(["Strategy", "Budget", "Share of actual defective modules found"], budget_rows, highlight_first=True))
        st.caption(
            "Of the modules that really were defective, this is how many sit in the top 10% of this split. "
            "The first row caught the most on this split. Random selection is a baseline."
        )

    table = metrics_table(result.metrics).sort_values("F1-Score", ascending=False)
    with section("Performance Metric Benchmarks", key="cmp_metrics"):
        rows = [
            [
                row["Classifier Model Architecture"],
                f"{row['Accuracy']:.3f}",
                f"{row['Precision']:.3f}",
                f"{row['Recall']:.3f}",
                f"{row['F1-Score']:.3f}",
                f"{row['ROC-AUC']:.3f}",
                f"{row['Matthews CC']:.3f}",
            ]
            for _, row in table.iterrows()
        ]
        html(
            html_table(
                ["Classifier Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Matthews CC"],
                rows,
            )
        )
        st.caption(
            "Ordered by F1 on this one split. The top F1 row is not the winner on precision, accuracy, or ROC-AUC."
        )

    left, right = st.columns(2, gap="small")
    with left:
        with section("Primary Metrics Grouped Comparison", key="cmp_grouped"):
            plotly(grouped_metrics(table))
    with right:
        with section("ROC Response Curves", key="cmp_roc"):
            aucs = {name: m.roc_auc for name, m in result.metrics.items()}
            plotly(roc_chart(result.roc, aucs))
            st.caption("The highest curve ranks defective modules above clean ones. The cutoff used in the table can still favor another model.")

    hybrid_metrics = result.metrics.get("Neuro-Fuzzy-GA")
    if hybrid_metrics:
        with section("Confusion Matrix (Neuro-Fuzzy-GA, test set)", key="cmp_cm"):
            html(
                html_table(
                    ["", "Predicted clean", "Predicted defective"],
                    [
                        ["Actual clean", str(hybrid_metrics.tn), str(hybrid_metrics.fp)],
                        [
                            "Actual defective",
                            f"<span style='color:#dc2626;font-weight:700'>{hybrid_metrics.fn}</span>",
                            str(hybrid_metrics.tp),
                        ],
                    ],
                )
            )
            st.caption(
                "Errors for Neuro-Fuzzy-GA on this same split. False negatives are highlighted because a missed defect costs more than an extra review. This matrix does not make the hybrid the best model."
            )



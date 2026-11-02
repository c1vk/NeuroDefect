from __future__ import annotations

import streamlit as st

from src.data.loader import describe_dataset_path, list_datasets, load_dataset
from src.data.quality import build_quality_report
from src.ui.charts import correlation_heatmap
from src.ui.components import empty_note, html, html_table, page_header, plotly, section
from src.utils.config import ROOT
from src.utils.exceptions import NeuroDefectError


def render() -> None:
    page_header("NeuroDefect / Dataset Analysis", "Dataset Exploration")
    directories = [ROOT / "data" / "sample", ROOT / "data" / "raw"]
    uploaded = st.file_uploader("Upload a NASA PROMISE CSV (optional)", type=["csv", "xlsx", "xls"])
    just_uploaded = None
    if uploaded is not None and st.session_state.get("_uploaded_name") != uploaded.name:
        dest = ROOT / "data" / "raw" / uploaded.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(uploaded.getvalue())
        st.session_state._uploaded_name = uploaded.name
        just_uploaded = str(dest.relative_to(ROOT))

    files = list_datasets(directories)
    names = [str(path.relative_to(ROOT)) for path in files]
    if not names:
        empty_note("Add a CSV under data/sample or data/raw, or upload one above.")
        return

    if just_uploaded in names:
        st.session_state["dataset_choice"] = just_uploaded

    col_a, col_b = st.columns((2, 1), vertical_alignment="bottom")
    with col_a:
        choice = st.selectbox(
            "Dataset",
            names,
            key="dataset_choice",
            format_func=lambda name: describe_dataset_path(ROOT / name, ROOT),
        )
    with col_b:
        load_clicked = st.button("Load dataset", type="primary", width="stretch")
    if just_uploaded:
        load_clicked = True

    if load_clicked and choice:
        try:
            loaded = load_dataset(ROOT / choice)
            st.session_state.dataset = loaded
            if loaded.target_column:
                st.session_state.quality = build_quality_report(
                    loaded.frame, loaded.target_column, loaded.name
                )
            else:
                st.session_state.quality = None
                st.warning("Target column was not found. Select it below before training.")
        except NeuroDefectError as exc:
            st.error(str(exc))

    dataset = st.session_state.get("dataset")
    if dataset is None:
        empty_note("Choose the practice sample or a NASA file, then press Load dataset.")
        return

    if dataset.target_column is None:
        target = st.selectbox("Select the defect target column", list(dataset.frame.columns))
        if st.button("Use this target column"):
            dataset.target_column = target
            dataset.target_detection = "user"
            st.session_state.dataset = dataset
            st.session_state.quality = build_quality_report(dataset.frame, target, dataset.name)
            st.rerun()
        return

    quality = st.session_state.get("quality") or build_quality_report(
        dataset.frame, dataset.target_column, dataset.name
    )
    st.session_state.quality = quality

    left, right = st.columns(2, gap="small")
    with left:
        html(
            f"""
            <div class="nd-card">
              <p class="nd-card-title">{dataset.name.replace('_', ' ')} Dataset Profile</p>
              <div class="nd-kv"><span>Total Module Instances</span><strong>{quality.n_rows} rows</strong></div>
              <div class="nd-kv"><span>Original Code Attributes</span><strong>{quality.n_features} software metrics</strong></div>
              <div class="nd-kv"><span>Target Defective Modules</span><strong>{quality.n_defective} modules</strong></div>
              <div class="nd-kv"><span>Baseline Defect Percent</span><strong>{quality.defect_rate*100:.1f}%</strong></div>
            </div>
            """
        )
    with right:
        imbalance_cls = "bad" if quality.n_defective and quality.n_clean / max(quality.n_defective, 1) >= 4 else "warn"
        html(
            f"""
            <div class="nd-card">
              <p class="nd-card-title">Data Quality &amp; Integrity</p>
              <div class="nd-kv"><span>Metric Completeness</span><strong class="good">{quality.completeness*100:.1f}%</strong></div>
              <div class="nd-kv"><span>Missing Parameters</span><strong class="warn">{quality.missing_fields} fields</strong></div>
              <div class="nd-kv"><span>Duplicate Signatures</span><strong class="{'good' if quality.duplicate_rows==0 else 'warn'}">{quality.duplicate_rows} detected</strong></div>
              <div class="nd-kv"><span>Class Imbalance Ratio</span><strong class="{imbalance_cls}">{quality.imbalance_ratio} {'(Highly Imbalanced)' if imbalance_cls=='bad' else ''}</strong></div>
            </div>
            """
        )

    with section("Code Metric Descriptive Statistics", key="ds_stats"):
        stats = quality.feature_stats
        rows = []
        for _, row in stats.iterrows():
            missing = int(row["Missing"])
            missing_html = f"<span style='color:#d97706'>{missing}</span>" if missing else "0"
            rows.append(
                [
                    row["Feature Name"],
                    f"{row['Mean']:.2f}",
                    f"{row['Std Dev']:.2f}",
                    f"{row['Min']:.1f}",
                    f"{row['Max']:.1f}",
                    missing_html,
                ]
            )
        html(html_table(["Feature Name", "Mean", "Std Dev", "Min", "Max", "Missing"], rows))

    with section("Feature Correlation Matrix (first 6 metrics)", key="ds_corr"):
        if quality.correlation is not None and not quality.correlation.empty:
            plotly(correlation_heatmap(quality.correlation))
        else:
            st.caption("Correlation is unavailable for this dataset.")

    with st.expander("Raw preview"):
        st.dataframe(dataset.frame.head(20), width="stretch")

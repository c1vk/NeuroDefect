from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.data.loader import load_dataset
from src.data.quality import build_quality_report
from src.ui.components import inject_css
from src.ui.layout import render_fallback_nav, render_sidebar
from src.ui.pages import comparison, dashboard, dataset, explainability, predictions, scan_predict, settings, training
from src.utils.config import load_config
from src.utils.exceptions import NeuroDefectError

PAGES = {
    "Dashboard": dashboard.render,
    "Dataset Analysis": dataset.render,
    "Model Training": training.render,
    "Predictions": predictions.render,
    "Scan & Predict": scan_predict.render,
    "Model Comparison": comparison.render,
    "Explainability": explainability.render,
    "Settings": settings.render,
}


def _bootstrap() -> None:
    if "config" not in st.session_state:
        st.session_state.config = load_config()
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    if "dataset" not in st.session_state:
        default = ROOT / st.session_state.config.dataset().get("path", "data/sample/NASA_CM1_sample.csv")
        if default.exists():
            try:
                loaded = load_dataset(default, name="Practice sample")
                st.session_state.dataset = loaded
                if loaded.target_column:
                    st.session_state.quality = build_quality_report(
                        loaded.frame, loaded.target_column, loaded.name
                    )
            except NeuroDefectError:
                st.session_state.dataset = None


def main() -> None:
    st.set_page_config(
        page_title="NeuroDefect Decision System",
        page_icon="▣",
        layout="wide",
        initial_sidebar_state="locked",
    )
    inject_css()
    _bootstrap()
    with st.sidebar:
        current = render_sidebar(st.session_state.config)
    render_fallback_nav()
    renderer = PAGES.get(current, dashboard.render)
    renderer()


if __name__ == "__main__":
    main()

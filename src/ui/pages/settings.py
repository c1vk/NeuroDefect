from __future__ import annotations

import streamlit as st

from src.ui.components import html, page_header, section
from src.utils.config import load_config


def render() -> None:
    page_header("Neurodefect / Settings", "Runtime Configuration")
    config = st.session_state.get("config") or load_config()

    with section("Runtime", key="set_runtime"):
        mode = st.selectbox("Operating mode", ["MVP", "RESEARCH"], index=0 if config.mode == "MVP" else 1)
        seed = st.number_input("Random seed", 0, 10_000, int(config.training().get("random_state", 42)))
        optimizers = ["sgd", "sgd_momentum", "adam"]
        current_optimizer = str(config.training().get("optimizer", "adam"))
        optimizer = st.selectbox(
            "BPN optimizer",
            optimizers,
            index=optimizers.index(current_optimizer) if current_optimizer in optimizers else optimizers.index("adam"),
        )
        if st.button("Apply settings", type="primary"):
            st.session_state.config = config.with_overrides(
                mode=mode,
                training={"random_state": int(seed), "optimizer": optimizer},
                ga={"random_state": int(seed)},
            )
            st.success("Settings applied for the next training run. Existing artifacts are not overwritten.")

    html(
        """
        <div class="nd-card">
          <p class="nd-card-title">Scientific notes</p>
          <p class="nd-empty">
            NASA PC5 / JM1 / CM1 provide binary defect labels. NeuroDefect therefore reports
            <b>defect prediction</b> and a derived <b>risk severity assessment</b>. It does not claim
            to predict experimentally validated bug severity (Low / Medium / High / Critical).
            MVP mode uses a smaller GA population and fewer epochs so the dashboard stays interactive.
            Research mode is intended for paper-quality runs.
          </p>
        </div>
        """
    )

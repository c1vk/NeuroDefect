from __future__ import annotations

import streamlit as st

from src.ui.theme import NAV_PAGES
from src.utils.config import AppConfig


LOGO_SVG = """
<svg width="18" height="18" viewBox="0 0 24 24" fill="none">
  <rect x="3" y="3" width="7" height="7" rx="1.6" fill="white"/>
  <rect x="14" y="3" width="7" height="7" rx="1.6" fill="white" opacity="0.85"/>
  <rect x="3" y="14" width="7" height="7" rx="1.6" fill="white" opacity="0.85"/>
  <rect x="14" y="14" width="7" height="7" rx="1.6" fill="white"/>
</svg>
"""


def render_sidebar(_config: AppConfig) -> str:
    current = st.session_state.get("page", "Dashboard")

    st.markdown(
        f"""
        <div class="nd-brand">
          <div class="nd-logo">{LOGO_SVG}</div>
          <div>
            <p class="nd-brand-name">NeuroDefect</p>
            <p class="nd-brand-tag">Decision System</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
        anchors=False,
    )

    for label, key in NAV_PAGES:
        is_active = current == key
        if st.button(label, key=f"nav_{key}", width="stretch", type="primary" if is_active else "secondary"):
            st.session_state.page = key
            st.rerun()

    return current


def render_fallback_nav() -> None:
    """Page switcher for viewports where the sidebar starts collapsed."""
    current = st.session_state.get("page", "Dashboard")
    with st.container(horizontal=True, key="fallback_nav"):
        for label, key in NAV_PAGES:
            if st.button(
                label,
                key=f"topnav_{key}",
                type="primary" if current == key else "secondary",
            ):
                st.session_state.page = key
                st.rerun()

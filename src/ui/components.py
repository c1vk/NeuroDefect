from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st

from src.utils.config import ROOT

CSS_PATH = ROOT / "assets" / "styles.css"
ANIM_PATH = ROOT / "assets" / "animations.css"
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def inject_css() -> None:
    css = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""
    anim = ANIM_PATH.read_text(encoding="utf-8") if ANIM_PATH.exists() else ""
    st.markdown(f"<style>{css}\n{anim}</style>", unsafe_allow_html=True, anchors=False)


def html(body: str) -> None:
    """Render HTML without Streamlit heading permalink icons."""
    st.markdown(body, unsafe_allow_html=True, anchors=False)


def page_header(crumb: str, title: str) -> None:
    html(
        f"""
        <div class="nd-header">
          <div>
            <div class="nd-crumb">{crumb}</div>
            <p class="nd-page-title">{title}</p>
          </div>
          <div class="nd-status"><span class="nd-dot"></span> System: Online</div>
        </div>
        """
    )


@contextmanager
def section(title: str, *, key: str | None = None, **container_kwargs) -> Iterator[None]:
    """Card that keeps a section title and its chart/table in the same box."""
    kwargs = {"border": True, **container_kwargs}
    if key is not None:
        kwargs["key"] = key
    with st.container(**kwargs):
        if title:
            html(f'<p class="nd-card-title">{title}</p>')
        yield


def plotly(fig, *, height: str | int = "content", **kwargs) -> None:
    st.plotly_chart(
        fig,
        width="stretch",
        height=height,
        config=PLOTLY_CONFIG,
        theme=None,
        **kwargs,
    )


def card_open(title: str | None = None) -> str:
    heading = f'<p class="nd-card-title">{title}</p>' if title else ""
    return f'<div class="nd-card">{heading}'


def card_close() -> str:
    return "</div>"


def kpi_card(label: str, value: str, meta: str = "", chip_class: str = "slate") -> str:
    meta_html = f'<span class="nd-chip {chip_class}">{meta}</span>' if meta else ""
    return f"""
    <div class="nd-card nd-kpi">
      <div class="nd-kpi-top">{label}</div>
      <div class="nd-kpi-row">
        <div class="nd-kpi-value">{value}</div>
        {meta_html}
      </div>
    </div>
    """


def empty_note(text: str) -> None:
    html(f'<div class="nd-card"><p class="nd-empty">{text}</p></div>')


def risk_chip(severity: str) -> str:
    mapping = {
        "CRITICAL": "red",
        "HIGH": "yellow",
        "MEDIUM": "slate",
        "LOW": "green",
    }
    cls = mapping.get(severity.upper(), "slate")
    label = severity.title()
    return f'<span class="nd-chip {cls}">{label}</span>'


def html_table(headers: list[str], rows: list[list[str]], highlight_first: bool = False) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for i, row in enumerate(rows):
        cls = ' class="active"' if highlight_first and i == 0 else ""
        cells = "".join(f"<td>{c}</td>" for c in row)
        body.append(f"<tr{cls}>{cells}</tr>")
    return f'<table class="nd-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'

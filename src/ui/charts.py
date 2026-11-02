from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.data.feature_map import display_name
from src.fuzzy.membership import Triangle
from src.ui.theme import BLUE, GREEN, MUTED, ORANGE, PLOTLY_LAYOUT, RED, YELLOW


def _layout(**overrides) -> dict:
    """Merge theme defaults with per-chart overrides. Later keys win."""
    return {**PLOTLY_LAYOUT, **overrides}


def apply_layout(fig: go.Figure, height: int = 280, **overrides) -> go.Figure:
    fig.update_layout(_layout(height=height, **overrides))
    fig.update_xaxes(showgrid=False, zeroline=False, color=MUTED)
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False, color=MUTED)
    return fig


def convergence_chart(
    history: list[float] | None = None,
    f1_hist: list[float] | None = None,
    height: int = 460,
    series_name: str = "Train accuracy",
) -> go.Figure:
    fig = go.Figure()
    series = list(history or [])
    if series:
        xs = list(range(1, len(series) + 1))
        fig.add_trace(
            go.Scatter(x=xs, y=series, mode="lines", name=series_name, line=dict(color=BLUE, width=3))
        )
        if f1_hist:
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(f1_hist) + 1)),
                    y=f1_hist,
                    mode="lines",
                    name="Train F1",
                    line=dict(color=GREEN, width=3),
                )
            )
    else:
        fig.add_trace(
            go.Scatter(x=[1, 8], y=[None, None], mode="lines", name=series_name, line=dict(color=BLUE, width=3))
        )
        fig.add_annotation(
            text="Train the hybrid engine to plot GA/BPN convergence",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=MUTED, size=13),
        )
    fig.update_layout(xaxis_title="", yaxis_title="", autosize=True)
    fig.update_xaxes(range=[1, max(len(series), 8)])
    fig.update_yaxes(range=[0, 1.05])
    return apply_layout(fig, height, margin=dict(l=36, r=16, t=12, b=48))


def _axis_font() -> dict:
    return dict(family="Inter, Segoe UI, sans-serif", color="#0f172a", size=13)


def ga_fitness_chart(history: list[float] | None, fuzzy_history: list[float] | None = None) -> go.Figure:
    """Fitness by generation. Margins keep the 0.xx ticks and the axis titles inside the card."""
    fig = go.Figure()
    series: list[float] = []
    if history:
        fig.add_scatter(y=history, mode="lines+markers", name="Feature GA", line=dict(color=BLUE, width=2))
        series.extend(history)
    if fuzzy_history:
        fig.add_scatter(y=fuzzy_history, mode="lines+markers", name="Fuzzy membership GA", line=dict(color=ORANGE, width=2))
        series.extend(fuzzy_history)
    y_range = None
    if series:
        low, high = min(series), max(series)
        pad = max(0.02, (high - low) * 0.25)
        y_range = [low - pad, high + pad]
    n_gen = max(len(history or []), len(fuzzy_history or []))
    fig.update_layout(
        height=340,
        margin=dict(l=88, r=36, t=56, b=72),
        title=dict(text="Best fitness by generation", x=0, xanchor="left"),
        font=_axis_font(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(
        title=dict(text="Generation", standoff=18),
        automargin=True,
        range=[-0.6, max(n_gen - 1, 1) + 0.6],
        tickfont=dict(size=13, color="#0f172a"),
        title_font=dict(size=13, color="#0f172a"),
    )
    fig.update_yaxes(
        title=dict(text="Best fitness", standoff=12),
        automargin=True,
        tickformat=".2f",
        range=y_range,
        tickfont=dict(size=13, color="#0f172a"),
        title_font=dict(size=13, color="#0f172a"),
    )
    return fig


def ga_frequency_chart(frequency: dict[str, float]) -> go.Figure:
    """Horizontal feature frequencies. The left margin grows with the longest metric name."""
    freq = sorted(frequency.items(), key=lambda kv: kv[1])
    longest = max((len(name) for name, _ in freq), default=4)
    fig = go.Figure(go.Bar(
        x=[value for _, value in freq],
        y=[name for name, _ in freq],
        orientation="h",
        marker=dict(color=BLUE),
        cliponaxis=False,
    ))
    fig.update_layout(
        height=max(360, 32 * len(freq) + 100),
        margin=dict(l=max(168, 10 * longest + 48), r=48, t=56, b=72),
        title=dict(text="Feature selection frequency", x=0, xanchor="left"),
        font=_axis_font(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(
        title=dict(text="Share of generations", standoff=16),
        automargin=True,
        range=[-0.08, 1.08],
        tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],
        tickformat=".1f",
        tickfont=dict(size=13, color="#0f172a"),
        title_font=dict(size=13, color="#0f172a"),
    )
    fig.update_yaxes(
        automargin=True,
        tickfont=dict(size=13, color="#0f172a"),
    )
    return fig


def loss_chart(loss: list[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss, mode="lines", name="Loss", line=dict(color=RED, width=3)))
    return apply_layout(fig, 180)


def donut(clean: int, defective: int) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            values=[clean, defective],
            labels=["Clean", "Defective"],
            hole=0.72,
            marker=dict(colors=[BLUE, ORANGE]),
            textinfo="none",
            sort=False,
        )
    )
    fig.update_layout(_layout(height=190, showlegend=False, margin=dict(l=10, r=10, t=10, b=10)))
    return fig


def probability_donut(prob: float) -> go.Figure:
    color = RED if prob >= 0.5 else BLUE
    fig = go.Figure(
        go.Pie(
            values=[prob, 1 - prob],
            labels=["p", "rest"],
            hole=0.74,
            marker=dict(colors=[color, "#f1f5f9"]),
            textinfo="none",
            sort=False,
        )
    )
    fig.update_layout(
        _layout(
            height=170,
            showlegend=False,
            margin=dict(l=8, r=8, t=8, b=8),
            annotations=[
                dict(
                    text=f"<b>{prob*100:.1f}%</b><br><span style='font-size:10px;color:#94a3b8'>PROBABILITY</span>",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
            ],
        )
    )
    return fig


def risk_bars(counts: dict[str, int]) -> go.Figure:
    order = ["Critical", "High", "Medium", "Low"]
    colors = {"Critical": RED, "High": ORANGE, "Medium": BLUE, "Low": GREEN}
    values = [counts.get(name.upper(), counts.get(name, 0)) for name in order]
    fig = go.Figure(
        go.Bar(
            y=order,
            x=values,
            orientation="h",
            marker_color=[colors[n] for n in order],
            text=values,
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(bargap=0.45)
    fig = apply_layout(fig, 210)
    # First category (Critical) belongs at the top of a horizontal bar chart.
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(visible=False)
    return fig


def grouped_metrics(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    names = table["Classifier Model Architecture"].tolist()
    for col, color in [("Precision", BLUE), ("Recall", GREEN), ("F1-Score", ORANGE)]:
        fig.add_trace(go.Bar(y=names, x=table[col], name=col, orientation="h", marker_color=color))
    fig.update_layout(barmode="group", bargap=0.35)
    fig = apply_layout(fig, 320, margin=dict(l=16, r=16, t=12, b=36))
    fig.update_yaxes(autorange="reversed", automargin=True)
    return fig


def roc_chart(roc: dict[str, tuple[np.ndarray, np.ndarray]], auc_values: dict[str, float]) -> go.Figure:
    fig = go.Figure()
    palette = [BLUE, GREEN, ORANGE, RED, MUTED]
    for i, (name, (fpr, tpr)) in enumerate(roc.items()):
        auc = auc_values.get(name, float("nan"))
        label = f"{name} (AUC = {auc:.2f})" if np.isfinite(auc) else name
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=label, line=dict(color=palette[i % len(palette)], width=3))
        )
    fig.update_xaxes(range=[0, 1], title="False positive rate")
    fig.update_yaxes(range=[0, 1], title="True positive rate")
    return apply_layout(
        fig,
        320,
        margin=dict(l=48, r=16, t=12, b=96),
        legend=dict(orientation="h", yanchor="top", y=-0.28, x=0, font=dict(size=11, color="#64748b")),
    )


def correlation_heatmap(corr: pd.DataFrame, limit: int = 6) -> go.Figure:
    subset = corr.iloc[:limit, :limit]
    labels = [display_name(c).replace("_", " ") for c in subset.columns]
    fig = go.Figure(
        go.Heatmap(
            z=subset.values,
            x=labels,
            y=labels,
            colorscale=[[0, "#b91c1c"], [0.5, "#f8fafc"], [1, "#1d4ed8"]],
            zmin=-1,
            zmax=1,
            text=np.round(subset.values, 2),
            texttemplate="%{text:.2f}",
            showscale=False,
        )
    )
    fig.update_layout(_layout(height=360, margin=dict(l=120, r=16, t=10, b=80)))
    fig.update_xaxes(tickangle=-35)
    return fig


def membership_chart(triangles: list[Triangle], colors: list[str], title_legend: list[str]) -> go.Figure:
    fig = go.Figure()
    for tri, color, label in zip(triangles, colors, title_legend):
        xs = [tri.a, tri.b, tri.c]
        ys = [0, 1, 0]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=3), name=label))
    fig.update_yaxes(visible=False, range=[-0.05, 1.15])
    fig.update_xaxes(showgrid=False)
    return apply_layout(fig, 210)


def contribution_bars(items: list[tuple[str, float]]) -> go.Figure:
    names = [display_name(n) for n, _ in items][::-1]
    vals = [v * 100 for _, v in items][::-1]
    colors = [RED if v >= 0 else GREEN for v in vals]
    fig = go.Figure(
        go.Bar(
            y=names,
            x=[abs(v) for v in vals],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in vals],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig = apply_layout(fig, 210, margin=dict(l=8, r=48, t=8, b=8))
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(visible=False)
    return fig

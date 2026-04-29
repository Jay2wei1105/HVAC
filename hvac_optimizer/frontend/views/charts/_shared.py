"""Shared design tokens and helpers for all HVAC dashboard chart modules."""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

# ── Design tokens ───────────────────────────────────────────────────────────
C = {
    "total":   "#2E5FB8",
    "chiller": "#4967A8",
    "chwp":    "#3D8FA2",
    "cwp":     "#B2854E",
    "ct":      "#8B6AA7",
    "oa_temp": "#B85B65",
    "oa_rh":   "#4C8F84",
    "delta":   "#6A7282",
    "cop":     "#355E9A",
}

_BASE_LAYOUT: dict[str, Any] = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=13, color="#172133"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#F3F6FB",
    margin=dict(t=52, b=48, l=62, r=24),
    legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center", font_size=12, font=dict(color="#172133")),
    hoverlabel=dict(bgcolor="#fbfcff", font_size=12, bordercolor="#ced8e6", font=dict(color="#172133")),
    xaxis=dict(
        tickfont=dict(color="#172133", size=12),
        title=dict(font=dict(color="#172133", size=13)),
    ),
    yaxis=dict(
        tickfont=dict(color="#172133", size=12),
        title=dict(font=dict(color="#172133", size=13)),
    ),
    colorway=list(C.values()),
)

RANGE_BUTTONS = [
    dict(count=1,  label="1D",  step="day",   stepmode="backward"),
    dict(count=7,  label="7D",  step="day",   stepmode="backward"),
    dict(count=14, label="14D", step="day",   stepmode="backward"),
    dict(step="all", label="全部"),
]

PLOTLY_CONFIG = {"displayModeBar": True, "scrollZoom": True,
                 "modeBarButtonsToRemove": ["lasso2d","select2d"], "displaylogo": False}


def layout(**overrides: Any) -> dict[str, Any]:
    """Return base layout merged with caller overrides."""
    base = dict(_BASE_LAYOUT)
    base.update(overrides)
    return base


def empty_fig(msg: str = "此欄位無資料") -> go.Figure:
    """Placeholder figure when required columns are absent."""
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font_size=14, font_color="#253044")
    fig.update_layout(**layout(height=300))
    return fig


def ts_axis(title: str = "") -> dict[str, Any]:
    """Standard time-series x-axis with range selector."""
    return dict(
        title=title,
        type="date",
        rangeselector=dict(buttons=RANGE_BUTTONS,
                           bgcolor="#edf2f9", activecolor="#2E5FB8",
                           font=dict(color="#1d2738", size=11)),
        rangeslider=dict(visible=True, thickness=0.04),
        gridcolor="#d3dcea",
    )

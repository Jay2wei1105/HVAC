"""Shared design tokens and helpers for all HVAC dashboard chart modules."""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

# ── Design tokens ───────────────────────────────────────────────────────────
C = {
    "total":   "#2F6DF6",
    "chiller": "#4F46E5",
    "chwp":    "#0EA5E9",
    "cwp":     "#F59E0B",
    "ct":      "#A855F7",
    "oa_temp": "#EF4444",
    "oa_rh":   "#14B8A6",
    "delta":   "#64748B",
    "cop":     "#2563EB",
}

_BASE_LAYOUT: dict[str, Any] = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=12, color="#111827"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#F8FAFD",
    margin=dict(t=52, b=48, l=62, r=24),
    legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center", font_size=11),
    hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#d9dfeb"),
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
                       x=0.5, y=0.5, showarrow=False, font_size=14, font_color="#70787c")
    fig.update_layout(**layout(height=300))
    return fig


def ts_axis(title: str = "") -> dict[str, Any]:
    """Standard time-series x-axis with range selector."""
    return dict(
        title=title,
        type="date",
        rangeselector=dict(buttons=RANGE_BUTTONS,
                           bgcolor="#f6f8fd", activecolor="#2F6DF6",
                           font=dict(color="#111827", size=11)),
        rangeslider=dict(visible=True, thickness=0.04),
        gridcolor="#e2e8f0",
    )

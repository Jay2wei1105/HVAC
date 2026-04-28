"""Environmental-impact charts: OA temperature correlation with system load."""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ._shared import C, PLOTLY_CONFIG, empty_fig, layout, ts_axis

MONTH_ZH = ["","一月","二月","三月","四月","五月","六月","七月","八月","九月","十月","十一月","十二月"]


def _col(recs: list[dict], key: str) -> list:
    return [r.get(key) for r in recs]


# ── figures ──────────────────────────────────────────────────────────────────

def fig_oa_dual_axis(data: dict[str, Any]) -> go.Figure:
    """Dual-axis: OA temperature and total kW over time."""
    recs    = data.get("series", {}).get("hourly") or []
    has_oa  = any(r.get("oa_temp")  is not None for r in recs)
    has_pwr = any(r.get("total_kw") is not None for r in recs)
    if not recs or not (has_oa and has_pwr):
        return empty_fig("缺少室外溫度或用電資料")
    ts = _col(recs, "ts")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ts, y=_col(recs,"total_kw"), name="系統用電",
                              mode="lines", line=dict(color=C["total"], width=2),
                              fill="tozeroy", fillcolor="rgba(15,76,92,0.07)",
                              hovertemplate="用電: %{y:.1f} kW<extra></extra>"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=ts, y=_col(recs,"oa_temp"), name="室外溫度",
                              mode="lines", line=dict(color=C["oa_temp"], width=1.8, dash="dot"),
                              hovertemplate="OA: %{y:.1f} °C<extra></extra>"),
                  secondary_y=True)
    if any(r.get("oa_rh") is not None for r in recs):
        fig.add_trace(go.Scatter(x=ts, y=_col(recs,"oa_rh"), name="室外濕度",
                                  mode="lines", line=dict(color=C["oa_rh"], width=1.4, dash="dash"),
                                  opacity=0.7,
                                  hovertemplate="RH: %{y:.1f}%%<extra></extra>"),
                      secondary_y=True)
    fig.update_layout(**layout(title="系統用電 vs 室外氣象（雙軸趨勢）",
                                xaxis=ts_axis(), height=380))
    fig.update_yaxes(title_text="系統用電 (kW)", gridcolor="#e2e8f0", secondary_y=False)
    fig.update_yaxes(title_text="溫度 (°C) / 濕度 (%)", gridcolor="#e2e8f0", secondary_y=True)
    return fig


def fig_oa_scatter(data: dict[str, Any]) -> go.Figure:
    """Scatter: OA temperature vs total kW, coloured by hour of day."""
    pts = data.get("scatter", {}).get("oa_vs_power") or []
    if not pts or all(r.get("oa_temp") is None for r in pts):
        return empty_fig("無環境散佈資料")
    oa   = [r.get("oa_temp")  for r in pts]
    pwr  = [r.get("total_kw") for r in pts]
    hour = [r.get("hour", 12) for r in pts]
    fig = go.Figure(go.Scatter(
        x=oa, y=pwr, mode="markers",
        marker=dict(size=5, color=hour, colorscale="RdYlBu_r", opacity=0.55,
                    colorbar=dict(title="時段", tickfont_size=10)),
        hovertemplate="OA: %{x:.1f} °C<br>Total: %{y:.1f} kW<br>時段 %{marker.color}:00<extra></extra>",
    ))
    # simple linear trendline
    import numpy as np
    oa_c   = [v for v in oa  if v is not None]
    pwr_c  = [v for v in pwr if v is not None]
    if len(oa_c) > 10:
        try:
            coeffs = np.polyfit(oa_c, pwr_c, 1)
            x_r    = [min(oa_c), max(oa_c)]
            y_r    = [coeffs[0]*x + coeffs[1] for x in x_r]
            fig.add_trace(go.Scatter(x=x_r, y=y_r, mode="lines", name="線性趨勢",
                                      line=dict(color="#E53935", width=2, dash="dash"),
                                      hoverinfo="skip"))
        except Exception:
            pass
    fig.update_layout(**layout(
        title="室外溫度 vs 系統用電（點顏色=時段）",
        xaxis=dict(title="室外溫度 (°C)", gridcolor="#e2e8f0"),
        yaxis=dict(title="系統用電 (kW)", gridcolor="#e2e8f0"),
        height=380,
    ))
    return fig


def fig_rh_scatter(data: dict[str, Any]) -> go.Figure:
    """Scatter: OA humidity vs total kW."""
    pts = data.get("scatter", {}).get("oa_vs_power") or []
    has_rh  = any(r.get("oa_rh")   is not None for r in pts)
    has_pwr = any(r.get("total_kw") is not None for r in pts)
    if not pts or not (has_rh and has_pwr):
        return empty_fig("缺少濕度資料")
    rh  = [r.get("oa_rh")   for r in pts]
    pwr = [r.get("total_kw") for r in pts]
    fig = go.Figure(go.Scatter(
        x=rh, y=pwr, mode="markers",
        marker=dict(size=4, color=C["oa_rh"], opacity=0.45),
        hovertemplate="RH: %{x:.1f}%%<br>Total: %{y:.1f} kW<extra></extra>",
    ))
    fig.update_layout(**layout(
        title="室外相對濕度 vs 系統用電",
        xaxis=dict(title="室外相對濕度 (%)", gridcolor="#e2e8f0"),
        yaxis=dict(title="系統用電 (kW)",    gridcolor="#e2e8f0"),
        height=320,
    ))
    return fig


def fig_monthly_overview(data: dict[str, Any]) -> go.Figure:
    """Bar + line: monthly energy consumption and average OA temperature."""
    monthly = data.get("profile", {}).get("by_month") or []
    if not monthly:
        return empty_fig("無月份彙整資料")
    months = [MONTH_ZH[int(r.get("month", 1))] for r in monthly]
    kwh    = _col(monthly, "total_kwh")
    avg_kw = _col(monthly, "avg_kw")
    oa     = _col(monthly, "avg_oa_temp")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=months, y=kwh, name="月用電量 (kWh)",
                          marker_color=C["total"], opacity=0.82,
                          hovertemplate="%{x}: %{y:,.0f} kWh<extra></extra>"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=months, y=avg_kw, name="月均負載 (kW)",
                              mode="lines+markers", line=dict(color=C["chwp"], width=2),
                              marker=dict(size=7),
                              hovertemplate="%{x}: 均值 %{y:.1f} kW<extra></extra>"),
                  secondary_y=False)
    if any(v is not None for v in oa):
        fig.add_trace(go.Scatter(x=months, y=oa, name="月均室外溫度",
                                  mode="lines+markers",
                                  line=dict(color=C["oa_temp"], width=2, dash="dot"),
                                  marker=dict(size=7),
                                  hovertemplate="%{x}: %{y:.1f} °C<extra></extra>"),
                      secondary_y=True)
    fig.update_layout(**layout(title="月份能耗彙整 + 室外氣溫", height=340))
    fig.update_yaxes(title_text="用電量 / kW", gridcolor="#e2e8f0", secondary_y=False)
    fig.update_yaxes(title_text="溫度 (°C)", gridcolor="#e2e8f0", secondary_y=True)
    return fig


# ── tab renderer ─────────────────────────────────────────────────────────────

def render_tab(data: dict[str, Any]) -> None:
    """Tab 4 — 🌡️ 環境相關性分析."""
    st.plotly_chart(fig_oa_dual_axis(data), use_container_width=True, config=PLOTLY_CONFIG)
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.plotly_chart(fig_oa_scatter(data), use_container_width=True, config=PLOTLY_CONFIG)
    with c2:
        st.plotly_chart(fig_rh_scatter(data), use_container_width=True, config=PLOTLY_CONFIG)
    st.plotly_chart(fig_monthly_overview(data), use_container_width=True, config=PLOTLY_CONFIG)

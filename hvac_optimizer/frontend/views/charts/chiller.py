"""Chiller-specific analysis charts: power, temperatures, COP, scatter."""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ._shared import C, PLOTLY_CONFIG, empty_fig, layout, ts_axis


def _col(recs: list[dict], key: str) -> list:
    return [r.get(key) for r in recs]


# ── figures ──────────────────────────────────────────────────────────────────

def fig_chiller_power(data: dict[str, Any]) -> go.Figure:
    """Chiller kW time series with CHWP kW overlay."""
    recs = data.get("series", {}).get("hourly") or []
    if not recs or all(r.get("ch_kw") is None for r in recs):
        return empty_fig("無冰水主機用電資料")
    ts = _col(recs, "ts")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=_col(recs, "ch_kw"), name="冰水主機 kW",
                              mode="lines", line=dict(color=C["chiller"], width=2),
                              fill="tozeroy", fillcolor="rgba(0,108,73,0.08)",
                              hovertemplate="CH: %{y:.1f} kW<extra></extra>"))
    if any(r.get("chwp_kw") is not None for r in recs):
        fig.add_trace(go.Scatter(x=ts, y=_col(recs, "chwp_kw"), name="冰水泵 kW",
                                  mode="lines", line=dict(color=C["chwp"], width=1.5, dash="dot"),
                                  hovertemplate="CHWP: %{y:.1f} kW<extra></extra>"))
    fig.update_layout(**layout(
        title="冰水主機用電趨勢（小時均值）",
        xaxis=ts_axis(), yaxis=dict(title="kW", gridcolor="#e2e8f0"), height=360,
    ))
    return fig


def fig_chw_temperatures(data: dict[str, Any]) -> go.Figure:
    """CHW supply / return temperatures and ΔT on secondary axis."""
    recs = data.get("series", {}).get("hourly") or []
    has_chw = any(r.get("chw_supply") is not None for r in recs)
    has_dt  = any(r.get("delta_chw") is not None for r in recs)
    if not recs or (not has_chw and not has_dt):
        return empty_fig("無冰水溫度資料")
    ts = _col(recs, "ts")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if has_chw:
        fig.add_trace(go.Scatter(x=ts, y=_col(recs, "chw_supply"), name="CHW 供水溫",
                                  mode="lines", line=dict(color="#3d8fa2", width=1.8),
                                  hovertemplate="供水: %{y:.2f} °C<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=ts, y=_col(recs, "chw_return"), name="CHW 回水溫",
                                  mode="lines", line=dict(color="#355e9a", width=1.8),
                                  hovertemplate="回水: %{y:.2f} °C<extra></extra>"), secondary_y=False)
    if has_dt:
        fig.add_trace(go.Scatter(x=ts, y=_col(recs, "delta_chw"), name="ΔT 冰水側",
                                  mode="lines", line=dict(color=C["delta"], width=2, dash="dash"),
                                  hovertemplate="ΔT: %{y:.2f} °C<extra></extra>"), secondary_y=True)
    fig.update_layout(**layout(title="冰水側溫度趨勢 + 溫差（ΔT）",
                                xaxis=ts_axis(), height=360))
    fig.update_yaxes(title_text="溫度 (°C)", gridcolor="#e2e8f0", secondary_y=False)
    fig.update_yaxes(title_text="ΔT (°C)", gridcolor="#e2e8f0", secondary_y=True)
    return fig


def fig_cop_trend(data: dict[str, Any]) -> go.Figure:
    """Estimated COP over time and 24h COP profile side-by-side."""
    recs   = data.get("series", {}).get("hourly") or []
    p_recs = data.get("profile", {}).get("by_hour") or []
    has_cop_series  = any(r.get("cop_est") is not None for r in recs)
    has_cop_profile = p_recs and p_recs[0].get("cop_est_mean") is not None
    if not has_cop_series and not has_cop_profile:
        return empty_fig("無法估算 COP（缺少冰水流量或主機電力資料）")
    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35],
                        subplot_titles=("COP 估算時序（小時均值）", "24h COP 平均分佈"),
                        shared_yaxes=True)
    if has_cop_series:
        ts = _col(recs, "ts")
        fig.add_trace(go.Scatter(x=ts, y=_col(recs, "cop_est"), name="COP 估算",
                                  mode="lines", line=dict(color=C["cop"], width=1.8),
                                  hovertemplate="COP: %{y:.2f}<extra></extra>"),
                      row=1, col=1)
    if has_cop_profile:
        hours   = [r["hour"] for r in p_recs]
        cop_avg = [r.get("cop_est_mean") for r in p_recs]
        cop_p10 = [r.get("cop_est_p10")  for r in p_recs]
        cop_p90 = [r.get("cop_est_p90")  for r in p_recs]
        fig.add_trace(go.Scatter(x=hours + hours[::-1], y=cop_p90 + cop_p10[::-1],
                                  fill="toself", fillcolor="rgba(21,101,192,0.10)",
                                  line=dict(width=0), name="P10–P90", showlegend=False),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=hours, y=cop_avg, mode="lines+markers",
                                  name="24h COP均值", line=dict(color=C["cop"], width=2),
                                  marker=dict(size=5)),
                      row=1, col=2)
    fig.update_layout(**layout(title="冰水主機 COP 分析（估算值，假設流量單位為 m³/h）", height=360))
    fig.update_xaxes(type="date", row=1, col=1)
    fig.update_xaxes(title_text="時段", dtick=4, row=1, col=2)
    fig.update_yaxes(title_text="COP", gridcolor="#e2e8f0")
    return fig


def fig_chiller_scatter(data: dict[str, Any]) -> go.Figure:
    """Scatter: CH_kW vs OA temperature, coloured by time of day."""
    pts = data.get("scatter", {}).get("oa_vs_power") or []
    if not pts or all(r.get("ch_kw") is None for r in pts):
        return empty_fig("無散佈圖資料")
    oa   = [r.get("oa_temp") for r in pts]
    ch   = [r.get("ch_kw")   for r in pts]
    hour = [r.get("hour", 0) for r in pts]
    fig = go.Figure(go.Scatter(
        x=oa, y=ch, mode="markers",
        marker=dict(size=5, color=hour, colorscale="Viridis", opacity=0.6,
                    colorbar=dict(title="時段", tickfont_size=10)),
        hovertemplate="OA: %{x:.1f} °C<br>CH: %{y:.1f} kW<br>時段 %{marker.color}:00<extra></extra>",
    ))
    fig.update_layout(**layout(
        title="冰水主機用電 vs 室外溫度（點顏色=時段）",
        xaxis=dict(title="室外溫度 (°C)", gridcolor="#e2e8f0"),
        yaxis=dict(title="冰水主機 kW", gridcolor="#e2e8f0"),
        height=380,
    ))
    return fig


def fig_chw_delta_histogram(data: dict[str, Any]) -> go.Figure:
    """Distribution of CHW temperature differential."""
    dt = data.get("distribution", {}).get("delta_chw") or {}
    if not dt:
        return empty_fig("無 ΔT 分佈資料")
    counts, edges = dt["counts"], dt["edges"]
    midpoints = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
    fig = go.Figure(go.Bar(x=midpoints, y=counts, marker_color=C["delta"], opacity=0.8,
                            hovertemplate="ΔT: %{x:.2f} °C<br>次數: %{y}<extra></extra>"))
    fig.update_layout(**layout(title="冰水側溫差（ΔT）分佈",
                                xaxis_title="ΔT (°C)", yaxis=dict(title="次數", gridcolor="#e2e8f0"),
                                height=300))
    return fig


# ── tab renderer ─────────────────────────────────────────────────────────────

def render_tab(data: dict[str, Any]) -> None:
    """Tab 2 — ❄️ 冰水主機 分析."""
    st.plotly_chart(fig_chiller_power(data), use_container_width=True, config=PLOTLY_CONFIG)
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.plotly_chart(fig_chw_temperatures(data), use_container_width=True, config=PLOTLY_CONFIG)
    with c2:
        st.plotly_chart(fig_chw_delta_histogram(data), use_container_width=True, config=PLOTLY_CONFIG)
    st.plotly_chart(fig_cop_trend(data), use_container_width=True, config=PLOTLY_CONFIG)
    st.plotly_chart(fig_chiller_scatter(data), use_container_width=True, config=PLOTLY_CONFIG)

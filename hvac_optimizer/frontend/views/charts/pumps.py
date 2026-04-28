"""Pump & cooling-tower analysis charts: power, frequency, CW temperatures."""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ._shared import C, PLOTLY_CONFIG, empty_fig, layout, ts_axis


def _col(recs: list[dict], key: str) -> list:
    return [r.get(key) for r in recs]


# ── figures ──────────────────────────────────────────────────────────────────

def fig_pump_power(data: dict[str, Any]) -> go.Figure:
    """Stacked area: CHWP + CWP + CT fan power over time."""
    recs = data.get("series", {}).get("hourly") or []
    layers = [("chwp_kw","冰水泵",C["chwp"]), ("cwp_kw","冷卻水泵",C["cwp"]), ("ct_kw","冷卻塔風扇",C["ct"])]
    available = [l for l in layers if any(r.get(l[0]) is not None for r in recs)]
    if not recs or not available:
        return empty_fig("無水泵/冷卻塔用電資料")
    ts = _col(recs, "ts")
    fig = go.Figure()
    for key, name, color in available:
        fig.add_trace(go.Scatter(x=ts, y=_col(recs, key), name=name,
                                  mode="lines", stackgroup="pumps",
                                  fillcolor=color, line=dict(width=0.5, color=color),
                                  hovertemplate=f"{name}: %{{y:.1f}} kW<extra></extra>"))
    fig.update_layout(**layout(title="水泵 & 冷卻塔 用電趨勢（堆疊面積）",
                                xaxis=ts_axis(), yaxis=dict(title="kW", gridcolor="#e2e8f0"),
                                height=360))
    return fig


def fig_frequency_profiles(data: dict[str, Any]) -> go.Figure:
    """24h average frequency profiles for CHWP, CWP, CT fan."""
    p_recs = data.get("profile", {}).get("by_hour") or []
    layers = [("chwp_hz_mean","冰水泵 Hz",C["chwp"]),
              ("cwp_hz_mean", "冷卻水泵 Hz",C["cwp"]),
              ("ct_hz_mean",  "冷卻塔 Hz", C["ct"])]
    available = [l for l in layers if p_recs and p_recs[0].get(l[0]) is not None]
    if not available:
        return empty_fig("無頻率剖面資料")
    hours = [r["hour"] for r in p_recs]
    fig = go.Figure()
    for key, name, color in available:
        fig.add_trace(go.Scatter(x=hours, y=[r.get(key) for r in p_recs],
                                  name=name, mode="lines+markers",
                                  line=dict(color=color, width=2),
                                  marker=dict(size=5),
                                  hovertemplate=f"{name}: %{{y:.1f}} Hz<extra></extra>"))
    fig.update_layout(**layout(title="24h 平均頻率分佈",
                                xaxis=dict(title="時段（時）", dtick=2, gridcolor="#e2e8f0"),
                                yaxis=dict(title="Hz", gridcolor="#e2e8f0"), height=320))
    return fig


def fig_freq_boxplots(data: dict[str, Any]) -> go.Figure:
    """Box-plot distribution of operating frequencies."""
    recs = data.get("series", {}).get("hourly") or []
    layers = [("chwp_hz","冰水泵",C["chwp"]), ("cwp_hz","冷卻水泵",C["cwp"]), ("ct_hz","冷卻塔",C["ct"])]
    available = [(k, n, c) for k, n, c in layers if any(r.get(k) is not None for r in recs)]
    if not available:
        return empty_fig("無頻率資料")
    fig = go.Figure()
    for key, name, color in available:
        vals = [r[key] for r in recs if r.get(key) is not None]
        fig.add_trace(go.Box(y=vals, name=name, marker_color=color,
                              boxmean="sd", hovertemplate=f"{name}: %{{y:.1f}} Hz<extra></extra>"))
    fig.update_layout(**layout(title="頻率操作分佈（箱型圖）",
                                yaxis=dict(title="Hz", gridcolor="#e2e8f0"), height=320))
    return fig


def fig_cw_temperatures(data: dict[str, Any]) -> go.Figure:
    """Cooling water supply/return + ΔT secondary axis."""
    recs   = data.get("series", {}).get("hourly") or []
    has_cw = any(r.get("cw_supply") is not None for r in recs)
    has_dt = any(r.get("delta_cw")  is not None for r in recs)
    if not recs or (not has_cw and not has_dt):
        return empty_fig("無冷卻水溫度資料")
    ts = _col(recs, "ts")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if has_cw:
        fig.add_trace(go.Scatter(x=ts, y=_col(recs,"cw_supply"), name="CW 供水溫",
                                  mode="lines", line=dict(color=C["cwp"], width=1.8),
                                  hovertemplate="CWS: %{y:.2f} °C<extra></extra>"), secondary_y=False)
        fig.add_trace(go.Scatter(x=ts, y=_col(recs,"cw_return"), name="CW 回水溫",
                                  mode="lines", line=dict(color=C["ct"], width=1.8),
                                  hovertemplate="CWR: %{y:.2f} °C<extra></extra>"), secondary_y=False)
    if has_dt:
        fig.add_trace(go.Scatter(x=ts, y=_col(recs,"delta_cw"), name="ΔT 冷卻水側",
                                  mode="lines", line=dict(color=C["delta"], width=2, dash="dash"),
                                  hovertemplate="ΔT: %{y:.2f} °C<extra></extra>"), secondary_y=True)
    fig.update_layout(**layout(title="冷卻水側溫度趨勢 + 溫差（ΔT）",
                                xaxis=ts_axis(), height=360))
    fig.update_yaxes(title_text="溫度 (°C)", gridcolor="#e2e8f0", secondary_y=False)
    fig.update_yaxes(title_text="ΔT (°C)",   gridcolor="#e2e8f0", secondary_y=True)
    return fig


def fig_ct_scatter(data: dict[str, Any]) -> go.Figure:
    """Scatter: CT Hz vs OA temperature."""
    recs  = data.get("series", {}).get("hourly") or []
    has   = lambda k: any(r.get(k) is not None for r in recs)  # noqa: E731
    if not recs or not (has("ct_hz") and has("oa_temp")):
        return empty_fig("無冷卻塔散佈圖資料")
    oa  = [r.get("oa_temp") for r in recs if r.get("ct_hz") is not None and r.get("oa_temp") is not None]
    hz  = [r.get("ct_hz")   for r in recs if r.get("ct_hz") is not None and r.get("oa_temp") is not None]
    cwp = [r.get("cwp_kw", 0) for r in recs if r.get("ct_hz") is not None and r.get("oa_temp") is not None]
    fig = go.Figure(go.Scatter(
        x=oa, y=hz, mode="markers",
        marker=dict(size=5, color=cwp, colorscale="Blues", opacity=0.65,
                    colorbar=dict(title="CWP kW", tickfont_size=10)),
        hovertemplate="OA: %{x:.1f} °C<br>CT Hz: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(**layout(title="冷卻塔頻率 vs 室外溫度（點顏色=冷卻水泵 kW）",
                                xaxis=dict(title="室外溫度 (°C)", gridcolor="#e2e8f0"),
                                yaxis=dict(title="CT 頻率 (Hz)", gridcolor="#e2e8f0"),
                                height=340))
    return fig


# ── tab renderer ─────────────────────────────────────────────────────────────

def render_tab(data: dict[str, Any]) -> None:
    """Tab 3 — 💧 水泵 & 水塔 分析."""
    st.plotly_chart(fig_pump_power(data), use_container_width=True, config=PLOTLY_CONFIG)
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.plotly_chart(fig_frequency_profiles(data), use_container_width=True, config=PLOTLY_CONFIG)
    with c2:
        st.plotly_chart(fig_freq_boxplots(data), use_container_width=True, config=PLOTLY_CONFIG)
    st.plotly_chart(fig_cw_temperatures(data), use_container_width=True, config=PLOTLY_CONFIG)
    st.plotly_chart(fig_ct_scatter(data), use_container_width=True, config=PLOTLY_CONFIG)

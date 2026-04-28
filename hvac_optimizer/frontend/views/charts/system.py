"""System-level analysis charts: total power, profiles, heatmap, LDC."""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ._shared import C, PLOTLY_CONFIG, empty_fig, layout, ts_axis


# ── helpers ──────────────────────────────────────────────────────────────────

def _col(recs: list[dict], key: str) -> list:
    return [r.get(key) for r in recs]


# ── figures ──────────────────────────────────────────────────────────────────

def fig_total_power(data: dict[str, Any]) -> go.Figure:
    """Full-period total power time series with component breakdown."""
    recs = data.get("series", {}).get("hourly") or []
    if not recs:
        return empty_fig("無時序資料")
    ts = _col(recs, "ts")
    fig = go.Figure()
    power_layers = [
        ("ch_kw",   "冰水主機",   C["chiller"]),
        ("chwp_kw", "冰水泵",     C["chwp"]),
        ("cwp_kw",  "冷卻水泵",   C["cwp"]),
        ("ct_kw",   "冷卻塔風扇", C["ct"]),
    ]
    has_components = any(recs[0].get(k) is not None for k, *_ in power_layers)
    if has_components:
        for key, name, color in power_layers:
            vals = _col(recs, key)
            if any(v is not None for v in vals):
                fig.add_trace(go.Scatter(x=ts, y=vals, name=name, mode="lines",
                                         stackgroup="pwr", fillcolor=color,
                                         line=dict(width=0.5, color=color),
                                         hovertemplate=f"{name}: %{{y:.1f}} kW<extra></extra>"))
    total = _col(recs, "total_kw")
    if any(v is not None for v in total):
        fig.add_trace(go.Scatter(x=ts, y=total, name="Total", mode="lines",
                                  line=dict(color=C["total"], width=2.5),
                                  hovertemplate="Total: %{y:.1f} kW<extra></extra>"))
    fig.update_layout(**layout(
        title="系統用電全期趨勢（小時均值）",
        xaxis=ts_axis(), yaxis=dict(title="kW", gridcolor="#e2e8f0"),
        height=380,
    ))
    return fig


def fig_hourly_profile(data: dict[str, Any]) -> go.Figure:
    """24-hour average load profile with P10–P90 confidence band."""
    recs = data.get("profile", {}).get("by_hour") or []
    if not recs or "total_kw_mean" not in recs[0]:
        return empty_fig("無逐時分析資料")
    hours  = _col(recs, "hour")
    mean_  = _col(recs, "total_kw_mean")
    p10_   = _col(recs, "total_kw_p10")
    p90_   = _col(recs, "total_kw_p90")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours + hours[::-1], y=p90_ + p10_[::-1],
        fill="toself", fillcolor="rgba(15,76,92,0.10)", line=dict(width=0),
        name="P10–P90 區間", showlegend=True,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=hours, y=mean_, mode="lines+markers",
                              name="平均負載", line=dict(color=C["total"], width=2.5),
                              marker=dict(size=6), hovertemplate="Hour %{x}: %{y:.1f} kW<extra></extra>"))
    if "ch_kw_mean" in recs[0]:
        fig.add_trace(go.Scatter(x=hours, y=_col(recs, "ch_kw_mean"), mode="lines",
                                  name="冰水主機均值", line=dict(color=C["chiller"], width=1.8, dash="dot")))
    fig.update_layout(**layout(
        title="24小時平均負載曲線（P10 / 均值 / P90）",
        xaxis=dict(title="時段（時）", dtick=2, gridcolor="#e2e8f0"),
        yaxis=dict(title="kW", gridcolor="#e2e8f0"),
        height=340,
    ))
    return fig


def fig_power_donut(data: dict[str, Any]) -> go.Figure:
    """Equipment average power share donut chart."""
    ps = data.get("power_share") or {}
    if not ps:
        return empty_fig("無子設備功率資料")
    labels_map = {"ch_kw":"冰水主機","chwp_kw":"冰水泵","cwp_kw":"冷卻水泵","ct_kw":"冷卻塔風扇"}
    colors = [C["chiller"], C["chwp"], C["cwp"], C["ct"]]
    labels = [labels_map[k] for k in ps]
    values = list(ps.values())
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55,
                            marker_colors=colors[:len(labels)],
                            textinfo="label+percent", textfont_size=12,
                            hovertemplate="%{label}: %{value:.1f} kW 平均<extra></extra>"))
    total_avg = sum(v for v in values if v)
    fig.add_annotation(text=f"<b>{total_avg:.0f}</b><br>kW 均值",
                        x=0.5, y=0.5, showarrow=False, font_size=14, font_color="#0b1c30")
    fig.update_layout(**layout(title="設備用電佔比（期間均值）",
                                showlegend=True, height=310, margin=dict(t=52,b=20,l=20,r=20)))
    return fig


def fig_heatmap(data: dict[str, Any]) -> go.Figure:
    """Total kW heatmap: hour of day × day of week."""
    hm = data.get("heatmap", {}).get("hour_vs_weekday") or {}
    if not hm:
        return empty_fig("無熱力圖資料")
    fig = go.Figure(go.Heatmap(
        z=hm["z"], x=hm["x"], y=hm["y"],
        colorscale=[[0,"#dce9ff"],[0.5,"#0F4C5C"],[1.0,"#003441"]],
        hovertemplate="時段 %{y}:00 / %{x}<br>平均用電: %{z:.1f} kW<extra></extra>",
        colorbar=dict(title="kW", tickfont_size=11),
    ))
    fig.update_layout(**layout(
        title="用電熱力圖（逐時 × 星期）",
        xaxis=dict(title="星期"), yaxis=dict(title="時段（時）", dtick=4, autorange="reversed"),
        height=380, margin=dict(t=52, b=48, l=52, r=40),
    ))
    return fig


def fig_load_duration(data: dict[str, Any]) -> go.Figure:
    """Load duration curve: sorted descending total kW."""
    ldc = data.get("ldc") or []
    if not ldc:
        return empty_fig("無 LDC 資料")
    pct = [i / (len(ldc) - 1) * 100 for i in range(len(ldc))]
    peak = ldc[0] if ldc else 0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pct, y=ldc, mode="lines", fill="tozeroy",
                              fillcolor="rgba(15,76,92,0.12)", line=dict(color=C["total"], width=2),
                              hovertemplate="累計時間 %{x:.1f}%<br>負載: %{y:.1f} kW<extra></extra>"))
    fig.add_hline(y=peak * 0.8, line_dash="dot", line_color="#E53935",
                  annotation_text=f"80% Peak ({peak*0.8:.0f} kW)", annotation_font_size=11)
    fig.update_layout(**layout(
        title="負載持續時間曲線（LDC）",
        xaxis=dict(title="累計時間百分比 (%)", gridcolor="#e2e8f0"),
        yaxis=dict(title="kW", gridcolor="#e2e8f0"),
        height=340,
    ))
    return fig


def fig_weekday_bar(data: dict[str, Any]) -> go.Figure:
    """Average and peak load by day of week."""
    recs = data.get("profile", {}).get("by_weekday") or []
    if not recs:
        return empty_fig("無每日分析資料")
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    wd_labels = [days[int(r["weekday"])] for r in recs]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="平均負載", x=wd_labels, y=_col(recs, "avg_kw"),
                          marker_color=C["total"], opacity=0.85,
                          hovertemplate="%{x}: 平均 %{y:.1f} kW<extra></extra>"))
    if recs[0].get("peak_kw") is not None:
        fig.add_trace(go.Scatter(name="峰值負載", x=wd_labels, y=_col(recs, "peak_kw"),
                                  mode="markers+lines", marker=dict(size=9, color=C["oa_temp"]),
                                  line=dict(color=C["oa_temp"], width=1.5, dash="dot"),
                                  hovertemplate="%{x}: 峰值 %{y:.1f} kW<extra></extra>"))
    fig.update_layout(**layout(title="逐星期 平均 / 峰值 負載",
                                xaxis_title="星期", yaxis=dict(title="kW", gridcolor="#e2e8f0"),
                                height=320, barmode="group"))
    return fig


# ── tab renderer ─────────────────────────────────────────────────────────────

def render_overview_tab(data: dict[str, Any]) -> None:
    """Tab 0 — 系統總覽: main time series + 24h profile + donut."""
    st.plotly_chart(fig_total_power(data), use_container_width=True, config=PLOTLY_CONFIG)
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.plotly_chart(fig_hourly_profile(data), use_container_width=True, config=PLOTLY_CONFIG)
    with c2:
        st.plotly_chart(fig_power_donut(data), use_container_width=True, config=PLOTLY_CONFIG)


def render_depth_tab(data: dict[str, Any]) -> None:
    """Tab 1 — 用電深度分析: heatmap + LDC + weekday bar."""
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.plotly_chart(fig_heatmap(data), use_container_width=True, config=PLOTLY_CONFIG)
    with c2:
        st.plotly_chart(fig_load_duration(data), use_container_width=True, config=PLOTLY_CONFIG)
    st.plotly_chart(fig_weekday_bar(data), use_container_width=True, config=PLOTLY_CONFIG)

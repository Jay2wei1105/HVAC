from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hvac_optimizer.frontend import api_client
from hvac_optimizer.frontend.views.charts import chiller as chiller_charts
from hvac_optimizer.frontend.views.charts import environment as environment_charts
from hvac_optimizer.frontend.views.charts import pumps as pump_charts
from hvac_optimizer.frontend.views.charts import system as system_charts
from hvac_optimizer.frontend.views.charts._shared import C, PLOTLY_CONFIG, layout
from hvac_optimizer.frontend.views.charts.system import fig_heatmap


def _load(site_id: str) -> dict[str, Any]:
    return api_client.get_analytics(site_id)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .ho-analysis-hero h1 {
            font-size: clamp(1.9rem, 3vw, 2.35rem);
            font-weight: 700;
            letter-spacing: -0.03em;
            color: var(--ho-on-surface);
            margin: 0 0 0.35rem 0;
        }
        .ho-analysis-hero p {
            margin: 0;
            color: var(--ho-on-variant);
            font-size: 0.98rem;
        }
        .ho-panel-title {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0 0 1rem 0;
            font-size: 1.06rem;
            font-weight: 600;
            color: var(--ho-on-surface);
        }
        .ho-kpi-card {
            background: var(--ho-surface);
            border: 1px solid var(--ho-outline);
            border-radius: 1rem;
            padding: 1.1rem 1.15rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05), 0 8px 20px rgba(15, 23, 42, 0.03);
            min-height: 168px;
        }
        .ho-kpi-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.76rem;
            font-weight: 600;
            color: var(--ho-on-variant);
            margin-bottom: 0.55rem;
        }
        .ho-kpi-value {
            font-size: 2.1rem;
            line-height: 1.08;
            letter-spacing: -0.03em;
            font-weight: 700;
            color: var(--ho-on-surface);
            margin: 0;
        }
        .ho-kpi-unit {
            font-size: 0.96rem;
            font-weight: 500;
            color: var(--ho-on-variant);
        }
        .ho-kpi-trend {
            margin-top: 0.9rem;
            display: flex;
            gap: 0.45rem;
            align-items: center;
            font-size: 0.84rem;
            color: var(--ho-on-variant);
        }
        .ho-kpi-good { color: var(--ho-primary); font-weight: 600; }
        .ho-kpi-bad { color: var(--ho-error); font-weight: 600; }
        .ho-meta-list {
            display: grid;
            gap: 0.5rem;
        }
        .ho-meta-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.6rem 0;
            border-bottom: 1px solid #e4e9f2;
        }
        .ho-meta-row:last-child {
            border-bottom: none;
            padding-bottom: 0;
        }
        .ho-meta-key {
            color: var(--ho-on-variant);
            font-size: 0.92rem;
        }
        .ho-meta-value {
            color: var(--ho-on-surface);
            font-size: 0.95rem;
            font-weight: 600;
            text-align: right;
            width: 180px;
        }
        .ho-spec-card {
            background: var(--ho-surface-low);
            border: 1px solid #d6deea;
            border-radius: 0.85rem;
            padding: 0.95rem;
        }
        .ho-spec-card h4 {
            margin: 0 0 0.75rem 0;
            color: var(--ho-primary-container);
            font-size: 1rem;
            font-weight: 700;
        }
        .ho-spec-item {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.45rem;
            font-size: 0.9rem;
            color: var(--ho-on-variant);
        }
        .ho-spec-item:last-child { margin-bottom: 0; }
        .ho-section-spacer { height: 0.7rem; }
        .ho-chart-card {
            background: var(--ho-surface);
            border: 1px solid var(--ho-outline);
            border-radius: 1rem;
            padding: 1rem 1rem 0.35rem 1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05), 0 8px 20px rgba(15, 23, 42, 0.03);
        }
        .ho-subtle-text {
            color: var(--ho-on-variant);
            font-size: 0.93rem;
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _fmt_number(value: Any, digits: int = 1) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return "--"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _site_label(site_meta: dict[str, Any], site_id: str) -> str:
    source = site_meta.get("source_filename") if isinstance(site_meta, dict) else None
    if isinstance(source, str) and source:
        return source.split("\\")[-1].split("/")[-1]
    return site_id


def _get_completed_projects() -> tuple[list[str], dict[str, str]]:
    projects = api_client.get_projects(completed_only=True)
    if not isinstance(projects, list):
        return [], {}
    labels = {p["site_id"]: str(p.get("label") or p["site_id"]) for p in projects if p.get("site_id")}
    return list(labels.keys()), labels


def _analysis_period(stats: dict[str, Any]) -> str:
    start = stats.get("date_start")
    end = stats.get("date_end")
    if start and end:
        return f"{start} - {end}"
    return "--"


def _equipment_blocks(equipment: dict[str, Any] | None) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    chillers = []
    towers = []
    if not isinstance(equipment, dict):
        return chillers, towers

    for idx, ch in enumerate(equipment.get("chillers") or [], start=1):
        chillers.append(
            {
                "label": f"CH-{ch.get('id', idx)}",
                "value": f"{_fmt_number(ch.get('rt'), 0)} RT / COP {_fmt_number(ch.get('cop'), 1)}",
            }
        )

    tower_specs = equipment.get("cooling_tower") or equipment.get("cooling_towers") or []
    for idx, ct in enumerate(tower_specs, start=1):
        towers.append(
            {
                "label": f"CT-{ct.get('id', idx)}",
                "value": f"{_fmt_number(ct.get('kw'), 1)} kW",
            }
        )

    return chillers, towers


def _render_header(stats: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="ho-analysis-hero">
            <h1>既有數據分析</h1>
            <p>Historical performance evaluation and system diagnostics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if stats.get("days"):
        st.markdown(
            f'<p class="ho-subtle-text" style="margin:0.35rem 0 0 0;">資料涵蓋約 {stats["days"]} 天，以下圖表均以既有歷史資料自動彙整。</p>',
            unsafe_allow_html=True,
        )


def _render_context_panels(data: dict[str, Any]) -> None:
    stats = data.get("stats") or {}
    site_meta = data.get("site_meta") or {}
    equipment = data.get("equipment") or {}
    chillers, towers = _equipment_blocks(equipment)
    site_ids, label_by_id = _get_completed_projects()
    current_site = st.session_state.get("site_id", "site_default")
    if current_site not in site_ids and site_ids:
        current_site = site_ids[0]
        st.session_state.site_id = current_site

    left, right = st.columns([1.0, 1.8], gap="large")
    with left:
        with st.container(border=True):
            st.markdown(
                """
                <div class="ho-panel-title">
                    <span class="material-symbols-outlined">location_city</span>
                    <span>Site Information</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            chosen_site = current_site
            if site_ids:
                chosen_site = st.selectbox(
                    "Facility",
                    options=site_ids,
                    format_func=lambda sid: label_by_id.get(sid, sid),
                    index=site_ids.index(current_site),
                    key="dashboard_site_select",
                )
            else:
                st.text_input("Facility", value=_site_label(site_meta, current_site), disabled=True)

            st.markdown(
                f"""
                <div class="ho-meta-list">
                    <div class="ho-meta-row">
                        <div class="ho-meta-key">Analysis Period</div>
                        <div class="ho-meta-value">{_analysis_period(stats)}</div>
                    </div>
                    <div class="ho-meta-row">
                        <div class="ho-meta-key">Samples</div>
                        <div class="ho-meta-value">{_fmt_int((data.get('quality_report') or {}).get('final_rows'))}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if chosen_site != current_site:
                st.session_state.site_id = chosen_site
                st.rerun()

    with right:
        chiller_rows = "".join(
            f'<div class="ho-spec-item"><span>{item["label"]}</span><span>{item["value"]}</span></div>'
            for item in chillers
        ) or '<div class="ho-spec-item"><span>Configured chillers</span><span>Pending</span></div>'
        tower_rows = "".join(
            f'<div class="ho-spec-item"><span>{item["label"]}</span><span>{item["value"]}</span></div>'
            for item in towers
        ) or '<div class="ho-spec-item"><span>Cooling towers</span><span>Pending</span></div>'
        with st.container(border=True):
            st.markdown(
                """
                <div class="ho-panel-title">
                    <span class="material-symbols-outlined">dns</span>
                    <span>Equipment Specifications</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem;">
                    <div class="ho-spec-card">
                        <h4>Chiller Plant</h4>
                        {chiller_rows}
                    </div>
                    <div class="ho-spec-card">
                        <h4>Cooling Towers</h4>
                        {tower_rows}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_kpi_cards(data: dict[str, Any]) -> None:
    stats = data.get("stats") or {}
    power_metrics = ((data.get("ml_results") or {}).get("power_metrics") or {})
    cop_avg = _safe_float(stats.get("cop_avg"))
    cop_design_gap = None if cop_avg is None else cop_avg - 5.0
    total_mwh = None
    total_kwh = _safe_float(stats.get("total_kwh"))
    if total_kwh is not None:
        total_mwh = total_kwh / 1000.0
    cost_est = _safe_float(stats.get("cost_est_ntd"))
    holdout_mape = _safe_float(power_metrics.get("holdout_mape"))

    cards = [
        {
            "label": "Total Energy Consumption",
            "value": _fmt_number(total_mwh, 1),
            "unit": "MWh",
            "trend_class": "ho-kpi-good",
            "trend_text": (
                f"Model holdout MAPE {_fmt_number(holdout_mape * 100 if holdout_mape is not None else None, 1)}%"
                if holdout_mape is not None
                else "Historical total energy"
            ),
            "icon": "bolt",
            "icon_bg": "rgba(46,95,184,0.12)",
            "icon_fg": "#2a4f90",
        },
        {
            "label": "Operating Cost",
            "value": _fmt_int(cost_est),
            "unit": "NT$",
            "trend_class": "ho-kpi-good",
            "trend_text": "Estimated from current tariff",
            "icon": "attach_money",
            "icon_bg": "rgba(75,122,112,0.14)",
            "icon_fg": "#315c54",
        },
        {
            "label": "Average COP",
            "value": _fmt_number(cop_avg, 2),
            "unit": "",
            "trend_class": "ho-kpi-good" if (cop_design_gap is not None and cop_design_gap >= 0) else "ho-kpi-bad",
            "trend_text": (
                f"{'+' if cop_design_gap >= 0 else ''}{_fmt_number(cop_design_gap, 2)} vs design COP 5.0"
                if cop_design_gap is not None
                else "Insufficient data"
            ),
            "icon": "speed",
            "icon_bg": "rgba(178,133,78,0.20)",
            "icon_fg": "#7c5a33",
        },
    ]

    cols = st.columns(3, gap="large")
    for col, card in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="ho-kpi-card">
                    <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;">
                        <div>
                            <div class="ho-kpi-label">{card['label']}</div>
                            <p class="ho-kpi-value">{card['value']} <span class="ho-kpi-unit">{card['unit']}</span></p>
                        </div>
                        <div style="width:40px;height:40px;border-radius:999px;background:{card['icon_bg']};display:flex;align-items:center;justify-content:center;color:{card['icon_fg']};">
                            <span class="material-symbols-outlined">{card['icon']}</span>
                        </div>
                    </div>
                    <div class="ho-kpi-trend">
                        <span class="{card['trend_class']}">{card['trend_text']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _fig_energy_vs_baseline(data: dict[str, Any]) -> go.Figure:
    hourly = pd.DataFrame(data.get("series", {}).get("hourly") or [])
    if hourly.empty or "ts" not in hourly.columns or "total_kw" not in hourly.columns:
        fig = go.Figure()
        fig.add_annotation(text="No hourly energy data available", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(**layout(height=390))
        return fig

    hourly["ts"] = pd.to_datetime(hourly["ts"], errors="coerce")
    hourly = hourly.dropna(subset=["ts", "total_kw"]).sort_values("ts")
    if hourly.empty:
        fig = go.Figure()
        fig.add_annotation(text="No hourly energy data available", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(**layout(height=390))
        return fig

    hourly["hour"] = hourly["ts"].dt.hour
    baseline_map = hourly.groupby("hour")["total_kw"].mean()
    hourly["baseline_kw"] = hourly["hour"].map(baseline_map)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hourly["ts"],
            y=hourly["total_kw"],
            name="Actual load",
            mode="lines",
            line=dict(color=C["total"], width=2.4),
            fill="tozeroy",
            fillcolor="rgba(46,95,184,0.09)",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Actual %{y:.1f} kW<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hourly["ts"],
            y=hourly["baseline_kw"],
            name="Hourly baseline",
            mode="lines",
            line=dict(color="#8b6aa7", width=1.8, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Baseline %{y:.1f} kW<extra></extra>",
        )
    )
    fig.update_layout(
        **layout(
            title="Energy Consumption vs Baseline",
            height=390,
            xaxis=dict(title="", type="date", gridcolor="#dde5f0"),
            yaxis=dict(title="kW", gridcolor="#dde5f0"),
            legend=dict(orientation="h", y=1.06, x=1, xanchor="right"),
        )
    )
    return fig


def _fig_shap_importance(data: dict[str, Any]) -> go.Figure:
    ml_results = data.get("ml_results") or {}
    power_features = list(ml_results.get("power_top_features") or [])
    q_features = list(ml_results.get("q_demand_top_features") or [])
    features = power_features or q_features
    if not features:
        fig = go.Figure()
        fig.add_annotation(text="No feature importance available yet", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        fig.update_layout(**layout(height=390))
        return fig

    frame = pd.DataFrame(features[:6])
    if "feature" not in frame.columns:
        frame["feature"] = [f"Feature {idx+1}" for idx in range(len(frame))]
    value_col = "importance"
    if value_col not in frame.columns:
        value_col = "value" if "value" in frame.columns else frame.columns[-1]
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0)
    frame = frame.sort_values(value_col, ascending=True)

    fig = go.Figure(
        go.Bar(
            x=frame[value_col],
            y=frame["feature"],
            orientation="h",
            marker=dict(color="#355e9a"),
            text=[f"{v:.3f}" for v in frame[value_col]],
            textposition="outside",
            hovertemplate="%{y}<br>Importance %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        **layout(
            title="Feature Importance (SHAP / Importance)",
            height=390,
            xaxis=dict(title="Impact score", gridcolor="#dde5f0"),
            yaxis=dict(title="", automargin=True),
            margin=dict(t=56, b=40, l=20, r=20),
        )
    )
    return fig


def _render_charts(data: dict[str, Any]) -> None:
    left, right = st.columns([2.0, 1.1], gap="large")
    with left:
        with st.container(border=True):
            st.plotly_chart(
                _fig_energy_vs_baseline(data),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="dashboard_energy_vs_baseline",
            )
    with right:
        with st.container(border=True):
            st.plotly_chart(
                _fig_shap_importance(data),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="dashboard_feature_importance",
            )

    st.markdown('<div class="ho-section-spacer"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.plotly_chart(
            fig_heatmap(data),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="dashboard_summary_heatmap",
        )


def _render_model_summary(data: dict[str, Any]) -> None:
    ml_results = data.get("ml_results") or {}
    power = ml_results.get("power_metrics") or {}
    qd = ml_results.get("q_demand_metrics") or {}
    opt = data.get("optimization_results") or {}

    st.markdown("### 模型與診斷摘要")
    cols = st.columns(4, gap="large")
    cols[0].metric("Power holdout MAPE", f"{(_safe_float(power.get('holdout_mape')) or 0.0):.2%}")
    cols[1].metric("Q coverage", f"{_fmt_number(qd.get('coverage'), 2)}" if qd.get("coverage") is not None else "--")
    cols[2].metric("Peak load", f"{_fmt_number((data.get('stats') or {}).get('peak_kw'))} kW")
    q_feasible = ((opt.get("q_constraint") or {}).get("feasible")) if isinstance(opt, dict) else None
    cols[3].metric("Q safety", "Pass" if q_feasible else ("Pending" if q_feasible is None else "Check"))


def _render_deep_dive(data: dict[str, Any]) -> None:
    st.markdown("### 深入分析圖表")
    tabs = st.tabs(["System", "Chiller", "Pumps & Towers", "Environment"])
    with tabs[0]:
        st.plotly_chart(
            system_charts.fig_total_power(data),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="deep_system_total_power",
        )
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            st.plotly_chart(
                system_charts.fig_hourly_profile(data),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="deep_system_hourly_profile",
            )
        with c2:
            st.plotly_chart(
                system_charts.fig_power_donut(data),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="deep_system_power_donut",
            )
        c3, c4 = st.columns([1, 1], gap="large")
        with c3:
            st.plotly_chart(
                system_charts.fig_heatmap(data),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="deep_system_heatmap",
            )
        with c4:
            st.plotly_chart(
                system_charts.fig_load_duration(data),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="deep_system_load_duration",
            )
        st.plotly_chart(
            system_charts.fig_weekday_bar(data),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="deep_system_weekday_bar",
        )
    with tabs[1]:
        st.plotly_chart(
            chiller_charts.fig_chiller_power(data),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="deep_chiller_power",
        )
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.plotly_chart(
                chiller_charts.fig_chw_temperatures(data),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="deep_chiller_temperatures",
            )
        with c2:
            st.plotly_chart(
                chiller_charts.fig_chw_delta_histogram(data),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="deep_chiller_delta_histogram",
            )
        st.plotly_chart(
            chiller_charts.fig_cop_trend(data),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="deep_chiller_cop_trend",
        )
        st.plotly_chart(
            chiller_charts.fig_chiller_scatter(data),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="deep_chiller_scatter",
        )
    with tabs[2]:
        st.plotly_chart(
            pump_charts.fig_pump_power(data),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="deep_pump_power",
        )
        c3, c4 = st.columns([1, 1], gap="large")
        with c3:
            st.plotly_chart(
                pump_charts.fig_frequency_profiles(data),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="deep_pump_frequency_profiles",
            )
        with c4:
            st.plotly_chart(
                pump_charts.fig_freq_boxplots(data),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="deep_pump_freq_boxplots",
            )
        st.plotly_chart(
            pump_charts.fig_cw_temperatures(data),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="deep_pump_cw_temperatures",
        )
        st.plotly_chart(
            pump_charts.fig_ct_scatter(data),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="deep_pump_ct_scatter",
        )
    with tabs[3]:
        st.plotly_chart(
            environment_charts.fig_oa_dual_axis(data),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="deep_environment_oa_dual_axis",
        )
        c5, c6 = st.columns([1, 1], gap="large")
        with c5:
            st.plotly_chart(
                environment_charts.fig_oa_scatter(data),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="deep_environment_oa_scatter",
            )
        with c6:
            st.plotly_chart(
                environment_charts.fig_rh_scatter(data),
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="deep_environment_rh_scatter",
            )
        st.plotly_chart(
            environment_charts.fig_monthly_overview(data),
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="deep_environment_monthly_overview",
        )


def render() -> None:
    _inject_styles()
    site_id = st.session_state.get("site_id", "site_default")
    data = _load(site_id)
    if "error" in data:
        st.error(data["error"])
        return

    _render_header(data.get("stats") or {})
    st.markdown('<div class="ho-section-spacer"></div>', unsafe_allow_html=True)
    _render_context_panels(data)
    st.markdown('<div class="ho-section-spacer"></div>', unsafe_allow_html=True)
    _render_kpi_cards(data)
    st.markdown('<div class="ho-section-spacer"></div>', unsafe_allow_html=True)
    _render_charts(data)
    st.markdown('<div class="ho-section-spacer"></div>', unsafe_allow_html=True)
    _render_model_summary(data)
    st.markdown('<div class="ho-section-spacer"></div>', unsafe_allow_html=True)
    _render_deep_dive(data)

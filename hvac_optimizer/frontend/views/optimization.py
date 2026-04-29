from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from hvac_optimizer.frontend import api_client

ELECTRICITY_RATE_NTD_PER_KWH = 4.0
GRID_CONTROL_COMBINATIONS = 2860
SECONDS_PER_EVAL_ESTIMATE = 0.00012
FAST_MODE_EFFECTIVE_COMBINATIONS = 96


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .ho-opt-hero h1 {
            margin: 0;
            font-size: clamp(1.8rem, 3vw, 2.25rem);
            letter-spacing: -0.03em;
            color: var(--ho-on-surface);
            font-weight: 700;
        }
        .ho-opt-hero p {
            margin: 0.35rem 0 0 0;
            color: var(--ho-on-variant);
            font-size: 0.97rem;
        }
        .ho-opt-card {
            background: var(--ho-surface);
            border: 1px solid var(--ho-outline);
            border-radius: 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05), 0 10px 24px rgba(15, 23, 42, 0.04);
            padding: 1.1rem 1.2rem;
        }
        .ho-opt-title {
            margin: 0 0 0.7rem 0;
            font-size: 1rem;
            font-weight: 650;
            color: var(--ho-on-surface);
        }
        .ho-opt-muted {
            color: var(--ho-on-variant);
            font-size: 0.88rem;
            line-height: 1.45;
        }
        .ho-opt-chart-title {
            margin: 0 0 0.35rem 0;
            font-size: 0.98rem;
            font-weight: 650;
            color: var(--ho-on-surface);
        }
        .ho-opt-chart-subtitle {
            margin: 0 0 0.7rem 0;
            color: var(--ho-on-variant);
            font-size: 0.84rem;
            line-height: 1.45;
        }
        div[data-testid="stTabs"] button[role="tab"] p,
        div[data-testid="stTabs"] button[role="tab"] {
            color: var(--ho-on-surface) !important;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] p,
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--ho-primary-container) !important;
        }
        div[data-testid="stMetricLabel"] *,
        div[data-testid="stMetricValue"] *,
        div[data-testid="stMetricDelta"] *,
        div[data-testid="stCaptionContainer"] *,
        div[data-testid="stMarkdownContainer"] *,
        div[data-testid="stText"] *,
        div[data-testid="stAlertContentInfo"] *,
        div[data-testid="stAlertContentWarning"] *,
        div[data-testid="stAlertContentError"] * {
            color: var(--ho-on-surface) !important;
        }
        div[data-testid="stJson"] *,
        div[data-testid="stCodeBlock"] * {
            color: #dbe7ff;
        }
        div[data-testid="stJson"] pre,
        div[data-testid="stCodeBlock"] pre {
            background: #111827 !important;
            border: 1px solid #1f2937;
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _grid_value_count(lo: float, hi: float, step: float) -> int:
    lo_f = float(min(lo, hi))
    hi_f = float(max(lo, hi))
    values = []
    current = lo_f
    while current <= hi_f + 1e-9:
        values.append(round(current, 4))
        current += step
    if not values or values[-1] < hi_f - 1e-9:
        values.append(round(hi_f, 4))
    return len(values)


def _estimate_control_combinations(bounds: dict[str, tuple[float, float]]) -> int:
    step_map = {
        "chws": 1.0,
        "chwp": 2.5,
        "cwp": 2.5,
        "ct_fan": 2.5,
    }
    total = 1
    for key, (lo, hi) in bounds.items():
        step = step_map.get(key)
        if step is None:
            continue
        total *= _grid_value_count(lo, hi, step)
    return max(total, 1)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_energy_summary(result: dict) -> dict[str, float]:
    interval_summary = result.get("interval_summary", {}) or {}
    baseline_kwh = max(_safe_float(interval_summary.get("baseline_kwh")), 0.0)
    optimized_kwh = max(_safe_float(interval_summary.get("optimized_kwh")), 0.0)
    saving_kwh = max(_safe_float(interval_summary.get("saving_kwh")), max(baseline_kwh - optimized_kwh, 0.0))
    saving_pct = _safe_float(interval_summary.get("saving_pct"), (saving_kwh / baseline_kwh * 100.0) if baseline_kwh > 0 else 0.0)
    saving_cost = _safe_float(interval_summary.get("saving_cost_ntd"), saving_kwh * ELECTRICITY_RATE_NTD_PER_KWH)
    return {
        "baseline_kwh": baseline_kwh,
        "optimized_kwh": optimized_kwh,
        "saving_kwh": saving_kwh,
        "saving_pct": saving_pct,
        "saving_cost": saving_cost,
    }


def _build_q_summary(result: dict) -> dict[str, float | bool | None]:
    interval_summary = result.get("interval_summary", {}) or {}
    q_constraint = result.get("q_constraint", {}) or {}
    q_required_min = _optional_float(q_constraint.get("q_required_min"))
    q_capability = _optional_float(q_constraint.get("q_capability"))
    avg_gap = _safe_float(interval_summary.get("avg_q_gap_pct"), 0.0)
    feasible_ratio = _safe_float(interval_summary.get("feasible_ratio"), 0.0)
    if q_required_min is None or q_required_min <= 0 or q_capability is None:
        return {
            "q_required_min": q_required_min,
            "q_capability": q_capability,
            "q_gap_pct": avg_gap,
            "feasible": feasible_ratio >= 0.95 if feasible_ratio > 0 else bool(q_constraint.get("feasible")),
            "feasible_ratio": feasible_ratio,
        }
    return {
        "q_required_min": q_required_min,
        "q_capability": q_capability,
        "q_gap_pct": avg_gap,
        "feasible": feasible_ratio >= 0.95 if feasible_ratio > 0 else bool(q_constraint.get("feasible")),
        "feasible_ratio": feasible_ratio,
    }


def _metric_row(result: dict) -> None:
    energy = _build_energy_summary(result)
    q_summary = _build_q_summary(result)
    interval_summary = result.get("interval_summary", {}) or {}
    no_feasible_points = int(_safe_float(interval_summary.get("no_feasible_points"), 0.0))
    cols = st.columns(4)
    cols[0].metric("節能率", f"{energy['saving_pct']:,.1f}%")
    cols[1].metric("總節能度數", f"{energy['saving_kwh']:,.1f}")
    cols[2].metric("估算節費", f"NT$ {energy['saving_cost']:,.0f}")
    cols[3].metric("舒適度(Q差異率)", f"{q_summary['q_gap_pct']:,.1f}%", delta=f"安全覆蓋 {q_summary['feasible_ratio'] * 100:,.1f}%")
    st.caption(
        f"四張卡片都是以整段 CSV 區間加總或平均後得到的結果；目前採每 1 小時逐點評估、"
        f"每點測試 {int(interval_summary.get('control_combinations', 0)):,} 組粗網格控制組合；"
        "節費以 4 元/度計算，Q差異率為整段區間的平均冷量裕度。"
    )
    if no_feasible_points > 0:
        st.warning(
            f"有 {no_feasible_points} 個時間點在目前控制範圍內找不到滿足 Q 的可行解；"
            "這些時間點已鎖回 baseline，不再當成節能結果。"
        )


def _estimate_hourly_points() -> int | None:
    stage1 = st.session_state.get("stage1_results") or {}
    quality = (stage1.get("quality_report") if isinstance(stage1, dict) else None) or {}
    final_rows = quality.get("final_rows")
    try:
        rows = int(final_rows)
    except (TypeError, ValueError):
        return None
    if rows <= 0:
        return None
    return max(1, round(rows / 4))


def _has_interval_results(result: dict) -> bool:
    return bool((result.get("interval_summary") or {})) and bool(result.get("interval_curve"))


def _load_active_optimization_result(site_id: str) -> dict | None:
    analytics = api_client.get_analytics(site_id)
    if not isinstance(analytics, dict) or analytics.get("error"):
        return None
    result = analytics.get("optimization_results")
    return result if isinstance(result, dict) else None


def _estimate_runtime_seconds(estimated_points: int | None, control_combinations: int, optimization_mode: str) -> float:
    if not estimated_points:
        return 60.0 if optimization_mode == "standard" else 45.0 if optimization_mode == "fast" else 30.0
    if optimization_mode == "standard":
        effective_combinations = control_combinations
    elif optimization_mode == "fast":
        effective_combinations = min(control_combinations, FAST_MODE_EFFECTIVE_COMBINATIONS)
    else:
        effective_combinations = min(control_combinations, 16)
    estimated_evals = estimated_points * effective_combinations
    return max(30.0, min(900.0, estimated_evals * SECONDS_PER_EVAL_ESTIMATE))


def _run_optimization_with_progress(site_id: str, bounds: dict, estimated_points: int | None, control_combinations: int, optimization_mode: str) -> dict:
    estimated_seconds = _estimate_runtime_seconds(estimated_points, control_combinations, optimization_mode)
    progress = st.progress(0, text="準備開始區間優化...")
    detail = st.empty()
    last_real_progress = 0
    last_status_text = "準備開始區間優化..."

    stage_messages = [
        (0.08, "正在載入清洗後資料與訓練模型..."),
        (0.22, "正在建立每小時取樣點與控制組合網格..."),
        (0.55, "正在逐點評估控制組合，計算整段 CSV 區間效益..."),
        (0.82, "正在整理節能摘要、Q 裕度與未來趨勢..."),
        (0.95, "正在封存結果並準備回傳圖表資料..."),
    ]

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(api_client.run_optimization, site_id, bounds)
        started_at = time.monotonic()

        while not future.done():
            status = api_client.get_optimization_status(site_id)
            if "error" not in status and isinstance(status, dict) and status:
                progress_value = min(99, max(last_real_progress, int(status.get("percent", 1) or 1)))
                stage_text = str(status.get("message") or "正在執行區間優化...")
                last_real_progress = progress_value
                last_status_text = stage_text
                completed_points = status.get("completed_points")
                total_points = status.get("total_points")
                current_timestamp = status.get("current_timestamp")
                if completed_points is not None and total_points:
                    detail_text = f"真實進度 {progress_value}%：已完成 {int(completed_points):,} / {int(total_points):,} 個每小時時間點。"
                    if current_timestamp:
                        detail_text += f" 目前時間點：{current_timestamp}"
                    detail.caption(detail_text)
                else:
                    detail.caption(f"真實進度 {progress_value}%：{stage_text}")
                progress.progress(progress_value, text=stage_text)
            else:
                elapsed = time.monotonic() - started_at
                ratio = min(elapsed / estimated_seconds, 1.0)
                # When live backend status is unavailable, keep the estimate conservative:
                # move smoothly toward 78% first, then creep toward 88% without ever jumping to 95%.
                eased = 1.0 - pow(2.718281828, -3.2 * ratio)
                estimated_cap = 78 if last_real_progress == 0 else 88
                estimated_progress = int(eased * estimated_cap)
                progress_value = max(last_real_progress, min(estimated_cap, max(3, estimated_progress)))

                stage_text = stage_messages[-1][1]
                for threshold, message in stage_messages:
                    if ratio <= threshold:
                        stage_text = message
                        break

                if estimated_points is not None:
                    approx_done = min(
                        estimated_points,
                        int(round(estimated_points * min(progress_value / max(estimated_cap, 1), 1.0))),
                    )
                    detail.caption(
                        f"目前無法取得後端即時進度，以下為估算值：{progress_value}% ，"
                        f"約完成 {approx_done:,} / {estimated_points:,} 個每小時時間點。"
                    )
                else:
                    detail.caption(f"目前無法取得後端即時進度，以下為估算值：{progress_value}% 。")

                progress.progress(progress_value, text=f"{stage_text}（估算）")
            time.sleep(0.2)

        result = future.result()

    if isinstance(result, dict) and "error" in result:
        progress.progress(100, text="後端回傳錯誤，請查看訊息。")
        detail.caption("這次最佳化沒有成功完成。")
    else:
        progress.progress(100, text="區間優化完成，正在整理結果...")
        detail.caption("已收到後端結果。")
    return result


def _build_power_trend_figure(result: dict) -> go.Figure:
    curve = result.get("interval_curve") or []
    if not curve:
        return go.Figure()
    df = pd.DataFrame(curve)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notnull()].sort_values("timestamp")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["baseline_power_kw"],
            name="原始能耗",
            mode="lines",
            line=dict(color="#c0cad8", width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["optimized_power_kw"],
            name="優化後能耗",
            mode="lines+markers",
            line=dict(color="#2f68f5", width=2.6),
            marker=dict(size=4),
        )
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title="kW", gridcolor="#e7ecf4", zeroline=False)
    fig.update_xaxes(title=None, gridcolor="#e7ecf4", zeroline=False)
    return fig


def _build_equipment_savings_figure(result: dict) -> go.Figure:
    data = result.get("equipment_savings_kwh") or {}
    labels = ["主機", "冰水泵", "冷卻泵", "冷卻塔"]
    keys = ["ch_kw", "chwp_kw", "cwp_kw", "ct_kw"]
    values = [_safe_float(data.get(key)) for key in keys]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color=["#2f68f5", "#4c7df0", "#6c96f5", "#8ab0ff"]),
            text=[f"{v:.1f} kWh" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title="kWh", gridcolor="#e7ecf4", zeroline=False)
    return fig


def _load_mpc_log(result: dict) -> pd.DataFrame | None:
    curve = (result.get("future_projection") or {}).get("curve") or []
    if not curve:
        return None
    df = pd.DataFrame(curve)
    if "timestamp" not in df.columns:
        return None
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notnull()].sort_values("timestamp")
    return df if not df.empty else None


def _build_q_safety_figure(result: dict) -> go.Figure | None:
    curve = result.get("interval_curve") or []
    if not curve:
        return go.Figure()
    mpc_df = pd.DataFrame(curve)
    mpc_df["timestamp"] = pd.to_datetime(mpc_df["timestamp"], errors="coerce")
    mpc_df = mpc_df[mpc_df["timestamp"].notnull()].sort_values("timestamp")

    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Q 裕度",),
    )
    q_margin = (mpc_df["q_capability"] - mpc_df["q_required_min"]).fillna(0.0)
    fig.add_trace(
        go.Scatter(
            x=mpc_df["timestamp"],
            y=mpc_df["q_required_min"],
            name="需求下限",
            mode="lines",
            line=dict(color="#ef8b2c", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=mpc_df["timestamp"],
            y=mpc_df["q_capability"],
            name="Q 能力",
            mode="lines+markers",
            line=dict(color="#1f8a5b", width=2.2),
            marker=dict(size=5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=mpc_df["timestamp"],
            y=q_margin,
            name="Q 裕度",
            marker=dict(color=["#1f8a5b" if v >= 0 else "#d64545" for v in q_margin]),
            opacity=0.28,
            hovertemplate="時間: %{x}<br>Q 裕度: %{y:.1f} kW<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=30, b=8),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(gridcolor="#e7ecf4", zeroline=False, row=1, col=1)
    fig.update_xaxes(gridcolor="#e7ecf4", zeroline=False, row=1, col=1)
    return fig


def _build_future_projection_figure(result: dict) -> go.Figure:
    future_df = _load_mpc_log(result)
    if future_df is None:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=future_df["timestamp"],
            y=future_df["baseline_power_kw"],
            name="未來基準",
            mode="lines",
            line=dict(color="#c0cad8", width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_df["timestamp"],
            y=future_df["optimized_power_kw"],
            name="MPC 預測後",
            mode="lines+markers",
            line=dict(color="#2f68f5", width=2.6),
            marker=dict(size=4),
        )
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title="kW", gridcolor="#e7ecf4", zeroline=False)
    fig.update_xaxes(title=None, gridcolor="#e7ecf4", zeroline=False)
    return fig


def render() -> None:
    _inject_styles()
    site_id = st.session_state.get("site_id", "site_default")
    last_site_id = st.session_state.get("opt_result_site_id")
    if last_site_id != site_id:
        st.session_state.pop("opt_results", None)
        st.session_state["opt_result_site_id"] = site_id

    result = st.session_state.get("opt_results")
    if not result:
        loaded = _load_active_optimization_result(site_id)
        if loaded:
            st.session_state["opt_results"] = loaded
            result = loaded

    st.markdown(
        """
        <div class="ho-opt-hero">
            <h1>優化分析（Advisory + MPC）</h1>
            <p>調整操作範圍後，系統會先找最佳控制點，再回放控制過程，最後輸出報表。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    opt_history = api_client.get_optimization_history(site_id)
    if isinstance(opt_history, dict) and not opt_history.get("error"):
        items = list(opt_history.get("items") or [])
        if items:
            label_by_id = {str(item["optimization_id"]): str(item.get("label") or item["optimization_id"]) for item in items}
            optimization_ids = [str(item["optimization_id"]) for item in items]
            active_optimization_id = str(opt_history.get("active_optimization_id") or optimization_ids[0])
            if active_optimization_id not in optimization_ids:
                active_optimization_id = optimization_ids[0]
            default_idx = optimization_ids.index(active_optimization_id)
            chosen_optimization_id = st.selectbox(
                "已封存優化分析",
                options=optimization_ids,
                index=default_idx,
                format_func=lambda oid, _m=label_by_id: _m.get(oid, oid),
                help="選取已封存的優化分析結果，直接載入控制範圍、分析模式與圖表。",
            )
            selected_item = next((item for item in items if str(item["optimization_id"]) == chosen_optimization_id), None)
            if selected_item:
                bounds_used = selected_item.get("bounds_used") or {}
                mode_label_saved = {"standard": "標準版", "fast": "加速版", "extreme": "極速版"}.get(str(selected_item.get("optimization_mode") or "standard"), str(selected_item.get("optimization_mode") or "standard"))
                st.caption(
                    f"已保存設定：模式 {mode_label_saved}；"
                    f"CHWS {bounds_used.get('chws')} / CHWP {bounds_used.get('chwp')} / "
                    f"CWP {bounds_used.get('cwp')} / CT fan {bounds_used.get('ct_fan')}"
                )
            if chosen_optimization_id != active_optimization_id:
                res = api_client.activate_optimization_history(site_id, chosen_optimization_id)
                if isinstance(res, dict) and not res.get("error"):
                    loaded = _load_active_optimization_result(site_id)
                    if loaded:
                        st.session_state["opt_results"] = loaded
                    st.rerun()
                else:
                    st.error(str((res or {}).get("error") or "無法切換優化分析結果"))

    saved_mode = str((result or {}).get("optimization_mode") or "fast")
    saved_mode_label = {"standard": "標準版", "fast": "加速版", "extreme": "極速版"}.get(saved_mode, "加速版")
    mode_label = st.segmented_control(
        "分析模式",
        options=["標準版", "加速版", "極速版"],
        default=saved_mode_label,
        help="標準版會逐點測完整粗網格；加速版會先做全域快篩；極速版會再縮小候選並跳過未來預測/MPC。",
    )
    optimization_mode = {"標準版": "standard", "加速版": "fast", "極速版": "extreme"}.get(mode_label, "fast")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown('<p class="ho-opt-title">冰水側控制範圍</p>', unsafe_allow_html=True)
            chws = st.slider("CHWS (C)", min_value=5.0, max_value=12.0, value=(6.0, 9.0), step=0.1)
            chwp = st.slider("CHWP (Hz)", min_value=20, max_value=60, value=(30, 55))
    with c2:
        with st.container(border=True):
            st.markdown('<p class="ho-opt-title">冷卻側控制範圍</p>', unsafe_allow_html=True)
            ct_fan = st.slider("CT fan (Hz)", min_value=20, max_value=60, value=(25, 55))
            cwp = st.slider("CWP (Hz)", min_value=20, max_value=60, value=(30, 55))

    current_bounds = {
        "chws": tuple(chws),
        "chwp": tuple(chwp),
        "cwp": tuple(cwp),
        "ct_fan": tuple(ct_fan),
    }
    control_combinations = _estimate_control_combinations(current_bounds)

    estimated_points = _estimate_hourly_points()
    if estimated_points is not None:
        estimated_evals = estimated_points * control_combinations
        if optimization_mode == "fast":
            st.info(
                f"加速版會先對 {control_combinations:,} 組控制組合做全域快篩，再只保留少量候選組合跑整段分析。"
                f"目前仍採每 1 小時取一點，預估約 {estimated_points:,} 個時間點。"
            )
        elif optimization_mode == "extreme":
            st.info(
                f"極速版會先對 {control_combinations:,} 組控制組合做更激進的快篩，只保留極少候選組合，"
                f"並跳過未來預測 / MPC。仍採每 1 小時取一點，預估約 {estimated_points:,} 個時間點。"
            )
        else:
            st.info(
                f"這次會用每 1 小時取一點的方式分析，預估約 {estimated_points:,} 個時間點，"
                f"每個時間點測試 {control_combinations:,} 組控制組合，合計約 {estimated_evals:,} 次模型評估。"
            )
    else:
        if optimization_mode == "fast":
            st.info(f"加速版會先對 {control_combinations:,} 組控制組合做全域快篩，再只保留少量候選組合跑整段分析。")
        elif optimization_mode == "extreme":
            st.info(f"極速版會先對 {control_combinations:,} 組控制組合做更激進的快篩，只保留極少候選組合，並跳過未來預測 / MPC。")
        else:
            st.info(f"這次會用每 1 小時取一點的方式分析，每個時間點測試 {control_combinations:,} 組控制組合。")

    if st.button("Run optimization", type="primary", use_container_width=True):
        bounds = {
            "chws": list(chws),
            "chwp": list(chwp),
            "cwp": list(cwp),
            "ct_fan": list(ct_fan),
            "optimization_mode": optimization_mode,
        }
        res = _run_optimization_with_progress(site_id, bounds, estimated_points, control_combinations, optimization_mode)
        if "error" in res:
            st.error(res["error"])
        elif not _has_interval_results(res):
            st.error("後端沒有回傳新版區間分析結果。請先重啟 backend，再重新執行一次最佳化。")
        else:
            st.session_state.opt_results = res
            st.session_state.opt_result_site_id = site_id
            st.rerun()

    if not result:
        st.info("請先執行一次最佳化，系統會產出建議設定、MPC 結果與報表資訊。")
        return

    if not _has_interval_results(result):
        st.warning("目前這份最佳化結果仍是舊格式，缺少整段 CSV 區間分析欄位。請重啟 backend 後重新執行最佳化。")
        return

    _metric_row(result)
    st.divider()

    st.markdown("#### 改善前後圖表")
    st.markdown(
        '<p class="ho-opt-muted">這裡改成整段 CSV 區間的結果：左圖是原始能耗和優化後能耗曲線，右圖是各設備節能效益；下方再看整段期間的 Q 裕度，以及未來一倍時長的 MPC 預測。</p>',
        unsafe_allow_html=True,
    )
    if not (result.get("interval_curve") or []):
        st.info("找不到區間優化資料，重新執行一次最佳化後，這裡會顯示整段 CSV 區間的分析圖。")
    else:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            with st.container(border=True):
                st.markdown('<p class="ho-opt-chart-title">原始能耗 vs 優化後能耗</p>', unsafe_allow_html=True)
                st.markdown('<p class="ho-opt-chart-subtitle">以整段 CSV 每個時間點跑優化後，得到的原始能耗曲線與優化後能耗曲線。</p>', unsafe_allow_html=True)
                st.plotly_chart(_build_power_trend_figure(result), use_container_width=True, config={"displayModeBar": False})
        with c2:
            with st.container(border=True):
                st.markdown('<p class="ho-opt-chart-title">各設備節能效益</p>', unsafe_allow_html=True)
                st.markdown('<p class="ho-opt-chart-subtitle">依整段區間累積的節能結果，估算各設備對總節能的貢獻。</p>', unsafe_allow_html=True)
                st.plotly_chart(_build_equipment_savings_figure(result), use_container_width=True, config={"displayModeBar": False})

        c3, c4 = st.columns(2, gap="large")
        with c3:
            with st.container(border=True):
                st.markdown('<p class="ho-opt-chart-title">整段區間的 Q 裕度</p>', unsafe_allow_html=True)
                st.markdown('<p class="ho-opt-chart-subtitle">橘線是需求下限，綠線是可提供的冷量；柱狀是每個時間點的冷量差距。</p>', unsafe_allow_html=True)
                st.plotly_chart(_build_q_safety_figure(result), use_container_width=True, config={"displayModeBar": False})
        with c4:
            future_curve = (result.get("future_projection") or {}).get("curve") or []
            with st.container(border=True):
                st.markdown('<p class="ho-opt-chart-title">未來一倍時長的 MPC 預測</p>', unsafe_allow_html=True)
                st.markdown('<p class="ho-opt-chart-subtitle">以目前資料區間長度為基準，外推接下來同等時長的趨勢效益。</p>', unsafe_allow_html=True)
                if future_curve:
                    st.plotly_chart(_build_future_projection_figure(result), use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("目前沒有未來預測曲線。")

    tab1, tab2, tab3 = st.tabs(["最佳控制點", "控制回放", "報表與產出"])

    with tab1:
        baseline = result.get("baseline", {})
        optimal = result.get("optimal_params", {})
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("#### 建議設定值")
            st.json(optimal)
        with c2:
            st.markdown("#### Baseline")
            st.json(baseline)
        st.markdown("#### Q 安全檢核")
        st.json(result.get("q_constraint", {}))

    with tab2:
        st.markdown("#### 控制回放摘要")
        st.json(result.get("mpc", {}))

    with tab3:
        st.markdown("#### 產出檔案")
        st.json(result.get("artifacts", {}))

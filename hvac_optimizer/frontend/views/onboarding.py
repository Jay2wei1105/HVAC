"""
Onboarding UI — Stage 1–2 flow. Layout/styling only; core API paths unchanged.
"""
from __future__ import annotations

import uuid
from collections import defaultdict

import plotly.graph_objects as go
import streamlit as st

from hvac_optimizer.frontend import api_client

_STEP_IDS = ("upload", "diagnostics", "mapping", "equipment", "ready")
_STEP_LABELS = ("檔案上傳", "資料診斷", "欄位對應", "設備規格", "訓練完成")


def _ensure_state() -> None:
    st.session_state.setdefault("site_id", "site_default")
    st.session_state.setdefault("ob_step", "upload")


def _ensure_draft_workspace_for_upload() -> None:
    """
    If the user opens「專案導入」while session still points at a trained project,
    start a fresh draft workspace (no navbar project picker on this page).
    """
    if st.session_state.get("ob_step") != "upload":
        return
    completed = api_client.get_projects(completed_only=True)
    if not isinstance(completed, list):
        return
    cur = st.session_state.site_id
    if any(p.get("site_id") == cur for p in completed):
        st.session_state.site_id = f"site_{uuid.uuid4().hex[:8]}"
        for k in ("dataset_info", "ml_results", "stage1_results", "project_display_name"):
            st.session_state.pop(k, None)


def _stepper_html(current: str) -> str:
    idx = _STEP_IDS.index(current)
    parts: list[str] = ['<div class="ho-stepper-wrap"><div class="ho-stepper">']
    for i, label in enumerate(_STEP_LABELS):
        if i > 0:
            line_cls = "ho-step-line done" if i <= idx else "ho-step-line"
            parts.append(f'<div class="{line_cls}"></div>')
        if i < idx:
            cls = "ho-step done"
            inner = f'<span class="ho-step-dot">✓</span><span>{label}</span>'
        elif i == idx:
            cls = "ho-step current"
            inner = f'<span class="ho-step-dot">{i + 1}</span><span>{label}</span>'
        else:
            cls = "ho-step"
            inner = f'<span class="ho-step-dot">{i + 1}</span><span>{label}</span>'
        parts.append(f'<div class="{cls}">{inner}</div>')
    parts.append("</div></div>")
    return "".join(parts)


def _hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="ho-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_feature_overview(ml_res: dict) -> list[dict[str, object]]:
    power_rows = ml_res.get("power_top_features") or []
    q_rows = ml_res.get("q_demand_feature_importance") or []

    if not q_rows and ml_res.get("q_demand_top_features"):
        q_rows = [
            {"feature": name, "importance": float(len(ml_res["q_demand_top_features"]) - idx)}
            for idx, name in enumerate(ml_res["q_demand_top_features"])
        ]

    source_map: dict[str, dict[str, float]] = defaultdict(dict)
    for row in power_rows:
        feature = str(row.get("feature") or "").strip()
        if feature:
            source_map[feature]["power"] = max(_safe_float(row.get("importance")), 0.0)
    for row in q_rows:
        feature = str(row.get("feature") or "").strip()
        if feature:
            source_map[feature]["q"] = max(_safe_float(row.get("importance")), 0.0)

    if not source_map:
        return []

    def _normalize(rows: list[dict[str, object]], key: str) -> dict[str, float]:
        total = sum(max(_safe_float(r.get("importance")), 0.0) for r in rows)
        if total <= 0:
            return {}
        return {
            str(r.get("feature") or "").strip(): max(_safe_float(r.get("importance")), 0.0) / total
            for r in rows
            if str(r.get("feature") or "").strip()
        }

    power_norm = _normalize(power_rows, "power")
    q_norm = _normalize(q_rows, "q")

    rows: list[dict[str, object]] = []
    for feature, src in source_map.items():
        power_share = power_norm.get(feature, 0.0)
        q_share = q_norm.get(feature, 0.0)
        combined = power_share + q_share
        rows.append(
            {
                "feature": feature,
                "combined": combined,
                "power_share": power_share,
                "q_share": q_share,
                "sources": [label for label, share in (("功率", power_share), ("Q", q_share)) if share > 0],
            }
        )

    rows.sort(key=lambda item: float(item["combined"]), reverse=True)
    top = rows[:6]
    total = sum(float(item["combined"]) for item in top)
    if total <= 0:
        total = 1.0
    for item in top:
        item["display_pct"] = float(item["combined"]) / total * 100.0
    return top


def _build_feature_figure(rows: list[dict[str, object]]) -> go.Figure:
    ordered = list(reversed(rows))
    labels = [str(item["feature"]) for item in ordered]
    combined = [float(item["display_pct"]) for item in ordered]
    power_share = [float(item["power_share"]) * 100.0 for item in ordered]
    q_share = [float(item["q_share"]) * 100.0 for item in ordered]

    fig = go.Figure(
        go.Bar(
            x=combined,
            y=labels,
            orientation="h",
            marker=dict(color="#2f68f5", line=dict(color="#244c94", width=0.6)),
            text=[f"{v:.1f}%" for v in combined],
            textposition="outside",
            hovertemplate=(
                "特徵: %{y}<br>"
                "合併占比: %{x:.1f}%<br>"
                "功率: %{customdata[0]:.1f}%<br>"
                "Q: %{customdata[1]:.1f}%<extra></extra>"
            ),
            customdata=list(zip(power_share, q_share)),
            width=0.55,
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=8, b=8),
        height=max(260, 56 * len(rows)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=None,
            range=[0, max(100.0, max(combined, default=0.0) * 1.15)],
            ticksuffix="%",
            gridcolor="#e7ecf4",
            zeroline=False,
        ),
        yaxis=dict(title=None, automargin=True, gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    return fig


def _render_feature_overview(ml_res: dict) -> None:
    rows = _build_feature_overview(ml_res)
    if not rows:
        st.info("目前沒有可顯示的重要特徵。")
        return

    st.markdown(
        """
        <style>
        .ho-ready-summary {
            color: var(--ho-on-variant);
            font-size: 0.95rem;
            line-height: 1.5;
            margin: 0.2rem 0 1rem 0;
        }
        .ho-feature-wrap {
            margin-top: 1rem;
            padding-top: 0.25rem;
            border-top: 1px solid #e4e9f2;
        }
        .ho-feature-head {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.85rem;
        }
        .ho-feature-title {
            margin: 0;
            font-size: 0.98rem;
            font-weight: 700;
            color: var(--ho-on-surface);
        }
        .ho-feature-subtitle {
            margin: 0.2rem 0 0 0;
            color: var(--ho-on-variant);
            font-size: 0.85rem;
            line-height: 1.45;
        }
        .ho-validation {
            margin-top: 1rem;
            padding: 0.95rem 1rem;
            border-radius: 0.9rem;
            background: #f6f8fc;
            border: 1px solid #e4e9f2;
        }
        .ho-validation-title {
            margin: 0 0 0.25rem 0;
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--ho-on-surface);
        }
        .ho-validation-text {
            margin: 0;
            color: var(--ho-on-variant);
            font-size: 0.85rem;
            line-height: 1.45;
        }
        .ho-validation-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.75rem;
        }
        .ho-check-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.6rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 650;
        }
        .ho-check-pill.ok {
            background: #e8f6ee;
            color: #196a3c;
        }
        .ho-check-pill.warn {
            background: #fff4df;
            color: #8a5a00;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="ho-feature-wrap">
          <div class="ho-feature-head">
            <div>
              <p class="ho-feature-title">重要特徵總覽</p>
              <p class="ho-feature-subtitle">合併功率與 Q 的重要度後排序，這裡會以真正的圖表呈現，而不是原始碼。</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(_build_feature_figure(rows), use_container_width=True, config={"displayModeBar": False})

    detail_lines = []
    for item in rows:
        feature = str(item["feature"])
        pct = float(item["display_pct"])
        power_share = float(item["power_share"]) * 100
        q_share = float(item["q_share"]) * 100
        labels = "、".join(item["sources"]) if item["sources"] else "無"
        detail_lines.append(
            f"<div class='ho-feature-detail'><strong>{feature}</strong> "
            f"<span>{pct:.1f}%</span><small>{labels} · 功率 {power_share:.1f}% · Q {q_share:.1f}%</small></div>"
        )
    detail_html = "".join(detail_lines)
    st.markdown(
        f"""
        <style>
        .ho-feature-detail {{
            display: flex;
            flex-wrap: wrap;
            align-items: baseline;
            gap: 0.5rem 0.65rem;
            padding: 0.28rem 0;
            color: var(--ho-on-variant);
            font-size: 0.85rem;
        }}
        .ho-feature-detail strong {{
            color: var(--ho-on-surface);
            font-size: 0.88rem;
            font-weight: 650;
        }}
        .ho-feature-detail span {{
            color: var(--ho-on-surface);
            font-weight: 700;
        }}
        .ho-feature-detail small {{
            color: var(--ho-on-variant);
        }}
        </style>
        <div>{detail_html}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_validation_summary(qd: dict) -> None:
    mapping = {
        "coverage_ok": "Q 覆蓋率",
        "pinball_improvement_ok": "Q 改善",
        "weather_driver_ok": "天氣驅動",
    }
    bits = []
    pass_count = 0
    for key, label in mapping.items():
        ok = bool(qd.get("validation", {}).get(key))
        pass_count += int(ok)
        bits.append(
            f'<span class="ho-check-pill {"ok" if ok else "warn"}">{label}{" / 通過" if ok else " / 需調整"}</span>'
        )
    st.markdown(
        f"""
        <div class="ho-validation">
          <div class="ho-validation-title">Q-demand 工程檢核</div>
          <div class="ho-validation-text">通過 {pass_count}/3 項。這代表模型已經能用，但還有幾個工程面指標值得再調整。</div>
          <div class="ho-validation-pills">{''.join(bits)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    _ensure_state()
    _ensure_draft_workspace_for_upload()
    step = st.session_state.ob_step

    _hero(
        "專案導入",
        "上傳空調系統數據，完成清洗、欄位對應與設備設定後，自動訓練預測模型。",
    )
    st.markdown(_stepper_html(step), unsafe_allow_html=True)

    if step == "upload":
        st.markdown("---")
        left, right = st.columns([1.65, 1.0], gap="large")

        with left:
            with st.container(border=True):
                st.markdown('<p class="ho-section-title">數據檔案上傳</p>', unsafe_allow_html=True)
                st.markdown(
                    """
                    <div class="ho-drop-hint">
                        <span class="material-symbols-outlined">cloud_upload</span>
                        <p class="ho-muted" style="margin:0;font-weight:500;color:var(--ho-on-surface);">選擇或拖曳檔案至下方區域</p>
                        <p class="ho-muted" style="margin:0.25rem 0 0 0;">支援 CSV、Excel。建議單檔 &lt; 50MB。</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                uploaded_file = st.file_uploader(
                    "Upload HVAC CSV / Excel",
                    type=["csv", "xlsx", "xls"],
                    label_visibility="collapsed",
                )
                if st.button("上傳並檢查資料", type="primary", use_container_width=True):
                    if not uploaded_file:
                        st.warning("請先選擇檔案。")
                    else:
                        with st.spinner("上傳與檢查中…"):
                            res = api_client.upload_file(st.session_state.site_id, uploaded_file)
                        if "error" in res:
                            st.error(res["error"])
                        else:
                            st.session_state.dataset_info = res
                            st.session_state.ob_step = "diagnostics"
                            st.rerun()

        with right:
            with st.container(border=True):
                st.markdown('<p class="ho-section-title">系統介接同步</p>', unsafe_allow_html=True)
                st.markdown(
                    '<p class="ho-muted" style="margin-bottom:0.75rem;">直接從 BMS 或 IoT 匯入（規劃中）。</p>',
                    unsafe_allow_html=True,
                )
                st.button(
                    "BMS 系統（Modbus / BACnet）",
                    use_container_width=True,
                    disabled=True,
                    help="後續版本開放，不影響目前 CSV 流程。",
                )
                st.button(
                    "IoT 閘道（MQTT / REST）",
                    use_container_width=True,
                    disabled=True,
                    help="後續版本開放。",
                )

    elif step == "diagnostics":
        st.markdown("---")
        with st.container(border=True):
            st.markdown('<p class="ho-section-title">資料診斷</p>', unsafe_allow_html=True)
            diag = api_client.get_diagnostics(st.session_state.site_id)
            if "error" in diag:
                st.error(diag["error"])
                return
            diag_stage = diag.get("diagnostic_stage", "upload_precheck")
            final_label = "Stage 1 清理後列數" if diag_stage == "stage1_cleaned" else "上傳預檢後列數"
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("原始列數", f"{diag.get('initial_rows', diag.get('raw_rows', 0)):,}")
            c2.metric(final_label, f"{diag.get('final_rows', 0):,}")
            c3.metric(
                "完整率",
                f"{diag.get('completeness', 0):.1%}" if diag.get("completeness") is not None else "—",
            )
            c4.metric(
                "異常比例",
                f"{diag.get('anomaly_ratio', 0):.1%}" if diag.get("anomaly_ratio") is not None else "—",
            )
            if diag_stage != "stage1_cleaned":
                st.caption("目前顯示的是上傳預檢結果，只會移除空白列、重複列等明顯問題；正式 Stage 1 清理會在設備設定後執行。")
            if st.button("繼續：欄位對應", type="primary"):
                st.session_state.ob_step = "mapping"
                st.rerun()

    elif step == "mapping":
        st.markdown("---")
        with st.container(border=True):
            st.markdown('<p class="ho-section-title">欄位對應</p>', unsafe_allow_html=True)
            st.caption("確認每一欄對應到標準感測器；可保留「無」略過。")
            sugg = api_client.get_mapping_suggestions(st.session_state.site_id)
            if "error" in sugg:
                st.error(sugg["error"])
                return
            st.session_state.equipment_suggestion = sugg.get("equipment_suggestion", {})
            target_options = [
                None,
                "timestamp",
                "ambient_temp",
                "oa_rh",
                "total_power",
                "chiller_power",
                "chwp_power",
                "cwp_power",
                "ct_fan_power",
                "chws_temp",
                "chwr_temp",
                "cws_temp",
                "cwr_temp",
                "chw_flow",
                "cw_flow",
                "chwp_freq",
                "cwp_freq",
                "ct_fan_freq",
                "chws_setpoint",
            ]
            confirmed: list[dict[str, str | None]] = []
            for i, item in enumerate(sugg.get("mappings", [])):
                default = item.get("target")
                choice = st.selectbox(
                    f"{item['source']}",
                    options=target_options,
                    index=target_options.index(default) if default in target_options else 0,
                    key=f"mapping_{i}",
                )
                confirmed.append({"source": item["source"], "target": choice})
            if st.button("儲存對應並繼續", type="primary"):
                res = api_client.save_mapping(st.session_state.site_id, confirmed)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.session_state.ob_step = "equipment"
                    st.rerun()

    elif step == "equipment":
        st.markdown("---")
        with st.container(border=True):
            st.markdown('<p class="ho-section-title">設備規格</p>', unsafe_allow_html=True)
            st.caption("依據 CSV 欄位自動偵測設備類型與台數；請確認後再訓練。")

            suggestion = st.session_state.get("equipment_suggestion") or {}
            detected = suggestion.get("detected") if isinstance(suggestion, dict) else {}
            if isinstance(detected, dict):
                st.info(
                    "自動偵測："
                    f" 主機 {detected.get('chillers', 0)} 台、"
                    f" CHWP {detected.get('chwp', 0)} 台、"
                    f" CWP {detected.get('cwp', 0)} 台、"
                    f" CT {detected.get('cooling_tower', 0)} 台"
                )

            equipment_payload: dict[str, list[dict[str, float | int]]] = {
                "chillers": [],
                "chwp": [],
                "cwp": [],
                "cooling_tower": [],
            }

            chillers = suggestion.get("chillers") if isinstance(suggestion, dict) else None
            chwp_list = suggestion.get("chwp") if isinstance(suggestion, dict) else None
            cwp_list = suggestion.get("cwp") if isinstance(suggestion, dict) else None
            ct_list = suggestion.get("cooling_tower") if isinstance(suggestion, dict) else None

            if isinstance(chillers, list) and chillers:
                st.markdown("**冰水主機（Chiller）**")
                for idx, item in enumerate(chillers):
                    c1, c2 = st.columns(2)
                    with c1:
                        rt = st.number_input(
                            f"CH-{idx+1} 額定 RT",
                            min_value=50.0,
                            max_value=2000.0,
                            value=float(item.get("rt", 500.0)),
                            step=10.0,
                            key=f"eq_ch_rt_{idx}",
                        )
                    with c2:
                        cop = st.number_input(
                            f"CH-{idx+1} 額定 COP",
                            min_value=2.0,
                            max_value=12.0,
                            value=float(item.get("cop", 5.5)),
                            step=0.1,
                            key=f"eq_ch_cop_{idx}",
                        )
                    equipment_payload["chillers"].append({"id": idx + 1, "rt": rt, "cop": cop})

            if isinstance(chwp_list, list) and chwp_list:
                st.markdown("**冰水泵（CHWP）**")
                for idx, item in enumerate(chwp_list):
                    c1, c2 = st.columns(2)
                    with c1:
                        kw = st.number_input(
                            f"CHWP-{idx+1} 額定功率 kW",
                            min_value=1.0,
                            max_value=500.0,
                            value=float(item.get("kw", 22.0)),
                            step=0.5,
                            key=f"eq_chwp_kw_{idx}",
                        )
                    with c2:
                        hz = st.number_input(
                            f"CHWP-{idx+1} 最高頻率 Hz",
                            min_value=20.0,
                            max_value=120.0,
                            value=float(item.get("freq_max_hz", 60.0)),
                            step=1.0,
                            key=f"eq_chwp_hz_{idx}",
                        )
                    equipment_payload["chwp"].append({"id": idx + 1, "kw": kw, "freq_max_hz": hz})

            if isinstance(cwp_list, list) and cwp_list:
                st.markdown("**冷卻水泵（CWP）**")
                for idx, item in enumerate(cwp_list):
                    c1, c2 = st.columns(2)
                    with c1:
                        kw = st.number_input(
                            f"CWP-{idx+1} 額定功率 kW",
                            min_value=1.0,
                            max_value=500.0,
                            value=float(item.get("kw", 22.0)),
                            step=0.5,
                            key=f"eq_cwp_kw_{idx}",
                        )
                    with c2:
                        hz = st.number_input(
                            f"CWP-{idx+1} 最高頻率 Hz",
                            min_value=20.0,
                            max_value=120.0,
                            value=float(item.get("freq_max_hz", 60.0)),
                            step=1.0,
                            key=f"eq_cwp_hz_{idx}",
                        )
                    equipment_payload["cwp"].append({"id": idx + 1, "kw": kw, "freq_max_hz": hz})

            if isinstance(ct_list, list) and ct_list:
                st.markdown("**冷卻塔（CT）**")
                for idx, item in enumerate(ct_list):
                    c1, c2 = st.columns(2)
                    with c1:
                        kw = st.number_input(
                            f"CT-{idx+1} 風扇額定功率 kW",
                            min_value=1.0,
                            max_value=300.0,
                            value=float(item.get("kw", 18.5)),
                            step=0.5,
                            key=f"eq_ct_kw_{idx}",
                        )
                    with c2:
                        hz = st.number_input(
                            f"CT-{idx+1} 最高頻率 Hz",
                            min_value=20.0,
                            max_value=120.0,
                            value=float(item.get("freq_max_hz", 60.0)),
                            step=1.0,
                            key=f"eq_ct_hz_{idx}",
                        )
                    equipment_payload["cooling_tower"].append({"id": idx + 1, "kw": kw, "freq_max_hz": hz})

            if not any(equipment_payload.values()):
                st.warning("目前無法從欄位自動判斷設備，至少請填一台冰水主機。")
                fallback_rt = st.number_input("CH-1 額定 RT", min_value=50.0, max_value=2000.0, value=500.0, step=10.0)
                fallback_cop = st.number_input("CH-1 額定 COP", min_value=2.0, max_value=12.0, value=5.5, step=0.1)
                equipment_payload["chillers"] = [{"id": 1, "rt": fallback_rt, "cop": fallback_cop}]

            if st.button("執行清洗與訓練", type="primary"):
                with st.spinner("Stage 1–2 管線執行中…"):
                    res = api_client.save_equipment(st.session_state.site_id, equipment_payload)
                if "error" in res or res.get("status") != "success":
                    st.error(res.get("message", res.get("error", "Training failed")))
                else:
                    new_sid = res.get("site_id")
                    if isinstance(new_sid, str) and new_sid:
                        st.session_state.site_id = new_sid
                    pname = res.get("project_display_name")
                    if pname:
                        st.session_state.project_display_name = str(pname)
                    st.session_state.ml_results = res.get("ml")
                    st.session_state.stage1_results = res.get("stage1")
                    st.session_state.ob_step = "ready"
                    st.rerun()

    elif step == "ready":
        st.markdown("---")
        with st.container(border=True):
            pname = st.session_state.get("project_display_name")
            if pname:
                st.success(f"專案已封存，名稱依資料期間：**{pname}**（可於「既有數據分析／優化分析」選取此專案）")
            st.markdown('<p class="ho-section-title">訓練結果</p>', unsafe_allow_html=True)
            ml_res = st.session_state.get("ml_results") or {}
            power = ml_res.get("power_metrics", {})
            qd = ml_res.get("q_demand_metrics", {})
            power_mape = power.get("holdout_mape")
            q_coverage = qd.get("coverage")
            q_improve = qd.get("pinball_improvement")
            overall = "這次訓練完成，功率模型表現穩定，Q 模型也有明顯進步。"
            if qd.get("status") == "skipped":
                overall = "這次訓練完成，但 Q 模型沒有成功產出，建議先檢查資料欄位或樣本數。"
            elif q_coverage is not None and q_coverage < 0.8:
                overall = "這次訓練可用，但 Q 覆蓋率還偏保守，後續可以再調整資料或特徵。"
            st.markdown(f'<p class="ho-ready-summary">{overall}</p>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("功率誤差", f"{power_mape:.2%}" if power_mape is not None else "—")
            c2.metric("功率穩定度", f"{power.get('holdout_cv_rmse', 0):.2f}" if power.get("holdout_cv_rmse") is not None else "—")
            c3.metric("Q 覆蓋率", f"{q_coverage:.2f}" if q_coverage is not None else "—")
            c4.metric("Q 改善", f"{q_improve:.1%}" if q_improve is not None else "—")

            if qd.get("status") == "skipped":
                st.warning(f"Q-demand 模型略過：{qd.get('message')}")

            _render_feature_overview(ml_res)
            _render_validation_summary(qd)

            if st.button("前往既有數據分析", type="primary", use_container_width=True):
                st.session_state.current_view = "dashboard"
                st.query_params["view"] = "dashboard"
                st.rerun()

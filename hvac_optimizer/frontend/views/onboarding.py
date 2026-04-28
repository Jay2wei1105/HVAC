"""
Onboarding UI — Stage 1–2 flow. Layout/styling only; core API paths unchanged.
"""
from __future__ import annotations

import uuid

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
                        <p class="ho-muted" style="margin:0;font-weight:500;color:#0b1c30;">選擇或拖曳檔案至下方區域</p>
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
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("原始列數", f"{diag.get('initial_rows', diag.get('raw_rows', 0)):,}")
            c2.metric("清洗後列數", f"{diag.get('final_rows', 0):,}")
            c3.metric(
                "完整率",
                f"{diag.get('completeness', 0):.1%}" if diag.get("completeness") is not None else "—",
            )
            c4.metric(
                "異常比例",
                f"{diag.get('anomaly_ratio', 0):.1%}" if diag.get("anomaly_ratio") is not None else "—",
            )
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
            st.caption("用於訓練與最佳化參考；請依實際機台填寫。")
            chillers = st.number_input("冰水主機台數", min_value=1, max_value=8, value=1, step=1)
            rated_rt = st.number_input("單台額定冷凍噸 RT", min_value=50.0, max_value=2000.0, value=500.0, step=50.0)
            rated_cop = st.number_input("額定 COP", min_value=2.0, max_value=10.0, value=5.5, step=0.1)
            equipment_payload = {
                "chillers": [{"id": idx + 1, "rt": rated_rt, "cop": rated_cop} for idx in range(int(chillers))]
            }
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
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("功率 Holdout MAPE", f"{power.get('holdout_mape', 0):.2%}")
            c2.metric("功率 CV(RMSE)", f"{power.get('holdout_cv_rmse', 0):.2f}")
            c3.metric("Q coverage", f"{qd.get('coverage', 0):.2f}" if qd.get("coverage") is not None else "—")
            c4.metric(
                "Q 改善",
                f"{qd.get('pinball_improvement', 0):.1%}" if qd.get("pinball_improvement") is not None else "—",
            )

            if qd.get("status") == "skipped":
                st.warning(f"Q-demand 模型略過：{qd.get('message')}")

            st.markdown("**功率模型重要特徵**")
            for item in ml_res.get("power_top_features", []):
                st.write(f"- `{item['feature']}`: {item['importance']:.3f}")

            st.markdown("**Q-demand 檢核**")
            st.json(qd.get("validation", {}))

            if st.button("前往既有數據分析", type="primary", use_container_width=True):
                st.session_state.current_view = "dashboard"
                st.query_params["view"] = "dashboard"
                st.rerun()

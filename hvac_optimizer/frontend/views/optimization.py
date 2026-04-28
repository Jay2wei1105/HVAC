from __future__ import annotations

import streamlit as st

from hvac_optimizer.frontend import api_client


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .ho-opt-hero h1 {
            margin: 0;
            font-size: clamp(1.8rem, 3vw, 2.25rem);
            letter-spacing: -0.03em;
            color: #111827;
            font-weight: 700;
        }
        .ho-opt-hero p {
            margin: 0.35rem 0 0 0;
            color: #5f6777;
            font-size: 0.97rem;
        }
        .ho-opt-card {
            background: #ffffff;
            border: 1px solid #d9dfeb;
            border-radius: 16px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05), 0 10px 24px rgba(15, 23, 42, 0.04);
            padding: 1.1rem 1.2rem;
        }
        .ho-opt-title {
            margin: 0 0 0.7rem 0;
            font-size: 1rem;
            font-weight: 650;
            color: #1f2937;
        }
        .ho-opt-muted {
            color: #667085;
            font-size: 0.88rem;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_row(result: dict) -> None:
    savings = result.get("savings", {})
    q_constraint = result.get("q_constraint", {})
    cols = st.columns(4)
    cols[0].metric("預估節能", f"{savings.get('total_kw', 0):,.1f} kW")
    cols[1].metric("節能比例", f"{savings.get('total_pct', 0):,.1f}%")
    cols[2].metric("Q 能力上限", f"{q_constraint.get('q_capability', 0) or 0:,.1f} kW")
    cols[3].metric("Q 安全可行", "可行" if q_constraint.get("feasible") else "不可行")


def render() -> None:
    _inject_styles()
    site_id = st.session_state.get("site_id", "site_default")
    result = st.session_state.get("opt_results")

    st.markdown(
        """
        <div class="ho-opt-hero">
            <h1>優化分析（Advisory + MPC）</h1>
            <p>調整操作範圍後執行 Stage 3–5，產生最佳化建議、MPC 回放與報表輸出。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

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
            cws = st.slider("CWS (C)", min_value=20.0, max_value=32.0, value=(23.0, 29.0), step=0.5)

    if st.button("Run optimization", type="primary", use_container_width=True):
        bounds = {
            "chws": list(chws),
            "chwp": list(chwp),
            "ct_fan": list(ct_fan),
            "cws": list(cws),
        }
        with st.spinner("Running Stage 3-5 flow..."):
            res = api_client.run_optimization(site_id, bounds)
        if "error" in res:
            st.error(res["error"])
        else:
            st.session_state.opt_results = res
            st.rerun()

    if not result:
        st.info("請先執行一次最佳化，系統會產出建議設定、MPC 結果與報表資訊。")
        return

    _metric_row(result)
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Stage 3 建議最佳化", "Stage 4 MPC 回放", "Stage 5 產出物"])

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
        st.markdown("#### 安全邊界（Q Constraint）")
        st.json(result.get("q_constraint", {}))

    with tab2:
        st.markdown("#### MPC 摘要")
        st.json(result.get("mpc", {}))

    with tab3:
        st.markdown("#### 產出檔案")
        st.json(result.get("artifacts", {}))


from __future__ import annotations

import streamlit as st

from hvac_optimizer.frontend import api_client


def _metric_row(result: dict) -> None:
    savings = result.get("savings", {})
    q_constraint = result.get("q_constraint", {})
    cols = st.columns(4)
    cols[0].metric("Predicted saving", f"{savings.get('total_kw', 0):,.1f} kW")
    cols[1].metric("Saving percent", f"{savings.get('total_pct', 0):,.1f}%")
    cols[2].metric("Q capability", f"{q_constraint.get('q_capability', 0) or 0:,.1f} kW")
    cols[3].metric("Q feasible", "Yes" if q_constraint.get("feasible") else "No")


def render() -> None:
    site_id = st.session_state.get("site_id", "site_default")
    result = st.session_state.get("opt_results")

    st.title("Advisory + MPC")
    st.caption("Stage 3 advisory optimization with Q-demand safety bound, plus Stage 4 MPC replay and Stage 5 report artifacts.")

    c1, c2 = st.columns(2)
    with c1:
        chws = st.slider("CHWS (C)", min_value=5.0, max_value=12.0, value=(6.0, 9.0), step=0.1)
        chwp = st.slider("CHWP (Hz)", min_value=20, max_value=60, value=(30, 55))
    with c2:
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
        st.info("Run an advisory search to generate optimization, MPC, and report outputs.")
        return

    _metric_row(result)
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Stage 3 Advisory", "Stage 4 MPC", "Stage 5 Artifacts"])

    with tab1:
        baseline = result.get("baseline", {})
        optimal = result.get("optimal_params", {})
        st.subheader("Recommended setpoints")
        st.json(optimal)
        st.subheader("Baseline")
        st.json(baseline)
        st.subheader("Safety boundary")
        st.json(result.get("q_constraint", {}))

    with tab2:
        st.subheader("MPC summary")
        st.json(result.get("mpc", {}))

    with tab3:
        st.subheader("Artifacts")
        st.json(result.get("artifacts", {}))


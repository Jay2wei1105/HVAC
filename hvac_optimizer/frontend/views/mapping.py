import streamlit as st
import httpx

API_BASE = "http://localhost:8000/api/v1"

def render():
    st.title("欄位對應 + 設備規格")
    st.markdown("我們已自動對應大部分欄位，請確認標 ⚠ 與 ✕ 的部分")
    
    sid = st.session_state.get("site_id", "demo")
    
    mappings = [
        {"source": "冷卻水回水溫", "target": "cwr_temp", "confidence": "medium", "reason": "中文名稱推測，請確認"},
        {"source": "SETP_C", "target": "chws_setpoint", "confidence": "medium", "reason": "可能是冰水設定點"},
        {"source": "CHWP_3_HZ", "target": None, "confidence": "low", "reason": "找不到，可選擇忽略或手動指定"}
    ]
    
    col1, col2, col3 = st.columns(3)
    col1.success("✓ 高信心（5）")
    col2.warning("⚠ 待確認（2）")
    col3.error("✕ 找不到（1）")
    
    st.markdown("### ⚠ 和 ✕ 的欄位")
    for m in mappings:
        with st.container(border=True):
            cols = st.columns([2, 2, 2])
            cols[0].text(f"{'⚠' if m['confidence']=='medium' else '✕'} {m['source']}")
            cols[1].selectbox("映射至", ["cwr_temp", "chws_setpoint", "（未對應）"], index=0 if m['target'] == 'cwr_temp' else 1 if m['target'] == 'chws_setpoint' else 2, key=f"sel_{m['source']}")
            st.caption(m['reason'])

    st.markdown("---")
    st.markdown("### 設備規格（依據映射結果自動偵測，請確認）")
    
    st.subheader("Chiller")
    st.info("偵測到 3 組 chiller 相關欄位")
    for i in range(3):
        col_rt, col_cop = st.columns(2)
        with col_rt:
            st.number_input(f"#{i+1} 冷凍噸 RT *", value=350 if i < 2 else 200)
        with col_cop:
            st.number_input(f"#{i+1} 額定 COP *", value=4.5 if i < 2 else 4.2)
            
    st.subheader("Cooling Tower")
    st.info("偵測到 3 組 CT 相關欄位")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.number_input("#1 功率 kW", value=18.5)
    with col_c2:
        st.number_input("#2 功率 kW", value=18.5)
    with col_c3:
        st.number_input("#3 功率 kW", value=18.5)

    st.markdown("---")
    c1, c2 = st.columns([1, 4])
    if c1.button("← 回 Onboarding"):
        st.session_state.current_view = "onboarding"
        st.rerun()
    if c2.button("確認並進入 Dashboard →"):
        try:
            httpx.post(f"{API_BASE}/sites/{sid}/data/mapping", json={"mappings": [], "equipment": {}})
        except:
            pass
        st.session_state.current_view = "dashboard"
        st.rerun()

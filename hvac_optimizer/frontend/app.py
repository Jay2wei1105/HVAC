import streamlit as st
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hvac_optimizer.frontend.views import onboarding, dashboard, optimization
from hvac_optimizer.frontend import api_client

st.set_page_config(
    page_title="⚡ HVAC Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Global styles injection (M3-inspired tokens + Streamlit overrides)
def inject_custom_styles():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">
    <style>
        .block-container {
            padding-top: 0 !important;
            padding-bottom: 2.5rem !important;
            max-width: 1440px !important;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {background: transparent;}

        :root {
            --ho-primary: #2f6df6;
            --ho-primary-container: #1f5ae0;
            --ho-secondary: #4f46e5;
            --ho-background: #f7f7f5;
            --ho-surface: #ffffff;
            --ho-surface-low: #f2f4f8;
            --ho-on-surface: #111827;
            --ho-on-variant: #596275;
            --ho-outline: #d8deea;
            --ho-error: #c24141;
            --ho-radius-sm: 10px;
            --ho-radius-md: 14px;
            --ho-radius-lg: 18px;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
            color: var(--ho-on-surface);
        }
        .stApp {
            background: radial-gradient(circle at 0% 0%, #eef3ff 0%, transparent 32%),
                        radial-gradient(circle at 100% 8%, #f4f6ff 0%, transparent 28%),
                        var(--ho-background) !important;
        }
        section[data-testid="stMain"] {
            color: rgba(255, 255, 255, 1) !important;
            background-color: rgba(255, 255, 255, 1) !important;
        }

        /* Top nav */
        .custom-nav {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 64px;
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border-bottom: 1px solid #e8ebf2;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 clamp(16px, 3vw, 32px);
            z-index: 1000020;
            box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
        }
        .nav-inner {
            display: flex;
            align-items: center;
            gap: clamp(0.75rem, 2vw, 1.5rem);
            max-width: 1440px;
            margin: 0 auto;
            width: 100%;
        }
        .nav-spacer {
            flex: 1 1 auto;
            min-width: 0;
        }
        .nav-logo {
            font-size: 1.2rem;
            font-weight: 700;
            color: rgba(5, 5, 5, 1);
            letter-spacing: -0.04em;
            white-space: nowrap;
        }
        .nav-links {
            display: flex;
            gap: 1.25rem;
            align-items: center;
            flex: 0 1 auto;
            flex-wrap: wrap;
        }
        .nav-item {
            font-size: 0.875rem;
            font-weight: 500;
            color: #5c667a;
            text-decoration: none;
            padding: 0.35rem 0;
            border-bottom: 2px solid transparent;
            transition: color 0.15s, border-color 0.15s;
        }
        .nav-item:hover { color: var(--ho-primary-container); }
        .nav-item.active {
            color: rgba(12, 13, 13, 1);
            border-bottom-color: #1d4ed8;
        }
        .nav-actions {
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        .nav-icon-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 40px;
            height: 40px;
            border-radius: 0.375rem;
            color: #64748b;
            cursor: default;
            opacity: 0.65;
        }

        /* Streamlit controls */
        .stButton > button {
            border-radius: var(--ho-radius-md) !important;
            font-weight: 600 !important;
            transition: transform 0.16s ease, box-shadow 0.16s ease, background 0.16s ease !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(180deg, #2f6df6 0%, #235bdf 100%) !important;
            color: #fff !important;
            border: none !important;
            box-shadow: 0 8px 20px rgba(47, 109, 246, 0.24) !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(180deg, #2b67eb 0%, #1f56d6 100%) !important;
            box-shadow: 0 10px 22px rgba(47, 109, 246, 0.30) !important;
        }
        .stButton > button[kind="secondary"] {
            background: var(--ho-surface) !important;
            border: 1px solid #d4d9e3 !important;
            color: var(--ho-on-surface) !important;
        }
        .stButton > button:active { transform: scale(0.98) !important; }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: var(--ho-radius-lg) !important;
            border-color: var(--ho-outline) !important;
            background: var(--ho-surface) !important;
            box-shadow: 0 1px 2px rgba(17, 24, 39, 0.05), 0 10px 24px rgba(17, 24, 39, 0.03) !important;
            padding: 0.45rem 0.7rem !important;
        }
        div.stVerticalBlock.st-emotion-cache-1gz5zxc {
            color: rgba(255, 255, 255, 1) !important;
            background-color: unset !important;
            background: unset !important;
            background-clip: unset !important;
            -webkit-background-clip: unset !important;
            border-color: rgba(0, 0, 0, 1) !important;
            border-image: none !important;
        }

        [data-testid="stFileUploader"] {
            border: 2px dashed var(--ho-outline) !important;
            border-radius: var(--ho-radius-md) !important;
            padding: 1rem !important;
            background: var(--ho-surface-low) !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background-color: var(--ho-surface-low) !important;
            background: unset !important;
            color: rgba(255, 255, 255, 1) !important;
        }
        span.st-emotion-cache-1x4hur2 {
            color: rgba(10, 10, 10, 0.6) !important;
        }
        [data-testid="stFileUploader"]:hover {
            background: #e5eeff !important;
        }

        div[data-testid="stMetric"] {
            background: var(--ho-surface);
            border: 1px solid var(--ho-outline);
            border-radius: var(--ho-radius-md);
            padding: 0.75rem 1rem !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        div[data-testid="stMetric"] label {
            color: var(--ho-on-variant) !important;
            font-size: 0.75rem !important;
            font-weight: 500 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: var(--ho-primary-container) !important;
            font-weight: 600 !important;
        }

        .stSelectbox label, .stNumberInput label, .stFileUploader label {
            font-weight: 500 !important;
            color: var(--ho-on-surface) !important;
        }
        div[data-baseweb="select"] > div {
            border-radius: var(--ho-radius-md) !important;
        }

        /* Onboarding stepper (injected HTML) */
        .ho-stepper-wrap {
            margin: 0 0 1.5rem 0;
            padding: 1rem 1.25rem;
            background: var(--ho-surface);
            border: 1px solid var(--ho-outline);
            border-radius: 0.75rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .ho-stepper {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.35rem;
            flex-wrap: wrap;
        }
        .ho-step {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--ho-on-variant);
        }
        .ho-step-dot {
            width: 28px;
            height: 28px;
            border-radius: 9999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            font-weight: 700;
            border: 2px solid var(--ho-outline);
            background: var(--ho-surface-low);
            color: var(--ho-on-variant);
        }
        .ho-step.done .ho-step-dot {
            background: rgba(0, 108, 73, 0.12);
            border-color: var(--ho-secondary);
            color: var(--ho-secondary);
        }
        .ho-step.current .ho-step-dot {
            background: var(--ho-primary-container);
            border-color: var(--ho-primary-container);
            color: #fff;
        }
        .ho-step-line {
            flex: 1;
            min-width: 12px;
            height: 2px;
            background: var(--ho-outline);
            border-radius: 1px;
            opacity: 0.55;
        }
        .ho-step-line.done { background: var(--ho-secondary); opacity: 0.5; }

        .ho-hero h1 {
            font-size: clamp(1.65rem, 3vw, 2rem);
            font-weight: 600;
            letter-spacing: -0.02em;
            color: var(--ho-on-surface);
            margin: 0 0 0.35rem 0;
        }
        .ho-hero p {
            margin: 0;
            color: var(--ho-on-variant);
            font-size: 1rem;
            line-height: 1.5;
        }

        .ho-section-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--ho-on-surface);
            margin: 0 0 0.75rem 0;
        }
        .ho-muted {
            font-size: 0.875rem;
            color: var(--ho-on-variant);
            line-height: 1.45;
        }
        .ho-drop-hint {
            text-align: center;
            padding: 0.5rem 0 1rem 0;
        }
        .ho-drop-hint .material-symbols-outlined {
            font-size: 2.75rem;
            color: #94a3b8;
            display: block;
            margin-bottom: 0.35rem;
        }

        .ho-sync-card button {
            opacity: 0.55;
            cursor: not-allowed !important;
        }

        .material-symbols-outlined {
            font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
            vertical-align: middle;
        }

        /* Hide sidebar — project picker lives under header on analysis views */
        section[data-testid="stSidebar"],
        div[data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }
        div[data-testid="stAppViewContainer"] > .main {
            margin-left: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header(current_view: str) -> None:
    """
    Top navigation: view links only (no Streamlit widgets — those sit below on analysis pages).
    """
    items = [
        ("onboarding", "專案導入"),
        ("dashboard", "既有數據分析"),
        ("optimization", "優化分析")
    ]

    links_html = ""
    for view_id, label in items:
        active_class = "active" if current_view == view_id else ""
        links_html += f'<a href="?view={view_id}" class="nav-item {active_class}" target="_self">{label}</a>'

    st.markdown(f"""
    <div class="custom-nav">
        <div class="nav-inner">
            <div class="nav-logo">HVAC Optimizer</div>
            <div class="nav-links">
                {links_html}
            </div>
            <div class="nav-spacer"></div>
            <div class="nav-actions" aria-hidden="true">
                <span class="nav-icon-btn material-symbols-outlined">notifications</span>
                <span class="nav-icon-btn material-symbols-outlined">settings</span>
            </div>
        </div>
    </div>
    <div style="height: 72px;"></div>
    """, unsafe_allow_html=True)


def _render_completed_project_selector() -> None:
    """Single dropdown: trained projects only (dashboard + optimization)."""
    projects = api_client.get_projects(completed_only=True)
    if not isinstance(projects, list):
        err = projects.get("error", "無法載入專案列表") if isinstance(projects, dict) else "無法載入專案列表"
        st.warning(str(err))
        return

    if not projects:
        st.info("尚無已訓練專案。請先到「專案導入」完成上傳、對應與模型訓練。")
        return

    label_by_id: dict[str, str] = {p["site_id"]: str(p.get("label") or p["site_id"]) for p in projects}
    site_ids = [p["site_id"] for p in projects]

    if st.session_state.site_id not in site_ids:
        st.session_state.site_id = site_ids[0]

    try:
        idx = site_ids.index(st.session_state.site_id)
    except ValueError:
        idx = 0

    with st.container(border=True):
        st.markdown('<p class="ho-section-title" style="margin-bottom:0.5rem;">已訓練專案</p>', unsafe_allow_html=True)
        st.caption("以資料期間自動命名；僅顯示已完成清洗與模型訓練的專案。")
        col_a, col_b = st.columns([3.2, 1.0])
        with col_a:
            chosen = st.selectbox(
                "選擇專案",
                options=site_ids,
                format_func=lambda sid, _m=label_by_id: _m.get(sid, sid),
                index=min(idx, len(site_ids) - 1),
                label_visibility="collapsed",
            )
        with col_b:
            st.markdown("<div style='height:1.65rem'></div>", unsafe_allow_html=True)
            if st.button("新專案匯入", use_container_width=True, help="開始新的 CSV 導入流程"):
                import uuid

                st.session_state.site_id = f"site_{uuid.uuid4().hex[:8]}"
                st.session_state.ob_step = "upload"
                for k in ("dataset_info", "ml_results", "stage1_results"):
                    st.session_state.pop(k, None)
                st.session_state.current_view = "onboarding"
                st.query_params["view"] = "onboarding"
                st.rerun()

    if chosen != st.session_state.site_id:
        st.session_state.site_id = chosen
        st.rerun()

# Initialize session state
if "current_view" not in st.session_state:
    st.session_state.current_view = "onboarding"
if "site_id" not in st.session_state:
    st.session_state.site_id = "site_default"

# Handle URL params for navigation if possible, otherwise use session_state
params = st.query_params
if "view" in params:
    st.session_state.current_view = params["view"]

def main():
    inject_custom_styles()
    render_header(st.session_state.current_view)

    view = st.session_state.current_view
    if view == "optimization":
        _render_completed_project_selector()

    if view == "onboarding":
        onboarding.render()
    elif view == "dashboard":
        dashboard.render()
    elif view == "optimization":
        optimization.render()

if __name__ == "__main__":
    main()

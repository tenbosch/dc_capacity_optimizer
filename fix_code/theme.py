"""Theme constants and CSS injection for the SCMO dashboard."""
import streamlit as st


def inject_css():
    """Inject the dashboard CSS (Databricks light theme). Call after st.set_page_config."""
    st.markdown("""
<style>
    /* === SCMO Cold Chain Dashboard — Databricks Light Theme === */
    :root {
        --primary-green: #00DC8C;
        --primary-yellow: #FAEB1E;
        --primary-red: #F03200;
        --alert-red: #F03200;
        --bright-red: #F03200;
        --gold: #FAEB1E;
        --soft-green: #00DC8C;
        --dark-blue: #2563EB;
        --crimson: #F03200;
        --tan: #94A3B8;
    }

    * { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important; }
    .stApp { background-color: #FFFFFF !important; }
    [data-testid="stSidebar"] { background-color: #F8FAFC !important; border-right: 1px solid #E2E8F0; }
    [data-testid="stSidebar"] * { color: #334155 !important; }
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSelectbox label { color: #64748B !important; font-weight: 500; }
    .stTabs [data-baseweb="tab"] { font-weight: 500; color: #64748B !important; padding: 12px 20px; }
    .stTabs [aria-selected="true"] { color: var(--dark-blue) !important; border-bottom: 2px solid var(--dark-blue) !important; }
    h1 { color: #1E293B !important; font-weight: 700; letter-spacing: -0.5px; }
    h2, h3 { color: #334155 !important; font-weight: 600; }
    .metric-card { background: #FFFFFF; border-radius: 8px; padding: 20px;
        box-shadow: none; border: 1px solid #E2E8F0; }
    .insight-box { background: #F0FDF4; border-left: 4px solid var(--primary-green);
        padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 8px 0; }
    .warning-box { background: #FFFBEB; border-left: 4px solid var(--primary-yellow);
        padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 8px 0; }
    .critical-box { background: #FEF2F2; border-left: 4px solid var(--bright-red);
        padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 8px 0; }

    /* KPI Card Styling */
    .kpi-row { display: flex; gap: 16px; margin-bottom: 16px; }
    .kpi-card {
        flex: 1; padding: 20px 24px; border-radius: 12px;
        border: 1px solid #E2E8F0; position: relative; overflow: hidden;
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
    }
    .kpi-card::before {
        content: ''; position: absolute; top: 0; left: 0;
        width: 4px; height: 100%; border-radius: 12px 0 0 12px;
    }
    .kpi-card.blue::before { background: var(--dark-blue); }
    .kpi-card.green::before { background: var(--primary-green); }
    .kpi-card.yellow::before { background: var(--primary-yellow); }
    .kpi-card.red::before { background: var(--bright-red); }
    .kpi-label {
        font-size: 11px; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.5px; color: #64748B; margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 28px; font-weight: 800; color: #1E293B;
        letter-spacing: -0.5px; line-height: 1.1;
    }
    .kpi-sub { font-size: 12px; color: #94A3B8; margin-top: 4px; }

    /* Recommendation Card Styling */
    .reco-card {
        border: 1px solid #E2E8F0; border-radius: 10px;
        padding: 16px 20px; margin-bottom: 12px;
        position: relative; overflow: hidden;
        transition: box-shadow 0.2s ease;
    }
    .reco-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
    .reco-card::before {
        content: ''; position: absolute; top: 0; left: 0;
        width: 4px; height: 100%;
    }
    .reco-card.high::before { background: var(--bright-red); }
    .reco-card.medium::before { background: var(--primary-yellow); }
    .reco-card .reco-header {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 8px;
    }
    .reco-card .reco-category {
        font-size: 14px; font-weight: 700; color: #1E293B;
    }
    .reco-card .reco-dc {
        font-size: 12px; color: #64748B; font-weight: 500;
    }
    .reco-card .reco-detail {
        font-size: 13px; color: #475569; line-height: 1.5; margin-bottom: 8px;
    }
    .reco-card .reco-impact {
        font-size: 11px; color: #16A34A; font-weight: 600;
        display: inline-flex; align-items: center; gap: 4px;
    }
    .reco-badge {
        padding: 3px 10px; border-radius: 12px;
        font-size: 11px; font-weight: 700; color: white;
    }
    .reco-badge.high { background: var(--bright-red); }
    .reco-badge.medium { background: var(--primary-yellow); }

    /* Sidebar Footer */
    .sidebar-footer {
        position: fixed; bottom: 0; left: 0;
        width: inherit; padding: 12px 20px;
        background: #F1F5F9; border-top: 1px solid #E2E8F0;
        font-size: 11px; color: #94A3B8;
    }

    /* Info Card Styling */
    .info-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #DBEAFE;
        border-radius: 10px;
        padding: 20px 24px;
        margin: 4px 0;
    }
    .info-card-header {
        display: flex; align-items: center; gap: 8px;
        margin-bottom: 14px; padding-bottom: 10px;
        border-bottom: 1px solid #E2E8F0;
    }
    .info-card-header .icon {
        font-size: 18px; width: 32px; height: 32px;
        display: flex; align-items: center; justify-content: center;
        background: #DBEAFE; border-radius: 8px;
    }
    .info-card-header .title {
        font-size: 14px; font-weight: 700; color: #1E40AF;
        letter-spacing: -0.2px;
    }
    .info-card-section {
        margin-bottom: 12px;
    }
    .info-card-section:last-child { margin-bottom: 0; }
    .info-card-label {
        font-size: 10px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.5px; color: #64748B; margin-bottom: 4px;
    }
    .info-card-text {
        font-size: 13px; color: #334155; line-height: 1.5;
    }
    .info-card-formula {
        display: inline-block; background: #1E293B; color: #E2E8F0;
        padding: 4px 10px; border-radius: 6px; font-family: 'JetBrains Mono', monospace;
        font-size: 12px; margin: 4px 0;
    }
    .info-card-pill {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 11px; font-weight: 600; margin-right: 6px; margin-bottom: 4px;
    }
    .pill-green { background: #D1FFF0; color: #006644; }
    .pill-yellow { background: #FFFDE0; color: #6B5E00; }
    .pill-red { background: #FFE5DE; color: #8B1A00; }
    .pill-blue { background: #DBEAFE; color: #1E40AF; }
    .info-card-insight {
        background: #F0FDF4; border-left: 3px solid #16A34A;
        padding: 10px 14px; border-radius: 0 8px 8px 0;
        margin-top: 12px; font-size: 12px; color: #166534;
    }
    .info-card-table {
        width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 12px;
    }
    .info-card-table th {
        background: #EFF6FF; color: #1E40AF; padding: 6px 10px;
        text-align: left; font-weight: 600; border-bottom: 2px solid #BFDBFE;
    }
    .info-card-table td {
        padding: 5px 10px; border-bottom: 1px solid #E2E8F0; color: #334155;
    }
    .info-card-table tr:last-child td { border-bottom: none; }

    /* Streamlit widget overrides */
    .stButton > button[kind="primary"] {
        background-color: var(--dark-blue) !important;
        border-color: var(--dark-blue) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
    }
    [data-testid="stMetricValue"] { color: #1E293B !important; font-weight: 700; }
    .stSuccess { border-left-color: var(--primary-green) !important; }
    .stWarning { border-left-color: var(--primary-yellow) !important; }
    .stError { border-left-color: var(--primary-red) !important; }
</style>
""", unsafe_allow_html=True)


# === Color Constants (Databricks Light Theme — mirrors CSS :root) ===
THEME = {
    "primary_green": "#00DC8C",
    "primary_yellow": "#FAEB1E",
    "primary_red": "#F03200",
    "alert_red": "#F03200",
    "bright_red": "#F03200",
    "gold": "#FAEB1E",
    "soft_green": "#00DC8C",
    "dark_blue": "#2563EB",
    "crimson": "#F03200",
    "tan": "#94A3B8",
}

# Shorthand aliases for readability in chart code
COLOR_PRIMARY = THEME["dark_blue"]
COLOR_SECONDARY = THEME["tan"]
COLOR_SUCCESS = THEME["primary_green"]
COLOR_WARNING = THEME["primary_yellow"]
COLOR_DANGER = THEME["bright_red"]
COLOR_THRESHOLD = THEME["alert_red"]
COLOR_VOLUME = THEME["tan"]
COLOR_PEL = THEME["crimson"]

# Chart series palette
COLOR_PALETTE = [
    THEME["dark_blue"], THEME["primary_green"], THEME["primary_yellow"],
    THEME["bright_red"], THEME["crimson"], THEME["tan"], THEME["soft_green"]
]

# Stock status colors
STOCK_COLORS = {
    "Overstock": THEME["bright_red"],
    "Within_Range": THEME["primary_green"],
    "Understock": THEME["primary_yellow"],
}

# DOH category colors (aligned with DOH Heatmap conditional formatting)
DOH_COLORS = {
    "🪦 Dead Stock": "#6B7280",
    "🐢 Very Slow Moving": THEME["bright_red"],
    "🚶 Slow Moving": THEME["gold"],
    "✅ Normal": THEME["primary_green"],
    "⚡ Fast Moving": THEME["gold"],
}

# DOH Heatmap conditional formatting bands
DOH_HEATMAP = {
    "critical_low": {"threshold": 3, "bg": THEME["bright_red"], "fg": "#FFFFFF"},
    "warning_low": {"threshold": 7, "bg": THEME["gold"], "fg": "#000000"},
    "healthy": {"threshold": 21, "bg": THEME["soft_green"], "fg": "#FFFFFF"},
    "warning_high": {"threshold": 28, "bg": THEME["gold"], "fg": "#000000"},
    "critical_high": {"threshold": 999, "bg": THEME["bright_red"], "fg": "#FFFFFF"},
}

# Demand variability / WMAPE gradient domain
WMAPE_GRADIENT = {"start": THEME["primary_green"], "mid": THEME["primary_yellow"], "end": THEME["primary_red"], "domain": [0, 250, 600]}

# DC Utilization gradient domain
UTIL_GRADIENT = {"start": THEME["primary_green"], "mid": THEME["primary_yellow"], "end": THEME["primary_red"], "domain": [30, 60, 100]}

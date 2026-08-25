"""SCMO Network Storage Capacity Forecasting Tool
Built by Supply Chain Management & Optimization Team
Target Users: Replenishment Operations, Engineering, Supply Chain Planning
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from databricks.sdk import WorkspaceClient
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    layout="wide",
    page_title="DC Capacity Forecasting",
    page_icon="\U0001F4E6",
)

# ============================================================
# THEME & STYLING
# ============================================================
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

# ============================================================
# DATA CONNECTION
# ============================================================
def get_user_token():
    """Retrieve the current user's access token from app headers."""
    token = st.context.headers.get("x-forwarded-access-token")
    if not token:
        st.error("User access token not available. Ensure the 'sql' scope is configured under User Authorization in the app settings.")
        st.stop()
    return token


def get_workspace_client():
    """Create a WorkspaceClient authenticated as the current app user."""
    token = get_user_token()
    # Remove platform-injected service principal OAuth creds to avoid
    # "more than one authorization method configured" conflict with PAT auth
    os.environ.pop("DATABRICKS_CLIENT_ID", None)
    os.environ.pop("DATABRICKS_CLIENT_SECRET", None)
    return WorkspaceClient(
        host=os.environ.get("DATABRICKS_HOST"),
        token=token
    )


@st.cache_data(ttl=600, show_spinner=False)
def run_query(query):
    """Execute SQL via Databricks Statement Execution API with type casting."""
    import time
    import requests as _requests
    from databricks.sdk.service.sql import Disposition, Format, StatementState

    w = get_workspace_client()
    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID", "")
    if not warehouse_id:
        raise ValueError("DATABRICKS_WAREHOUSE_ID environment variable is not set.")

    # Use EXTERNAL_LINKS to avoid the 26MB inline byte limit
    result = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=query,
        wait_timeout="30s",
        disposition=Disposition.EXTERNAL_LINKS,
        format=Format.JSON_ARRAY,
    )

    # Poll if the query hasn't finished within the wait timeout
    while result.status and result.status.state in (
        StatementState.PENDING,
        StatementState.RUNNING,
    ):
        time.sleep(2)
        result = w.statement_execution.get_statement(result.statement_id)

    if result.status and result.status.error:
        raise RuntimeError(f"Query failed: {result.status.error.message}")

    columns = [col.name for col in result.manifest.schema.columns]
    col_types = [col.type_name for col in result.manifest.schema.columns]

    # Fetch data from external links (EXTERNAL_LINKS never uses data_array)
    data = []
    for chunk_info in result.manifest.chunks:
        chunk_result = w.statement_execution.get_statement_result_chunk_n(
            statement_id=result.statement_id, chunk_index=chunk_info.chunk_index
        )
        if chunk_result.external_links:
            for link in chunk_result.external_links:
                resp = _requests.get(link.external_link)
                resp.raise_for_status()
                data.extend(resp.json())
        elif chunk_result.data_array:
            data.extend(chunk_result.data_array)

    df = pd.DataFrame(data, columns=columns)
    INT_TYPES = ("INT", "BIGINT", "SMALLINT", "TINYINT", "LONG", "SHORT")
    FLOAT_TYPES = ("DOUBLE", "FLOAT", "DECIMAL")
    for col_name, col_type in zip(columns, col_types):
        ct = str(getattr(col_type, 'value', col_type)).upper()
        if ct in INT_TYPES:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype("Int64")
        elif ct in FLOAT_TYPES or ct.startswith("DECIMAL"):
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    return df


# ============================================================
# ML FORECASTING ENGINE
# ============================================================
def linear_forecast(series, horizon=4):
    """Simple linear regression forecast with confidence intervals."""
    y = series.dropna().values.astype(float)
    if len(y) < 3:
        return None, None, None
    x = np.arange(len(y))
    # Fit linear model: y = mx + b
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs[0], coeffs[1]
    # Forecast future points
    future_x = np.arange(len(y), len(y) + horizon)
    forecast = slope * future_x + intercept
    # Confidence interval (based on residual std)
    fitted = slope * x + intercept
    residual_std = np.std(y - fitted)
    ci_upper = forecast + 1.96 * residual_std
    ci_lower = forecast - 1.96 * residual_std
    return forecast, ci_lower, ci_upper


def exponential_smoothing(series, alpha=0.3, horizon=4):
    """Simple exponential smoothing forecast."""
    y = series.dropna().values.astype(float)
    if len(y) < 3:
        return None, None, None
    # Compute smoothed values
    smoothed = np.zeros(len(y))
    smoothed[0] = y[0]
    for i in range(1, len(y)):
        smoothed[i] = alpha * y[i] + (1 - alpha) * smoothed[i - 1]
    # Forecast is the last smoothed value projected forward
    last_smooth = smoothed[-1]
    forecast = np.full(horizon, last_smooth)
    # Confidence grows with horizon
    residual_std = np.std(y - smoothed)
    ci_factors = np.array([1.96 * residual_std * np.sqrt(h + 1) for h in range(horizon)])
    ci_upper = forecast + ci_factors
    ci_lower = forecast - ci_factors
    return forecast, ci_lower, ci_upper


def detect_outliers_iqr(series, multiplier=1.5):
    """Detect outliers using IQR method. Returns boolean mask."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (series < lower) | (series > upper), lower, upper, q1, q3, iqr


def compute_capacity_score(row):
    """Compute a 0-100 capacity efficiency score for a DC."""
    score = 100
    if row.get("overstock_pct", 0) > 30:
        score -= 25
    elif row.get("overstock_pct", 0) > 15:
        score -= 10
    if row.get("dead_stock_pct", 0) > 10:
        score -= 20
    elif row.get("dead_stock_pct", 0) > 5:
        score -= 10
    if row.get("median_doh", 0) > 60:
        score -= 15
    elif row.get("median_doh", 0) > 30:
        score -= 5
    return max(0, min(100, score))


# ============================================================
# SQL QUERIES
# ============================================================
QUERY_CORE_WEEKLY = """
WITH mat_dedup AS (
  SELECT mat_nbr, mat_grp
  FROM (
    SELECT mat_nbr, mat_grp, ROW_NUMBER() OVER (PARTITION BY mat_nbr ORDER BY mat_nbr) AS rn
    FROM adh_genpro_use2_prd.s_products.material_details_consolidated_hhd
    WHERE temp_cond_desc IN ('Keep Frozen', 'Refrge/Do Not Freeze')
  ) WHERE rn = 1
),
item_key_lookup AS (
  SELECT DISTINCT di.item_key, mm.mat_nbr8 AS mat
  FROM adh_genpro_use2_prd.s_products.material_details_consolidated_hhd mm
  INNER JOIN adh_genpro_use2_prd.s_distribution.a_dim_items_hhd di ON di.item_number_nk = mm.mat_nbr8
  WHERE mm.temp_cond_desc IN ('Keep Frozen', 'Refrge/Do Not Freeze')
),
reserve_items AS (
  SELECT DISTINCT il.division_lookup AS plant, ik.mat AS item
  FROM adh_genpro_use2_prd.s_distribution.inventory_locations_hhd il
  INNER JOIN item_key_lookup ik ON il.item_key = ik.item_key
  INNER JOIN adh_genpro_use2_prd.s_distribution.a_dim_locations_hhd loc ON loc.loc_key = il.location_key
  WHERE loc.active_reserve_casepick IN ('Reserve')
    AND YEAR(DATE(il.snapshot_date_key)) >= 2026
),
dc_lookup AS (
  SELECT DISTINCT division_number, division_name AS DC_Name
  FROM adh_genpro_use2_prd.s_distribution.a_dim_divisions_hhd
),
rep_dedup AS (
  SELECT dc_formatted, itm_formatted, order_up_to_level_qty, order_up_to_level_days,
    safety_stock_qty, safety_stock_days, repln_invtry, repln_days, deseasonalized_fcst_for_period
  FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY dc_formatted, itm_formatted ORDER BY adh_updated DESC) AS rn
    FROM adh_genpro_use2_prd.s_warehouse_ops.replenish_material_hist_hhd
  ) WHERE rn = 1
),
inventory_weekly AS (
  SELECT
    CONCAT('WK-', LPAD(WEEKOFYEAR(inv.invtry_date), 2, '0')) AS week_label,
    YEAR(inv.invtry_date) AS yr,
    WEEKOFYEAR(inv.invtry_date) AS wk,
    inv.plant,
    inv.plant || '-' || LTRIM('0', TRIM(inv.mat_nbr)) AS SKU,
    AVG(inv.on_hand_qty) AS Avg_OH_Qty,
    AVG(inv.deseasonalized_fcst_qty) AS Avg_Deseasoned_Fcst_Qty,
    TRY_DIVIDE(AVG(inv.on_hand_qty), AVG(inv.deseasonalized_fcst_qty) / 28) AS DOH,
    CASE
      WHEN AVG(inv.on_hand_qty) > AVG(rep.order_up_to_level_qty) THEN 'Overstock'
      WHEN AVG(inv.on_hand_qty) < AVG(rep.safety_stock_qty) THEN 'Understock'
      ELSE 'Within_Range'
    END AS Stock_Status,
    AVG(rep.order_up_to_level_qty) AS Avg_Order_Up_To_Level_Qty,
    AVG(rep.order_up_to_level_days) AS Avg_Order_Up_To_Level_Days,
    AVG(rep.safety_stock_qty) AS Avg_Safety_Stock_Qty,
    AVG(rep.safety_stock_days) AS Avg_Safety_Stock_Days,
    AVG(rep.repln_invtry) AS Avg_Repln_Invtry,
    AVG(rep.repln_days) AS Avg_Repln_Days,
    AVG(rep.deseasonalized_fcst_for_period) AS Forecast_Demand,
    mat.mat_grp
  FROM adh_genpro_use2_prd.s_warehouse_ops.custom_mat_fcst_position_hhd inv
  INNER JOIN mat_dedup mat ON inv.mat_nbr = mat.mat_nbr
  INNER JOIN rep_dedup rep ON rep.itm_formatted = LTRIM('0', TRIM(inv.mat_nbr)) AND inv.plant = rep.dc_formatted
  INNER JOIN reserve_items ri ON ri.plant = inv.plant AND ri.item = LTRIM('0', TRIM(inv.mat_nbr))
  WHERE inv.invtry_date >= '2026-01-01'
    AND (inv.on_hand_qty + inv.on_ord_qty) > 0
    AND inv.plant NOT IN ('087', '203', '204', '200', '220', '230')
  GROUP BY WEEKOFYEAR(inv.invtry_date), YEAR(inv.invtry_date), inv.plant,
    inv.plant || '-' || LTRIM('0', TRIM(inv.mat_nbr)), mat.mat_grp
)
SELECT iw.week_label, iw.yr, iw.wk, iw.plant, iw.SKU, iw.Avg_OH_Qty,
  iw.Avg_Deseasoned_Fcst_Qty, iw.DOH, iw.Stock_Status,
  iw.Avg_Order_Up_To_Level_Qty, iw.Avg_Order_Up_To_Level_Days,
  iw.Avg_Safety_Stock_Qty, iw.Avg_Safety_Stock_Days,
  iw.Avg_Repln_Invtry, iw.Avg_Repln_Days, iw.Forecast_Demand,
  dc.DC_Name, iw.mat_grp
FROM inventory_weekly iw
INNER JOIN dc_lookup dc ON iw.plant = dc.division_number
WHERE (iw.DOH <= 365 OR iw.DOH IS NULL)
"""

QUERY_VOLUME_DATA = """
WITH item_key_lookup AS (
  SELECT DISTINCT di.item_key, mm.mat_nbr8 AS mat
  FROM adh_genpro_use2_prd.s_products.material_details_consolidated_hhd mm
  INNER JOIN adh_genpro_use2_prd.s_distribution.a_dim_items_hhd di ON di.item_number_nk = mm.mat_nbr8
  WHERE mm.temp_cond_desc IN ('Keep Frozen', 'Refrge/Do Not Freeze')
),
sku_location AS (
  SELECT DISTINCT il.location_key, ik.mat AS item, il.division_lookup AS division,
    YEAR(DATE(il.snapshot_date_key)) AS yr, WEEKOFYEAR(DATE(il.snapshot_date_key)) AS wk
  FROM adh_genpro_use2_prd.s_distribution.inventory_locations_hhd il
  INNER JOIN item_key_lookup ik ON il.item_key = ik.item_key
  WHERE YEAR(DATE(il.snapshot_date_key)) >= 2026
),
location_data AS (
  SELECT
    loc.division_nbr_nk || '-' || sw.item AS SKU,
    loc.division_nbr_nk AS plant,
    sw.yr, sw.wk,
    CONCAT('WK-', LPAD(sw.wk, 2, '0')) AS week_label,
    AVG(loc.height * loc.width * loc.length) AS Total_Volume_Cubic_Inches,
    AVG(loc.height * loc.width * loc.length) / 75000 AS PEL,
    SUM(CASE WHEN loc.active_reserve_casepick = 'Active' THEN 1 ELSE 0 END) AS active_locations,
    SUM(CASE WHEN loc.active_reserve_casepick = 'Reserve' THEN 1 ELSE 0 END) AS reserve_locations
  FROM adh_genpro_use2_prd.s_distribution.a_dim_locations_hhd loc
  INNER JOIN sku_location sw ON loc.loc_key = sw.location_key
  WHERE loc.active_reserve_casepick IN ('Reserve')
    AND TRIM(loc.class) = 'Refrig'
    AND loc.height < 9999 AND loc.width < 9999 AND loc.length < 9999
    
  GROUP BY loc.division_nbr_nk || '-' || sw.item, loc.division_nbr_nk, sw.yr, sw.wk
)
SELECT * FROM location_data
WHERE Total_Volume_Cubic_Inches <= 1000000
  AND plant NOT IN ('087', '203', '204', '200', '220', '230')
"""

QUERY_BILLING = """
WITH mat_dedup AS (
  SELECT mat_nbr, mat_grp
  FROM (
    SELECT mat_nbr, mat_grp, ROW_NUMBER() OVER (PARTITION BY mat_nbr ORDER BY mat_nbr) AS rn
    FROM adh_genpro_use2_prd.s_products.material_details_consolidated_hhd
    WHERE temp_cond_desc IN ('Keep Frozen', 'Refrge/Do Not Freeze')
  ) WHERE rn = 1
),
item_key_lookup AS (
  SELECT DISTINCT di.item_key, mm.mat_nbr8 AS mat
  FROM adh_genpro_use2_prd.s_products.material_details_consolidated_hhd mm
  INNER JOIN adh_genpro_use2_prd.s_distribution.a_dim_items_hhd di ON di.item_number_nk = mm.mat_nbr8
  WHERE mm.temp_cond_desc IN ('Keep Frozen', 'Refrge/Do Not Freeze')
),
reserve_items AS (
  SELECT DISTINCT il.division_lookup AS plant, ik.mat AS item
  FROM adh_genpro_use2_prd.s_distribution.inventory_locations_hhd il
  INNER JOIN item_key_lookup ik ON il.item_key = ik.item_key
  INNER JOIN adh_genpro_use2_prd.s_distribution.a_dim_locations_hhd loc ON loc.loc_key = il.location_key
  WHERE loc.active_reserve_casepick IN ('Reserve')
    AND YEAR(DATE(il.snapshot_date_key)) >= 2026
)
SELECT
  bt.ship_plant || '-' || LTRIM('0', TRIM(bt.mat_nbr)) AS SKU,
  bt.ship_plant AS plant,
  YEAR(bt.invc_created_date) AS yr,
  WEEKOFYEAR(bt.invc_created_date) AS wk,
  CONCAT('WK-', LPAD(WEEKOFYEAR(bt.invc_created_date), 2, '0')) AS week_label,
  SUM(bt.invc_qty) AS Total_Billing_Qty,
  SUM(bt.sales_qty) AS Total_Sales_Qty,
  mat.mat_grp
FROM adh_genpro_use2_prd.g_order360.billing_transactions_hhd bt
INNER JOIN mat_dedup mat ON TRIM(bt.mat_nbr) = TRIM(mat.mat_nbr)
INNER JOIN reserve_items ri ON ri.plant = bt.ship_plant AND ri.item = LTRIM('0', TRIM(bt.mat_nbr))
WHERE bt.invc_created_date >= '2026-01-01'
  AND bt.sales_qty > 0
  AND bt.ship_plant NOT IN ('087', '203', '204', '200', '220', '230')
GROUP BY bt.ship_plant || '-' || LTRIM('0', TRIM(bt.mat_nbr)),
  bt.ship_plant, YEAR(bt.invc_created_date), WEEKOFYEAR(bt.invc_created_date), mat.mat_grp
"""

QUERY_DC_UTILIZATION = """
SELECT dc_name, utilization_pct, lat, lon
FROM VALUES
  ('Mansfield', 90, 42.7588, -71.21),
  ('Corona', 89, 33.8753, -117.5664),
  ('Seattle', 83, 47.6062, -122.3321),
  ('Puerto Rico', 82, 18.2208, -66.5901),
  ('Whitestown', 80, 39.9970, -86.3458),
  ('Chicago', 80, 41.8781, -87.6298),
  ('Richmond', 79, 37.5407, -77.4360),
  ('Newburgh', 76, 41.5034, -74.0104),
  ('Buford', 75, 34.1207, -83.9810),
  ('MONTCLAIR D&S', 74, 40.8259, -74.2090),
  ('Sacramento', 72, 38.5816, -121.4944),
  ('Olive Branch', 72, 34.9618, -89.8295),
  ('Bethlehem', 70, 40.6259, -75.3705),
  ('Columbus', 69, 39.9612, -82.9988),
  ('Kansas City', 68, 39.0997, -94.5786),
  ('Phoenix-B', 68, 33.4484, -112.0740),
  ('BROOKS D&S', 67, 31.7541, -83.5434),
  ('Denver', 66, 39.7392, -104.9903),
  ('Dallas', 65, 32.7767, -96.7970),
  ('Orlando Vista', 64, 28.5383, -81.3792),
  ('Williamston', 62, 42.6890, -84.2830),
  ('Raleigh', 61, 35.7796, -78.6382),
  ('Amityville', 60, 40.6790, -73.4171),
  ('Louisville', 57, 38.2527, -85.7585),
  ('Honolulu', 57, 21.3069, -157.8583),
  ('Shakopee', 55, 44.7974, -93.5272),
  ('Houston', 49, 29.7604, -95.3698),
  ('Salt Lake City', 45, 40.7608, -111.8910),
  ('DOTHAN D&S', 39, 31.2232, -85.3905)
  AS t(dc_name, utilization_pct, lat, lon)
"""

QUERY_FILTER_OPTIONS = """
WITH dc_lookup AS (
  SELECT DISTINCT division_number AS plant, division_name AS DC_Name
  FROM adh_genpro_use2_prd.s_distribution.a_dim_divisions_hhd
  WHERE division_number IN ( '003','004','008','010','012','014','015','016','017','018',
            '019','021','022','023','026','027','028','029','030','037',
            '038','039','041','049','055','063')
    AND adh_delete_flag = FALSE
    AND end_date IS NULL
),
date_range AS (
  SELECT DISTINCT YEAR(invtry_date) AS yr, MONTH(invtry_date) AS mn, WEEKOFYEAR(invtry_date) AS wk
  FROM adh_genpro_use2_prd.s_warehouse_ops.custom_mat_fcst_position_hhd
  WHERE invtry_date >= '2026-01-01'
)
SELECT 'dc' AS filter_type, plant AS val, DC_Name AS label FROM dc_lookup
UNION ALL
SELECT 'year', CAST(yr AS STRING), CAST(yr AS STRING) FROM (SELECT DISTINCT yr FROM date_range)
UNION ALL
SELECT 'month', CAST(yr AS STRING) || '-' || LPAD(CAST(mn AS STRING), 2, '0'), CAST(yr AS STRING) || '-' || LPAD(CAST(mn AS STRING), 2, '0') FROM (SELECT DISTINCT yr, mn FROM date_range)
UNION ALL
SELECT 'month_week', CAST(yr AS STRING) || '-' || LPAD(CAST(mn AS STRING), 2, '0'), CAST(wk AS STRING) FROM date_range
UNION ALL
SELECT 'week', CAST(wk AS STRING), CAST(wk AS STRING) FROM (SELECT DISTINCT wk FROM date_range)
UNION ALL
SELECT 'mat_grp', mat_grp,
  CASE mat_grp WHEN 'BRX' THEN 'Brand Rx' WHEN 'GRX' THEN 'Generic Rx'
  WHEN 'OTC' THEN 'OTC Brand' WHEN 'OTG' THEN 'OTC Generic'
  WHEN 'MSU' THEN 'Medical/Surgical' WHEN 'GMR' THEN 'Generic Medical'
  WHEN 'HBC' THEN 'Health & Beauty' WHEN 'HBG' THEN 'H&B Generic'
  WHEN 'SSU' THEN 'Surgical Supply' WHEN 'HHC' THEN 'Home Health'
  ELSE mat_grp END
FROM (SELECT DISTINCT mat_grp FROM adh_genpro_use2_prd.s_products.material_details_consolidated_hhd
      WHERE temp_cond_desc IN ('Keep Frozen', 'Refrge/Do Not Freeze') AND mat_grp IS NOT NULL)
"""


# ============================================================
# FILTER UTILITIES
# ============================================================
def apply_filters(df, plant_col="plant", week_col="wk", year_col="yr", mat_grp_col="mat_grp"):
    """Apply global sidebar filters to a DataFrame."""
    result = df.copy()
    if st.session_state.get("selected_mat_grps") and mat_grp_col in result.columns:
        result = result[result[mat_grp_col].isin(st.session_state["selected_mat_grps"])]
    if st.session_state.get("selected_plants") and plant_col in result.columns:
        result = result[result[plant_col].isin(st.session_state["selected_plants"])]
    if st.session_state.get("selected_years") and year_col and year_col in result.columns:
        result = result[result[year_col].astype(str).isin([str(y) for y in st.session_state["selected_years"]])]
    # Month filter: if months selected but no specific weeks, apply weeks from selected months
    if st.session_state.get("selected_months") and not st.session_state.get("selected_weeks") and week_col and week_col in result.columns:
        m2w = st.session_state.get("month_to_weeks", {})
        month_weeks = sorted(set().union(*(m2w.get(m, set()) for m in st.session_state["selected_months"])))
        if month_weeks:
            result = result[result[week_col].astype(int).isin(month_weeks)]
    if st.session_state.get("selected_weeks") and week_col and week_col in result.columns:
        result = result[result[week_col].astype(int).isin(st.session_state["selected_weeks"])]
    return result


def apply_mat_grp_sku_filter(df, df_core_filtered):
    """Filter a DataFrame (e.g. volume) by SKUs in the mat_grp-filtered core data.
    Only applies when a material group filter is active and the df lacks mat_grp column."""
    if st.session_state.get("selected_mat_grps") and "mat_grp" not in df.columns and "SKU" in df.columns:
        valid_skus = set(df_core_filtered["SKU"].unique())
        return df[df["SKU"].isin(valid_skus)]
    return df


def format_number(val, decimals=1):
    """Format large numbers for display."""
    if pd.isna(val):
        return "N/A"
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:.{decimals}f}M"
    if abs(val) >= 1_000:
        return f"{val/1_000:.{decimals}f}K"
    return f"{val:.{decimals}f}"


def empty_state(message="No data available for current filter selection."):
    """Show a friendly empty state."""
    st.info(f"\U0001F4AD {message}")


def chart_layout(fig, height=450):
    """Apply consistent chart styling."""
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        height=height, margin=dict(t=40, b=40, l=40, r=40),
        font=dict(family="Inter, sans-serif", size=12),
    )
    return fig


def info_card(title, icon, sections, insight=None):
    """Generate styled HTML info card.
    sections: list of dicts with keys 'label', 'content' (HTML string)
    """
    html = f"""<div class='info-card'>
    <div class='info-card-header'>
        <div class='icon'>{icon}</div>
        <div class='title'>{title}</div>
    </div>"""
    for s in sections:
        html += f"""<div class='info-card-section'>
        <div class='info-card-label'>{s['label']}</div>
        <div class='info-card-text'>{s['content']}</div>
    </div>"""
    if insight:
        html += f"""<div class='info-card-insight'>✨ <b>Insight:</b> {insight}</div>"""
    html += "</div>"
    return html


# ============================================================
# APP HEADER & SIDEBAR
# ============================================================
st.title("\U0001F4E6 Network Storage Capacity Forecasting")
st.caption("Supply Chain Management & Optimization | Cold Chain Distribution Network")

# Load filter options
MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
              7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

try:
    with st.spinner("Loading filters..."):
        df_filters = run_query(QUERY_FILTER_OPTIONS)
    dc_options = df_filters[df_filters["filter_type"] == "dc"].sort_values("label")
    year_options = sorted(df_filters[df_filters["filter_type"] == "year"]["val"].unique())
    week_options = sorted(df_filters[df_filters["filter_type"] == "week"]["val"].astype(int).unique())

    # Month options: raw values like "2026-01" → display as "Jan-26"
    month_raw_options = sorted(df_filters[df_filters["filter_type"] == "month"]["val"].unique())

    def month_raw_to_label(raw):
        """Convert '2026-01' → 'Jan-26'"""
        yr_str, mn_str = raw.split("-")
        return f"{MONTH_ABBR[int(mn_str)]}-{yr_str[2:]}"

    # Build month-to-weeks mapping from month_week rows
    df_month_week = df_filters[df_filters["filter_type"] == "month_week"]
    month_to_weeks = {}
    for _, row in df_month_week.iterrows():
        month_key = row["val"]  # e.g. "2026-01"
        wk = int(row["label"])  # week number
        month_to_weeks.setdefault(month_key, set()).add(wk)

    # Material group options
    mat_grp_options = df_filters[df_filters["filter_type"] == "mat_grp"].sort_values("label")
except Exception as e:
    st.sidebar.error(f"Failed to load filters: {e}")
    dc_options = pd.DataFrame(columns=["val", "label"])
    year_options = ["2025", "2026"]
    week_options = list(range(1, 53))
    month_raw_options = []
    month_to_weeks = {}
    mat_grp_options = pd.DataFrame(columns=["val", "label"])
    def month_raw_to_label(raw):
        yr_str, mn_str = raw.split("-")
        return f"{MONTH_ABBR[int(mn_str)]}-{yr_str[2:]}"

with st.sidebar:
    st.markdown("### \U0001F50D Filters")
    st.markdown("---")

    if not dc_options.empty:
        dc_display = dict(zip(dc_options["val"], dc_options["label"] + " (" + dc_options["val"] + ")"))
        selected_plants = st.multiselect(
            "Distribution Center", options=list(dc_display.keys()),
            format_func=lambda x: dc_display.get(x, x),
            default=None, placeholder="All DCs"
        )
    else:
        selected_plants = []

    # --- Material Group / Product Category filter ---
    if not mat_grp_options.empty:
        mat_grp_display = dict(zip(mat_grp_options["val"], mat_grp_options["label"] + " (" + mat_grp_options["val"] + ")"))
        selected_mat_grps = st.multiselect(
            "Material Group", options=list(mat_grp_display.keys()),
            format_func=lambda x: mat_grp_display.get(x, x),
            default=None, placeholder="All Material Groups"
        )
    else:
        selected_mat_grps = []

    selected_years = st.multiselect(
        "Calendar Year", options=year_options, default=None, placeholder="All Years"
    )

    # --- Month filter (synced with year) ---
    # If year is selected, only show months for that year
    if selected_years:
        available_months = [m for m in month_raw_options if m.split("-")[0] in [str(y) for y in selected_years]]
    else:
        available_months = list(month_raw_options)

    selected_months = st.multiselect(
        "Month", options=available_months,
        format_func=month_raw_to_label,
        default=None, placeholder="All Months"
    )

    # --- Week filter (synced with month) ---
    # If month is selected, only show weeks that fall in those months
    if selected_months:
        available_weeks = sorted(set().union(*(month_to_weeks.get(m, set()) for m in selected_months)))
    else:
        available_weeks = list(week_options)

    selected_weeks = st.multiselect(
        "Calendar Week", options=available_weeks,
        format_func=lambda x: f"WK-{x:02d}", default=None, placeholder="All Weeks"
    )

    # --- Settings hidden for now ---
    # st.markdown("---")
    # st.markdown("### \U00002699\uFE0F Settings")
    # forecast_horizon = st.slider("Forecast Horizon (weeks)", 2, 12, 4)
    # forecast_method = st.selectbox("Forecast Method", ["Linear Regression", "Exponential Smoothing"])
    # outlier_sensitivity = st.select_slider(
    #     "Outlier Sensitivity", options=[1.0, 1.5, 2.0, 2.5, 3.0], value=1.5
    # )

    # --- Sidebar Footer ---
    st.markdown("---")
    st.markdown(
        "<div style='padding:8px 0;'>"
        "<div style='font-size:11px;color:#94A3B8;font-weight:600;'>SCMO Capacity Forecasting</div>"
        "<div style='font-size:10px;color:#CBD5E1;margin-top:2px;'>v1.1 · Cold Chain Analytics</div>"
        "<div style='font-size:10px;color:#CBD5E1;margin-top:4px;'>"
        "🔄 Data refreshes every week</div>"
        "</div>",
        unsafe_allow_html=True
    )

# Default settings (UI hidden)
forecast_horizon = 4
forecast_method = "Linear Regression"
outlier_sensitivity = 1.5

# Store in session state for filter functions
st.session_state["selected_plants"] = selected_plants
st.session_state["selected_mat_grps"] = selected_mat_grps
st.session_state["selected_years"] = selected_years
st.session_state["selected_months"] = selected_months
st.session_state["month_to_weeks"] = month_to_weeks
st.session_state["selected_weeks"] = selected_weeks


# ============================================================
# LOAD CORE DATA
# ============================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_core_data():
    """Load and cache the primary datasets."""
    df_inv = run_query(QUERY_CORE_WEEKLY)
    return df_inv


try:
    with st.spinner("Loading inventory data..."):
        df_core_raw = load_core_data()
    df_core = apply_filters(df_core_raw)
except Exception as e:
    st.error(f"\u274C Failed to load core data: {e}")
    st.stop()


# ============================================================
# TAB LAYOUT
# ============================================================
tab1, tab3, tab2, tab6, tab4 = st.tabs([
    "\U0001F4CA Network Overview",
    "\U0001F4E6 Inventory Health",
    "\U0001F4C8 Demand Vs Forecast(DFIO)",
    "\U0001F4A1 Recommendations",
    "\U0001F4D6 Readme"
])


# ======================== TAB 1: NETWORK OVERVIEW ========================
with tab1:
    st.header("Network Overview")
    st.caption("Current operational snapshot — latest week only")

    if df_core.empty:
        empty_state()
    else:
        # --- KPI Metrics Row (latest week snapshot) ---
        latest_wk = df_core["wk"].max()
        latest_yr = df_core[df_core["wk"] == latest_wk]["yr"].max()
        df_latest = df_core[(df_core["wk"] == latest_wk) & (df_core["yr"] == latest_yr)]

        # Exclude null-DOH SKUs (zero forecast) from KPI counts to align with DOH Category chart
        df_latest_valid = df_latest[df_latest["DOH"].notna()]
        distinct_skus = df_latest["SKU"].nunique()  # All SKUs including no-forecast
        unique_materials = df_latest["SKU"].str.split("-", n=1).str[1].nunique()  # Unique material numbers (plant-agnostic)
        total_skus = df_latest_valid["SKU"].nunique()
        total_dcs = df_latest_valid["plant"].nunique()
        median_doh = df_latest_valid["DOH"].median()
        null_doh_skus = df_latest[df_latest["DOH"].isna()]["SKU"].nunique()
        overstock_pct = (df_latest_valid[df_latest_valid["Stock_Status"] == "Overstock"]["SKU"].nunique() / total_skus * 100) if total_skus > 0 else 0
        understock_pct = (df_latest_valid[df_latest_valid["Stock_Status"] == "Understock"]["SKU"].nunique() / total_skus * 100) if total_skus > 0 else 0

        # Compute network health score
        health_score = 100
        if overstock_pct > 30: health_score -= 25
        elif overstock_pct > 15: health_score -= 10
        if median_doh and median_doh > 60: health_score -= 15
        elif median_doh and median_doh > 30: health_score -= 5
        if understock_pct > 15: health_score -= 15
        elif understock_pct > 5: health_score -= 5
        health_score = max(0, min(100, health_score))
        health_color = THEME["primary_green"] if health_score >= 75 else (THEME["gold"] if health_score >= 50 else THEME["bright_red"])
        health_label = "Healthy" if health_score >= 75 else ("Needs Attention" if health_score >= 50 else "Critical")

        # --- Styled KPI Cards ---
        doh_display = f"{median_doh:.1f}" if pd.notna(median_doh) else "N/A"
        doh_color = "green" if (pd.notna(median_doh) and median_doh <= 21) else ("yellow" if (pd.notna(median_doh) and median_doh <= 30) else "red")
        overstock_color = "green" if overstock_pct < 15 else ("yellow" if overstock_pct < 30 else "red")

        st.markdown(f"""
        <div class='kpi-row'>
            <div class='kpi-card blue'>
                <div class='kpi-label'>SKU Count</div>
                <div class='kpi-value'>{distinct_skus:,}</div>
                <div class='kpi-sub'>Latest week only (WK-{latest_wk:02d}, {latest_yr})</div>
            </div>
            <div class='kpi-card blue'>
                <div class='kpi-label'>Unique SKU</div>
                <div class='kpi-value'>{unique_materials:,}</div>
                <div class='kpi-sub'>Distinct material numbers</div>
            </div>
            <div class='kpi-card blue'>
                <div class='kpi-label'>Distribution Centers</div>
                <div class='kpi-value'>{total_dcs}</div>
                <div class='kpi-sub'>Active cold chain sites</div>
            </div>
            <div class='kpi-card {doh_color}'>
                <div class='kpi-label'>Median DOH</div>
                <div class='kpi-value'>{doh_display} <span style='font-size:14px;font-weight:500;color:#64748B;'>days</span></div>
                <div class='kpi-sub'>Target: 7–21 days</div>
            </div>
            <div class='kpi-card {overstock_color}'>
                <div class='kpi-label'>Overstock Rate</div>
                <div class='kpi-value'>{overstock_pct:.1f}%</div>
                <div class='kpi-sub'>SKUs above OUL</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # --- DC Utilization Map (full width) ---
        st.subheader("DC Utilization Map")
        st.caption("Bubble size and color indicate DC capacity utilization % | green (low) → red (high) | DC names labeled")
        try:
            df_util = run_query(QUERY_DC_UTILIZATION)
            # Filter map using DC names from QUERY_DC_UTILIZATION directly
            if selected_plants:
                # Build plant-to-dc_name mapping from utilization data + dc_options
                if not dc_options.empty:
                    plant_to_util_name = dict(zip(dc_options["val"], dc_options["label"]))
                    selected_util_names = [plant_to_util_name.get(p, "") for p in selected_plants]
                    df_util = df_util[df_util["dc_name"].isin(selected_util_names)]
            if not df_util.empty:
                df_util["size"] = df_util["utilization_pct"].clip(lower=25)
                fig = px.scatter_mapbox(
                    df_util, lat="lat", lon="lon",
                    size="size", color="utilization_pct",
                    text="dc_name",
                    color_continuous_scale=[
                        [0, "#059669"],
                        [0.5, "#D97706"],
                        [1, "#DC2626"]
                    ],
                    range_color=[30, 100],
                    hover_name="dc_name",
                    hover_data={"utilization_pct": True, "lat": False, "lon": False, "size": False},
                    size_max=28,
                    zoom=3.2,
                    center={"lat": 39.0, "lon": -98.0},
                    mapbox_style="carto-positron"
                )
                fig.update_traces(textposition="top center", textfont=dict(size=10, color="#1E293B"))
                fig.update_layout(
                    height=580, margin=dict(t=0, b=0, l=0, r=0),
                    coloraxis_colorbar=dict(
                        title="Util %", thickness=14, len=0.5,
                        bgcolor="rgba(255,255,255,0.8)", borderwidth=0,
                        tickfont=dict(color="#334155"), title_font=dict(color="#334155")
                    ),
                    font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=11, color="#1E293B")
                )
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("ℹ️ About DC Utilization Map", expanded=False):
                    st.markdown(info_card(
                        "DC Utilization Map", "🗺️",
                        [
                            {"label": "Formula", "content": "<span class='info-card-formula'>Utilization % = Occupied Capacity / Total Capacity × 100</span>"},
                            {"label": "Visual Encoding", "content":
                                "<span class='info-card-pill pill-green'>Bubble size</span> proportional to utilization %<br>"
                                "<span class='info-card-pill pill-green'>&lt; 50%</span>"
                                "<span class='info-card-pill pill-yellow'>50–75%</span>"
                                "<span class='info-card-pill pill-red'>&gt; 75%</span>"
                            },
                        ],
                        insight="DCs above 80% utilization are nearing capacity constraints and may need overflow planning or rebalancing."
                    ), unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not load utilization data: {e}")

        st.markdown("---")

        # --- Stock Status Distribution (latest week snapshot, excl. null-DOH) ---
        st.subheader("Stock Status Distribution")
        st.caption(f"Point-in-time snapshot: WK-{latest_wk:02d}, {latest_yr} · Excludes {null_doh_skus} no-forecast SKUs")
        col_pie, col_legend = st.columns([2, 1])
        df_stock = df_latest_valid.groupby("Stock_Status")["SKU"].nunique().reset_index(name="Count")
        total = df_stock["Count"].sum()
        df_stock["Percentage"] = (df_stock["Count"] / total * 100).round(1)

        with col_pie:
            fig = go.Figure(data=[go.Pie(
                labels=df_stock["Stock_Status"],
                values=df_stock["Count"],
                hole=0.55,
                marker=dict(
                    colors=[STOCK_COLORS.get(s, THEME["tan"]) for s in df_stock["Stock_Status"]],
                    line=dict(color="#FFFFFF", width=2.5)
                ),
                textinfo="percent",
                textfont=dict(size=14, color="#1E293B"),
                hovertemplate="<b>%{label}</b><br>SKUs: %{value:,}<br>Share: %{percent}<extra></extra>",
                pull=[0.03 if s == "Overstock" else 0 for s in df_stock["Stock_Status"]]
            )])
            fig.update_layout(
                height=380, margin=dict(t=20, b=20, l=20, r=20),
                plot_bgcolor="white", paper_bgcolor="white",
                showlegend=False,
                font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
                annotations=[dict(
                    text=f"<b>{total:,}</b><br><span style='font-size:12px;color:#64748B'>Total SKUs</span>",
                    x=0.5, y=0.5, font_size=20, font_color="#1E293B",
                    showarrow=False
                )]
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_legend:
            st.markdown("<br><br>", unsafe_allow_html=True)
            for _, row in df_stock.iterrows():
                color = STOCK_COLORS.get(row["Stock_Status"], THEME["tan"])
                st.markdown(
                    f"<div style='display:flex;align-items:center;margin-bottom:12px;'>"
                    f"<div style='width:14px;height:14px;border-radius:4px;background:{color};margin-right:10px;'></div>"
                    f"<div><span style='font-weight:600;color:#1E293B;'>{row['Stock_Status'].replace('_', ' ')}</span>"
                    f"<br><span style='font-size:13px;color:#64748B;'>{row['Count']:,} SKUs &middot; {row['Percentage']}%</span></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        with st.expander("ℹ️ About Stock Status Distribution", expanded=False):
            st.markdown(info_card(
                "Stock Status Distribution", "📦",
                [
                    {"label": "Classification Logic (per SKU per week)", "content":
                        "<table class='info-card-table'>"
                        "<tr><th>Status</th><th>Condition</th></tr>"
                        "<tr><td><span class='info-card-pill pill-red'>Overstock</span></td><td>Avg On-Hand Qty &gt; Order-Up-To Level</td></tr>"
                        "<tr><td><span class='info-card-pill pill-green'>Within Range</span></td><td>Safety Stock ≤ On-Hand ≤ Order-Up-To Level</td></tr>"
                        "<tr><td><span class='info-card-pill pill-yellow'>Understock</span></td><td>Avg On-Hand Qty &lt; Safety Stock Qty</td></tr>"
                        "</table>"
                    },
                    {"label": "Exclusion Note", "content":
                        "SKUs with <b>zero forecast demand</b> (deseasonalized_fcst_qty = 0) are excluded from this chart and KPI counts. "
                        "These SKUs have NULL DOH (division by zero) and would artificially inflate Overstock counts since any positive "
                        "on-hand inventory exceeds their Order-Up-To Level of 0. They are reported separately as 'No Forecast' SKUs in the subtitle."
                    },
                ],
                insight="High overstock % indicates excess inventory tying up capital. Understock signals potential service failures."
            ), unsafe_allow_html=True)

        st.markdown("---")

        # --- DOH Heatmap (styled table with conditional formatting) ---
        st.subheader("DOH Heatmap by DC & Week")
        col_heatmap, col_heatmap_legend = st.columns([5, 1])

        pivot_df = df_core.pivot_table(index="DC_Name", columns="week_label", values="DOH", aggfunc="median")
        if not pivot_df.empty:
            # Sort columns chronologically (ascending)
            sorted_cols = sorted(pivot_df.columns, key=lambda x: int(x.split("-")[1]) if "-" in x else 0)
            pivot_df = pivot_df[sorted_cols].round(1)

            # Conditional formatting function
            def doh_cell_style(val):
                if pd.isna(val):
                    return "background-color: #F8FAFC; color: #94A3B8;"
                elif val < 0 or val > 26:
                    return f"background-color: {THEME['bright_red']}; color: #FFFFFF; font-weight: 600;"
                elif val < 5 or val > 19:
                    return f"background-color: {THEME['gold']}; color: #000000; font-weight: 500;"
                else:
                    return f"background-color: {THEME['primary_green']}; color: #FFFFFF; font-weight: 500;"

            with col_heatmap:
                styled = pivot_df.style.map(doh_cell_style).format("{:.1f}", na_rep="—")
                styled = styled.set_properties(**{
                    "text-align": "center",
                    "font-size": "12px",
                    "padding": "6px 10px",
                    "border": "1px solid #E2E8F0"
                })
                styled = styled.set_table_styles([
                    {"selector": "th", "props": [
                        ("background-color", "#F8FAFC"), ("color", "#334155"),
                        ("font-weight", "600"), ("text-align", "center"),
                        ("padding", "8px 10px"), ("font-size", "11px"),
                        ("border", "1px solid #E2E8F0")
                    ]},
                    {"selector": "th.row_heading", "props": [
                        ("background-color", "#F8FAFC"), ("color", "#1E293B"),
                        ("font-weight", "600"), ("text-align", "left"),
                        ("padding", "8px 12px"), ("font-size", "12px"),
                        ("min-width", "120px"),
                        ("position", "sticky"), ("left", "0"), ("z-index", "2")
                    ]},
                    {"selector": "th.blank.level0", "props": [
                        ("position", "sticky"), ("left", "0"), ("z-index", "3"),
                        ("background-color", "#F8FAFC")
                    ]},
                    {"selector": "td", "props": [
                        ("white-space", "nowrap"), ("min-width", "65px")
                    ]},
                    {"selector": "table", "props": [
                        ("border-collapse", "collapse"), ("width", "max-content"),
                        ("font-family", "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif")
                    ]}
                ])
                st.markdown(
                    f"<div style='overflow-x:auto; max-height:700px; overflow-y:auto; border:1px solid #E2E8F0; border-radius:8px;'>"
                    f"{styled.to_html()}</div>",
                    unsafe_allow_html=True
                )

            with col_heatmap_legend:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Conditional Formatting**", unsafe_allow_html=True)
                legend_items = [
                    (THEME["bright_red"], "#FFF", "Red", "< 0 or > 26 days"),
                    (THEME["gold"], "#000", "Yellow", "< 5 or > 19 days"),
                    (THEME["primary_green"], "#FFF", "Green", "5 – 19 days"),
                ]
                for bg, fg, label, rule in legend_items:
                    st.markdown(
                        f"<div style='display:flex;align-items:center;margin-bottom:10px;'>"
                        f"<div style='width:36px;height:22px;border-radius:4px;background:{bg};margin-right:10px;"
                        f"display:flex;align-items:center;justify-content:center;'>"
                        f"<span style='font-size:9px;color:{fg};font-weight:600;'>DOH</span></div>"
                        f"<div><span style='font-weight:600;color:#1E293B;font-size:13px;'>{label}</span>"
                        f"<br><span style='font-size:12px;color:#64748B;'>{rule}</span></div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
        else:
            empty_state()

        with st.expander("ℹ️ About DOH Heatmap", expanded=False):
            st.markdown(info_card(
                "DOH Heatmap by DC & Week", "📊",
                [
                    {"label": "Formula", "content": "<span class='info-card-formula'>DOH = Avg On-Hand Qty / (Avg Deseasonalized Forecast / 28)</span><br>Estimates how many days current inventory lasts at the forecasted consumption rate."},
                    {"label": "Aggregation", "content": "Median DOH per DC per week across all SKUs."},
                    {"label": "Color Bands", "content":
                        "<span class='info-card-pill pill-red'>Red: &lt; 0 or &gt; 26d</span>"
                        "<span class='info-card-pill pill-yellow'>Yellow: &lt; 5 or &gt; 19d</span>"
                        "<span class='info-card-pill pill-green'>Green: 5–19d</span>"
                    },
                ],
                insight="Persistent red cells indicate structural misalignment between replenishment policies and actual demand."
            ), unsafe_allow_html=True)

        st.markdown("---")

        # --- PEL Section ---
        st.subheader("Pallet Equivalent Locations (PEL)")
        try:
            with st.spinner("Loading volume data for PEL trends..."):
                df_vol_pel_raw = run_query(QUERY_VOLUME_DATA)
            df_vol_pel = apply_mat_grp_sku_filter(apply_filters(df_vol_pel_raw), df_core)

            if not df_vol_pel.empty:
                # Map plant to DC_Name
                dc_name_map_pel = df_core[["plant", "DC_Name"]].drop_duplicates().set_index("plant")["DC_Name"]

                # --- PEL by DC (Latest Week) ---
                st.markdown("**PEL Utilization by DC (Latest Week)**")
                st.caption("Stacked: PEL Utilized vs PEL Available per DC")
                latest_vol_wk = df_vol_pel["wk"].max()
                df_pel_dc = df_vol_pel[df_vol_pel["wk"] == latest_vol_wk].groupby("plant").agg(
                    PEL=("PEL", "sum")
                ).reset_index()
                df_pel_dc["DC_Name"] = df_pel_dc["plant"].map(dc_name_map_pel)
                df_pel_dc = df_pel_dc.dropna(subset=["DC_Name"]).sort_values("PEL", ascending=False).head(15)

                if not df_pel_dc.empty:
                    # Join with DC utilization to compute PEL utilized vs available
                    df_util_pel = run_query(QUERY_DC_UTILIZATION)[["dc_name", "utilization_pct"]]
                    df_pel_dc = df_pel_dc.merge(df_util_pel, left_on="DC_Name", right_on="dc_name", how="left")
                    df_pel_dc["utilization_pct"] = df_pel_dc["utilization_pct"].fillna(50)  # default if no match
                    df_pel_dc["PEL_Utilized"] = (df_pel_dc["PEL"] * df_pel_dc["utilization_pct"] / 100).round(0)
                    df_pel_dc["PEL_Available"] = (df_pel_dc["PEL"] - df_pel_dc["PEL_Utilized"]).clip(lower=0).round(0)
                    df_pel_dc = df_pel_dc.sort_values("PEL", ascending=False)

                    fig_pel_dc = go.Figure()
                    fig_pel_dc.add_trace(go.Bar(
                        x=df_pel_dc["DC_Name"], y=df_pel_dc["PEL_Utilized"],
                        name="PEL Utilized", marker_color=COLOR_DANGER,
                        text=df_pel_dc["PEL_Utilized"].astype(int), textposition="inside"
                    ))
                    fig_pel_dc.add_trace(go.Bar(
                        x=df_pel_dc["DC_Name"], y=df_pel_dc["PEL_Available"],
                        name="PEL Available", marker_color=COLOR_SUCCESS,
                        text=df_pel_dc["PEL_Available"].astype(int), textposition="inside"
                    ))
                    fig_pel_dc = chart_layout(fig_pel_dc, 420)
                    fig_pel_dc.update_layout(
                        barmode="stack", xaxis_title="", yaxis_title="Pallet Equivalent Locations",
                        legend=dict(orientation="h", y=1.08)
                    )
                    st.plotly_chart(fig_pel_dc, use_container_width=True)

                with st.expander("ℹ️ About PEL by DC", expanded=False):
                    st.markdown(info_card(
                        "PEL Utilization by DC", "📦",
                        [
                            {"label": "Formula", "content":
                                "<span class='info-card-formula'>PEL = AVG(H × W × L) / 75,000</span><br>"
                                "<span class='info-card-formula'>PEL Utilized = Total PEL × Utilization %</span><br>"
                                "<span class='info-card-formula'>PEL Available = Total PEL − PEL Utilized</span>"
                            },
                            {"label": "Visual Encoding", "content":
                                "<span class='info-card-pill pill-red'>Red</span> PEL Utilized — space actively consumed<br>"
                                "<span class='info-card-pill pill-green'>Green</span> PEL Available — remaining capacity headroom"
                            },
                        ],
                        insight="DCs with minimal green (available) space are at capacity risk. Consider overflow planning or inventory reduction."
                    ), unsafe_allow_html=True)


            else:
                empty_state("No volume data available for PEL analysis.")
        except Exception as e:
            st.warning(f"Could not load PEL data: {e}")


# ======================== TAB 2: DEMAND VS FORECAST ========================
with tab2:
    st.header("Demand Vs Forecast")

    if df_core.empty:
        empty_state()
    else:
        st.markdown("""
        <div class='insight-box'>
        <b>About this module:</b> Compare actual billing demand (Total_Billing_Qty) against
        forecasted demand across DCs and weeks. Identify forecast accuracy gaps, WMAPE trends,
        and demand variability to improve planning.
        </div>
        """, unsafe_allow_html=True)

        # --- Load Billing Data (Actual Demand) ---
        try:
            with st.spinner("Loading billing data..."):
                df_billing_raw = run_query(QUERY_BILLING)
            df_billing = apply_filters(df_billing_raw)
        except Exception as e:
            st.warning(f"Could not load billing data: {e}")
            df_billing = pd.DataFrame()

        if not df_billing.empty:
            # --- Demand Variability (CV) — shown first ---
            dc_name_map = df_core[["plant", "DC_Name"]].drop_duplicates().set_index("plant")["DC_Name"]
            st.subheader("Demand Variability (Coefficient of Variation)")
            # Aggregate to weekly totals per plant first, then compute CV across weeks
            df_weekly_demand = df_billing.groupby(["plant", "wk"]).agg(
                Weekly_Demand=("Total_Billing_Qty", "sum")
            ).reset_index()
            df_cv = df_weekly_demand.groupby("plant").agg(
                Mean_Demand=("Weekly_Demand", "mean"),
                Std_Demand=("Weekly_Demand", "std")
            ).reset_index()
            df_cv["DC_Name"] = df_cv["plant"].map(dc_name_map)
            df_cv["CV"] = (df_cv["Std_Demand"] / df_cv["Mean_Demand"].clip(lower=1) * 100).round(1)
            df_cv = df_cv.sort_values("CV", ascending=False).head(20)

            if not df_cv.empty:
                fig = px.bar(
                    df_cv, x="DC_Name", y="CV", text="CV",
                    color="CV",
                    color_continuous_scale=[[0, THEME["primary_green"]], [0.5, THEME["gold"]], [1, THEME["bright_red"]]]
                )
                fig.update_traces(texttemplate="%{text}%")
                fig = chart_layout(fig, 400)
                fig.update_layout(xaxis_title="", yaxis_title="CV %", coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("ℹ️ About Demand Variability (CV)", expanded=False):
                st.markdown(info_card(
                    "Demand Variability (Coefficient of Variation)", "🌊",
                    [
                        {"label": "Formula", "content": "<span class='info-card-formula'>CV = (Std Dev of Weekly Demand / Mean Weekly Demand) × 100</span><br>Weekly Demand = SUM(Total_Billing_Qty) per plant per week. CV is then computed across weeks to measure temporal volatility."},
                        {"label": "Interpretation", "content":
                            "<span class='info-card-pill pill-green'>CV &lt; 50%</span> Stable, predictable demand<br>"
                            "<span class='info-card-pill pill-yellow'>CV 50–100%</span> Moderate variability, review safety stock<br>"
                            "<span class='info-card-pill pill-red'>CV &gt; 100%</span> Highly erratic, consider different forecasting methods"
                        },
                    ],
                    insight="DCs with high CV benefit from higher safety stock or more responsive replenishment policies."
                ), unsafe_allow_html=True)

            st.markdown("---")

            # --- Demand vs Forecast Comparison ---
            st.subheader("Weekly Actual Demand vs Forecast")

            # deseasonalized_fcst_for_period is already a weekly forecast
            df_core["Weekly_Forecast"] = df_core["Forecast_Demand"]

            # Aggregate billing (actual demand) by week
            df_actual_wk = df_billing.groupby(["week_label", "yr", "wk"]).agg(
                Actual_Demand=("Total_Billing_Qty", "sum")
            ).reset_index()

            # Aggregate prorated weekly forecast by week
            df_forecast_wk = df_core.groupby(["week_label", "yr", "wk"]).agg(
                Forecast_Demand=("Weekly_Forecast", "sum")
            ).reset_index()

            # Merge actual and forecast
            df_dvf = df_actual_wk.merge(df_forecast_wk, on=["week_label", "yr", "wk"], how="outer").sort_values(["yr", "wk"])

            if not df_dvf.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_dvf["week_label"], y=df_dvf["Actual_Demand"],
                    mode="lines+markers", name="Actual Demand (Billing)",
                    line=dict(color=COLOR_PRIMARY, width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=df_dvf["week_label"], y=df_dvf["Forecast_Demand"],
                    mode="lines+markers", name="Forecast Demand",
                    line=dict(color=COLOR_WARNING, width=2, dash="dash")
                ))
                fig = chart_layout(fig, 420)
                fig.update_layout(xaxis_title="Week", yaxis_title="Total Quantity", legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("ℹ️ About Demand vs Forecast", expanded=False):
                st.markdown(info_card(
                    "Weekly Actual Demand vs Forecast", "📈",
                    [
                        {"label": "Actual Demand", "content": "<span class='info-card-formula'>SUM(Total_Billing_Qty)</span> from <code>billing_transactions_hhd</code> aggregated by week."},
                        {"label": "Forecast Demand", "content": "<span class='info-card-formula'>deseasonalized_fcst_for_period</span> (weekly forecast from replenishment parameters)."},
                        {"label": "Data Sources", "content":
                            "<span class='info-card-pill pill-blue'>Actuals</span> billing_transactions_hhd (invoice qty)<br>"
                            "<span class='info-card-pill pill-blue'>Forecast</span> replenish_material_hist_hhd"
                        },
                    ],
                    insight="Persistent gaps indicate systematic bias — above = under-forecast, below = over-forecast."
                ), unsafe_allow_html=True)

            st.markdown("---")

            # --- Forecast Accuracy by DC (WMAPE) ---
            st.subheader("Forecast Error by DC (WMAPE)")
            col_wmape_chart, col_wmape_formula = st.columns([3, 1])

            # Aggregate billing by plant, then join with DC names
            df_actual_dc = df_billing.groupby("plant").agg(
                Total_Actual=("Total_Billing_Qty", "sum")
            ).reset_index()

            df_forecast_dc = df_core.groupby("plant").agg(
                Total_Forecast=("Weekly_Forecast", "sum")
            ).reset_index()

            df_acc = df_actual_dc.merge(df_forecast_dc, on="plant", how="inner")
            # Add DC names
            dc_name_map = df_core[["plant", "DC_Name"]].drop_duplicates().set_index("plant")["DC_Name"]
            df_acc["DC_Name"] = df_acc["plant"].map(dc_name_map)
            df_acc["WMAPE"] = (abs(df_acc["Total_Actual"] - df_acc["Total_Forecast"]) / df_acc[["Total_Actual", "Total_Forecast"]].max(axis=1).clip(lower=1) * 100).round(1)
            df_acc = df_acc.sort_values("WMAPE", ascending=False)

            selected_dc_drill = None
            with col_wmape_chart:
                if not df_acc.empty:
                    fig = px.bar(
                        df_acc, x="DC_Name", y="WMAPE", text="WMAPE",
                        color="WMAPE",
                        color_continuous_scale=[[0, THEME["primary_green"]], [0.5, THEME["gold"]], [1, THEME["bright_red"]]],
                        range_color=[0, df_acc["WMAPE"].quantile(0.95)]
                    )
                    fig.update_traces(texttemplate="%{text}%")
                    fig = chart_layout(fig, 420)
                    fig.update_layout(xaxis_title="", yaxis_title="WMAPE %", coloraxis_showscale=False)
                    # Enable click-to-drill: selecting a bar filters downstream charts
                    wmape_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="wmape_dc_select")

                    # Capture selected DC from bar click
                    selected_dc_drill = None
                    if wmape_event and wmape_event.selection and wmape_event.selection.points:
                        selected_dc_drill = wmape_event.selection.points[0].get("x")

                    if selected_dc_drill:
                        st.markdown(
                            f"<div style='display:flex;align-items:center;gap:10px;padding:8px 14px;"
                            f"background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;margin-top:8px;'>"
                            f"<span style='font-size:13px;color:#1E40AF;font-weight:600;'>"
                            f"🔍 Drilled into: {selected_dc_drill}</span>"
                            f"<span style='font-size:11px;color:#64748B;'>(click empty area to reset)</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

            with col_wmape_formula:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("\U0001F4D0 Formulas", expanded=False):
                    st.markdown(
                        "<div style='background:#F8FAFC;border-radius:6px;padding:12px;'>"
                        "<p style='font-size:12px;color:#334155;margin-bottom:10px;'>"
                        "<b>WMAPE</b><br>"
                        "<code style='background:#E2E8F0;padding:2px 6px;border-radius:4px;font-size:11px;'>"
                        "|Actual − Forecast| / Actual × 100</code></p>"
                        "<p style='font-size:12px;color:#334155;margin-bottom:10px;'>"
                        "<b>Bias %</b><br>"
                        "<code style='background:#E2E8F0;padding:2px 6px;border-radius:4px;font-size:11px;'>"
                        "(Forecast − Actual) / Actual × 100</code><br>"
                        "<span style='font-size:11px;color:#64748B;'>+ve = over-forecast, −ve = under-forecast</span></p>"
                        "<p style='font-size:12px;color:#334155;margin-bottom:10px;'>"
                        "<b>Absolute Error</b><br>"
                        "<code style='background:#E2E8F0;padding:2px 6px;border-radius:4px;font-size:11px;'>"
                        "|Actual − Forecast|</code></p>"
                        "<hr style='border:none;border-top:1px solid #E2E8F0;margin:12px 0;'>"
                        "<p style='font-size:11px;color:#64748B;margin:0;'>"
                        "<b>Actual</b> = Total_Billing_Qty<br>"
                        "<b>Forecast</b> = deseasonalized_fcst_for_period (weekly rate)</p>"
                        "</div>",
                        unsafe_allow_html=True
                    )

            # --- Apply drill-down filter from WMAPE chart selection ---
            df_billing_drill = df_billing.copy()
            df_core_drill = df_core.copy()
            if selected_dc_drill:
                drill_plant = dc_name_map[dc_name_map == selected_dc_drill].index[0] if selected_dc_drill in dc_name_map.values else None
                if drill_plant:
                    df_billing_drill = df_billing_drill[df_billing_drill["plant"] == drill_plant]
                    df_core_drill = df_core_drill[df_core_drill["plant"] == drill_plant]

            # --- Top 20 SKUs Impacting Forecast Accuracy ---
            drill_label = f" — {selected_dc_drill}" if selected_dc_drill else ""
            st.subheader(f"Top 20 SKUs Impacting Forecast Error{drill_label}")
            st.caption("SKUs with the highest WMAPE % (descending) | Minimum 100 units actual & forecast demand")

            # SKU-level actual demand from billing
            df_actual_sku = df_billing_drill.groupby(["SKU", "plant"]).agg(
                Actual_Qty=("Total_Billing_Qty", "sum")
            ).reset_index()

            # SKU-level forecast from core
            df_forecast_sku = df_core_drill.groupby(["SKU", "plant"]).agg(
                Forecast_Qty=("Forecast_Demand", "sum")
            ).reset_index()

            df_sku_acc = df_actual_sku.merge(df_forecast_sku, on=["SKU", "plant"], how="inner")
            df_sku_acc["DC_Name"] = df_sku_acc["plant"].map(dc_name_map)
            df_sku_acc["Abs_Error"] = abs(df_sku_acc["Actual_Qty"] - df_sku_acc["Forecast_Qty"])
            # Use MAX(Actual, Forecast) as denominator to bound WMAPE between 0-100%
            df_sku_acc["WMAPE"] = (df_sku_acc["Abs_Error"] / df_sku_acc[["Actual_Qty", "Forecast_Qty"]].max(axis=1).clip(lower=1) * 100).round(1)
            # Filter out low-volume SKUs and missing forecasts that inflate WMAPE unrealistically
            df_sku_acc_filtered = df_sku_acc[(df_sku_acc["Actual_Qty"] >= 100) & (df_sku_acc["Forecast_Qty"] >= 100)]
            df_sku_top = df_sku_acc_filtered.nlargest(20, "WMAPE")

            if not df_sku_top.empty:
                df_sku_top = df_sku_top.sort_values("WMAPE", ascending=False)
                fig = px.bar(
                    df_sku_top, x="SKU", y="WMAPE", text="WMAPE",
                    color="WMAPE",
                    color_continuous_scale=[[0, THEME["primary_green"]], [0.5, THEME["gold"]], [1, THEME["bright_red"]]],
                    range_color=[0, 100]
                )
                fig.update_traces(texttemplate="%{text}%")
                fig = chart_layout(fig, 450)
                fig.update_layout(
                    xaxis_title="", yaxis_title="WMAPE %",
                    xaxis_tickangle=-45,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("ℹ️ About Top 20 SKUs", expanded=False):
                st.markdown(info_card(
                    "Top 20 SKUs by WMAPE", "🎯",
                    [
                        {"label": "Ranking", "content": "Top 20 SKUs ordered descending by WMAPE %."},
                        {"label": "Formula", "content": "<span class='info-card-formula'>WMAPE = |Actual − Forecast| / MAX(Actual, Forecast) × 100</span>"},
                    ],
                    insight="These SKUs have the largest relative forecast error. Improving their forecasts yields the greatest accuracy gain."
                ), unsafe_allow_html=True)

            # --- Volume for Top 20 SKUs ---
            if not df_sku_top.empty:
                st.subheader("Volume for Top 20 Forecast-Impact SKUs")
                st.caption("Average cubic inches and PEL for the same SKUs with highest WMAPE")
                try:
                    df_vol_top20_raw = run_query(QUERY_VOLUME_DATA)
                    df_vol_top20 = apply_filters(df_vol_top20_raw)
                    top20_skus = df_sku_top["SKU"].tolist()
                    df_vol_top20 = df_vol_top20[df_vol_top20["SKU"].isin(top20_skus)]

                    if not df_vol_top20.empty:
                        df_vol_agg = df_vol_top20.groupby("SKU").agg(
                            Avg_Volume=("Total_Volume_Cubic_Inches", "mean"),
                            Avg_PEL=("PEL", "mean")
                        ).reset_index()
                        df_vol_agg = df_vol_agg.sort_values("Avg_Volume", ascending=False)

                        fig = px.bar(
                            df_vol_agg, x="SKU", y="Avg_Volume", text_auto=".0f",
                            color="Avg_PEL",
                            color_continuous_scale=[[0, "#DBEAFE"], [1, "#1D4ED8"]],
                            labels={"Avg_Volume": "Avg Volume (cu in)", "Avg_PEL": "Avg PEL"}
                        )
                        fig = chart_layout(fig, 420)
                        fig.update_layout(
                            xaxis_title="", yaxis_title="Avg Volume (Cubic Inches)",
                            xaxis_tickangle=-45, coloraxis_colorbar_title="PEL"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # --- KPI Table for Top 20 SKUs ---
                        st.markdown("**SKU Detail Table**")
                        df_doh_sku = df_core_drill.groupby("SKU").agg(Median_DOH=("DOH", "median")).reset_index()
                        df_kpi_table = df_sku_top[["SKU", "DC_Name", "WMAPE"]].copy()
                        df_kpi_table = df_kpi_table.merge(df_vol_agg[["SKU", "Avg_Volume", "Avg_PEL"]], on="SKU", how="left")
                        df_kpi_table = df_kpi_table.merge(df_doh_sku, on="SKU", how="left")

                        # Fetch material descriptions for these SKUs
                        try:
                            item_numbers = [sku.split("-", 1)[1] for sku in df_kpi_table["SKU"].tolist() if "-" in sku]
                            if item_numbers:
                                items_sql = ", ".join([f"'{i}'" for i in item_numbers])
                                df_desc = run_query(
                                    f"SELECT DISTINCT mat_nbr8, mat_desc FROM adh_genpro_use2_prd.s_products.material_details_consolidated_hhd WHERE mat_nbr8 IN ({items_sql})"
                                )
                                df_kpi_table["Item"] = df_kpi_table["SKU"].str.split("-", n=1).str[1]
                                df_kpi_table = df_kpi_table.merge(
                                    df_desc.rename(columns={"mat_nbr8": "Item", "mat_desc": "Description"}),
                                    on="Item", how="left"
                                )
                                df_kpi_table.drop(columns=["Item"], inplace=True)
                            else:
                                df_kpi_table["Description"] = "—"
                        except Exception:
                            df_kpi_table["Description"] = "—"

                        # Format and display
                        display_cols = ["SKU", "Description", "DC_Name", "Avg_Volume", "Avg_PEL", "Median_DOH", "WMAPE"]
                        df_display = df_kpi_table[[c for c in display_cols if c in df_kpi_table.columns]].copy()
                        col_rename = {"DC_Name": "DC", "Avg_Volume": "Volume (cu in)", "Avg_PEL": "PEL", "Median_DOH": "DOH (days)", "WMAPE": "WMAPE %"}
                        df_display.rename(columns=col_rename, inplace=True)
                        df_display = df_display.round(1)
                        st.dataframe(df_display, use_container_width=True, hide_index=True)
                    else:
                        st.info("No volume data available for the top 20 WMAPE SKUs.")
                except Exception as e:
                    st.warning(f"Could not load volume data for top SKUs: {e}")

        else:
            empty_state("No billing data available for current filters.")


# ======================== TAB 3: INVENTORY ========================
with tab3:
    st.header("Inventory")
    st.caption("Broader categorization across the full filtered time window (all weeks)")

    if df_core.empty:
        empty_state()
    else:
        st.markdown("""
        <div class='insight-box'>
        <b>Key Insight:</b> Capacity is driven by three primary factors: Days on Hand (DOH),
        replenishment policy thresholds (Order-Up-To levels), and physical volume per SKU.
        </div>
        """, unsafe_allow_html=True)

        # --- DOH Category Breakdown by DC ---
        st.subheader("DOH Category Breakdown by DC")
        # Categorize ALL SKUs including null-DOH (Dead Stock)
        df_cat = df_core.copy()
        # Dead Stock: Demand = 0 AND Forecast = 0 (DOH is NULL)
        # Remaining: bin by DOH aligned with heatmap thresholds
        conditions = [
            df_cat["DOH"].isna(),                              # Dead Stock
            df_cat["DOH"] > 26,                                # Very Slow Moving
            (df_cat["DOH"] > 19) & (df_cat["DOH"] <= 26),     # Slow Moving
            (df_cat["DOH"] >= 5) & (df_cat["DOH"] <= 19),     # Normal
            (df_cat["DOH"] >= 0) & (df_cat["DOH"] < 5),       # Fast Moving
        ]
        choices = ["🪦 Dead Stock", "🐢 Very Slow Moving", "🚶 Slow Moving", "✅ Normal", "⚡ Fast Moving"]
        df_cat["DOH_Category"] = np.select(conditions, choices, default="🪦 Dead Stock")
        # Exclude extreme DOH outliers (>365) from non-dead categories
        df_cat = df_cat[~((df_cat["DOH"].notna()) & (df_cat["DOH"] > 365))]

        df_dc_cat = df_cat.groupby(["DC_Name", "DOH_Category"], observed=True)["SKU"].nunique().reset_index(name="SKU_Count")
        if not df_dc_cat.empty:
            # Compute percentage for 100% stacked bar
            df_dc_total = df_dc_cat.groupby("DC_Name")["SKU_Count"].sum().reset_index(name="Total")
            df_dc_cat = df_dc_cat.merge(df_dc_total, on="DC_Name")
            df_dc_cat["Pct"] = (df_dc_cat["SKU_Count"] / df_dc_cat["Total"] * 100).round(1)

            # Sort DCs by Very Slow Moving % (worst first)
            slow_pct = df_dc_cat[df_dc_cat["DOH_Category"] == "🐢 Very Slow Moving"].set_index("DC_Name")["Pct"]
            dc_order = slow_pct.sort_values(ascending=False).index.tolist()
            # Include DCs with no very slow at the end
            all_dcs = df_dc_cat["DC_Name"].unique().tolist()
            dc_order += [dc for dc in all_dcs if dc not in dc_order]

            category_order = ["🪦 Dead Stock", "🐢 Very Slow Moving", "🚶 Slow Moving", "✅ Normal", "⚡ Fast Moving"]
            fig = go.Figure()
            for cat in category_order:
                df_c = df_dc_cat[df_dc_cat["DOH_Category"] == cat].set_index("DC_Name")[["Pct", "SKU_Count"]].reindex(dc_order).fillna(0)
                fig.add_trace(go.Bar(
                    x=dc_order, y=df_c["Pct"].values,
                    name=cat, marker_color=DOH_COLORS.get(cat, "#94A3B8"),
                    hovertemplate="<b>%{x}</b><br>" + cat + ": %{y:.1f}%<br>SKUs: %{customdata}<extra></extra>",
                    customdata=df_c["SKU_Count"].astype(int).values
                ))
            fig = chart_layout(fig, 480)
            fig.update_layout(
                barmode="stack", xaxis_title="", yaxis_title="% of SKUs",
                xaxis_tickangle=-40,
                yaxis=dict(range=[0, 100], dtick=25, ticksuffix="%"),
                legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center",
                    font=dict(size=11), traceorder="normal"),
                bargap=0.15
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary metrics row
            distinct_skus_cat = df_cat["SKU"].nunique()  # unique SKUs regardless of DC
            network_dead = df_dc_cat[df_dc_cat["DOH_Category"] == "🪦 Dead Stock"]["SKU_Count"].sum()
            network_normal = df_dc_cat[df_dc_cat["DOH_Category"] == "✅ Normal"]["SKU_Count"].sum()
            network_very_slow = df_dc_cat[df_dc_cat["DOH_Category"] == "🐢 Very Slow Moving"]["SKU_Count"].sum()
            network_total = df_dc_cat["SKU_Count"].sum()
            normal_pct = (network_normal / network_total * 100) if network_total > 0 else 0
            very_slow_pct = (network_very_slow / network_total * 100) if network_total > 0 else 0
            dead_pct = (network_dead / network_total * 100) if network_total > 0 else 0
            col_m0, col_m1, col_m2, col_m3 = st.columns(4)
            col_m0.metric("SKU Count", f"{distinct_skus_cat:,}", help="Counts unique SKUs across all weeks in the filtered range (not just the latest week). This may be higher than Network Overview because it includes SKUs that were active in earlier weeks but have since been depleted or discontinued.")
            col_m1.metric("✅ Normal (5-19d)", f"{normal_pct:.1f}%")
            col_m2.metric("🐢 Very Slow (>26d)", f"{very_slow_pct:.1f}%")
            col_m3.metric("🪦 Dead Stock", f"{dead_pct:.1f}%")

        with st.expander("ℹ️ About DOH Categories", expanded=False):
            st.markdown(info_card(
                "DOH Category Breakdown", "📊",
                [
                    {"label": "DOH Bins (aligned with DOH Heatmap)", "content":
                        "<table class='info-card-table'>"
                        "<tr><th>Category</th><th>Condition</th><th>Color</th><th>Rationale</th></tr>"
                        "<tr><td>🪦 Dead Stock</td><td>Demand = 0 &amp; Forecast = 0</td><td>Grey</td><td>No movement, no expectation — flagged separately</td></tr>"
                        "<tr><td>🐢 Very Slow Moving</td><td>DOH &gt; 26 days</td><td><span class='info-card-pill pill-red'>Red</span></td><td>Critically excess inventory</td></tr>"
                        "<tr><td>🚶 Slow Moving</td><td>DOH 19–26 days</td><td><span class='info-card-pill pill-yellow'>Yellow</span></td><td>Elevated stock, warning zone</td></tr>"
                        "<tr><td>✅ Normal</td><td>DOH 5–19 days</td><td><span class='info-card-pill pill-green'>Green</span></td><td>Healthy range, aligned with target</td></tr>"
                        "<tr><td>⚡ Fast Moving</td><td>DOH 0–5 days</td><td><span class='info-card-pill pill-yellow'>Yellow</span></td><td>Low stock, replenishment needed soon</td></tr>"
                        "</table>"
                    },
                    {"label": "Dead Stock Logic", "content":
                        "Dead Stock SKUs have <b>zero forecast demand</b> (deseasonalized_fcst_qty = 0), resulting in NULL DOH. "
                        "These items have physical inventory but no demand signal — they are potential candidates for "
                        "liquidation, redistribution, or write-off."
                    },
                ],
                insight="DOH thresholds match the DOH Heatmap on the Network Overview tab. Dead Stock is identified by zero demand/forecast rather than DOH duration."
            ), unsafe_allow_html=True)

        st.markdown("---")

        # --- On-Hand Qty vs Order-Up-To Level by DC (Per SKU Average) ---
        st.subheader("On-Hand Qty vs Order-Up-To Level by DC")
        st.caption("Average per-SKU inventory vs. replenishment policy ceiling per DC")

        # DOH Category filter
        category_order_filter = ["🪦 Dead Stock", "🐢 Very Slow Moving", "🚶 Slow Moving", "✅ Normal", "⚡ Fast Moving"]
        selected_doh_cats = st.multiselect(
            "Filter by DOH Category", options=category_order_filter,
            default=None, placeholder="All Categories", key="oh_oul_doh_filter"
        )
        df_oh_source = df_cat[df_cat["DOH_Category"].isin(selected_doh_cats)] if selected_doh_cats else df_cat

        # Mean per SKU across all SKU-weeks at each DC
        df_oh_vs_oul = df_oh_source.groupby("DC_Name").agg(
            Avg_OH_Qty=("Avg_OH_Qty", "mean"),
            Avg_Order_Up_To_Level_Qty=("Avg_Order_Up_To_Level_Qty", "mean")
        ).reset_index().sort_values("Avg_OH_Qty", ascending=False)

        if not df_oh_vs_oul.empty:
            fig_oh_oul = go.Figure()
            fig_oh_oul.add_trace(go.Bar(
                x=df_oh_vs_oul["DC_Name"], y=df_oh_vs_oul["Avg_OH_Qty"],
                name="Avg On-Hand Qty (per SKU)", marker_color=COLOR_PRIMARY,
                text=df_oh_vs_oul["Avg_OH_Qty"].round(0).astype(int),
                textposition="outside", textfont=dict(size=10)
            ))
            fig_oh_oul.add_trace(go.Bar(
                x=df_oh_vs_oul["DC_Name"], y=df_oh_vs_oul["Avg_Order_Up_To_Level_Qty"],
                name="Avg Order-Up-To Level (per SKU)", marker_color=COLOR_DANGER,
                text=df_oh_vs_oul["Avg_Order_Up_To_Level_Qty"].round(0).astype(int),
                textposition="outside", textfont=dict(size=10)
            ))
            fig_oh_oul = chart_layout(fig_oh_oul, 480)
            fig_oh_oul.update_layout(
                barmode="group", xaxis_title="", yaxis_title="Quantity (Eaches) — Per SKU Avg",
                xaxis_tickangle=-40, bargap=0.2, bargroupgap=0.05,
                legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center")
            )
            st.plotly_chart(fig_oh_oul, use_container_width=True)

            with st.expander("ℹ️ About On-Hand vs Order-Up-To Level", expanded=False):
                st.markdown(info_card(
                    "On-Hand Qty vs Order-Up-To Level", "📊",
                    [
                        {"label": "Aggregation", "content":
                            "<span class='info-card-formula'>MEAN(Avg_OH_Qty) per DC</span> — average on-hand inventory per SKU across all weeks<br>"
                            "<span class='info-card-formula'>MEAN(Avg_Order_Up_To_Level_Qty) per DC</span> — average policy ceiling per SKU"
                        },
                        {"label": "Metrics", "content":
                            "<span class='info-card-pill pill-blue'>Blue bar</span> Avg On-Hand Quantity per SKU at each DC<br>"
                            "<span class='info-card-pill pill-red'>Red bar</span> Avg Order-Up-To Level per SKU at each DC (replenishment ceiling)"
                        },
                        {"label": "Interpretation", "content":
                            "DCs where the blue bar exceeds the red bar have per-SKU overstock on average — "
                            "typical items carry more inventory than their policy target. Use the DOH Category filter "
                            "to isolate how specific segments (e.g. Very Slow Moving) compare against targets."
                        },
                    ],
                    insight="Filter by DOH Category to see how problematic inventory segments compare against their policy thresholds."
                ), unsafe_allow_html=True)
        else:
            empty_state()

        # --- Weekly Trend (linked to DOH Category filter above) ---
        st.subheader("Weekly Trend")
        doh_filter_label = ", ".join(selected_doh_cats) if selected_doh_cats else "All Categories"
        st.caption(f"Weekly average on-hand qty vs order-up-to level | Filter: {doh_filter_label}")

        df_trend_source = df_cat[df_cat["DOH_Category"].isin(selected_doh_cats)] if selected_doh_cats else df_cat
        df_trend_wk = df_trend_source.groupby(["week_label", "yr", "wk"]).agg(
            Avg_OH_Qty=("Avg_OH_Qty", "mean"),
            Avg_OUL=("Avg_Order_Up_To_Level_Qty", "mean")
        ).reset_index().sort_values(["yr", "wk"])

        if not df_trend_wk.empty:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=df_trend_wk["week_label"], y=df_trend_wk["Avg_OH_Qty"],
                mode="lines+markers", name="Avg On-Hand Qty",
                line=dict(color=COLOR_PRIMARY, width=2.5),
                marker=dict(size=6)
            ))
            fig_trend.add_trace(go.Scatter(
                x=df_trend_wk["week_label"], y=df_trend_wk["Avg_OUL"],
                mode="lines+markers", name="Avg Order-Up-To Level",
                line=dict(color=COLOR_DANGER, width=2.5, dash="dash"),
                marker=dict(size=6, symbol="diamond")
            ))
            fig_trend = chart_layout(fig_trend, 420)
            fig_trend.update_layout(
                xaxis_title="Week", yaxis_title="Quantity (Eaches) — Per SKU Avg",
                legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center")
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            with st.expander("ℹ️ About Weekly Trend", expanded=False):
                st.markdown(info_card(
                    "Weekly On-Hand Qty Trend", "📈",
                    [
                        {"label": "Aggregation", "content":
                            "<span class='info-card-formula'>MEAN(Avg_OH_Qty) per week</span> — average on-hand per SKU across all DCs for each week<br>"
                            "<span class='info-card-formula'>MEAN(Avg_Order_Up_To_Level_Qty) per week</span> — average policy ceiling per SKU"
                        },
                        {"label": "Filter Interaction", "content":
                            "This chart respects the <b>Filter by DOH Category</b> selector above. "
                            "Select a category (e.g. Very Slow Moving) to see how that segment's inventory "
                            "trends over time relative to its policy target."
                        },
                    ],
                    insight="A widening gap between the blue and red lines over time indicates growing overstock or understock pressure."
                ), unsafe_allow_html=True)
        else:
            empty_state("No data available for selected DOH categories.")

        st.markdown("---")

        # --- Top 10 Overstock SKU Analysis ---
        st.subheader("Top 10 Overstock SKU Analysis")
        overstock_filter_note = f"Filter: {', '.join(selected_doh_cats)}" if selected_doh_cats else "Filter: All Categories"
        st.caption(f"SKUs with highest excess inventory above Order-Up-To Level | {overstock_filter_note}")

        df_excess_source = df_cat[df_cat["DOH_Category"].isin(selected_doh_cats)].copy() if selected_doh_cats else df_cat.copy()
        if not df_excess_source.empty:
            df_excess = df_excess_source.groupby("SKU").agg(
                Avg_OH_Qty=("Avg_OH_Qty", "mean"),
                Avg_OUL=("Avg_Order_Up_To_Level_Qty", "mean"),
                Median_DOH=("DOH", "median")
            ).reset_index()
            df_excess["Avg_OH_Qty"] = df_excess["Avg_OH_Qty"].round(0).astype(int)
            df_excess["Avg_OUL"] = df_excess["Avg_OUL"].round(0).astype(int)
            # Filter: only SKUs where OH > OUL (overstock)
            df_excess = df_excess[df_excess["Avg_OH_Qty"] > df_excess["Avg_OUL"]]
            df_excess["Excess_Qty"] = (df_excess["Avg_OH_Qty"] - df_excess["Avg_OUL"]).astype(int)
            df_excess["Median_DOH"] = df_excess["Median_DOH"].fillna(0).round(1)

            df_excess_top = df_excess.nlargest(10, "Excess_Qty").sort_values("Excess_Qty", ascending=False)

            if not df_excess_top.empty:
                # Bar chart
                text_fmt = df_excess_top["Excess_Qty"].apply(lambda x: f"{x:,.0f}")

                fig_excess = go.Figure()
                fig_excess.add_trace(go.Bar(
                    x=df_excess_top["SKU"],
                    y=df_excess_top["Excess_Qty"],
                    marker=dict(
                        color=df_excess_top["Excess_Qty"],
                        colorscale=[[0, THEME["gold"]], [1, THEME["bright_red"]]],
                        cmin=df_excess_top["Excess_Qty"].min(),
                        cmax=df_excess_top["Excess_Qty"].max()
                    ),
                    text=text_fmt,
                    textposition="outside",
                    customdata=np.stack([
                        df_excess_top["Avg_OH_Qty"],
                        df_excess_top["Avg_OUL"],
                        df_excess_top["Excess_Qty"],
                        df_excess_top["Median_DOH"]
                    ], axis=-1),
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "On-Hand Qty: %{customdata[0]:,.0f}<br>"
                        "Order-Up-To Level: %{customdata[1]:,.0f}<br>"
                        "Excess Qty: %{customdata[2]:,.0f}<br>"
                        "DOH: %{customdata[3]:.1f} days"
                        "<extra></extra>"
                    )
                ))
                fig_excess = chart_layout(fig_excess, 450)
                fig_excess.update_layout(
                    xaxis_title="", yaxis_title="Excess Quantity (Eaches)",
                    xaxis_tickangle=-40
                )
                st.plotly_chart(fig_excess, use_container_width=True)

                # Detail table
                with st.expander("📋 View Top 10 Overstock Detail Table", expanded=False):
                    df_table_display = df_excess_top[["SKU", "Avg_OH_Qty", "Avg_OUL", "Excess_Qty", "Median_DOH"]].copy()
                    df_table_display.rename(columns={
                        "Avg_OH_Qty": "Avg On-Hand Qty",
                        "Avg_OUL": "Avg Order-Up-To Level",
                        "Excess_Qty": "Excess Quantity",
                        "Median_DOH": "DOH (days)"
                    }, inplace=True)
                    st.dataframe(df_table_display, use_container_width=True, hide_index=True)

                with st.expander("ℹ️ About Top 10 Overstock SKU Analysis", expanded=False):
                    st.markdown(info_card(
                        "Top 10 Overstock SKU Analysis", "📦",
                        [
                            {"label": "Formula", "content":
                                "<span class='info-card-formula'>Excess Quantity = Avg On-Hand Qty − Avg Order-Up-To Level</span>"
                            },
                            {"label": "Filter", "content": "Only SKUs where Avg On-Hand Qty &gt; Avg Order-Up-To Level (physical overstock condition)."},
                            {"label": "Ranking", "content": "Top 10 SKUs with the highest Excess Quantity, ranked in descending order."},
                        ],
                        insight="SKUs at the top of this list contribute the most to physical overstock. Prioritize these for redistribution, Order-Up-To level review, or demand re-evaluation."
                    ), unsafe_allow_html=True)
                # --- PEL vs DOH for Top 10 Overstock SKUs ---
                st.subheader("Top 10 Overstock SKUs — PEL vs DOH")
                st.caption("Bar = Avg PEL (ordered by PEL descending) | Line = Median DOH | Hover for volume")
                try:
                    top10_skus = df_excess_top["SKU"].tolist()
                    df_vol_top10_raw = run_query(QUERY_VOLUME_DATA)
                    df_vol_top10 = apply_filters(df_vol_top10_raw)
                    df_vol_top10 = df_vol_top10[df_vol_top10["SKU"].isin(top10_skus)]

                    if not df_vol_top10.empty:
                        df_vol_agg_top10 = df_vol_top10.groupby("SKU").agg(
                            Avg_PEL=("PEL", "mean"),
                            Avg_Volume=("Total_Volume_Cubic_Inches", "mean")
                        ).reset_index()
                        # Merge DOH from excess data
                        df_pel_doh = df_vol_agg_top10.merge(
                            df_excess_top[["SKU", "Median_DOH"]], on="SKU", how="inner"
                        )
                        df_pel_doh["Avg_PEL"] = df_pel_doh["Avg_PEL"].round(2)
                        df_pel_doh["Avg_Volume"] = df_pel_doh["Avg_Volume"].round(0).astype(int)
                        df_pel_doh = df_pel_doh.sort_values("Avg_PEL", ascending=False)

                        if not df_pel_doh.empty:
                            fig_pel_doh = go.Figure()
                            # Bar: PEL
                            fig_pel_doh.add_trace(go.Bar(
                                x=df_pel_doh["SKU"],
                                y=df_pel_doh["Avg_PEL"],
                                name="Avg PEL",
                                marker_color=COLOR_PRIMARY,
                                text=df_pel_doh["Avg_PEL"].apply(lambda x: f"{x:.2f}"),
                                textposition="outside",
                                customdata=np.stack([
                                    df_pel_doh["Avg_Volume"],
                                    df_pel_doh["Median_DOH"],
                                    df_pel_doh["Avg_PEL"]
                                ], axis=-1),
                                hovertemplate=(
                                    "<b>%{x}</b><br>"
                                    "PEL: %{customdata[2]:.2f}<br>"
                                    "Volume: %{customdata[0]:,.0f} cu in<br>"
                                    "DOH: %{customdata[1]:.1f} days"
                                    "<extra></extra>"
                                ),
                                yaxis="y"
                            ))
                            # Line: DOH on secondary axis
                            fig_pel_doh.add_trace(go.Scatter(
                                x=df_pel_doh["SKU"],
                                y=df_pel_doh["Median_DOH"],
                                name="Median DOH",
                                mode="lines+markers",
                                line=dict(color=COLOR_DANGER, width=2.5),
                                marker=dict(size=8, symbol="diamond"),
                                customdata=np.stack([
                                    df_pel_doh["Avg_Volume"],
                                    df_pel_doh["Median_DOH"],
                                    df_pel_doh["Avg_PEL"]
                                ], axis=-1),
                                hovertemplate=(
                                    "<b>%{x}</b><br>"
                                    "DOH: %{customdata[1]:.1f} days<br>"
                                    "PEL: %{customdata[2]:.2f}<br>"
                                    "Volume: %{customdata[0]:,.0f} cu in"
                                    "<extra></extra>"
                                ),
                                yaxis="y2"
                            ))
                            fig_pel_doh = chart_layout(fig_pel_doh, 450)
                            fig_pel_doh.update_layout(
                                xaxis_title="", xaxis_tickangle=-40,
                                yaxis=dict(title="Avg PEL", side="left"),
                                yaxis2=dict(title="Median DOH (days)", side="right", overlaying="y", showgrid=False),
                                legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                                barmode="group"
                            )
                            st.plotly_chart(fig_pel_doh, use_container_width=True)

                            with st.expander("ℹ️ About PEL vs DOH Chart", expanded=False):
                                st.markdown(info_card(
                                    "Top 10 Overstock SKUs — PEL vs DOH", "📦",
                                    [
                                        {"label": "Axes", "content":
                                            "<span class='info-card-pill pill-blue'>Bar (left axis)</span> Avg PEL = AVG(H × W × L) / 75,000 cu in<br>"
                                            "<span class='info-card-pill pill-red'>Line (right axis)</span> Median DOH (days)"
                                        },
                                        {"label": "Tooltip", "content": "Volume (cubic inches), PEL, and DOH for each SKU."},
                                        {"label": "Sort Order", "content": "SKUs ordered by Avg PEL descending (highest space consumers first)."},
                                    ],
                                    insight="SKUs with both high PEL and high DOH are the biggest capacity offenders — they consume the most space AND sit the longest. Prioritize these for immediate action."
                                ), unsafe_allow_html=True)
                        else:
                            st.info("No volume/PEL data found for the top 10 overstock SKUs.")
                    else:
                        st.info("No volume data available for the top 10 overstock SKUs.")
                except Exception as e:
                    st.warning(f"Could not load volume data for PEL vs DOH chart: {e}")

            else:
                empty_state("No overstock SKUs found (all SKUs are within or below Order-Up-To Level).")
        else:
            empty_state("No data available for selected DOH categories.")

        st.markdown("---")

        # --- SKU Detail Table (linked to DOH Category filter) ---
        st.subheader("SKU Detail")
        sku_filter_note = f"Showing: {', '.join(selected_doh_cats)}" if selected_doh_cats else "Showing: All Categories"
        st.caption(f"Overstock SKUs — Top 10 highlighted, then remaining overstock | {sku_filter_note}")
        df_sku_detail_source = df_cat[df_cat["DOH_Category"].isin(selected_doh_cats)].copy() if selected_doh_cats else df_cat.copy()
        if not df_sku_detail_source.empty:
            df_sku_detail = df_sku_detail_source.groupby("SKU").agg(
                Avg_OH_Qty=("Avg_OH_Qty", "mean"),
                DOH=("DOH", "median"),
                Forecast_Demand=("Forecast_Demand", "mean"),
                Avg_OUL=("Avg_Order_Up_To_Level_Qty", "mean"),
                Stock_Status=("Stock_Status", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
            ).reset_index()
            df_sku_detail["Avg_OH_Qty"] = df_sku_detail["Avg_OH_Qty"].round(0).astype(int)
            df_sku_detail["DOH"] = df_sku_detail["DOH"].round(1)
            df_sku_detail["Forecast_Demand"] = df_sku_detail["Forecast_Demand"].round(2)
            df_sku_detail["Avg_OUL"] = df_sku_detail["Avg_OUL"].round(0).astype(int)

            # Filter to overstock SKUs only (On-Hand > OUL)
            df_sku_overstock = df_sku_detail[df_sku_detail["Avg_OH_Qty"] > df_sku_detail["Avg_OUL"]].copy()
            df_sku_overstock["Excess_Qty"] = (df_sku_overstock["Avg_OH_Qty"] - df_sku_overstock["Avg_OUL"]).astype(int)

            if not df_sku_overstock.empty:
                # Identify top 10 by Excess Qty
                top10_detail_skus = set(df_sku_overstock.nlargest(10, "Excess_Qty")["SKU"])
                df_sku_overstock["Rank"] = df_sku_overstock["SKU"].apply(
                    lambda s: "⭐ Top 10" if s in top10_detail_skus else ""
                )
                # Sort: top 10 first (by Excess_Qty desc), then remaining (by Excess_Qty desc)
                df_sku_overstock["_is_top10"] = df_sku_overstock["SKU"].isin(top10_detail_skus)
                df_sku_overstock = df_sku_overstock.sort_values(
                    ["_is_top10", "Excess_Qty"], ascending=[False, False]
                )
                df_sku_overstock.drop(columns=["_is_top10"], inplace=True)

                # Rename and display
                df_sku_overstock.rename(columns={
                    "Avg_OH_Qty": "On-Hand Qty",
                    "DOH": "DOH (days)", "Forecast_Demand": "Weekly Forecast",
                    "Avg_OUL": "Order-Up-To Level", "Stock_Status": "Status",
                    "Excess_Qty": "Excess Qty"
                }, inplace=True)
                display_cols = ["Rank", "SKU", "On-Hand Qty", "Order-Up-To Level", "Excess Qty", "DOH (days)", "Weekly Forecast", "Status"]
                st.dataframe(df_sku_overstock[display_cols], use_container_width=True, hide_index=True)
                st.caption(f"Total overstock SKUs: {len(df_sku_overstock):,} | Top 10 marked with ⭐")
            else:
                empty_state("No overstock SKUs found for selected DOH categories.")
        else:
            empty_state("No SKUs found for selected DOH categories.")

        st.markdown("---")

        # --- SKU Count by Material Group (linked to DOH Category filter) ---
        st.subheader("SKU Count by Material Group")
        mat_grp_filter_label = ", ".join(selected_doh_cats) if selected_doh_cats else "All Categories"
        st.caption(f"Unique SKU count per material group | Filter: {mat_grp_filter_label}")
        MAT_GRP_LABELS = {
            "BRX": "Brand Rx", "GRX": "Generic Rx", "OTC": "OTC Brand",
            "OTG": "OTC Generic", "MSU": "Medical/Surgical", "GMR": "Generic Medical",
            "HBC": "Health & Beauty", "HBG": "H&B Generic", "SSU": "Surgical Supply",
            "HHC": "Home Health"
        }
        # Use DOH Category filter if active, otherwise show all categories
        if selected_doh_cats:
            df_mat_grp = df_cat[df_cat["DOH_Category"].isin(selected_doh_cats)].copy()
        else:
            df_mat_grp = df_cat.copy()
        df_mat_grp["Group_Description"] = df_mat_grp["mat_grp"].map(lambda x: MAT_GRP_LABELS.get(x, x) if pd.notna(x) else "Unknown")
        df_mat_summary = df_mat_grp.groupby(["mat_grp", "Group_Description"])["SKU"].nunique().reset_index(name="SKU_Count")
        df_mat_summary = df_mat_summary.sort_values("SKU_Count", ascending=False)
        df_mat_summary.rename(columns={"mat_grp": "Material Group", "Group_Description": "Description", "SKU_Count": "SKU Count"}, inplace=True)
        st.dataframe(df_mat_summary, use_container_width=True, hide_index=True)


# ======================== TAB 4: README ========================
with tab4:
    st.header("Readme")

    # --- Features Section ---
    st.subheader("Features")
    st.markdown("""
    1. **Sidebar Filters** — DC, Material Group, Year, Month (synced with Year), Week (synced with Month)
    2. **Network Overview Tab** — KPI cards (SKU Count, Unique SKU, DCs, Median DOH, Overstock Rate), Network Health Score
    3. **DC Utilization Map** — Geo bubble map with utilization % color gradient, responds to DC filter
    4. **Stock Status Distribution** — Donut chart (Overstock / Within Range / Understock) with legend
    5. **DOH Heatmap** — Pivot table by DC × Week with conditional formatting (Red/Yellow/Green bands)
    6. **PEL Utilization by DC** — Stacked bar (PEL Utilized vs PEL Available) using utilization % split
    7. **Demand vs Forecast (DFIO) Tab** — Weekly Actual Demand vs Forecast line chart
    8. **Forecast Error by DC (WMAPE)** — Bar chart with click-to-drill into a specific DC
    9. **Top 20 SKUs by WMAPE** — Highest forecast error SKUs with volume and DOH detail table
    10. **Demand Variability (CV)** — Coefficient of Variation by DC
    11. **Inventory Health Tab** — DOH Category Breakdown (100% stacked bar: Dead/Very Slow/Slow/Normal/Fast)
    12. **On-Hand vs Order-Up-To Level** — Grouped bar by DC with DOH Category filter interaction
    13. **Weekly Trend** — OH Qty vs OUL over time, linked to DOH Category filter
    14. **Top 10 SKUs by DOH** — Grouped bar with DOH annotations
    15. **SKU Detail Table** — Filterable table with inventory metrics
    16. **SKU Count by Material Group** — Summary table with label descriptions
    17. **Recommendations Tab** — Auto-generated actionable recommendations (Overstock, Dead Stock, Stockout, Policy)
    18. **DC Efficiency Scorecard** — 0–100 composite score with horizontal bar chart
    19. **Readme Tab** — Assumptions documentation + DOH/Volume Outlier Analysis (IQR method with box plots & histograms)
    20. **Info Cards** — Expandable methodology explanations on every chart
    21. **Custom Theming** — Databricks Light Theme with styled KPI cards, recommendation cards, color system
    """)

    st.markdown("---")

    # --- Assumptions Section ---
    st.subheader("Assumptions")
    st.markdown("""
    - **Temperature Condition Filter:** Only SKUs classified as *Keep Frozen* or *Refrge/Do Not Freeze* are included in this analysis.
    - **Location Type:** Only *Reserve* locations are considered for inventory and volume calculations.
    - **Date Range:** All data is filtered to January 2026 onward.
    - **Excluded Plants:** Plants 087, 203, 204, 200, 220, and 230 are excluded from all queries.
    - **DOH Cap:** SKUs with DOH > 365 days are excluded to remove extreme outliers from aggregations.
    - **Minimum Volume Threshold:** For demand vs forecast accuracy (WMAPE), SKUs must have at least 100 units in both actual and forecast demand to be included in the Top 20 ranking.
    - **Forecast Source:** Forecast demand uses the `deseasonalized_fcst_for_period` field from replenishment parameters (weekly rate).
    - **Actual Demand Source:** Actual demand is sourced from billing transactions (`invc_qty` / `sales_qty`).
    - **Utilization Data:** DC utilization percentages are static reference values (not dynamically computed).
    - **PEL Calculation:** Pallet Equivalent Locations = AVG(H × W × L) / 75,000 cubic inches.
    - **Volume Cap:** SKUs with total volume > 1,000,000 cubic inches are excluded as data quality outliers.
    """)

    st.markdown("---")

    # --- Outlier Analysis Section ---
    st.subheader("Outlier Analysis")
    if df_core.empty:
        empty_state()
    else:
        # --- DOH Outlier Analysis ---
        st.markdown("**DOH Outlier Detection**")
        df_doh_valid = df_core[df_core["DOH"].notna() & (df_core["DOH"] <= 365)].copy()

        if not df_doh_valid.empty:
            outlier_mask, lower, upper, q1, q3, iqr = detect_outliers_iqr(
                df_doh_valid["DOH"], multiplier=outlier_sensitivity
            )
            n_outliers = outlier_mask.sum()
            pct_outliers = n_outliers / len(df_doh_valid) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("DOH Outliers Detected", f"{n_outliers:,}", f"{pct_outliers:.1f}% of records")
            col2.metric("IQR Range", f"{q1:.0f} - {q3:.0f} days")
            col3.metric("Outlier Threshold", f"> {upper:.0f} days")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**DOH Distribution by DC (Box Plot)**")
                top_dcs = df_doh_valid.groupby("DC_Name")["DOH"].median().nlargest(12).index.tolist()
                df_box = df_doh_valid[df_doh_valid["DC_Name"].isin(top_dcs)]
                fig = px.box(
                    df_box, x="DC_Name", y="DOH", color="DC_Name",
                    color_discrete_sequence=COLOR_PALETTE
                )
                fig.add_hline(y=upper, line_dash="dash", line_color=COLOR_DANGER,
                    annotation_text=f"Outlier Threshold ({upper:.0f})")
                fig = chart_layout(fig, 450)
                fig.update_layout(showlegend=False, xaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                st.markdown("**DOH Distribution (Histogram)**")
                fig = px.histogram(
                    df_doh_valid, x="DOH", nbins=50,
                    color_discrete_sequence=[COLOR_PRIMARY]
                )
                fig.add_vline(x=upper, line_dash="dash", line_color="red",
                    annotation_text=f"Outlier: {upper:.0f}")
                fig.add_vline(x=df_doh_valid["DOH"].median(), line_dash="dot",
                    line_color="green", annotation_text="Median")
                fig = chart_layout(fig, 450)
                fig.update_layout(xaxis_title="Days on Hand", yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True)

            # --- Outlier Detail Table ---
            with st.expander("\U0001F50D View DOH Outlier Details", expanded=False):
                df_outliers = df_doh_valid[outlier_mask].groupby(["DC_Name", "SKU"]).agg(
                    Median_DOH=("DOH", "median"),
                    Avg_OH_Qty=("Avg_OH_Qty", "mean"),
                    Weeks_as_Outlier=("DOH", "count")
                ).reset_index().sort_values("Median_DOH", ascending=False).head(50)
                st.dataframe(df_outliers, use_container_width=True, hide_index=True)

            with st.expander("ℹ️ About DOH Outlier Detection", expanded=False):
                st.markdown(
                    "**Method:** Interquartile Range (IQR)\n\n"
                    "**Formula:**\n"
                    "- `IQR = Q3 − Q1`\n"
                    "- `Upper Threshold = Q3 + (multiplier × IQR)`\n"
                    "- `Lower Threshold = Q1 − (multiplier × IQR)`\n\n"
                    f"**Current Sensitivity:** {outlier_sensitivity}× IQR (default: 1.5×)\n\n"
                    "**Insight:** DOH outliers represent SKUs with abnormally high or low inventory relative to their forecast. "
                    "These often indicate demand signal failures, stale safety stock parameters, or discontinued items still in stock."
                )

        st.markdown("---")

        # --- Volume Outlier Analysis ---
        st.subheader("Volume Outlier Detection")
        try:
            with st.spinner("Loading volume data..."):
                df_vol_raw = run_query(QUERY_VOLUME_DATA)
            df_vol = apply_mat_grp_sku_filter(apply_filters(df_vol_raw), df_core)

            if not df_vol.empty:
                vol_outlier_mask, v_lower, v_upper, v_q1, v_q3, v_iqr = detect_outliers_iqr(
                    df_vol["Total_Volume_Cubic_Inches"], multiplier=outlier_sensitivity
                )
                n_vol_out = vol_outlier_mask.sum()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Volume Outliers", f"{n_vol_out:,}")
                    st.markdown("**Volume Distribution by DC**")
                    df_vol_dc = df_vol.groupby("plant")["Total_Volume_Cubic_Inches"].median().reset_index()
                    df_vol_dc = df_vol_dc.sort_values("Total_Volume_Cubic_Inches", ascending=False).head(15)
                    fig = px.bar(
                        df_vol_dc, x="plant", y="Total_Volume_Cubic_Inches",
                        text_auto=".0f", color="Total_Volume_Cubic_Inches",
                        color_continuous_scale=[[0, "#ccfbf1"], [1, "#0d9488"]]
                    )
                    fig = chart_layout(fig, 400)
                    fig.update_layout(xaxis_title="Plant", yaxis_title="Median Volume (cu in)", coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.metric("Threshold", f"{v_upper:,.0f} cu in")
                    st.markdown("**Top SKUs by Volume**")
                    df_top_vol = df_vol.groupby("SKU").agg(
                        Avg_Volume=("Total_Volume_Cubic_Inches", "mean"),
                        Avg_PEL=("PEL", "mean")
                    ).reset_index().nlargest(15, "Avg_Volume")
                    fig = px.bar(
                        df_top_vol, x="SKU", y="Avg_Volume", text_auto=".0f",
                        color="Avg_PEL", color_continuous_scale=[[0, "#dbeafe"], [1, "#1d4ed8"]]
                    )
                    fig = chart_layout(fig, 400)
                    fig.update_layout(xaxis_title="", yaxis_title="Avg Volume (cu in)")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                empty_state("No volume data available.")
        except Exception as e:
            st.warning(f"Could not load volume data: {e}")




# ======================== TAB 5: FORECASTING & ML (HIDDEN) ========================
if False:  # Tab hidden for now — keep code intact
    st.header("Capacity Forecasting")

    if df_core.empty:
        empty_state()
    else:
        st.markdown("""
        <div class='insight-box'>
        <b>About this module:</b> This section provides ML-based capacity forecasting.
        Configure your parameters in the sidebar, select a DC, and click <b>Run Forecast</b>
        to generate projections with confidence intervals.
        </div>
        """, unsafe_allow_html=True)

        # --- Forecasting Configuration ---
        st.subheader("Forecast Configuration")
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            st.markdown(f"**Method:** {forecast_method}")
        with col_cfg2:
            st.markdown(f"**Horizon:** {forecast_horizon} weeks")
        with col_cfg3:
            st.markdown(f"**Outlier Sensitivity:** {outlier_sensitivity}")

        st.markdown("---")

        # --- DC Selector for Forecasting ---
        available_dcs = sorted(df_core["DC_Name"].dropna().unique())
        forecast_dc = st.selectbox(
            "Select DC for Forecast", options=["All Network"] + available_dcs,
            index=0
        )

        if forecast_dc == "All Network":
            df_fc = df_core.copy()
        else:
            df_fc = df_core[df_core["DC_Name"] == forecast_dc].copy()

        # --- Aggregate weekly time series ---
        df_ts = df_fc.groupby(["yr", "wk", "week_label"]).agg(
            Median_DOH=("DOH", "median"),
            Avg_OH_Qty=("Avg_OH_Qty", "sum"),
            Total_Forecast_Demand=("Forecast_Demand", "sum"),
            SKU_Count=("SKU", "nunique")
        ).reset_index().sort_values(["yr", "wk"])

        # --- Show Historical Trends (always visible) ---
        st.subheader(f"Historical DOH Trend ({forecast_dc})")
        if not df_ts.empty:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=df_ts["week_label"], y=df_ts["Median_DOH"],
                mode="lines+markers", name="Median DOH",
                line=dict(color=COLOR_PRIMARY, width=2)
            ))
            fig_hist = chart_layout(fig_hist, 350)
            fig_hist.update_layout(xaxis_title="Week", yaxis_title="Median DOH")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            empty_state("No historical data available for selected filters.")

        st.markdown("---")

        # --- Forecast Execution (button-gated) ---
        run_forecast = st.button("\U0001F680 Run Forecast", type="primary", use_container_width=True)

        if run_forecast:
            if len(df_ts) < 4:
                st.warning("Insufficient data points for forecasting (need at least 4 weeks).")
            else:
                with st.spinner("Running forecast model..."):
                    # --- Run Forecast ---
                    forecast_func = linear_forecast if forecast_method == "Linear Regression" else exponential_smoothing
                    alpha_param = {"alpha": 0.3} if forecast_method == "Exponential Smoothing" else {}

                    doh_forecast, doh_ci_low, doh_ci_high = forecast_func(
                        df_ts["Median_DOH"], horizon=forecast_horizon, **alpha_param
                    )
                    oh_forecast, oh_ci_low, oh_ci_high = forecast_func(
                        df_ts["Avg_OH_Qty"], horizon=forecast_horizon, **alpha_param
                    )

                # Build forecast labels
                last_wk = df_ts["wk"].iloc[-1]
                forecast_labels = [f"WK-{((last_wk + i) % 52) + 1:02d}" for i in range(forecast_horizon)]

                # --- DOH Forecast Chart ---
                st.subheader(f"DOH Forecast ({forecast_dc})")
                col1, col2 = st.columns([3, 1])

                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_ts["week_label"], y=df_ts["Median_DOH"],
                        mode="lines+markers", name="Historical DOH",
                        line=dict(color=COLOR_PRIMARY, width=2)
                    ))
                    if doh_forecast is not None:
                        fig.add_trace(go.Scatter(
                            x=forecast_labels, y=doh_forecast,
                            mode="lines+markers", name="Forecast",
                            line=dict(color=COLOR_DANGER, width=2, dash="dash")
                        ))
                        fig.add_trace(go.Scatter(
                            x=forecast_labels + forecast_labels[::-1],
                            y=np.concatenate([doh_ci_high, doh_ci_low[::-1]]).tolist(),
                            fill="toself", fillcolor="rgba(239,68,68,0.1)",
                            line=dict(color="rgba(0,0,0,0)"), name="95% CI"
                        ))
                    fig = chart_layout(fig, 400)
                    fig.update_layout(xaxis_title="Week", yaxis_title="Median DOH")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("**Forecast Summary**")
                    if doh_forecast is not None:
                        current_doh = df_ts["Median_DOH"].iloc[-1]
                        projected_doh = doh_forecast[-1]
                        change = projected_doh - current_doh
                        st.metric("Current DOH", f"{current_doh:.1f}")
                        st.metric("Projected DOH", f"{projected_doh:.1f}", delta=f"{change:+.1f} days")
                        if projected_doh > 45:
                            st.markdown("<div class='warning-box'>\u26A0\uFE0F DOH trending above target (45 days). Consider rebalancing.</div>", unsafe_allow_html=True)
                        elif projected_doh < 14:
                            st.markdown("<div class='warning-box'>\u26A0\uFE0F DOH dropping below safety threshold (14 days).</div>", unsafe_allow_html=True)
                    else:
                        st.info("Insufficient data for forecast.")

                # --- On-Hand Quantity Forecast ---
                st.subheader(f"On-Hand Inventory Forecast ({forecast_dc})")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=df_ts["week_label"], y=df_ts["Avg_OH_Qty"],
                    mode="lines+markers", name="Historical On-Hand",
                    line=dict(color=COLOR_SUCCESS, width=2)
                ))
                if oh_forecast is not None:
                    fig2.add_trace(go.Scatter(
                        x=forecast_labels, y=oh_forecast,
                        mode="lines+markers", name="Forecast",
                        line=dict(color=COLOR_WARNING, width=2, dash="dash")
                    ))
                    fig2.add_trace(go.Scatter(
                        x=forecast_labels + forecast_labels[::-1],
                        y=np.concatenate([oh_ci_high, oh_ci_low[::-1]]).tolist(),
                        fill="toself", fillcolor="rgba(245,158,11,0.1)",
                        line=dict(color="rgba(0,0,0,0)"), name="95% CI"
                    ))
                fig2 = chart_layout(fig2, 380)
                fig2.update_layout(xaxis_title="Week", yaxis_title="Total On-Hand Qty")
                st.plotly_chart(fig2, use_container_width=True)

                st.success("Forecast complete. Adjust parameters in the sidebar and re-run to compare.")
        else:
            # --- Show available capabilities when forecast hasn't been run ---
            st.markdown("""\n#### Available Forecast Capabilities\n\n| Feature | Description |
| --- | --- |
| **DOH Projection** | Forecast Days on Hand with 95% confidence intervals |
| **On-Hand Qty Forecast** | Project total inventory levels by DC or network |
| **Linear Regression** | Trend-based projection using least squares fit |
| **Exponential Smoothing** | Weighted recent data for adaptive forecasting |
| **Configurable Horizon** | 2 to 12 week lookahead window |
""")
            st.info("\U0001F4A1 Click **Run Forecast** above to generate projections based on your current filter selections.")


# ======================== TAB 6: RECOMMENDATIONS ========================
with tab6:
    st.header("Actionable Recommendations")

    if df_core.empty:
        empty_state()
    else:
        # --- Generate Recommendations (latest week snapshot) ---
        latest_wk_t6 = df_core["wk"].max()
        latest_yr_t6 = df_core[df_core["wk"] == latest_wk_t6]["yr"].max()
        df_latest_t6 = df_core[(df_core["wk"] == latest_wk_t6) & (df_core["yr"] == latest_yr_t6)]
        st.caption(f"Recommendations based on latest week snapshot: WK-{latest_wk_t6:02d}, {latest_yr_t6}")

        recommendations = []

        # 1. Overstock analysis
        df_over_dc = df_latest_t6[df_latest_t6["Stock_Status"] == "Overstock"].groupby("DC_Name")["SKU"].nunique().reset_index(name="Overstock_SKUs")
        total_sku_per_dc = df_latest_t6.groupby("DC_Name")["SKU"].nunique().reset_index(name="Total_SKUs")
        df_over_dc = df_over_dc.merge(total_sku_per_dc, on="DC_Name")
        df_over_dc["Overstock_Pct"] = (df_over_dc["Overstock_SKUs"] / df_over_dc["Total_SKUs"] * 100)
        high_overstock = df_over_dc[df_over_dc["Overstock_Pct"] > 25].sort_values("Overstock_Pct", ascending=False)
        for _, row in high_overstock.head(5).iterrows():
            recommendations.append({
                "priority": "High",
                "category": "Overstock Reduction",
                "dc": row["DC_Name"],
                "detail": f"{row['Overstock_SKUs']} SKUs ({row['Overstock_Pct']:.0f}%) are overstocked. Review Order-Up-To policies and consider redistribution.",
                "impact": "Reduce carrying cost, free warehouse space"
            })

        # 2. Dead stock
        df_dead_reco = df_core[df_core["DOH"].notna() & (df_core["DOH"] > 180)].groupby("DC_Name")["SKU"].nunique().reset_index(name="Dead_SKUs")
        high_dead = df_dead_reco[df_dead_reco["Dead_SKUs"] > 10].sort_values("Dead_SKUs", ascending=False)
        for _, row in high_dead.head(5).iterrows():
            recommendations.append({
                "priority": "High",
                "category": "Dead Stock Liquidation",
                "dc": row["DC_Name"],
                "detail": f"{row['Dead_SKUs']} SKUs with DOH > 180 days. Initiate returns, transfers, or write-offs.",
                "impact": "Recover space, reduce obsolescence risk"
            })

        # 3. Understock risk
        df_under_dc = df_core[df_core["Stock_Status"] == "Understock"].groupby("DC_Name")["SKU"].nunique().reset_index(name="Understock_SKUs")
        high_under = df_under_dc[df_under_dc["Understock_SKUs"] > 20].sort_values("Understock_SKUs", ascending=False)
        for _, row in high_under.head(3).iterrows():
            recommendations.append({
                "priority": "Medium",
                "category": "Stockout Prevention",
                "dc": row["DC_Name"],
                "detail": f"{row['Understock_SKUs']} SKUs below safety stock. Prioritize replenishment to avoid service disruptions.",
                "impact": "Improve fill rate, reduce customer impact"
            })

        # 4. Policy misalignment
        df_policy_gap = df_core.groupby("DC_Name").agg(
            Avg_DOH=("DOH", "median"),
            Avg_OUL_Days=("Avg_Order_Up_To_Level_Days", "mean")
        ).reset_index()
        df_policy_gap["Gap"] = df_policy_gap["Avg_DOH"] - df_policy_gap["Avg_OUL_Days"]
        misaligned = df_policy_gap[df_policy_gap["Gap"] > 15].sort_values("Gap", ascending=False)
        for _, row in misaligned.head(3).iterrows():
            recommendations.append({
                "priority": "Medium",
                "category": "Policy Optimization",
                "dc": row["DC_Name"],
                "detail": f"Actual DOH ({row['Avg_DOH']:.0f}d) exceeds Order-Up-To target ({row['Avg_OUL_Days']:.0f}d) by {row['Gap']:.0f} days. Review replenishment parameters.",
                "impact": "Align inventory to policy, reduce excess"
            })

        # --- Display Recommendations ---
        if recommendations:
            # Priority filter
            priority_filter = st.radio("Filter by Priority", ["All", "High", "Medium"], horizontal=True)
            filtered_recos = recommendations if priority_filter == "All" else [r for r in recommendations if r["priority"] == priority_filter]

            # Summary badges
            n_high = sum(1 for r in filtered_recos if r["priority"] == "High")
            n_med = sum(1 for r in filtered_recos if r["priority"] == "Medium")
            high_badge = f"<span class='reco-badge high'>{n_high} High</span>" if n_high else ""
            med_badge = f"<span class='reco-badge medium'>{n_med} Medium</span>" if n_med else ""
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;margin:8px 0 16px;'>"
                f"<span style='font-size:14px;font-weight:700;color:#1E293B;'>{len(filtered_recos)} recommendation(s)</span>"
                f"{high_badge}"
                f"{med_badge}"
                f"</div>", unsafe_allow_html=True
            )

            # Category icons for visual differentiation
            category_icons = {
                "Overstock Reduction": "📦",
                "Dead Stock Liquidation": "☠️",
                "Stockout Prevention": "⚠️",
                "Policy Optimization": "⚙️",
            }

            for i, reco in enumerate(filtered_recos):
                priority_class = "high" if reco["priority"] == "High" else "medium"
                icon = category_icons.get(reco["category"], "📌")
                st.markdown(f"""
                <div class='reco-card {priority_class}'>
                    <div class='reco-header'>
                        <div>
                            <span class='reco-category'>{icon} {reco['category']}</span>
                            <span class='reco-dc'>&nbsp;&mdash;&nbsp;{reco['dc']}</span>
                        </div>
                        <span class='reco-badge {priority_class}'>{reco['priority']}</span>
                    </div>
                    <div class='reco-detail'>{reco['detail']}</div>
                    <div class='reco-impact'>→ {reco['impact']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("\u2705 No critical issues detected. Network is operating within expected parameters.")

        # --- Efficiency Score by DC ---
        st.subheader("DC Efficiency Scorecard")
        dc_scores = []
        for dc in df_core["DC_Name"].dropna().unique():
            dc_data = df_core[df_core["DC_Name"] == dc]
            total = dc_data["SKU"].nunique()
            if total == 0:
                continue
            overstock_p = dc_data[dc_data["Stock_Status"] == "Overstock"]["SKU"].nunique() / total * 100
            dead_p = dc_data[dc_data["DOH"].notna() & (dc_data["DOH"] > 180)]["SKU"].nunique() / total * 100
            med_doh = dc_data["DOH"].median()
            score = compute_capacity_score({"overstock_pct": overstock_p, "dead_stock_pct": dead_p, "median_doh": med_doh})
            dc_scores.append({"DC_Name": dc, "Score": score, "Overstock%": round(overstock_p, 1), "Dead Stock%": round(dead_p, 1), "Median DOH": round(med_doh, 1) if pd.notna(med_doh) else 0})

        if dc_scores:
            df_scores = pd.DataFrame(dc_scores).sort_values("Score", ascending=True)
            fig = px.bar(
                df_scores, x="Score", y="DC_Name", orientation="h",
                color="Score", text="Score",
                color_continuous_scale=[[0, "#FF8A70"], [0.5, "#FDF68E"], [1, "#80F0C8"]],
                hover_data=["Overstock%", "Dead Stock%", "Median DOH"]
            )
            fig.update_traces(texttemplate="%{text}/100")
            fig = chart_layout(fig, max(300, len(dc_scores) * 28))
            fig.update_layout(xaxis_title="Efficiency Score (0-100)", yaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("\U0001F4CB View Full Scorecard Table"):
                st.dataframe(
                    df_scores.sort_values("Score", ascending=False),
                    use_container_width=True, hide_index=True
                )

            with st.expander("ℹ️ About Efficiency Score", expanded=False):
                st.markdown(info_card(
                    "DC Efficiency Scorecard", "🏆",
                    [
                        {"label": "Methodology (0–100 scale)", "content": "Starts at 100, with deductions based on compounding inefficiencies."},
                        {"label": "Deduction Rules", "content":
                            "<table class='info-card-table'>"
                            "<tr><th>Factor</th><th>Condition</th><th>Deduction</th></tr>"
                            "<tr><td>Overstock %</td><td>&gt; 30%</td><td><span class='info-card-pill pill-red'>−25</span></td></tr>"
                            "<tr><td>Overstock %</td><td>15–30%</td><td><span class='info-card-pill pill-yellow'>−10</span></td></tr>"
                            "<tr><td>Dead Stock %</td><td>&gt; 10%</td><td><span class='info-card-pill pill-red'>−20</span></td></tr>"
                            "<tr><td>Dead Stock %</td><td>5–10%</td><td><span class='info-card-pill pill-yellow'>−10</span></td></tr>"
                            "<tr><td>Median DOH</td><td>&gt; 60 days</td><td><span class='info-card-pill pill-red'>−15</span></td></tr>"
                            "<tr><td>Median DOH</td><td>30–60 days</td><td><span class='info-card-pill pill-yellow'>−5</span></td></tr>"
                            "</table>"
                        },
                        {"label": "Interpretation", "content": "<span class='info-card-pill pill-green'>Score 100</span> = no excess, no dead stock, healthy DOH. Lower scores = compounding issues."},
                    ],
                    insight="Focus improvement efforts on DCs scoring below 60 — they typically have multiple overlapping issues."
                ), unsafe_allow_html=True)

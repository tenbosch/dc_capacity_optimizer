"""SCMO Network Storage Capacity Forecasting Tool
Built by Supply Chain Management & Optimization Team
Target Users: Replenishment Operations, Engineering, Supply Chain Planning

Thin orchestrator: page config, CSS, sidebar, core-data load, and tab dispatch.
Concerns live in sibling modules (theme, connection, queries, analytics, utils,
settings, sidebar); each screen lives in tabs/<name>.py as render(df_core).
"""
import streamlit as st

import theme
import sidebar
from connection import run_query
from queries import QUERY_CORE_WEEKLY
from utils import apply_filters
from tabs import (
    network_overview,
    demand_forecast,
    inventory_health,
    readme,
    recommendations,
    forecasting_ml,
)

# Page config MUST be the first Streamlit call.
st.set_page_config(
    layout="wide",
    page_title="DC Capacity Forecasting",
    page_icon="\U0001F4E6",
)

theme.inject_css()

st.title("\U0001F4E6 Network Storage Capacity Forecasting")
st.caption("Supply Chain Management & Optimization | Cold Chain Distribution Network")

sidebar.render_sidebar()


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


tab1, tab3, tab2, tab6, tab4 = st.tabs([
    "\U0001F4CA Network Overview",
    "\U0001F4E6 Inventory Health",
    "\U0001F4C8 Demand Vs Forecast(DFIO)",
    "\U0001F4A1 Recommendations",
    "\U0001F4D6 Readme"
])

with tab1:
    network_overview.render(df_core)
with tab2:
    demand_forecast.render(df_core)
with tab3:
    inventory_health.render(df_core)
with tab4:
    readme.render(df_core)
with tab6:
    recommendations.render(df_core)

# Tab 5 (Forecasting & ML) is hidden — code kept intact but not executed.
if False:
    forecasting_ml.render(df_core)


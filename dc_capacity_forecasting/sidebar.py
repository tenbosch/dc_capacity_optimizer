"""Sidebar filter widgets. Writes the selected_* / dc_options keys into
st.session_state that utils.apply_filters and the tabs read."""
import pandas as pd
import streamlit as st

from connection import run_query
from queries import QUERY_FILTER_OPTIONS
from utils import MONTH_ABBR
from genie_chat import render_chat_dialog


def render_sidebar():
    """Render the sidebar filters and persist selections to st.session_state."""
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

        # --- AI Chat Assistant ---
        st.markdown("---")
        if st.button("\U0001F4AC AI Assistant", key="genie_chat_open", type="primary", use_container_width=True):
            render_chat_dialog()

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

    # Store in session state for filter functions
    st.session_state["selected_plants"] = selected_plants
    st.session_state["selected_mat_grps"] = selected_mat_grps
    st.session_state["selected_years"] = selected_years
    st.session_state["selected_months"] = selected_months
    st.session_state["month_to_weeks"] = month_to_weeks
    st.session_state["selected_weeks"] = selected_weeks
    st.session_state["dc_options"] = dc_options

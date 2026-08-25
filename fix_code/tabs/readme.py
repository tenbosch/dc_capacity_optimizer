"""Readme tab (feature list, assumptions, DOH/volume outlier analysis)."""
import streamlit as st
import pandas as pd
import plotly.express as px

from settings import outlier_sensitivity
from connection import run_query
from queries import QUERY_VOLUME_DATA
from utils import (apply_filters, apply_mat_grp_sku_filter, chart_layout, empty_state)
from theme import (COLOR_DANGER, COLOR_PALETTE, COLOR_PRIMARY)
from analytics import detect_outliers_iqr


def render(df_core):
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

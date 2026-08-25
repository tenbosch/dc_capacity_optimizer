"""Demand vs Forecast tab (variability, WMAPE accuracy, drill-down)."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from connection import run_query
from queries import (QUERY_BILLING, QUERY_VOLUME_DATA)
from utils import (apply_filters, chart_layout, empty_state, info_card)
from theme import (COLOR_PRIMARY, COLOR_WARNING, THEME)


def render(df_core):
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

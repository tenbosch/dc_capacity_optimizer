"""Network Overview tab (KPIs, DC utilization map, stock status, DOH heatmap, PEL)."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from connection import run_query
from queries import (QUERY_DC_UTILIZATION, QUERY_VOLUME_DATA)
from utils import (apply_filters, apply_mat_grp_sku_filter, chart_layout, empty_state, info_card)
from theme import (COLOR_DANGER, COLOR_SUCCESS, STOCK_COLORS, THEME)


def render(df_core):
    selected_plants = st.session_state.get("selected_plants", [])
    dc_options = st.session_state.get(
        "dc_options", pd.DataFrame(columns=["val", "label"])
    )
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

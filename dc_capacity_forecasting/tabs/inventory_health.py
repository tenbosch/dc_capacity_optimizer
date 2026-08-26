"""Inventory Health tab (DOH categories, on-hand vs OUL, overstock analysis)."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from connection import run_query
from queries import QUERY_VOLUME_DATA
from utils import (apply_filters, chart_layout, empty_state, info_card)
from theme import (COLOR_DANGER, COLOR_PRIMARY, DOH_COLORS, THEME)


def render(df_core):
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

"""Recommendations tab (actionable alerts + DC efficiency scorecard + nearest DCs)."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math

from connection import run_query
from queries import QUERY_DC_UTILIZATION
from utils import (chart_layout, empty_state, info_card)
from analytics import compute_capacity_score
from theme import THEME


def render(df_core):
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

        st.markdown("---")

        # --- Nearest Distribution Centers (Overflow / Capacity Relief) ---
        st.subheader("\U0001F3ED Nearest Distribution Centers")
        st.caption(
            "Identify nearby DCs that can absorb overflow when a facility is at or nearing capacity."
        )

        try:
            df_dc_util = run_query(QUERY_DC_UTILIZATION)
        except Exception as e:
            st.warning(f"Could not load DC utilization data: {e}")
            df_dc_util = pd.DataFrame()

        if not df_dc_util.empty:
            # Use the DC selected in the sidebar filter
            selected_plants = st.session_state.get("selected_plants", [])
            dc_opts_df = st.session_state.get("dc_options", pd.DataFrame())
            selected_origin_dc = None

            if not selected_plants or len(selected_plants) != 1:
                st.info(
                    "\U0001F449 Please select **exactly one** Distribution Center from the "
                    "sidebar filter to identify nearby DCs for overflow routing."
                )
            else:
                # Resolve the plant code to DC name
                plant_code = selected_plants[0]
                if not dc_opts_df.empty:
                    dc_label_match = dc_opts_df[dc_opts_df["val"] == plant_code]["label"]
                    selected_origin_dc = dc_label_match.iloc[0] if not dc_label_match.empty else None
                else:
                    selected_origin_dc = None

                # Fall back: try matching directly in utilization data
                if selected_origin_dc is None or selected_origin_dc not in df_dc_util["dc_name"].values:
                    # Try matching plant code to dc_name as-is
                    dc_name_match = df_dc_util[df_dc_util["dc_name"].str.contains(plant_code, case=False, na=False)]
                    if not dc_name_match.empty:
                        selected_origin_dc = dc_name_match.iloc[0]["dc_name"]
                    else:
                        st.warning(
                            f"The selected DC (plant {plant_code}) could not be matched to "
                            "the utilization dataset. Please verify the selection."
                        )
                        selected_origin_dc = None

            if not df_dc_util.empty and selected_plants and len(selected_plants) == 1 and selected_origin_dc:
                max_distance_miles = st.slider(
                    "Maximum search radius (miles)",
                    min_value=50,
                    max_value=2000,
                    value=500,
                    step=50,
                    key="nearest_dc_radius",
                )

                # --- Haversine distance calculation ---
                def _haversine(lat1, lon1, lat2, lon2):
                    """Return distance in miles between two lat/lon points."""
                    R = 3958.8  # Earth radius in miles
                    phi1, phi2 = math.radians(lat1), math.radians(lat2)
                    dphi = math.radians(lat2 - lat1)
                    dlambda = math.radians(lon2 - lon1)
                    a = (
                        math.sin(dphi / 2) ** 2
                        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
                    )
                    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

                origin_row = df_dc_util[df_dc_util["dc_name"] == selected_origin_dc].iloc[0]
                origin_lat, origin_lon = origin_row["lat"], origin_row["lon"]
                origin_util = origin_row["utilization_pct"]

                # Calculate distances to all other DCs
                df_neighbors = df_dc_util[df_dc_util["dc_name"] != selected_origin_dc].copy()
                df_neighbors["distance_miles"] = df_neighbors.apply(
                    lambda r: round(_haversine(origin_lat, origin_lon, r["lat"], r["lon"]), 1),
                    axis=1,
                )
                df_neighbors = df_neighbors[
                    df_neighbors["distance_miles"] <= max_distance_miles
                ].sort_values("distance_miles")

                # Show origin DC utilization as context
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:12px;padding:10px 16px;"
                    f"background:#FEF3C7;border:1px solid #F59E0B;border-radius:8px;margin:12px 0;'>"
                    f"<span style='font-size:20px;'>\u26a0\ufe0f</span>"
                    f"<span style='font-size:13px;color:#92400E;'>"
                    f"<b>{selected_origin_dc}</b> is at <b>{origin_util}%</b> utilization. "
                    f"Showing <b>{len(df_neighbors)}</b> DCs within <b>{max_distance_miles}</b> miles.</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                if not df_neighbors.empty:
                    # Color-code by available capacity (lower utilization = greener)
                    df_display_nearest = df_neighbors[
                        ["dc_name", "utilization_pct", "distance_miles"]
                    ].copy()
                    df_display_nearest.columns = ["Distribution Center", "Utilization %", "Distance (miles)"]
                    df_display_nearest["Available Capacity"] = (
                        100 - df_display_nearest["Utilization %"]
                    ).apply(lambda x: f"{x}%")

                    # --- Map visualization ---
                    st.markdown("**DC Proximity Map**")

                    # Build map dataframe: origin + neighbors
                    df_map = pd.concat([
                        pd.DataFrame([{
                            "dc_name": selected_origin_dc,
                            "lat": origin_lat,
                            "lon": origin_lon,
                            "utilization_pct": origin_util,
                            "distance_miles": 0,
                            "role": "Origin (at capacity)",
                        }]),
                        df_neighbors.assign(role="Nearby DC"),
                    ], ignore_index=True)

                    df_map["size"] = df_map["role"].apply(
                        lambda r: 18 if r.startswith("Origin") else 12
                    )
                    df_map["label"] = df_map.apply(
                        lambda r: (
                            f"{r['dc_name']} \u2014 {r['utilization_pct']:.0f}% util"
                            if r["distance_miles"] == 0
                            else f"{r['dc_name']} \u2014 {r['utilization_pct']:.0f}% util, {r['distance_miles']:.0f} mi"
                        ),
                        axis=1,
                    )

                    fig_map = go.Figure()

                    # Nearby DCs — colored by utilization
                    df_nearby_map = df_map[df_map["role"] == "Nearby DC"]
                    fig_map.add_trace(go.Scattergeo(
                        lat=df_nearby_map["lat"],
                        lon=df_nearby_map["lon"],
                        text=df_nearby_map["label"],
                        hoverinfo="text",
                        marker=dict(
                            size=df_nearby_map["size"],
                            color=df_nearby_map["utilization_pct"],
                            colorscale=[
                                [0, THEME["primary_green"]],
                                [0.5, THEME["gold"]],
                                [1, THEME["bright_red"]],
                            ],
                            cmin=0,
                            cmax=100,
                            colorbar=dict(title="Util %", thickness=12, len=0.5),
                            line=dict(width=1, color="white"),
                        ),
                        name="Nearby DCs",
                    ))

                    # Origin DC — distinct marker
                    df_origin_map = df_map[df_map["role"].str.startswith("Origin")]
                    fig_map.add_trace(go.Scattergeo(
                        lat=df_origin_map["lat"],
                        lon=df_origin_map["lon"],
                        text=df_origin_map["label"],
                        hoverinfo="text",
                        marker=dict(
                            size=16,
                            color="#DC2626",
                            symbol="diamond",
                            line=dict(width=2, color="white"),
                        ),
                        name="Origin DC",
                    ))

                    # Lines connecting origin to each neighbor
                    for _, nb in df_nearby_map.iterrows():
                        fig_map.add_trace(go.Scattergeo(
                            lat=[origin_lat, nb["lat"]],
                            lon=[origin_lon, nb["lon"]],
                            mode="lines",
                            line=dict(width=1, color="rgba(100,116,139,0.4)"),
                            hoverinfo="skip",
                            showlegend=False,
                        ))

                    fig_map.update_geos(
                        scope="north america",
                        showland=True,
                        landcolor="#F8FAFC",
                        showlakes=True,
                        lakecolor="#DBEAFE",
                        showcountries=True,
                        countrycolor="#CBD5E1",
                        showsubunits=True,
                        subunitcolor="#E2E8F0",
                        projection_type="albers usa",
                    )
                    fig_map.update_layout(
                        height=480,
                        margin=dict(t=10, b=10, l=10, r=10),
                        legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center"),
                        paper_bgcolor="white",
                    )
                    st.plotly_chart(fig_map, use_container_width=True)

                    # Data table
                    st.dataframe(
                        df_display_nearest,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info(
                        f"No other DCs found within {max_distance_miles} miles of {selected_origin_dc}. "
                        "Try increasing the search radius."
                    )

                with st.expander("\u2139\ufe0f About Nearest DC Finder", expanded=False):
                    st.markdown(info_card(
                        "Nearest Distribution Centers", "\U0001F3ED",
                        [
                            {"label": "Purpose", "content": "Identify alternative DCs within a given radius that have available capacity to absorb overflow product."},
                            {"label": "Distance", "content": "<span class='info-card-formula'>Haversine formula</span> \u2014 great-circle distance between DC coordinates."},
                            {"label": "Interpretation", "content":
                                "<span class='info-card-pill pill-green'>Util &lt; 60%</span> High available capacity<br>"
                                "<span class='info-card-pill pill-yellow'>Util 60\u201380%</span> Moderate capacity remaining<br>"
                                "<span class='info-card-pill pill-red'>Util &gt; 80%</span> Limited overflow capacity"
                            },
                        ],
                        insight="Prioritize nearby DCs with low utilization for fastest, most cost-effective overflow routing."
                    ), unsafe_allow_html=True)
        else:
            st.info("DC utilization data unavailable.")

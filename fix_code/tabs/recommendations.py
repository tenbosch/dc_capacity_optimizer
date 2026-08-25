"""Recommendations tab (actionable alerts + DC efficiency scorecard)."""
import streamlit as st
import pandas as pd
import plotly.express as px

from utils import (chart_layout, empty_state, info_card)
from analytics import compute_capacity_score


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

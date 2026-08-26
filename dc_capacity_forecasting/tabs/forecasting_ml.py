"""Forecasting & ML tab (hidden; DOH/inventory projections)."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from settings import (forecast_horizon, forecast_method, outlier_sensitivity)
from utils import (chart_layout, empty_state)
from theme import (COLOR_DANGER, COLOR_PRIMARY, COLOR_SUCCESS, COLOR_WARNING)
from analytics import (exponential_smoothing, linear_forecast)


def render(df_core):
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

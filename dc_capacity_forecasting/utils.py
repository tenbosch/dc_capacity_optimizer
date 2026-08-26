"""Shared filter and display helpers."""
import pandas as pd
import streamlit as st

MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
              7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


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

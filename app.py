import os
import yaml
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pulp
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

# =============================================================================
# LOAD CONFIG
# =============================================================================
with open("config.yaml") as _f:
    CONFIG = yaml.safe_load(_f)

SCHEMA = CONFIG["schema"]
OPT = CONFIG["optimization"]
PEN_DEFAULTS = OPT["penalties"]
FALLBACKS = OPT["fallback_capacities"]
APP_CFG = CONFIG["app"]
PSLIDERS = APP_CFG["penalty_sliders"]
CMULTS = APP_CFG["capacity_multipliers"]
FACILITY_FILTER = CONFIG["synthetic_data"]["distribution_centers"]["facility_type"]
SOLVER_PREFERENCE = OPT.get("solver_preference", ["cbc"])

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="DC Capacity Optimizer",
    page_icon="\U0001f3ed",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SESSION STATE
# =============================================================================
if "view" not in st.session_state:
    st.session_state["view"] = "map"
if "selected_dc" not in st.session_state:
    st.session_state["selected_dc"] = None

# =============================================================================
# DATABASE CONNECTION + QUERY HELPERS
# =============================================================================
@st.cache_resource
def get_connection():
    cfg = Config()
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        raise ValueError(
            "DATABRICKS_WAREHOUSE_ID environment variable is not set. "
            "Ensure the SQL warehouse resource is configured in app.yaml."
        )
    return sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        credentials_provider=lambda: cfg.authenticate,
        _socket_timeout=30,
    )


_STALE_CONN_KEYWORDS = (
    "closed", "expired", "invalid session", "connection",
    "timeout", "eof", "broken pipe", "reset by peer",
    "ssl", "token", "unauthorized", "403", "401",
)


def _run_query(sql_text, params=None):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(sql_text, params or {})
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        return pd.DataFrame(data, columns=columns)
    finally:
        cursor.close()


def run_query(sql_text, params=None):
    """Run a SQL query with automatic reconnect on stale connections."""
    try:
        return _run_query(sql_text, params)
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in _STALE_CONN_KEYWORDS):
            get_connection.clear()
            return _run_query(sql_text, params)
        raise


# =============================================================================
# DATA LOADERS
# =============================================================================
@st.cache_data(ttl=APP_CFG["cache_ttl_seconds"], show_spinner=False)
def load_network_summary():
    """One row per Wholesale DC. Powers the map landing screen.

    Aggregates inventory + demand at the SQL warehouse so we never hydrate
    per-SKU tables to render the map.
    """
    query = f"""
    SELECT
      d.dc_id,
      d.facility_name,
      d.city,
      d.state_code,
      d.country,
      d.latitude,
      d.longitude,
      d.region,
      d.automation_level,
      c.total_storage_cube_ft3,
      c.max_daily_throughput_units,
      c.max_daily_inbound_pallets,
      c.max_daily_outbound_pallets,
      COALESCE(inv.on_hand_units, 0)      AS on_hand_units,
      COALESCE(inv.inventory_cube_ft3, 0) AS inventory_cube_ft3,
      COALESCE(dem.total_demand_units, 0) AS total_demand_units,
      COALESCE(dem.n_skus, 0)             AS n_skus,
      COALESCE(dem.n_periods, 0)          AS n_periods,
      ROUND(
        100.0 * COALESCE(inv.inventory_cube_ft3, 0)
              / NULLIF(c.total_storage_cube_ft3, 0),
        1
      ) AS storage_util_pct
    FROM {SCHEMA}.distribution_centers d
    JOIN {SCHEMA}.dc_capacity c USING (dc_id)
    LEFT JOIN (
      SELECT dc_id,
             SUM(on_hand_units)  AS on_hand_units,
             SUM(total_cube_ft3) AS inventory_cube_ft3
      FROM {SCHEMA}.inventory_levels
      GROUP BY dc_id
    ) inv USING (dc_id)
    LEFT JOIN (
      SELECT dc_id,
             SUM(demand_units)              AS total_demand_units,
             COUNT(DISTINCT sku_id)         AS n_skus,
             COUNT(DISTINCT forecast_date)  AS n_periods
      FROM {SCHEMA}.demand_forecast
      GROUP BY dc_id
    ) dem USING (dc_id)
    WHERE d.facility_type = %(facility_filter)s
    ORDER BY d.dc_id
    """
    df = run_query(query, {"facility_filter": FACILITY_FILTER})
    # Coerce numeric columns to plain floats — the SQL warehouse returns some
    # of these as `Decimal` which Plotly's JSON serializer chokes on, causing
    # the figure to silently render empty.
    numeric_cols = [
        "latitude", "longitude",
        "total_storage_cube_ft3", "max_daily_throughput_units",
        "max_daily_inbound_pallets", "max_daily_outbound_pallets",
        "on_hand_units", "inventory_cube_ft3",
        "total_demand_units", "n_skus", "n_periods",
        "storage_util_pct",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
    return df


@st.cache_data(ttl=APP_CFG["cache_ttl_seconds"], show_spinner=False)
def load_product_master():
    return run_query(f"SELECT * FROM {SCHEMA}.product_master")


@st.cache_data(ttl=APP_CFG["cache_ttl_seconds"], show_spinner=False)
def load_dc_data(dc_id):
    """Load every per-DC table filtered to the chosen DC. Cached per dc_id."""
    tables = {
        "demand_forecast": "SELECT * FROM {schema}.demand_forecast WHERE dc_id = %(dc_id)s",
        "inventory_levels": "SELECT * FROM {schema}.inventory_levels WHERE dc_id = %(dc_id)s",
        "dc_capacity": "SELECT * FROM {schema}.dc_capacity WHERE dc_id = %(dc_id)s",
        "labor_availability": "SELECT * FROM {schema}.labor_availability WHERE dc_id = %(dc_id)s",
        "throughput_rates": "SELECT * FROM {schema}.throughput_rates WHERE dc_id = %(dc_id)s",
        "inbound_plan": "SELECT * FROM {schema}.inbound_plan WHERE dc_id = %(dc_id)s",
        "outbound_plan": "SELECT * FROM {schema}.outbound_plan WHERE dc_id = %(dc_id)s",
        "optimization_parameters":
            "SELECT * FROM {schema}.optimization_parameters WHERE dc_id = %(dc_id)s",
        "distribution_centers":
            "SELECT * FROM {schema}.distribution_centers WHERE dc_id = %(dc_id)s",
    }
    out = {}
    for name, tpl in tables.items():
        out[name] = run_query(tpl.format(schema=SCHEMA), {"dc_id": dc_id})
    return out


@st.cache_data(ttl=APP_CFG["cache_ttl_seconds"], show_spinner=False)
def load_ndc_summary():
    """One row per NDC: capacity + recent flow + SLA attainment.

    Returns an empty DataFrame if `ndc_capacity`/`ndc_inbound`/`ndc_outbound`
    don't exist yet (generator notebook not re-run with NDC support).
    """
    query = f"""
    SELECT
      d.dc_id,
      d.facility_name,
      d.city,
      d.state_code,
      d.latitude,
      d.longitude,
      d.region,
      c.max_hourly_inbound_pallets,
      c.max_hourly_outbound_pallets,
      c.max_hourly_throughput_pallets,
      c.sla_dwell_hours,
      COALESCE(inb.avg_daily_inbound, 0)    AS avg_daily_inbound_pallets,
      COALESCE(o.avg_daily_outbound, 0)     AS avg_daily_outbound_pallets,
      COALESCE(o.avg_dwell_hours, 0)        AS avg_dwell_hours,
      COALESCE(o.sla_attainment_pct, 0)     AS sla_attainment_pct,
      COALESCE(o.n_destinations, 0)         AS n_destinations
    FROM {SCHEMA}.distribution_centers d
    JOIN {SCHEMA}.ndc_capacity c USING (dc_id)
    LEFT JOIN (
      SELECT dc_id,
             SUM(inbound_pallets) / NULLIF(COUNT(DISTINCT DATE(arrival_hour_utc)), 0)
               AS avg_daily_inbound
      FROM {SCHEMA}.ndc_inbound
      GROUP BY dc_id
    ) inb USING (dc_id)
    LEFT JOIN (
      SELECT dc_id,
             SUM(outbound_pallets) / NULLIF(COUNT(DISTINCT DATE(departure_hour_utc)), 0)
               AS avg_daily_outbound,
             AVG(dwell_hours) AS avg_dwell_hours,
             100.0 * AVG(CAST(met_sla AS INT)) AS sla_attainment_pct,
             COUNT(DISTINCT dest_dc_id) AS n_destinations
      FROM {SCHEMA}.ndc_outbound
      GROUP BY dc_id
    ) o USING (dc_id)
    WHERE d.facility_type = 'NDC'
    ORDER BY d.dc_id
    """
    try:
        df = run_query(query)
    except Exception:
        return pd.DataFrame()
    numeric_cols = [
        "latitude", "longitude",
        "max_hourly_inbound_pallets", "max_hourly_outbound_pallets",
        "max_hourly_throughput_pallets", "sla_dwell_hours",
        "avg_daily_inbound_pallets", "avg_daily_outbound_pallets",
        "avg_dwell_hours", "sla_attainment_pct", "n_destinations",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
    return df


@st.cache_data(ttl=APP_CFG["cache_ttl_seconds"], show_spinner=False)
def load_ndc_data(dc_id):
    """Load the three NDC tables filtered to the chosen NDC + the meta row."""
    tables = {
        "ndc_capacity":
            "SELECT * FROM {schema}.ndc_capacity WHERE dc_id = %(dc_id)s",
        "ndc_inbound":
            "SELECT * FROM {schema}.ndc_inbound WHERE dc_id = %(dc_id)s",
        "ndc_outbound":
            "SELECT * FROM {schema}.ndc_outbound WHERE dc_id = %(dc_id)s",
        "distribution_centers":
            "SELECT * FROM {schema}.distribution_centers WHERE dc_id = %(dc_id)s",
    }
    out = {}
    for name, tpl in tables.items():
        out[name] = run_query(tpl.format(schema=SCHEMA), {"dc_id": dc_id})
    return out


# =============================================================================
# CONNECTION DIAGNOSTICS (rendered once at top)
# =============================================================================
def render_diagnostics():
    with st.expander("⚙️ Connection diagnostics", expanded=False):
        warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
        if warehouse_id:
            st.success(f"Warehouse ID: `{warehouse_id}`")
        else:
            st.error(
                "DATABRICKS_WAREHOUSE_ID is **not set**. "
                "Configure the SQL warehouse resource in app.yaml."
            )
            st.stop()
        try:
            _cfg = Config()
            st.success(f"Host: `{_cfg.host}`")
        except Exception as e:
            st.error(f"SDK Config error: {e}")
            st.stop()
        st.info(f"Schema: `{SCHEMA}`")


# =============================================================================
# ABOUT
# =============================================================================
def render_about():
    with st.expander("ℹ️ About this app", expanded=False):
        st.markdown(f"""
        ### What does this app do?

        Models a pharma distribution network with two tiers. The landing
        page shows every facility on a map:

        * **Wholesale DCs** (green/red circles) — storage + outbound nodes;
          click a pin to drill into per-DC inventory and run the LP optimizer.
        * **National DCs / NDCs** (dark-blue diamonds) — 24-hour cross-dock
          hubs that receive product from pharma manufacturers and dispatch
          it onward to Wholesale DCs. Click a pin to open the NDC throughput
          simulation with interactive what-if controls.

        ### Where does the DC list come from?

        `{SCHEMA}.distribution_centers` filtered to
        `facility_type = '{FACILITY_FILTER}'`, inner-joined with
        `{SCHEMA}.dc_capacity` so only DCs with both metadata and capacity data
        are selectable.

        ### Solver

        Prefers **HiGHS** (`highspy`) and falls back to **CBC** if HiGHS is not
        installed. Configurable via `optimization.solver_preference` in
        `config.yaml`.

        ### Data sources
        All input data lives in the `{SCHEMA}` schema.
        """)


# =============================================================================
# MAP VIEW
# =============================================================================
def render_map(df_summary, df_ndc_summary=None):
    st.markdown("### \U0001f5fa️ Distribution Center Network")
    st.caption(
        "Green/red circles = Wholesale DCs (color = current storage utilization, "
        "size = storage capacity). Dark-blue diamonds = National DCs (cross-dock). "
        "Hover any pin for stats; pick a DC below to drill in."
    )

    if df_summary.empty:
        st.error(
            f"No Wholesale DCs found in `{SCHEMA}.distribution_centers` joined "
            f"with `{SCHEMA}.dc_capacity`. Run the 'Generate Synthetic Data' "
            "notebook."
        )
        return

    df = df_summary.copy()
    df["marker_size"] = np.clip(df["total_storage_cube_ft3"] / 2500.0, 10, 32)

    customdata = df[[
        "dc_id", "facility_name", "city", "state_code", "region",
        "on_hand_units", "storage_util_pct", "total_demand_units",
        "n_skus", "n_periods", "automation_level",
        "total_storage_cube_ft3", "max_daily_throughput_units",
    ]].values

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        name="Wholesale DC",
        lon=df["longitude"].astype(float),
        lat=df["latitude"].astype(float),
        text=df["facility_name"],
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[1]}</b> (%{customdata[0]})<br>"
            "%{customdata[2]}, %{customdata[3]} · %{customdata[4]}<br>"
            "<br>"
            "On-hand: %{customdata[5]:,.0f} units<br>"
            "Storage util: %{customdata[6]:.1f}%<br>"
            "Storage capacity: %{customdata[11]:,.0f} cu ft<br>"
            "Max throughput: %{customdata[12]:,.0f} units/day<br>"
            "Demand (%{customdata[9]}d horizon): %{customdata[7]:,.0f} units<br>"
            "SKUs: %{customdata[8]}<br>"
            "Automation: %{customdata[10]}"
            "<extra></extra>"
        ),
        marker=dict(
            size=df["marker_size"],
            color=df["storage_util_pct"].astype(float),
            colorscale="RdYlGn_r",
            cmin=0, cmax=100,
            colorbar=dict(title="WDC<br>Storage<br>Util %", thickness=12, len=0.6,
                          x=1.02),
            line=dict(width=1, color="white"),
        ),
        mode="markers",
    ))

    # NDC trace — dark blue diamond, distinct hovertemplate, larger marker.
    if df_ndc_summary is not None and not df_ndc_summary.empty:
        ndc = df_ndc_summary.copy()
        ndc_custom = ndc[[
            "dc_id", "facility_name", "city", "state_code", "region",
            "avg_daily_inbound_pallets", "avg_daily_outbound_pallets",
            "avg_dwell_hours", "sla_attainment_pct",
            "max_hourly_throughput_pallets", "sla_dwell_hours",
            "n_destinations",
        ]].values
        fig.add_trace(go.Scattergeo(
            name="National DC",
            lon=ndc["longitude"].astype(float),
            lat=ndc["latitude"].astype(float),
            text=ndc["facility_name"],
            customdata=ndc_custom,
            hovertemplate=(
                "<b>%{customdata[1]}</b> (%{customdata[0]})<br>"
                "%{customdata[2]}, %{customdata[3]} · %{customdata[4]}<br>"
                "<br>"
                "Cross-dock SLA: %{customdata[10]:.0f}h<br>"
                "Avg daily inbound: %{customdata[5]:,.0f} pallets<br>"
                "Avg daily outbound: %{customdata[6]:,.0f} pallets<br>"
                "Avg dwell: %{customdata[7]:.1f}h<br>"
                "SLA attainment: %{customdata[8]:.1f}%<br>"
                "Hourly throughput cap: %{customdata[9]:,.0f} pallets<br>"
                "Downstream DCs: %{customdata[11]}"
                "<extra></extra>"
            ),
            marker=dict(
                symbol="diamond",
                size=24,
                color="#0B3D91",
                line=dict(width=2, color="white"),
            ),
            mode="markers",
        ))
    # No `scope` — let lataxis/lonaxis define the viewport so Hawaii (~-157 lon)
    # and Puerto Rico (~-66 lon, ~18 lat) are both inside the visible area.
    fig.update_layout(
        geo=dict(
            projection_type="equirectangular",
            showland=True, landcolor="rgb(243, 243, 243)",
            showsubunits=True, subunitcolor="rgb(217, 217, 217)",
            showcountries=True, countrycolor="rgb(204, 204, 204)",
            showlakes=True, lakecolor="rgb(255, 255, 255)",
            lataxis=dict(range=[15, 52]),
            lonaxis=dict(range=[-162, -64]),
        ),
        height=600,
        margin=dict(l=0, r=0, t=10, b=0),
    )

    # Try Streamlit's on_select if the runtime supports it (Streamlit 1.30+).
    # Older runtimes silently render nothing, so we fall back to a plain chart
    # plus the deterministic dropdown+button below.
    try:
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            key="dc_map",
            on_select="rerun",
            selection_mode="points",
        )
        points = []
        if event is not None:
            sel = getattr(event, "selection", None)
            if sel is None and isinstance(event, dict):
                sel = event.get("selection")
            if sel is not None:
                points = (sel.get("points", [])
                          if isinstance(sel, dict)
                          else getattr(sel, "points", []))
        if points:
            p = points[0]
            cd = p.get("customdata") if isinstance(p, dict) else p["customdata"]
            st.session_state["selected_dc"] = cd[0]
            st.session_state["view"] = "detail"
            st.rerun()
    except TypeError:
        # Streamlit too old for on_select — render the chart non-interactively;
        # the dropdown+button below handles drill-in.
        st.plotly_chart(fig, use_container_width=True)

    # Below-the-map quick-pick fallback. Streamlit click events occasionally do
    # not propagate on first interaction; this gives the user a deterministic
    # way to drill in. Includes both WDCs and NDCs.
    st.markdown("---")
    pick_col, _ = st.columns([1, 2])
    with pick_col:
        rows = []
        for r in df.itertuples():
            rows.append((r.dc_id,
                         f"[WDC] {r.dc_id} — {r.facility_name} ({r.city}, {r.state_code})"))
        if df_ndc_summary is not None and not df_ndc_summary.empty:
            for r in df_ndc_summary.itertuples():
                rows.append((r.dc_id,
                             f"[NDC] {r.dc_id} — {r.facility_name} ({r.city}, {r.state_code})"))
        labels = [lbl for _, lbl in rows]
        ids = [d for d, _ in rows]
        choice = st.selectbox(
            "Or pick a DC from the list",
            options=range(len(labels)),
            format_func=lambda i: labels[i],
            key="map_quick_pick",
        )
        if st.button("Open DC detail", use_container_width=True):
            st.session_state["selected_dc"] = ids[choice]
            st.session_state["view"] = "detail"
            st.rerun()


# =============================================================================
# NDC CROSS-DOCK SIMULATION
# =============================================================================
_ARRIVAL_PATTERN_PRESETS = {
    "Uniform":        [1] * 24,
    "Morning-heavy":  [1, 1, 1, 2, 4, 7, 10, 10, 9, 8, 6, 5,
                       4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1],
    "Evening-heavy":  [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5,
                       6, 7, 8, 9, 10, 10, 9, 7, 5, 3, 2, 1],
}


def simulate_ndc_dispatch(df_inbound_day, capacity_per_hour, vol_mult=1.0,
                          arrival_pattern="Default", sla_hours=24.0):
    """Pure-Python queue simulator for a single day at the NDC.

    Replays the inbound pallets through a per-hour throughput ceiling and
    measures dwell time per pallet. Used by the what-if sliders so the user
    sees instant reactions to volume/capacity/pattern changes.
    """
    base = df_inbound_day.copy()
    if base.empty:
        zeros = np.zeros(24, dtype=int)
        return {
            "hourly_inbound": zeros, "hourly_outbound": zeros, "hourly_queue": zeros,
            "dwell_hours": np.array([]), "sla_pct": 100.0,
            "peak_queue": 0, "avg_dwell": 0.0,
            "total_inbound": 0, "total_outbound": 0,
        }
    base["arrival_hr"] = pd.to_datetime(base["arrival_hour_utc"]).dt.hour
    base_hourly = (
        base.groupby("arrival_hr")["inbound_pallets"].sum()
            .reindex(range(24), fill_value=0)
            .values.astype(float)
    )
    if arrival_pattern in _ARRIVAL_PATTERN_PRESETS:
        weights = np.array(_ARRIVAL_PATTERN_PRESETS[arrival_pattern], dtype=float)
        total = base_hourly.sum() * vol_mult
        hourly_arrivals = (total * (weights / weights.sum())).round().astype(int)
    else:
        hourly_arrivals = (base_hourly * vol_mult).round().astype(int)

    queue = []
    hourly_inb = np.zeros(24, dtype=int)
    hourly_out = np.zeros(24, dtype=int)
    hourly_q = np.zeros(24, dtype=int)
    dwells = []
    for hr in range(24):
        n_arr = int(hourly_arrivals[hr])
        queue.extend([hr] * n_arr)
        n_dep = min(len(queue), int(capacity_per_hour))
        for _ in range(n_dep):
            arr_hr = queue.pop(0)
            dwells.append(hr - arr_hr + 0.5)  # +0.5 to avoid zero-dwell artifacts
        hourly_inb[hr] = n_arr
        hourly_out[hr] = n_dep
        hourly_q[hr] = len(queue)

    # Anything still in queue at hour 24 — extrapolate dwell so SLA reflects misses.
    for arr_hr in queue:
        dwells.append(max(24 - arr_hr, sla_hours + 1))

    dwell_arr = np.array(dwells, dtype=float)
    sla_pct = (100.0 * (dwell_arr <= sla_hours).mean()) if len(dwell_arr) else 100.0
    return {
        "hourly_inbound": hourly_inb,
        "hourly_outbound": hourly_out,
        "hourly_queue": hourly_q,
        "dwell_hours": dwell_arr,
        "sla_pct": float(sla_pct),
        "peak_queue": int(hourly_q.max()),
        "avg_dwell": float(dwell_arr.mean()) if len(dwell_arr) else 0.0,
        "total_inbound": int(hourly_inb.sum()),
        "total_outbound": int(hourly_out.sum()),
    }


# =============================================================================
# NDC DETAIL VIEW
# =============================================================================
def render_ndc_detail(dc_id):
    top = st.columns([1, 4])
    with top[0]:
        if st.button("← Back to map", use_container_width=True, key="ndc_back"):
            st.session_state["view"] = "map"
            st.session_state["selected_dc"] = None
            st.rerun()

    with st.status(f"Loading NDC data for {dc_id}...", expanded=False) as status:
        try:
            data = load_ndc_data(dc_id)
            df_inb = data["ndc_inbound"]
            df_out = data["ndc_outbound"]
            df_cap = data["ndc_capacity"]
            df_meta = data["distribution_centers"]
            if df_cap is None or df_cap.empty:
                status.update(label=f"No ndc_capacity for {dc_id}", state="error")
                st.error(
                    f"`{SCHEMA}.ndc_capacity` has no row for `{dc_id}`. "
                    "Re-run the 'Generate Synthetic Data' notebook with the NDC "
                    "section enabled (`ndc.active_dc_ids` in config.yaml)."
                )
                return
            status.update(
                label=f"Loaded {dc_id}: {len(df_inb):,} inbound, {len(df_out):,} outbound",
                state="complete",
            )
        except Exception as e:
            status.update(label="Failed to load NDC data", state="error")
            st.error(f"Could not load NDC data for {dc_id}: {e}")
            return

    meta = df_meta.iloc[0]
    cap_row = df_cap.iloc[0]
    facility_name = meta["facility_name"]
    sla_hours = float(cap_row["sla_dwell_hours"])
    base_capacity = int(cap_row["max_hourly_throughput_pallets"])

    with top[1]:
        st.markdown(
            f"## \U0001f3ed {dc_id} — {facility_name} (NDC)\n"
            f"**{meta['city']}, {meta['state_code']}** · {sla_hours:.0f}h cross-dock · "
            f"hourly throughput cap **{base_capacity:,} pallets**"
        )

    if df_inb is None or df_inb.empty:
        st.warning(f"No ndc_inbound rows for {dc_id} yet.")
        return

    df_inb = df_inb.copy()
    df_inb["arrival_hour_utc"] = pd.to_datetime(df_inb["arrival_hour_utc"])
    df_inb["arrival_date"] = df_inb["arrival_hour_utc"].dt.date
    df_inb["inbound_pallets"] = pd.to_numeric(df_inb["inbound_pallets"], errors="coerce").fillna(0).astype(int)

    df_out = df_out.copy() if df_out is not None else pd.DataFrame()
    if not df_out.empty:
        df_out["departure_hour_utc"] = pd.to_datetime(df_out["departure_hour_utc"])
        df_out["departure_date"] = df_out["departure_hour_utc"].dt.date
        df_out["outbound_pallets"] = pd.to_numeric(df_out["outbound_pallets"], errors="coerce").fillna(0).astype(int)
        df_out["dwell_hours"] = pd.to_numeric(df_out["dwell_hours"], errors="coerce").fillna(0.0)

    available_days = sorted(df_inb["arrival_date"].unique())

    # Sidebar header
    st.sidebar.caption(f"**{dc_id}** — {facility_name}")
    st.sidebar.markdown(f"{meta['city']}, {meta['state_code']}")
    st.sidebar.markdown("---")
    st.sidebar.header("\U0001f4c5 Day")
    if len(available_days) == 1:
        selected_day = available_days[0]
        st.sidebar.markdown(f"`{selected_day}`")
    else:
        selected_day = st.sidebar.select_slider(
            "Planning day",
            options=available_days,
            value=available_days[0],
            format_func=lambda d: str(d),
        )

    st.sidebar.header("⚙️ What-If")
    vol_mult = st.sidebar.slider("Inbound volume multiplier", 0.5, 2.0, 1.0, 0.1,
                                  help="Scale today's pharma inbound by this factor.")
    cap_mult = st.sidebar.slider("Hourly throughput capacity multiplier", 0.5, 2.0, 1.0, 0.1,
                                  help="Scale the NDC's per-hour throughput ceiling.")
    pattern = st.sidebar.radio(
        "Arrival pattern",
        ["Default", "Uniform", "Morning-heavy", "Evening-heavy"],
        index=0,
        help="Default = whatever pattern was synthesized into ndc_inbound for this day.",
    )

    adjusted_capacity = max(1, int(round(base_capacity * cap_mult)))

    df_inb_day = df_inb[df_inb["arrival_date"] == selected_day]
    sim = simulate_ndc_dispatch(
        df_inb_day, adjusted_capacity,
        vol_mult=vol_mult, arrival_pattern=pattern, sla_hours=sla_hours,
    )

    # KPI row
    st.markdown("---")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Daily Inbound", f"{sim['total_inbound']:,} pallets")
    k2.metric("Daily Outbound", f"{sim['total_outbound']:,} pallets")
    k3.metric(
        "SLA Attainment", f"{sim['sla_pct']:.1f}%",
        delta=f"-{100 - sim['sla_pct']:.1f}% miss" if sim['sla_pct'] < 100 else None,
        delta_color="inverse",
    )
    k4.metric(
        "Avg Dwell", f"{sim['avg_dwell']:.1f}h",
        delta=f"{sim['avg_dwell'] - sla_hours:+.1f}h vs SLA" if sim['avg_dwell'] > sla_hours else None,
        delta_color="inverse",
    )
    k5.metric("Peak Queue", f"{sim['peak_queue']:,} pallets")

    # Hourly flow
    st.subheader(f"\U0001f4ca Hourly Flow — {selected_day}")
    hours = list(range(24))
    fig_flow = go.Figure()
    fig_flow.add_trace(go.Bar(
        x=hours, y=sim["hourly_inbound"], name="Inbound", marker_color="#00CC96"))
    fig_flow.add_trace(go.Bar(
        x=hours, y=sim["hourly_outbound"], name="Outbound", marker_color="#636EFA"))
    fig_flow.add_trace(go.Scatter(
        x=hours, y=sim["hourly_queue"], name="Queue depth (EOH)",
        mode="lines+markers", line=dict(color="#EF553B", width=3),
        yaxis="y2",
    ))
    fig_flow.add_hline(
        y=adjusted_capacity, line_dash="dash", line_color="orange",
        annotation_text=f"Hourly capacity ({adjusted_capacity})",
    )
    fig_flow.update_layout(
        xaxis_title="Hour of day (UTC)",
        yaxis=dict(title="Pallets / hour"),
        yaxis2=dict(title="Queue depth (pallets)", overlaying="y", side="right"),
        barmode="group", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_flow, use_container_width=True)

    # Sankey: pharma -> NDC -> WDC
    if not df_out.empty:
        st.subheader("\U0001f504 Pharma → NDC → Wholesale DCs")
        df_inb_day_for_sankey = df_inb[df_inb["arrival_date"] == selected_day]
        df_out_day = df_out[df_out["departure_date"] == selected_day]

        suppliers = (
            df_inb_day_for_sankey.groupby("supplier_id")["inbound_pallets"].sum()
            .sort_values(ascending=False).reset_index()
        )
        dests = (
            df_out_day.groupby("dest_dc_id")["outbound_pallets"].sum()
            .sort_values(ascending=False).reset_index()
        )

        if not suppliers.empty and not dests.empty:
            ndc_label = facility_name
            node_labels = list(suppliers["supplier_id"]) + [ndc_label] + list(dests["dest_dc_id"])
            ndc_idx = len(suppliers)
            src = list(range(len(suppliers)))
            tgt = [ndc_idx] * len(suppliers)
            vals = list(suppliers["inbound_pallets"].astype(float))
            src += [ndc_idx] * len(dests)
            tgt += [ndc_idx + 1 + i for i in range(len(dests))]
            vals += list(dests["outbound_pallets"].astype(float))

            colors = (
                ["#00CC96"] * len(suppliers)
                + ["#0B3D91"]
                + ["#636EFA"] * len(dests)
            )
            fig_sankey = go.Figure(go.Sankey(
                node=dict(label=node_labels, pad=15, thickness=15, color=colors),
                link=dict(source=src, target=tgt, value=vals),
            ))
            fig_sankey.update_layout(height=520)
            st.plotly_chart(fig_sankey, use_container_width=True)
        else:
            st.caption("Not enough data for Sankey on this day.")

    # Dwell histogram
    if len(sim["dwell_hours"]) > 0:
        st.subheader("⏱️ Dwell Time Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=sim["dwell_hours"], nbinsx=24, marker_color="#636EFA",
            name="Pallets",
        ))
        fig_hist.add_vline(
            x=sla_hours, line_dash="dash", line_color="red",
            annotation_text=f"{sla_hours:.0f}h SLA",
        )
        fig_hist.update_layout(
            xaxis_title="Dwell time (hours)", yaxis_title="Pallet count",
            height=350,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("\U0001f4cb Daily breakdown table", expanded=False):
        per_day_in = (
            df_inb.groupby("arrival_date")["inbound_pallets"].sum().reset_index()
            .rename(columns={"inbound_pallets": "inbound"})
        )
        if not df_out.empty:
            per_day_out = (
                df_out.groupby("departure_date").agg(
                    outbound=("outbound_pallets", "sum"),
                    avg_dwell=("dwell_hours", "mean"),
                    sla_pct=("met_sla", lambda s: 100.0 * s.astype(float).mean()),
                ).reset_index()
                .rename(columns={"departure_date": "arrival_date"})
            )
            per_day = per_day_in.merge(per_day_out, on="arrival_date", how="outer").fillna(0)
        else:
            per_day = per_day_in
        per_day.columns = [c.replace("_", " ").title() for c in per_day.columns]
        st.dataframe(per_day, use_container_width=True, hide_index=True)


# =============================================================================
# OPTIMIZATION MODEL (per-DC)
# =============================================================================
def _make_solver():
    """Return the first solver in solver_preference whose `.available()` is True.

    `pulp.HiGHS_CMD` always constructs, but spawns a `highs` CLI subprocess at
    solve time — that fails on Databricks Apps because only the `highspy`
    Python binding is installed, no CLI binary. Prefer `pulp.HiGHS` (uses the
    binding directly); fall back to `HiGHS_CMD` then CBC. Each candidate is
    gated on `.available()` so we never hand back a solver that will fail at
    `model.solve` time.
    """
    candidates = []
    for name in SOLVER_PREFERENCE:
        n = str(name).lower()
        if n == "highs":
            highs_cls = getattr(pulp, "HiGHS", None)
            if highs_cls is not None:
                candidates.append((highs_cls, "HiGHS"))
            candidates.append((pulp.HiGHS_CMD, "HiGHS_CMD"))
        elif n == "cbc":
            candidates.append((pulp.PULP_CBC_CMD, "CBC"))

    for cls, label in candidates:
        try:
            solver = cls(msg=0)
            if solver.available():
                return solver, label
        except Exception:
            continue
    # Last resort
    return pulp.PULP_CBC_CMD(msg=0), "CBC"


def run_optimization(df_products, df_demand, df_inventory, df_capacity,
                     df_labor, df_throughput, df_inbound, df_params,
                     penalty_stor, penalty_tp, penalty_ob, penalty_ib,
                     cap_mult=1.0, tp_mult=1.0):
    """Build and solve the LP for a single DC. Returns results dict or None."""
    skus = sorted(df_products["sku_id"].unique())
    periods = sorted(df_demand["forecast_date"].unique())

    revenue = dict(zip(df_products["sku_id"], df_products["revenue_per_unit"]))
    holding_cost = dict(zip(df_products["sku_id"], df_products["holding_cost_per_unit_per_day"]))
    cube = dict(zip(df_products["sku_id"], df_products["unit_cube_ft3"]))
    upp = dict(zip(df_products["sku_id"],
                   df_products["units_per_case"] * df_products["cases_per_pallet"]))

    regular_labor_cost = OPT["default_regular_labor_cost_per_hour"]
    overtime_labor_cost = OPT["default_overtime_labor_cost_per_hour"]
    if df_params is not None and not df_params.empty:
        p_dict = dict(zip(df_params["param_name"], df_params["param_value"]))
        regular_labor_cost = float(p_dict.get("regular_labor_cost_per_hour", regular_labor_cost))
        overtime_labor_cost = float(p_dict.get("overtime_labor_cost_per_hour", overtime_labor_cost))

    demand = {(r["sku_id"], r["forecast_date"]): r["demand_units"]
              for _, r in df_demand.iterrows()}
    init_inv = dict(zip(df_inventory["sku_id"], df_inventory["on_hand_units"]))

    inbound_units, inbound_pallets_map = {}, {}
    if df_inbound is not None and not df_inbound.empty:
        for _, r in df_inbound.iterrows():
            k = (r["sku_id"], r["scheduled_date"])
            inbound_units[k] = inbound_units.get(k, 0) + r["inbound_units"]
            inbound_pallets_map[k] = inbound_pallets_map.get(k, 0) + r["inbound_pallets"]

    tp_rate = {}
    if df_throughput is not None and not df_throughput.empty:
        tp_rate = dict(zip(df_throughput["sku_id"], df_throughput["units_per_labor_hour"]))
    default_tp = OPT["default_throughput_rate_units_per_hour"]

    cap = df_capacity.iloc[0]
    max_storage = float(cap["total_storage_cube_ft3"]) * cap_mult
    max_throughput = float(cap["max_daily_throughput_units"]) * tp_mult
    max_inb_pallets = float(cap["max_daily_inbound_pallets"])
    max_ob_pallets = float(cap["max_daily_outbound_pallets"])

    max_reg_hrs, max_ot_hrs = {}, {}
    if df_labor is not None and not df_labor.empty:
        agg = df_labor.groupby("work_date").agg(
            reg=("available_hours", "sum"),
            ot=("overtime_hours_available", "sum"),
        ).reset_index()
        for _, r in agg.iterrows():
            max_reg_hrs[r["work_date"]] = r["reg"]
            max_ot_hrs[r["work_date"]] = r["ot"]
    for t in periods:
        max_reg_hrs.setdefault(t, OPT["default_max_regular_hours_per_period"])
        max_ot_hrs.setdefault(t, OPT["default_max_overtime_hours_per_period"])

    model = pulp.LpProblem("DC_Opt", pulp.LpMaximize)
    f = pulp.LpVariable.dicts("f", ((i, t) for i in skus for t in periods), lowBound=0)
    I = pulp.LpVariable.dicts("I", ((i, t) for i in skus for t in periods), lowBound=0)
    L = pulp.LpVariable.dicts("L", periods, lowBound=0)
    O = pulp.LpVariable.dicts("O", periods, lowBound=0)
    ss = pulp.LpVariable.dicts("ss", periods, lowBound=0)
    st_var = pulp.LpVariable.dicts("st", periods, lowBound=0)
    so = pulp.LpVariable.dicts("so", periods, lowBound=0)
    si = pulp.LpVariable.dicts("si", periods, lowBound=0)

    model += (
        pulp.lpSum(revenue[i] * f[i, t] for i in skus for t in periods)
        - pulp.lpSum(holding_cost[i] * I[i, t] for i in skus for t in periods)
        - pulp.lpSum(regular_labor_cost * L[t] + overtime_labor_cost * O[t] for t in periods)
        - pulp.lpSum(penalty_stor * ss[t] for t in periods)
        - pulp.lpSum(penalty_tp * st_var[t] for t in periods)
        - pulp.lpSum(penalty_ob * so[t] for t in periods)
        - pulp.lpSum(penalty_ib * si[t] for t in periods)
    )

    for i in skus:
        for t_idx, t in enumerate(periods):
            ib_qty = inbound_units.get((i, t), 0)
            if t_idx == 0:
                model += I[i, t] == init_inv.get(i, 0) + ib_qty - f[i, t]
            else:
                model += I[i, t] == I[i, periods[t_idx - 1]] + ib_qty - f[i, t]
            model += f[i, t] <= demand.get((i, t), 0)

    for t in periods:
        model += pulp.lpSum(cube[i] * I[i, t] for i in skus) <= max_storage + ss[t]
        model += pulp.lpSum(f[i, t] / tp_rate.get(i, default_tp) for i in skus) <= L[t] + O[t]
        model += L[t] <= max_reg_hrs[t]
        model += O[t] <= max_ot_hrs[t]
        model += pulp.lpSum(f[i, t] / upp[i] for i in skus) <= max_ob_pallets + so[t]
        model += pulp.lpSum(f[i, t] for i in skus) <= max_throughput + st_var[t]
        total_ib_pallets = sum(inbound_pallets_map.get((i, t), 0) for i in skus)
        model += total_ib_pallets <= max_inb_pallets + si[t]

    solver, solver_name = _make_solver()
    model.solve(solver)
    if model.status != pulp.constants.LpStatusOptimal:
        return None

    records = []
    for i in skus:
        for t in periods:
            d_val = demand.get((i, t), 0)
            f_val = f[i, t].varValue or 0
            records.append({
                "sku_id": i, "period": str(t),
                "fulfilled": round(f_val, 1), "demand": d_val,
                "fill_rate": round(f_val / d_val * 100, 1) if d_val > 0 else 100.0,
                "inventory": round(I[i, t].varValue or 0, 1),
                "revenue": round(f_val * revenue[i], 2),
            })

    overflow = []
    for t in periods:
        total_cube = sum((I[i, t].varValue or 0) * cube[i] for i in skus)
        total_ful = sum((f[i, t].varValue or 0) for i in skus)
        outbound_pallets = sum((f[i, t].varValue or 0) / upp[i] for i in skus)
        inbound_pallets = sum(inbound_pallets_map.get((i, t), 0) for i in skus)
        ob_util_pct = (outbound_pallets / max_ob_pallets * 100) if max_ob_pallets > 0 else 0
        ib_util_pct = (inbound_pallets / max_inb_pallets * 100) if max_inb_pallets > 0 else 0
        overflow.append({
            "period": str(t),
            "storage_util_pct":    round(total_cube / max_storage * 100, 1) if max_storage > 0 else 0,
            "throughput_util_pct": round(total_ful / max_throughput * 100, 1) if max_throughput > 0 else 0,
            "outbound_util_pct":   round(ob_util_pct, 1),
            "inbound_util_pct":    round(ib_util_pct, 1),
            "outbound_pallets":    round(outbound_pallets, 1),
            "inbound_pallets":     round(inbound_pallets, 1),
            "throughput_units":    round(total_ful, 1),
            "storage_overflow":    round(ss[t].varValue or 0, 1),
            "throughput_overflow": round(st_var[t].varValue or 0, 1),
            "outbound_overflow":   round(so[t].varValue or 0, 1),
            "inbound_overflow":    round(si[t].varValue or 0, 1),
            "storage_penalty":    round((ss[t].varValue or 0) * penalty_stor, 2),
            "throughput_penalty": round((st_var[t].varValue or 0) * penalty_tp, 2),
            "outbound_penalty":   round((so[t].varValue or 0) * penalty_ob, 2),
            "inbound_penalty":    round((si[t].varValue or 0) * penalty_ib, 2),
        })

    labor_out = []
    for t in periods:
        labor_out.append({
            "period": str(t),
            "regular_hrs":  round(L[t].varValue or 0, 1),
            "overtime_hrs": round(O[t].varValue or 0, 1),
            "reg_util_pct": round((L[t].varValue or 0) / max_reg_hrs[t] * 100, 1)
                            if max_reg_hrs[t] > 0 else 0,
        })

    df_ful = pd.DataFrame(records)
    df_ov = pd.DataFrame(overflow)
    df_lab = pd.DataFrame(labor_out)

    total_rev = df_ful["revenue"].sum()
    total_penalty = (df_ov["storage_penalty"].sum() + df_ov["throughput_penalty"].sum()
                     + df_ov["outbound_penalty"].sum() + df_ov["inbound_penalty"].sum())
    total_labor_cost = sum(
        (L[t].varValue or 0) * regular_labor_cost
        + (O[t].varValue or 0) * overtime_labor_cost
        for t in periods
    )
    total_holding = sum((I[i, t].varValue or 0) * holding_cost[i]
                        for i in skus for t in periods)
    total_demand_units = df_ful["demand"].sum()

    return {
        "fulfillment": df_ful, "overflow": df_ov, "labor": df_lab,
        "objective": pulp.value(model.objective),
        "total_revenue": float(total_rev),
        "total_penalty": float(total_penalty),
        "total_labor_cost": float(total_labor_cost),
        "total_holding_cost": float(total_holding),
        "net_profit": float(total_rev - total_holding - total_labor_cost - total_penalty),
        "fill_rate": float(df_ful["fulfilled"].sum() / total_demand_units * 100)
                     if total_demand_units > 0 else 0.0,
        "solver_name": solver_name,
        "max_storage": float(max_storage),
        "max_throughput": float(max_throughput),
        "max_ob_pallets": float(max_ob_pallets),
        "max_inb_pallets": float(max_inb_pallets),
    }


# =============================================================================
# DETAIL VIEW
# =============================================================================
def render_detail(dc_id, df_summary):
    # Back button + header
    top = st.columns([1, 4])
    with top[0]:
        if st.button("← Back to map", use_container_width=True):
            st.session_state["view"] = "map"
            st.session_state["selected_dc"] = None
            st.rerun()

    # Load DC-scoped data (cached per dc_id)
    with st.status(f"Loading data for {dc_id}...", expanded=False) as status:
        try:
            dc_tables = load_dc_data(dc_id)
            df_products = load_product_master()
            df_demand = dc_tables["demand_forecast"]
            df_inventory = dc_tables["inventory_levels"]
            df_capacity = dc_tables["dc_capacity"]
            df_labor = dc_tables["labor_availability"]
            df_throughput = dc_tables["throughput_rates"]
            df_inbound = dc_tables["inbound_plan"]
            df_outbound = dc_tables["outbound_plan"]
            df_params = dc_tables["optimization_parameters"]
            df_dc_meta = dc_tables["distribution_centers"]

            if df_capacity is None or df_capacity.empty:
                status.update(label=f"No capacity data for {dc_id}", state="error")
                st.error(f"`{SCHEMA}.dc_capacity` has no row for `{dc_id}`.")
                return
            if df_dc_meta is None or df_dc_meta.empty:
                status.update(label=f"No metadata for {dc_id}", state="error")
                st.error(f"`{SCHEMA}.distribution_centers` has no row for `{dc_id}`.")
                return

            status.update(
                label=f"Loaded {dc_id}: {len(df_demand):,} demand, "
                      f"{len(df_inventory):,} inventory, "
                      f"{len(df_inbound):,} inbound rows",
                state="complete",
            )
        except Exception as e:
            status.update(label="Failed to load DC data", state="error")
            st.error(f"Could not load data for {dc_id}: {e}")
            return

    dc_meta = df_dc_meta.iloc[0]
    facility_name = dc_meta["facility_name"]
    city = dc_meta.get("city", "")
    state_code = dc_meta.get("state_code", "")
    region = dc_meta.get("region", "")

    with top[1]:
        st.markdown(
            f"## \U0001f3ed {dc_id} — {facility_name}\n"
            f"**{city}, {state_code}** · {region} · "
            f"automation: {dc_meta.get('automation_level', 'n/a')}"
        )

    st.sidebar.caption(f"**{dc_id}** — {facility_name}")
    st.sidebar.markdown(f"{city}, {state_code}")
    st.sidebar.markdown("---")

    # --- Current Inventory ---
    st.markdown("---")
    st.subheader("\U0001f4e6 Current Inventory Position")

    df_inv_enriched = df_inventory.merge(
        df_products[["sku_id", "unit_cube_ft3", "revenue_per_unit",
                     "holding_cost_per_unit_per_day",
                     "units_per_case", "cases_per_pallet"]],
        on="sku_id", how="left",
    )
    df_inv_enriched["total_cube_ft3"] = (
        df_inv_enriched["on_hand_units"] * df_inv_enriched["unit_cube_ft3"]
    )
    df_inv_enriched["inventory_value"] = (
        df_inv_enriched["on_hand_units"] * df_inv_enriched["revenue_per_unit"]
    )
    df_inv_enriched["daily_holding_cost"] = (
        df_inv_enriched["on_hand_units"] * df_inv_enriched["holding_cost_per_unit_per_day"]
    )

    cap_row = df_capacity.iloc[0]
    total_storage_capacity = float(cap_row["total_storage_cube_ft3"])
    current_total_cube = df_inv_enriched["total_cube_ft3"].sum()
    storage_util_pct = (
        current_total_cube / total_storage_capacity * 100
        if total_storage_capacity > 0 else 0.0
    )

    inv_col1, inv_col2, inv_col3, inv_col4 = st.columns(4)
    inv_col1.metric("Total Units On-Hand", f"{df_inv_enriched['on_hand_units'].sum():,.0f}")
    inv_col2.metric(
        "Storage Utilization", f"{storage_util_pct:.1f}%",
        delta=f"{storage_util_pct - 100:.1f}% vs capacity" if storage_util_pct > 100 else None,
        delta_color="inverse",
    )
    inv_col3.metric("Inventory Value", f"${df_inv_enriched['inventory_value'].sum():,.0f}")
    inv_col4.metric("Daily Holding Cost", f"${df_inv_enriched['daily_holding_cost'].sum():,.0f}")

    inv_left, inv_right = st.columns(2)
    with inv_left:
        fig_inv_units = px.bar(
            df_inv_enriched.sort_values("on_hand_units", ascending=False),
            x="sku_id", y="on_hand_units",
            color="days_of_supply",
            color_continuous_scale="RdYlGn",
            title="On-Hand Inventory by SKU",
            labels={"on_hand_units": "Units", "sku_id": "SKU",
                    "days_of_supply": "Days of Supply"},
        )
        fig_inv_units.update_layout(height=350)
        st.plotly_chart(fig_inv_units, use_container_width=True)

    with inv_right:
        fig_cube = px.bar(
            df_inv_enriched.sort_values("total_cube_ft3", ascending=False),
            x="sku_id", y="total_cube_ft3",
            title="Cube Utilization by SKU (cu ft)",
            labels={"total_cube_ft3": "Cubic Feet", "sku_id": "SKU"},
            color_discrete_sequence=["#636EFA"],
        )
        if len(df_inv_enriched) > 0:
            fig_cube.add_hline(
                y=total_storage_capacity / len(df_inv_enriched),
                line_dash="dash", line_color="red",
                annotation_text="Avg capacity per SKU",
            )
        fig_cube.update_layout(height=350)
        st.plotly_chart(fig_cube, use_container_width=True)

    with st.expander("\U0001f4cb Inventory detail table", expanded=False):
        inv_display = df_inv_enriched[[
            "sku_id", "on_hand_units", "available_units", "allocated_units",
            "days_of_supply", "total_cube_ft3", "inventory_value", "daily_holding_cost",
        ]].copy()
        inv_display.columns = [
            "SKU", "On-Hand Units", "Available Units", "Allocated Units",
            "Days of Supply", "Cube (cu ft)", "Inventory Value ($)", "Daily Holding Cost ($)",
        ]
        inv_display["Cube (cu ft)"] = inv_display["Cube (cu ft)"].round(1)
        inv_display["Inventory Value ($)"] = inv_display["Inventory Value ($)"].round(2)
        inv_display["Daily Holding Cost ($)"] = inv_display["Daily Holding Cost ($)"].round(2)
        st.dataframe(inv_display, use_container_width=True, hide_index=True)

    # --- Demand vs Coverage ---
    st.markdown("---")
    st.subheader("\U0001f4ca Demand vs. Inventory Coverage")
    st.caption(
        "Compares current on-hand inventory against total forecasted demand "
        "over the planning horizon."
    )

    demand_by_sku = df_demand.groupby("sku_id")["demand_units"].sum().reset_index()
    demand_by_sku.columns = ["sku_id", "total_demand"]

    inbound_by_sku = pd.DataFrame({"sku_id": df_inv_enriched["sku_id"], "inbound_units": 0})
    if df_inbound is not None and not df_inbound.empty:
        inbound_by_sku = df_inbound.groupby("sku_id")["inbound_units"].sum().reset_index()

    df_coverage = (
        df_inv_enriched[["sku_id", "on_hand_units"]]
        .merge(demand_by_sku, on="sku_id", how="left")
        .merge(inbound_by_sku, on="sku_id", how="left")
    )
    df_coverage["inbound_units"] = df_coverage["inbound_units"].fillna(0)
    df_coverage["total_demand"] = df_coverage["total_demand"].fillna(0)
    df_coverage["total_supply"] = df_coverage["on_hand_units"] + df_coverage["inbound_units"]
    df_coverage["coverage_ratio"] = np.where(
        df_coverage["total_demand"] > 0,
        (df_coverage["total_supply"] / df_coverage["total_demand"]).round(2),
        np.inf,
    )
    df_coverage["surplus_deficit"] = df_coverage["total_supply"] - df_coverage["total_demand"]
    df_coverage["at_risk"] = df_coverage["coverage_ratio"] < 1.0

    n_planning_days = df_demand["forecast_date"].nunique()

    dc_col1, dc_col2, dc_col3, dc_col4 = st.columns(4)
    total_demand_all = df_coverage["total_demand"].sum()
    total_supply_all = df_coverage["total_supply"].sum()
    overall_coverage = total_supply_all / total_demand_all if total_demand_all > 0 else 0
    at_risk_count = int(df_coverage["at_risk"].sum())

    dc_col1.metric(f"Total Demand ({n_planning_days}-day)", f"{total_demand_all:,.0f} units")
    dc_col2.metric("Total Supply (On-Hand + Inbound)", f"{total_supply_all:,.0f} units")
    dc_col3.metric(
        "Overall Coverage Ratio", f"{overall_coverage:.2f}x",
        delta=f"{overall_coverage - 1.0:+.2f}x vs breakeven" if overall_coverage < 1.0 else None,
        delta_color="inverse" if overall_coverage < 1.0 else "normal",
    )
    dc_col4.metric(
        "SKUs At Risk", f"{at_risk_count} of {len(df_coverage)}",
        delta=f"{at_risk_count} under-stocked" if at_risk_count > 0 else "All covered",
        delta_color="inverse" if at_risk_count > 0 else "off",
    )

    dem_left, dem_right = st.columns(2)
    with dem_left:
        df_coverage_sorted = df_coverage.sort_values("coverage_ratio")
        fig_cov = go.Figure()
        fig_cov.add_trace(go.Bar(
            x=df_coverage_sorted["sku_id"], y=df_coverage_sorted["on_hand_units"],
            name="On-Hand Inventory", marker_color="#636EFA"))
        fig_cov.add_trace(go.Bar(
            x=df_coverage_sorted["sku_id"], y=df_coverage_sorted["inbound_units"],
            name="Scheduled Inbound", marker_color="#00CC96"))
        fig_cov.add_trace(go.Bar(
            x=df_coverage_sorted["sku_id"], y=df_coverage_sorted["total_demand"],
            name=f"Total Demand ({n_planning_days}d)", marker_color="#EF553B",
            opacity=0.7))
        fig_cov.update_layout(
            title="Inventory + Inbound vs. Forecasted Demand by SKU",
            yaxis_title="Units", barmode="group", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_cov, use_container_width=True)

    with dem_right:
        demand_daily = df_demand.groupby("forecast_date")["demand_units"].sum().reset_index()
        demand_daily.columns = ["date", "total_demand"]
        demand_daily["date"] = pd.to_datetime(demand_daily["date"])
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=demand_daily["date"], y=demand_daily["total_demand"],
            mode="lines+markers", name="Daily Demand",
            line=dict(width=3, color="#EF553B"),
            fill="tozeroy", fillcolor="rgba(239, 85, 59, 0.1)",
        ))
        avg_daily_supply = total_supply_all / max(n_planning_days, 1)
        fig_daily.add_hline(
            y=avg_daily_supply, line_dash="dash", line_color="#636EFA",
            annotation_text=f"Avg daily supply: {avg_daily_supply:,.0f}",
        )
        fig_daily.update_layout(
            title="Daily Demand Over Planning Horizon",
            xaxis_title="Date", yaxis_title="Units", height=400,
        )
        st.plotly_chart(fig_daily, use_container_width=True)

    with st.expander("\U0001f50d Coverage detail by SKU", expanded=False):
        cov_display = df_coverage[[
            "sku_id", "on_hand_units", "inbound_units", "total_supply",
            "total_demand", "surplus_deficit", "coverage_ratio", "at_risk",
        ]].copy()
        cov_display.columns = [
            "SKU", "On-Hand", "Inbound", "Total Supply",
            "Total Demand", "Surplus / Deficit", "Coverage Ratio", "At Risk",
        ]
        cov_display["At Risk"] = cov_display["At Risk"].map({True: "⚠️ YES", False: "✅ No"})
        cov_display = cov_display.sort_values("Coverage Ratio")
        st.dataframe(cov_display, use_container_width=True, hide_index=True)

    # --- Outbound Throughput (WDC → Local Couriers) ---
    if df_outbound is not None and not df_outbound.empty:
        st.markdown("---")
        st.subheader("\U0001f69a Outbound Throughput to Local Couriers")
        st.caption(
            "Product leaving this DC for last-mile delivery. Each pallet is "
            "tendered to a local courier (FedEx, UPS, XPO, or SAIA)."
        )

        df_ob = df_outbound.copy()
        df_ob["scheduled_ship_date"] = pd.to_datetime(df_ob["scheduled_ship_date"])
        df_ob["outbound_pallets"] = pd.to_numeric(df_ob["outbound_pallets"], errors="coerce").fillna(0)
        df_ob["outbound_units"] = pd.to_numeric(df_ob["outbound_units"], errors="coerce").fillna(0).astype(int)

        daily_ob = (
            df_ob.groupby("scheduled_ship_date")
                .agg(pallets=("outbound_pallets", "sum"),
                     units=("outbound_units", "sum"),
                     orders=("order_number", "nunique"))
                .reset_index()
                .sort_values("scheduled_ship_date")
        )

        max_ob_pallets = float(df_capacity.iloc[0].get("max_daily_outbound_pallets", 0) or 0)
        avg_daily = daily_ob["pallets"].mean() if len(daily_ob) else 0.0
        peak_daily = daily_ob["pallets"].max() if len(daily_ob) else 0.0
        peak_util = (100.0 * peak_daily / max_ob_pallets) if max_ob_pallets > 0 else 0.0
        avg_util = (100.0 * avg_daily / max_ob_pallets) if max_ob_pallets > 0 else 0.0
        total_units = int(df_ob["outbound_units"].sum())
        total_orders = int(df_ob["order_number"].nunique())

        ob_col1, ob_col2, ob_col3, ob_col4, ob_col5 = st.columns(5)
        ob_col1.metric("Avg Daily Outbound", f"{avg_daily:,.0f} pallets")
        ob_col2.metric(
            "Peak Daily Outbound", f"{peak_daily:,.0f} pallets",
            delta=f"{peak_util:.0f}% of dock capacity" if max_ob_pallets > 0 else None,
            delta_color="inverse" if peak_util > 100 else "normal",
        )
        ob_col3.metric(
            "Avg Dock Utilization", f"{avg_util:.1f}%" if max_ob_pallets > 0 else "n/a",
            help=f"Outbound dock capacity: {max_ob_pallets:,.0f} pallets/day",
        )
        ob_col4.metric("Total Units Shipped", f"{total_units:,}")
        ob_col5.metric("Total Orders", f"{total_orders:,}")

        ob_left, ob_right = st.columns([2, 1])
        with ob_left:
            fig_ob = go.Figure()
            fig_ob.add_trace(go.Bar(
                x=daily_ob["scheduled_ship_date"], y=daily_ob["pallets"],
                name="Pallets shipped", marker_color="#636EFA",
            ))
            if max_ob_pallets > 0:
                fig_ob.add_hline(
                    y=max_ob_pallets, line_dash="dash", line_color="red",
                    annotation_text=f"Dock capacity ({int(max_ob_pallets):,}/day)",
                )
            fig_ob.update_layout(
                title="Daily Outbound Volume to Local Couriers",
                xaxis_title="Ship date", yaxis_title="Pallets",
                height=400,
            )
            st.plotly_chart(fig_ob, use_container_width=True)

        with ob_right:
            if "carrier_id" in df_ob.columns:
                carrier_mix = (
                    df_ob.groupby("carrier_id")["outbound_pallets"].sum()
                        .sort_values(ascending=False)
                        .reset_index()
                )
                fig_carrier = px.pie(
                    carrier_mix, values="outbound_pallets", names="carrier_id",
                    title="Courier Mix",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    hole=0.45,
                )
                fig_carrier.update_layout(height=400, showlegend=True,
                                          legend=dict(orientation="h", y=-0.1))
                st.plotly_chart(fig_carrier, use_container_width=True)

        with st.expander("\U0001f4cb Daily outbound detail", expanded=False):
            disp = daily_ob.copy()
            disp.columns = ["Ship Date", "Pallets", "Units", "Orders"]
            disp["Pallets"] = disp["Pallets"].round(1)
            if max_ob_pallets > 0:
                disp["Capacity %"] = (100.0 * disp["Pallets"] / max_ob_pallets).round(1)
            st.dataframe(disp, use_container_width=True, hide_index=True)

    # --- Sidebar: Scenario Parameters ---
    st.sidebar.header("⚙️ Scenario Parameters")
    st.sidebar.markdown("Adjust penalty costs to model different operational strategies.")

    st.sidebar.subheader("Penalty Costs")
    penalty_storage = st.sidebar.slider(
        "Storage overflow ($/cu ft/day)",
        min_value=float(PSLIDERS["storage"]["min"]),
        max_value=float(PSLIDERS["storage"]["max"]),
        value=float(PSLIDERS["storage"]["default"]),
        step=float(PSLIDERS["storage"]["step"]),
        help="Cost of renting overflow storage per cubic foot per day",
    )
    penalty_throughput = st.sidebar.slider(
        "Throughput overflow ($/unit)",
        min_value=float(PSLIDERS["throughput"]["min"]),
        max_value=float(PSLIDERS["throughput"]["max"]),
        value=float(PSLIDERS["throughput"]["default"]),
        step=float(PSLIDERS["throughput"]["step"]),
        help="Cost of expediting beyond normal throughput capacity",
    )
    penalty_outbound = st.sidebar.slider(
        "Outbound dock overflow ($/pallet)",
        min_value=float(PSLIDERS["outbound"]["min"]),
        max_value=float(PSLIDERS["outbound"]["max"]),
        value=float(PSLIDERS["outbound"]["default"]),
        step=float(PSLIDERS["outbound"]["step"]),
        help="Cost of trailer detention and staging delays",
    )
    penalty_inbound = st.sidebar.slider(
        "Inbound dock overflow ($/pallet)",
        min_value=float(PSLIDERS["inbound"]["min"]),
        max_value=float(PSLIDERS["inbound"]["max"]),
        value=float(PSLIDERS["inbound"]["default"]),
        step=float(PSLIDERS["inbound"]["step"]),
        help="Cost of carrier detention and yard holds",
    )

    st.sidebar.subheader("Capacity Overrides")
    capacity_multiplier = st.sidebar.slider(
        "Storage capacity multiplier",
        min_value=float(CMULTS["storage"]["min"]),
        max_value=float(CMULTS["storage"]["max"]),
        value=float(CMULTS["storage"]["default"]),
        step=float(CMULTS["storage"]["step"]),
        help="Simulate expanded or reduced storage (1.0 = current)",
    )
    throughput_multiplier = st.sidebar.slider(
        "Throughput capacity multiplier",
        min_value=float(CMULTS["throughput"]["min"]),
        max_value=float(CMULTS["throughput"]["max"]),
        value=float(CMULTS["throughput"]["default"]),
        step=float(CMULTS["throughput"]["step"]),
        help="Simulate expanded or reduced processing capacity",
    )

    st.sidebar.subheader("Scenario Comparison")
    enable_comparison = st.sidebar.checkbox("Enable scenario comparison", value=False)
    if enable_comparison:
        st.sidebar.markdown("**Comparison scenario penalties:**")
        comp_penalty_storage = st.sidebar.number_input(
            "Comp: Storage ($/cu ft)", value=2 * float(PSLIDERS["storage"]["default"]))
        comp_penalty_throughput = st.sidebar.number_input(
            "Comp: Throughput ($/unit)", value=2 * float(PSLIDERS["throughput"]["default"]))
        comp_penalty_outbound = st.sidebar.number_input(
            "Comp: Outbound ($/pallet)", value=2 * float(PSLIDERS["outbound"]["default"]))
        comp_penalty_inbound = st.sidebar.number_input(
            "Comp: Inbound ($/pallet)", value=2 * float(PSLIDERS["inbound"]["default"]))

    RESULTS_KEY = f"results::{dc_id}"
    COMP_KEY = f"comp_results::{dc_id}"

    if st.sidebar.button("\U0001f680 Run Optimization", type="primary", use_container_width=True):
        with st.spinner(f"Solving optimization model for {dc_id}..."):
            results = run_optimization(
                df_products, df_demand, df_inventory, df_capacity,
                df_labor, df_throughput, df_inbound, df_params,
                penalty_storage, penalty_throughput, penalty_outbound, penalty_inbound,
                capacity_multiplier, throughput_multiplier,
            )
            st.session_state[RESULTS_KEY] = results
            if enable_comparison:
                comp_results = run_optimization(
                    df_products, df_demand, df_inventory, df_capacity,
                    df_labor, df_throughput, df_inbound, df_params,
                    comp_penalty_storage, comp_penalty_throughput,
                    comp_penalty_outbound, comp_penalty_inbound,
                    capacity_multiplier, throughput_multiplier,
                )
                st.session_state[COMP_KEY] = comp_results

    # --- Results ---
    if RESULTS_KEY in st.session_state and st.session_state[RESULTS_KEY] is not None:
        results = st.session_state[RESULTS_KEY]
        df_ov = results["overflow"]
        df_ful = results["fulfillment"]
        df_lab = results["labor"]

        st.markdown("---")
        st.subheader(f"\U0001f4c8 Optimization Results — {dc_id}")
        st.caption(f"Solver: **{results.get('solver_name', 'unknown')}**")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Net Profit", f"${results['net_profit']:,.0f}")
        col2.metric("Revenue", f"${results['total_revenue']:,.0f}")
        col3.metric("Fill Rate", f"{results['fill_rate']:.1f}%")
        col4.metric(
            "Overflow Penalty", f"${results['total_penalty']:,.0f}",
            delta=f"-${results['total_penalty']:,.0f}" if results['total_penalty'] > 0 else None,
            delta_color="inverse",
        )
        col5.metric("Labor Cost", f"${results['total_labor_cost']:,.0f}")

        if enable_comparison and COMP_KEY in st.session_state and st.session_state[COMP_KEY]:
            comp = st.session_state[COMP_KEY]
            st.markdown("### Scenario Comparison")
            comp_df = pd.DataFrame({
                "Metric": ["Net Profit", "Revenue", "Fill Rate (%)",
                           "Overflow Penalty", "Labor Cost"],
                "Primary Scenario": [
                    f"${results['net_profit']:,.0f}", f"${results['total_revenue']:,.0f}",
                    f"{results['fill_rate']:.1f}%", f"${results['total_penalty']:,.0f}",
                    f"${results['total_labor_cost']:,.0f}",
                ],
                "Comparison Scenario": [
                    f"${comp['net_profit']:,.0f}", f"${comp['total_revenue']:,.0f}",
                    f"{comp['fill_rate']:.1f}%", f"${comp['total_penalty']:,.0f}",
                    f"${comp['total_labor_cost']:,.0f}",
                ],
                "Delta": [
                    f"${comp['net_profit'] - results['net_profit']:+,.0f}",
                    f"${comp['total_revenue'] - results['total_revenue']:+,.0f}",
                    f"{comp['fill_rate'] - results['fill_rate']:+.1f}%",
                    f"${comp['total_penalty'] - results['total_penalty']:+,.0f}",
                    f"${comp['total_labor_cost'] - results['total_labor_cost']:+,.0f}",
                ],
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "\U0001f4ca Capacity Utilization", "\U0001f69a Outbound Volume",
            "⚠️ Overflow Analysis",
            "\U0001f4e6 Fill Rate by SKU", "\U0001f477 Labor Allocation",
            "\U0001f4e6 Inventory Trajectory",
        ])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_ov["period"], y=df_ov["storage_util_pct"],
                                 name="Storage %", marker_color="#636EFA"))
            fig.add_trace(go.Bar(x=df_ov["period"], y=df_ov["throughput_util_pct"],
                                 name="Throughput %", marker_color="#EF553B"))
            fig.add_trace(go.Bar(x=df_ov["period"], y=df_ov.get("outbound_util_pct", 0),
                                 name="Outbound dock %", marker_color="#00CC96"))
            fig.add_trace(go.Bar(x=df_ov["period"], y=df_ov.get("inbound_util_pct", 0),
                                 name="Inbound dock %", marker_color="#FFA15A"))
            fig.add_hline(y=100, line_dash="dash", line_color="red",
                          annotation_text="Capacity limit")
            fig.update_layout(title="Capacity Utilization by Period (all 4 constraints)",
                              yaxis_title="Utilization %", barmode="group", height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Each bar shows how much of that constraint's daily capacity the "
                "optimizer's plan consumes. Anything over 100% becomes overflow "
                "(see the Overflow Analysis tab)."
            )

        with tab2:
            # Real outbound volume from the LP solution, vs the dock capacity.
            max_ob = float(results.get("max_ob_pallets", 0) or 0)
            max_inb = float(results.get("max_inb_pallets", 0) or 0)
            max_tp = float(results.get("max_throughput", 0) or 0)

            ob_col1, ob_col2, ob_col3, ob_col4 = st.columns(4)
            ob_col1.metric("Total Outbound (LP)",
                           f"{df_ov['outbound_pallets'].sum():,.0f} pallets")
            ob_col2.metric("Total Outbound (units)",
                           f"{df_ov['throughput_units'].sum():,.0f}")
            ob_col3.metric("Peak Day Outbound",
                           f"{df_ov['outbound_pallets'].max():,.0f} pallets",
                           delta=f"{df_ov['outbound_util_pct'].max():.0f}% of dock"
                                  if max_ob > 0 else None,
                           delta_color="inverse" if df_ov['outbound_util_pct'].max() > 100 else "normal")
            ob_col4.metric("Avg Daily Outbound",
                           f"{df_ov['outbound_pallets'].mean():,.0f} pallets")

            st.markdown(
                "**LP-planned outbound volume per day** — what the optimizer "
                "chose to ship downstream to local couriers. Red dashed line is "
                "the WDC's outbound dock capacity."
            )
            fig_ob = go.Figure()
            fig_ob.add_trace(go.Bar(
                x=df_ov["period"], y=df_ov["outbound_pallets"],
                name="Outbound pallets (LP plan)", marker_color="#636EFA",
            ))
            if max_ob > 0:
                fig_ob.add_hline(
                    y=max_ob, line_dash="dash", line_color="red",
                    annotation_text=f"Dock capacity ({int(max_ob):,}/day)",
                )
            fig_ob.update_layout(
                title="Outbound Pallets per Day (LP Solution)",
                xaxis_title="Period", yaxis_title="Pallets", height=400,
            )
            st.plotly_chart(fig_ob, use_container_width=True)

            st.markdown("**Throughput (units) per day**")
            fig_tp = go.Figure()
            fig_tp.add_trace(go.Bar(
                x=df_ov["period"], y=df_ov["throughput_units"],
                name="Units shipped", marker_color="#00CC96",
            ))
            if max_tp > 0:
                fig_tp.add_hline(
                    y=max_tp, line_dash="dash", line_color="red",
                    annotation_text=f"Throughput capacity ({int(max_tp):,} units/day)",
                )
            fig_tp.update_layout(
                xaxis_title="Period", yaxis_title="Units", height=350,
            )
            st.plotly_chart(fig_tp, use_container_width=True)

        with tab3:
            penalty_breakdown = pd.DataFrame({
                "Period": df_ov["period"],
                "Storage ($)": df_ov["storage_penalty"],
                "Throughput ($)": df_ov["throughput_penalty"],
                "Outbound ($)": df_ov["outbound_penalty"],
                "Inbound ($)": df_ov["inbound_penalty"],
            })
            fig2 = px.bar(
                penalty_breakdown.melt(id_vars="Period", var_name="Type", value_name="Cost"),
                x="Period", y="Cost", color="Type",
                title="Overflow Penalty Cost by Type and Period",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Physical Overflow Quantities (amounts exceeding capacity)")
            st.caption(
                "These are the LP's *slack* variables — quantities the plan went "
                "past capacity. Zero means the plan stayed inside the constraint."
            )
            overflow_qty = pd.DataFrame({
                "Period": df_ov["period"],
                "Storage overflow (cu ft)":    df_ov["storage_overflow"],
                "Throughput overflow (units)": df_ov["throughput_overflow"],
                "Outbound overflow (pallets)": df_ov["outbound_overflow"],
                "Inbound overflow (pallets)":  df_ov["inbound_overflow"],
            })
            st.dataframe(overflow_qty, use_container_width=True, hide_index=True)

        with tab4:
            sku_agg = df_ful.groupby("sku_id").agg(
                total_demand=("demand", "sum"),
                total_fulfilled=("fulfilled", "sum"),
                total_revenue=("revenue", "sum"),
            ).reset_index()
            sku_agg["fill_rate"] = np.where(
                sku_agg["total_demand"] > 0,
                (sku_agg["total_fulfilled"] / sku_agg["total_demand"] * 100).round(1),
                100.0,
            )
            sku_agg = sku_agg.sort_values("fill_rate")
            fig3 = px.bar(
                sku_agg, x="sku_id", y="fill_rate",
                color="fill_rate", color_continuous_scale="RdYlGn",
                range_color=[0, 100],
                title="Demand Fill Rate by SKU",
            )
            fig3.add_hline(y=95, line_dash="dash", annotation_text="95% Target")
            fig3.update_layout(height=400, yaxis_title="Fill Rate %")
            st.plotly_chart(fig3, use_container_width=True)

        with tab5:
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(x=df_lab["period"], y=df_lab["regular_hrs"],
                                  name="Regular Hours", marker_color="#00CC96"))
            fig4.add_trace(go.Bar(x=df_lab["period"], y=df_lab["overtime_hrs"],
                                  name="Overtime Hours", marker_color="#FFA15A"))
            fig4.update_layout(title="Labor Hours Allocation",
                               barmode="stack", yaxis_title="Hours", height=400)
            st.plotly_chart(fig4, use_container_width=True)

        with tab6:
            st.markdown("**Projected Inventory Over Planning Horizon**")
            st.caption("Shows how inventory evolves based on the optimized fulfillment plan.")

            inv_total = df_ful.groupby("period")["inventory"].sum().reset_index()
            fig_inv_traj = go.Figure()
            fig_inv_traj.add_trace(go.Scatter(
                x=inv_total["period"], y=inv_total["inventory"],
                mode="lines+markers", name="Total Inventory",
                line=dict(width=3, color="#636EFA"),
                fill="tozeroy", fillcolor="rgba(99, 110, 250, 0.1)",
            ))
            fig_inv_traj.update_layout(
                title="Total Inventory Trajectory (Optimized Plan)",
                xaxis_title="Period", yaxis_title="Units", height=350,
            )
            st.plotly_chart(fig_inv_traj, use_container_width=True)

            inv_by_sku = df_ful.pivot_table(
                index="sku_id", columns="period",
                values="inventory", aggfunc="sum",
            )
            fig_heat = px.imshow(
                inv_by_sku.values,
                labels=dict(x="Period", y="SKU", color="Units"),
                x=list(inv_by_sku.columns),
                y=list(inv_by_sku.index),
                color_continuous_scale="YlOrRd",
                title="Inventory Heatmap by SKU and Period",
                aspect="auto",
            )
            fig_heat.update_layout(height=400)
            st.plotly_chart(fig_heat, use_container_width=True)

            st.markdown("**Demand vs. Fulfillment vs. Ending Inventory by Period**")
            period_summary = df_ful.groupby("period").agg(
                total_demand=("demand", "sum"),
                total_fulfilled=("fulfilled", "sum"),
                total_inventory=("inventory", "sum"),
            ).reset_index()

            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(
                x=period_summary["period"], y=period_summary["total_demand"],
                name="Demand", marker_color="#AB63FA"))
            fig_compare.add_trace(go.Bar(
                x=period_summary["period"], y=period_summary["total_fulfilled"],
                name="Fulfilled", marker_color="#00CC96"))
            fig_compare.add_trace(go.Scatter(
                x=period_summary["period"], y=period_summary["total_inventory"],
                name="Ending Inventory", mode="lines+markers",
                line=dict(width=3, color="#EF553B"), yaxis="y2"))
            fig_compare.update_layout(
                title="Demand / Fulfillment / Inventory by Period",
                yaxis=dict(title="Units (Demand & Fulfilled)"),
                yaxis2=dict(title="Ending Inventory (units)",
                            overlaying="y", side="right"),
                barmode="group", height=400,
                legend=dict(orientation="h", yanchor="bottom",
                            y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_compare, use_container_width=True)
    else:
        st.info(
            f"\U0001f448 Adjust parameters in the sidebar and click **Run Optimization** "
            f"to see results for **{dc_id}**."
        )


# =============================================================================
# MAIN
# =============================================================================
st.title("\U0001f3ed DC Capacity Utilization Optimizer")
st.markdown("Interactive scenario planning with penalty-based soft constraints.")

render_diagnostics()
render_about()

# Always load the network summary — it's tiny (one row per DC) and powers both
# the map and the "Back to map" trip. The NDC summary is optional: empty if the
# generator hasn't been re-run with the NDC section yet.
try:
    df_summary = load_network_summary()
except Exception as e:
    st.error(f"Failed to load network summary: {e}")
    st.stop()

df_ndc_summary = load_ndc_summary()
ndc_ids = set(df_ndc_summary["dc_id"]) if not df_ndc_summary.empty else set()

if st.session_state["view"] == "map" or not st.session_state.get("selected_dc"):
    render_map(df_summary, df_ndc_summary if not df_ndc_summary.empty else None)
else:
    selected = st.session_state["selected_dc"]
    if selected in ndc_ids:
        render_ndc_detail(selected)
    else:
        render_detail(selected, df_summary)

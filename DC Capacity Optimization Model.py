# Databricks notebook source
# MAGIC %md
# MAGIC ## Genie Code
# MAGIC **Initial Prompt**:
# MAGIC
# MAGIC *Create a mathematical model where we need to optimize capacity utilization of a distribution center to achieve sales given the levels of inventory, demand, capacity positions inside the distribution center, labor hours, distribution center throughput, inbound and outbound volumes*
# MAGIC
# MAGIC *Then create the tables needed that would provide inputs into this formule.*
# MAGIC
# MAGIC *The tables should be created in the existing schema: tenbosch.scmo_poc*
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Multi-DC + config-driven refactor
# MAGIC
# MAGIC - All tunables (penalties, fallback capacities, schema name, etc.) live in `config.yaml` in this same workspace folder.
# MAGIC - The LP is now solved **per DC in a loop** over every DC found in `dc_capacity`.
# MAGIC - The synthetic-data generator has been moved to the separate **`Generate Synthetic Data`** notebook — run that first if any input table is empty.

# COMMAND ----------

# DBTITLE 1,Mathematical Model Formulation
# MAGIC %md
# MAGIC # DC Capacity Utilization Optimization Model
# MAGIC
# MAGIC ## Objective Function
# MAGIC
# MAGIC **Maximize total profit** minus penalty costs for capacity overflow:
# MAGIC
# MAGIC $$\max Z = \underbrace{\sum_{i} \sum_{t} r_i \cdot f_{it}}_{\text{revenue}} - \underbrace{\sum_{i} \sum_{t} h_i \cdot I_{it}}_{\text{holding cost}} - \underbrace{\sum_{t} (c_l \cdot L_t + c_o \cdot O_t)}_{\text{labor cost}} - \underbrace{\sum_{t} \sum_{k} p_k \cdot s_{kt}}_{\text{overflow penalties}}$$
# MAGIC
# MAGIC ### Decision Variables
# MAGIC | Variable | Description |
# MAGIC |----------|-------------|
# MAGIC | $f_{it}$ | Units fulfilled for SKU $i$ in period $t$ |
# MAGIC | $I_{it}$ | Inventory of SKU $i$ at end of period $t$ |
# MAGIC | $L_t$ | Regular labor hours used in period $t$ |
# MAGIC | $O_t$ | Overtime labor hours used in period $t$ |
# MAGIC | $s_{kt}$ | **Slack variable**: overflow amount for constraint $k$ in period $t$ |
# MAGIC
# MAGIC ### Slack Variables (Soft Constraints)
# MAGIC | Slack Variable | Measures | Penalty Cost (default) |
# MAGIC |----------------|----------|---|
# MAGIC | $s_{\text{storage},t}$ | Cubic feet exceeding storage capacity | $5.00/\text{cu ft}/\text{day}$ |
# MAGIC | $s_{\text{throughput},t}$ | Units exceeding daily throughput limit | $2.00/\text{unit}$ |
# MAGIC | $s_{\text{outbound},t}$ | Pallets exceeding outbound dock capacity | $50.00/\text{pallet}$ |
# MAGIC | $s_{\text{inbound},t}$ | Pallets exceeding inbound dock capacity | $75.00/\text{pallet}$ |
# MAGIC
# MAGIC > **Key Insight**: Slack variables ensure the model **always returns a feasible solution**. When a constraint must be violated, the penalty cost quantifies the real-world expense of that violation (overflow storage, carrier detention, expediting). A non-zero slack value = an actionable alert.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Hard Constraints (always enforced)
# MAGIC
# MAGIC ### 1. Inventory Balance
# MAGIC $$I_{it} = I_{i,t-1} + \text{inbound}_{it} - f_{it} \quad \forall \, i, t$$
# MAGIC
# MAGIC ### 2. Demand Ceiling
# MAGIC $$f_{it} \leq d_{it} \quad \forall \, i, t$$
# MAGIC
# MAGIC ### 3. Labor Capacity
# MAGIC $$\sum_{i} \frac{f_{it}}{\text{throughput\_rate}_i} \leq L_t + O_t \quad \forall \, t$$
# MAGIC
# MAGIC ### 4. Regular Labor Limit
# MAGIC $$L_t \leq \text{MaxRegularHours}_t \quad \forall \, t$$
# MAGIC
# MAGIC ### 5. Overtime Labor Limit
# MAGIC $$O_t \leq \text{MaxOvertimeHours}_t \quad \forall \, t$$
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Soft Constraints (with slack/penalty)
# MAGIC
# MAGIC ### 6. Storage Capacity
# MAGIC $$\sum_{i} (I_{it} \times \text{cube}_i) \leq \text{MaxStorage} + s_{\text{storage},t} \quad \forall \, t$$
# MAGIC
# MAGIC ### 7. Outbound Dock Capacity
# MAGIC $$\sum_{i} \frac{f_{it}}{\text{units\_per\_pallet}_i} \leq \text{MaxOutboundPallets} + s_{\text{outbound},t} \quad \forall \, t$$
# MAGIC
# MAGIC ### 8. Throughput Ceiling
# MAGIC $$\sum_{i} f_{it} \leq \text{MaxDailyThroughput} + s_{\text{throughput},t} \quad \forall \, t$$
# MAGIC
# MAGIC ### 9. Inbound Dock Capacity
# MAGIC $$\text{InboundPallets}_t \leq \text{MaxInboundPallets} + s_{\text{inbound},t} \quad \forall \, t$$
# MAGIC
# MAGIC ### 10. Non-negativity
# MAGIC $$f_{it}, \; I_{it}, \; L_t, \; O_t, \; s_{kt} \geq 0 \quad \forall \, i, t, k$$
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Solution Approach
# MAGIC This is a **Linear Program (LP)** solved using the **CBC** (Coin-or Branch and Cut) solver via PuLP. One LP is built and solved **per DC** — DCs are independent in this formulation.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install pulp highspy pyyaml --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load config.yaml
import yaml

CONFIG_PATH = "/Workspace/Users/jeff.tenbosch@databricks.com/dc-capacity-optimizer/config.yaml"

with open(CONFIG_PATH) as _f:
    cfg = yaml.safe_load(_f)

SCHEMA = cfg["schema"]
OPT = cfg["optimization"]
PENALTIES = OPT["penalties"]
FALLBACKS = OPT["fallback_capacities"]

print(f"Loaded config — schema={SCHEMA}")
print(f"  Penalty (amb-storage/cold-storage/throughput/outbound/inbound): "
      f"${PENALTIES['storage_ambient']}/cuft, ${PENALTIES['storage_cold']}/cuft, "
      f"${PENALTIES['throughput']}/unit, "
      f"${PENALTIES['outbound']}/pallet, ${PENALTIES['inbound']}/pallet")

# COMMAND ----------

# DBTITLE 1,Load data from input tables
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


def load_table(table_name):
    """Load a table from the schema into a pandas DataFrame. Returns empty DF on error."""
    try:
        df = spark.table(f"{SCHEMA}.{table_name}").toPandas()
        print(f"  ✓ {table_name}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"  ✗ {table_name}: {e}")
        return pd.DataFrame()


print(f"Loading input tables from {SCHEMA}...\n")

df_products = load_table("product_master")
df_demand = load_table("demand_forecast")
df_inventory = load_table("inventory_levels")
df_capacity = load_table("dc_capacity")
df_labor = load_table("labor_availability")
df_throughput = load_table("throughput_rates")
df_inbound = load_table("inbound_plan")
df_outbound = load_table("outbound_plan")
df_params = load_table("optimization_parameters")
df_dc_meta = load_table("distribution_centers")

# Restrict DC metadata to Wholesale DCs only (the only valid type for this model).
FACILITY_FILTER = cfg["synthetic_data"]["distribution_centers"]["facility_type"]
if not df_dc_meta.empty:
    df_dc_meta = df_dc_meta[df_dc_meta["facility_type"] == FACILITY_FILTER].copy()

# Restrict demand to the forward planning horizon — historical rows (those with
# actual_units populated) drive the Forecast Accuracy dashboard, not the LP.
# Filter on actual_units IS NULL to match the contract used by app.py and the
# inbound/outbound filters below; the generator no longer pins a start_date in
# config, so date comparisons would need a runtime computation.
if not df_demand.empty and "actual_units" in df_demand.columns:
    df_demand = df_demand[df_demand["actual_units"].isna()].copy()
    print(f"  → filtered demand_forecast to forward horizon: {len(df_demand)} rows")

# Inbound/outbound history rows have actual_date / actual_ship_date populated;
# the LP only plans the forward horizon, so drop those.
if not df_inbound.empty and "actual_date" in df_inbound.columns:
    df_inbound = df_inbound[df_inbound["actual_date"].isna()].copy()
if not df_outbound.empty and "actual_ship_date" in df_outbound.columns:
    df_outbound = df_outbound[df_outbound["actual_ship_date"].isna()].copy()

required = {
    "product_master": df_products,
    "demand_forecast": df_demand,
    "inventory_levels": df_inventory,
    "dc_capacity": df_capacity,
    "labor_availability": df_labor,
    "throughput_rates": df_throughput,
    "inbound_plan": df_inbound,
    "distribution_centers": df_dc_meta,
}
empty = [n for n, d in required.items() if d.empty]
if empty:
    print(f"\n⚠️  Required tables are empty: {empty}")
    print("   Run the 'Generate Synthetic Data' notebook first.")
else:
    print("\n✓ All required tables loaded.")

# COMMAND ----------

# DBTITLE 1,Define per-DC optimization function
import pulp


SOLVER_PREFERENCE = OPT.get("solver_preference", ["highs", "cbc"])


def _make_solver():
    """Return the first solver in solver_preference whose `.available()` is True.

    Mirrors the app's _make_solver: prefers `pulp.HiGHS` (Python binding via
    highspy — no CLI binary needed), then `pulp.HiGHS_CMD`, then CBC.
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
    return pulp.PULP_CBC_CMD(msg=0), "CBC"


def optimize_dc(dc_id, cfg, *,
                df_products, df_demand, df_inventory, df_capacity,
                df_labor, df_throughput, df_inbound, df_params):
    """Build and solve the LP for a single DC. Returns a results dict or {'status': 'failed'}."""

    OPT = cfg["optimization"]
    PEN = OPT["penalties"]
    FBK = OPT["fallback_capacities"]

    # ---- Filter every DC-scoped table to this DC ----
    dem = df_demand[df_demand["dc_id"] == dc_id]
    inv = df_inventory[df_inventory["dc_id"] == dc_id]
    cap = df_capacity[df_capacity["dc_id"] == dc_id]
    lab = df_labor[df_labor["dc_id"] == dc_id]
    tpr = df_throughput[df_throughput["dc_id"] == dc_id]
    inb = df_inbound[df_inbound["dc_id"] == dc_id]
    par = df_params[df_params["dc_id"] == dc_id] if (df_params is not None and not df_params.empty) else pd.DataFrame()

    if dem.empty or inv.empty:
        return {"status": "skipped", "reason": "no demand or inventory rows for DC"}

    # ---- Indices ----
    skus = sorted(dem["sku_id"].unique())
    periods = sorted(dem["forecast_date"].unique())

    # ---- SKU-level params from product_master (shared across DCs) ----
    revenue, holding_cost, cube, units_per_pallet, storage_type = {}, {}, {}, {}, {}
    for _, row in df_products.iterrows():
        s = row["sku_id"]
        revenue[s] = row["revenue_per_unit"]
        holding_cost[s] = row["holding_cost_per_unit_per_day"]
        cube[s] = row["unit_cube_ft3"]
        units_per_pallet[s] = row["units_per_case"] * row["cases_per_pallet"]
        storage_type[s] = row.get("storage_type", "AMBIENT")

    # ---- Labor cost params (per-DC override, else config default) ----
    regular_labor_cost = OPT["default_regular_labor_cost_per_hour"]
    overtime_labor_cost = OPT["default_overtime_labor_cost_per_hour"]
    if not par.empty:
        pd_lookup = dict(zip(par["param_name"], par["param_value"]))
        regular_labor_cost = float(pd_lookup.get("regular_labor_cost_per_hour", regular_labor_cost))
        overtime_labor_cost = float(pd_lookup.get("overtime_labor_cost_per_hour", overtime_labor_cost))

    # ---- Demand, initial inventory, inbound, throughput ----
    demand = {(r["sku_id"], r["forecast_date"]): r["demand_units"] for _, r in dem.iterrows()}
    init_inv = dict(zip(inv["sku_id"], inv["on_hand_units"]))

    inbound_units, inbound_pallets_map = {}, {}
    for _, r in inb.iterrows():
        k = (r["sku_id"], r["scheduled_date"])
        inbound_units[k] = inbound_units.get(k, 0) + r["inbound_units"]
        inbound_pallets_map[k] = inbound_pallets_map.get(k, 0) + r["inbound_pallets"]

    tp_rate = {}
    if not tpr.empty:
        tp_rate = dict(zip(tpr["sku_id"], tpr["units_per_labor_hour"]))
    default_tp = OPT["default_throughput_rate_units_per_hour"]

    # ---- DC capacity (with fallbacks) ----
    if not cap.empty:
        cap_row = cap.iloc[0]
        max_ambient = float(cap_row["ambient_cube_ft3"])
        max_cold = float(cap_row["cold_cube_ft3"])
        max_throughput = float(cap_row["max_daily_throughput_units"])
        max_inb_pallets = float(cap_row["max_daily_inbound_pallets"])
        max_ob_pallets = float(cap_row["max_daily_outbound_pallets"])
    else:
        max_ambient = FBK["max_ambient_storage"]
        max_cold = FBK["max_cold_storage"]
        max_throughput = FBK["max_daily_throughput"]
        max_inb_pallets = FBK["max_inbound_pallets"]
        max_ob_pallets = FBK["max_outbound_pallets"]

    skus_ambient = [s for s in skus if storage_type.get(s, "AMBIENT") == "AMBIENT"]
    skus_cold = [s for s in skus if storage_type.get(s, "AMBIENT") == "COLD"]

    # ---- Labor availability per period ----
    max_reg_hrs, max_ot_hrs = {}, {}
    if not lab.empty:
        agg = lab.groupby("work_date").agg(
            reg=("available_hours", "sum"),
            ot=("overtime_hours_available", "sum"),
        ).reset_index()
        for _, r in agg.iterrows():
            max_reg_hrs[r["work_date"]] = r["reg"]
            max_ot_hrs[r["work_date"]] = r["ot"]
    for t in periods:
        max_reg_hrs.setdefault(t, OPT["default_max_regular_hours_per_period"])
        max_ot_hrs.setdefault(t, OPT["default_max_overtime_hours_per_period"])

    # ---- Build LP ----
    model = pulp.LpProblem(f"DC_Opt_{dc_id}", pulp.LpMaximize)

    f = pulp.LpVariable.dicts("f", ((i, t) for i in skus for t in periods), lowBound=0)
    I = pulp.LpVariable.dicts("I", ((i, t) for i in skus for t in periods), lowBound=0)
    L = pulp.LpVariable.dicts("L", periods, lowBound=0)
    O = pulp.LpVariable.dicts("O", periods, lowBound=0)
    ss_amb = pulp.LpVariable.dicts("ss_amb", periods, lowBound=0)
    ss_cold = pulp.LpVariable.dicts("ss_cold", periods, lowBound=0)
    st_v = pulp.LpVariable.dicts("st", periods, lowBound=0)
    so = pulp.LpVariable.dicts("so", periods, lowBound=0)
    si = pulp.LpVariable.dicts("si", periods, lowBound=0)

    model += (
        pulp.lpSum(revenue[i] * f[i, t] for i in skus for t in periods)
        - pulp.lpSum(holding_cost[i] * I[i, t] for i in skus for t in periods)
        - pulp.lpSum(regular_labor_cost * L[t] + overtime_labor_cost * O[t] for t in periods)
        - pulp.lpSum(PEN["storage_ambient"] * ss_amb[t] for t in periods)
        - pulp.lpSum(PEN["storage_cold"] * ss_cold[t] for t in periods)
        - pulp.lpSum(PEN["throughput"] * st_v[t] for t in periods)
        - pulp.lpSum(PEN["outbound"] * so[t] for t in periods)
        - pulp.lpSum(PEN["inbound"] * si[t] for t in periods)
    )

    for i in skus:
        for t_idx, t in enumerate(periods):
            ib = inbound_units.get((i, t), 0)
            if t_idx == 0:
                model += I[i, t] == init_inv.get(i, 0) + ib - f[i, t]
            else:
                model += I[i, t] == I[i, periods[t_idx - 1]] + ib - f[i, t]
            model += f[i, t] <= demand.get((i, t), 0)

    for t in periods:
        model += pulp.lpSum(cube[i] * I[i, t] for i in skus_ambient) <= max_ambient + ss_amb[t]
        model += pulp.lpSum(cube[i] * I[i, t] for i in skus_cold) <= max_cold + ss_cold[t]
        model += pulp.lpSum(f[i, t] / tp_rate.get(i, default_tp) for i in skus) <= L[t] + O[t]
        model += L[t] <= max_reg_hrs[t]
        model += O[t] <= max_ot_hrs[t]
        model += pulp.lpSum(f[i, t] / units_per_pallet[i] for i in skus) <= max_ob_pallets + so[t]
        model += pulp.lpSum(f[i, t] for i in skus) <= max_throughput + st_v[t]
        total_inb = sum(inbound_pallets_map.get((i, t), 0) for i in skus)
        model += total_inb <= max_inb_pallets + si[t]

    solver, solver_name = _make_solver()
    model.solve(solver)

    if model.status != pulp.constants.LpStatusOptimal:
        return {"status": "failed", "reason": pulp.LpStatus[model.status],
                "dc_id": dc_id, "solver_used": solver_name}

    # ---- Parse results ----
    fulfillment, overflow, labor_out = [], [], []
    for i in skus:
        for t in periods:
            f_val = f[i, t].varValue or 0
            I_val = I[i, t].varValue or 0
            d_val = demand.get((i, t), 0)
            fulfillment.append({
                "dc_id": dc_id, "sku_id": i, "period": str(t),
                "fulfilled_qty": round(f_val, 2),
                "ending_inventory": round(I_val, 2),
                "demand_qty": d_val,
                "fill_rate": round(f_val / d_val, 4) if d_val > 0 else 1.0,
                "revenue_contribution": round(f_val * revenue[i], 2),
                "holding_cost_incurred": round(I_val * holding_cost[i], 2),
            })

    for t in periods:
        amb_cube = sum((I[i, t].varValue or 0) * cube[i] for i in skus_ambient)
        cold_cube = sum((I[i, t].varValue or 0) * cube[i] for i in skus_cold)
        total_ful = sum((f[i, t].varValue or 0) for i in skus)
        total_ob_pallets = sum((f[i, t].varValue or 0) / units_per_pallet[i] for i in skus)
        total_lab_need = sum((f[i, t].varValue or 0) / tp_rate.get(i, default_tp) for i in skus)
        total_lab_avail = max_reg_hrs[t] + max_ot_hrs[t]
        total_inb = sum(inbound_pallets_map.get((i, t), 0) for i in skus)

        overflow.append({
            "dc_id": dc_id, "period": str(t),
            "ambient_storage_utilization_pct": round(amb_cube / max_ambient * 100, 1) if max_ambient > 0 else 0.0,
            "cold_storage_utilization_pct":    round(cold_cube / max_cold * 100, 1) if max_cold > 0 else 0.0,
            "throughput_utilization_pct":   round(total_ful / max_throughput * 100, 1),
            "outbound_dock_utilization_pct": round(total_ob_pallets / max_ob_pallets * 100, 1),
            "inbound_dock_utilization_pct": round(total_inb / max_inb_pallets * 100, 1) if max_inb_pallets > 0 else 0.0,
            "labor_utilization_pct":        round(total_lab_need / total_lab_avail * 100, 1) if total_lab_avail > 0 else 0.0,
            "outbound_pallets_planned":     round(total_ob_pallets, 1),
            "inbound_pallets_planned":      round(total_inb, 1),
            "throughput_units_planned":     round(total_ful, 1),
            "ambient_storage_overflow_cuft": round(ss_amb[t].varValue or 0, 1),
            "cold_storage_overflow_cuft":    round(ss_cold[t].varValue or 0, 1),
            "throughput_overflow_units":    round(st_v[t].varValue or 0, 1),
            "outbound_overflow_pallets":    round(so[t].varValue or 0, 1),
            "inbound_overflow_pallets":     round(si[t].varValue or 0, 1),
            "ambient_storage_penalty_cost": round((ss_amb[t].varValue or 0) * PEN["storage_ambient"], 2),
            "cold_storage_penalty_cost":    round((ss_cold[t].varValue or 0) * PEN["storage_cold"], 2),
            "throughput_penalty_cost":      round((st_v[t].varValue or 0) * PEN["throughput"], 2),
            "outbound_penalty_cost":        round((so[t].varValue or 0) * PEN["outbound"], 2),
            "inbound_penalty_cost":         round((si[t].varValue or 0) * PEN["inbound"], 2),
            "ambient_storage_binding": amb_cube >= max_ambient * 0.99 if max_ambient > 0 else False,
            "cold_storage_binding":    cold_cube >= max_cold * 0.99 if max_cold > 0 else False,
            "throughput_binding":   total_ful >= max_throughput * 0.99,
            "outbound_dock_binding": total_ob_pallets >= max_ob_pallets * 0.99,
            "inbound_dock_binding": total_inb >= max_inb_pallets * 0.99 if max_inb_pallets > 0 else False,
            "labor_binding":        total_lab_need >= total_lab_avail * 0.99 if total_lab_avail > 0 else False,
        })

    for t in periods:
        L_val = L[t].varValue or 0
        O_val = O[t].varValue or 0
        labor_out.append({
            "dc_id": dc_id, "period": str(t),
            "regular_hours_used":      round(L_val, 2),
            "overtime_hours_used":     round(O_val, 2),
            "regular_hours_available": max_reg_hrs[t],
            "overtime_hours_available": max_ot_hrs[t],
            "regular_utilization_pct":  round(L_val / max_reg_hrs[t] * 100, 1) if max_reg_hrs[t] > 0 else 0.0,
            "overtime_utilization_pct": round(O_val / max_ot_hrs[t] * 100, 1) if max_ot_hrs[t] > 0 else 0.0,
            "labor_cost":              round(L_val * regular_labor_cost + O_val * overtime_labor_cost, 2),
        })

    df_ful = pd.DataFrame(fulfillment)
    df_ov = pd.DataFrame(overflow)
    df_lab = pd.DataFrame(labor_out)

    total_rev = df_ful["revenue_contribution"].sum()
    total_hold = df_ful["holding_cost_incurred"].sum()
    total_lab_cost = df_lab["labor_cost"].sum()
    total_penalty = (df_ov["ambient_storage_penalty_cost"].sum()
                     + df_ov["cold_storage_penalty_cost"].sum()
                     + df_ov["throughput_penalty_cost"].sum()
                     + df_ov["outbound_penalty_cost"].sum()
                     + df_ov["inbound_penalty_cost"].sum())
    total_demand = sum(demand.values())
    total_fulfilled = df_ful["fulfilled_qty"].sum()

    return {
        "status": "optimal",
        "dc_id": dc_id,
        "fulfillment": df_ful,
        "overflow": df_ov,
        "labor": df_lab,
        "objective_value": pulp.value(model.objective),
        "total_revenue": float(total_rev),
        "total_holding_cost": float(total_hold),
        "total_labor_cost": float(total_lab_cost),
        "total_penalty_cost": float(total_penalty),
        "net_profit": float(total_rev - total_hold - total_lab_cost - total_penalty),
        "fill_rate": float(total_fulfilled / total_demand * 100) if total_demand > 0 else 0.0,
        "solver_used": solver_name,
    }

# COMMAND ----------

# DBTITLE 1,Solve every DC
if df_capacity.empty or df_dc_meta.empty:
    print("⚠️  Missing prerequisite data:")
    if df_capacity.empty:
        print(f"     • {SCHEMA}.dc_capacity is empty — run the Generate Synthetic Data notebook.")
    if df_dc_meta.empty:
        print(f"     • {SCHEMA}.distribution_centers has no rows with "
              f"facility_type='{FACILITY_FILTER}'.")
    results_by_dc = {}
else:
    valid_dc_ids = set(df_dc_meta["dc_id"])
    capacity_dc_ids = set(df_capacity["dc_id"])
    dc_ids = sorted(valid_dc_ids & capacity_dc_ids)
    skipped = sorted(capacity_dc_ids - valid_dc_ids)
    dc_name_lookup = dict(zip(df_dc_meta["dc_id"], df_dc_meta["facility_name"]))

    print(f"Solving optimization for {len(dc_ids)} Wholesale DCs "
          f"(filtered from {len(capacity_dc_ids)} dc_capacity rows; "
          f"{len(skipped)} skipped as non-Wholesale)...\n")

    results_by_dc = {}
    for dc in dc_ids:
        label = f"{dc} ({dc_name_lookup.get(dc, '?')})"
        print(f"  → {label} … ", end="", flush=True)
        r = optimize_dc(
            dc, cfg,
            df_products=df_products, df_demand=df_demand,
            df_inventory=df_inventory, df_capacity=df_capacity,
            df_labor=df_labor, df_throughput=df_throughput,
            df_inbound=df_inbound, df_params=df_params,
        )
        results_by_dc[dc] = r
        if r["status"] == "optimal":
            print(f"OK [{r['solver_used']:<10}]  net=${r['net_profit']:>14,.2f}  "
                  f"fill={r['fill_rate']:5.1f}%  penalty=${r['total_penalty_cost']:>10,.2f}")
        else:
            print(f"FAILED ({r.get('reason', '?')})")

    print(f"\n✓ Completed {sum(1 for r in results_by_dc.values() if r['status'] == 'optimal')} of "
          f"{len(results_by_dc)} DCs.")

# COMMAND ----------

# DBTITLE 1,Write per-DC results back to Delta
from datetime import datetime

optimal = [r for r in results_by_dc.values() if r["status"] == "optimal"]

if not optimal:
    print("⚠️  No optimal DC results to write.")
else:
    run_ts = datetime.now().isoformat()

    df_fulfillment_all = pd.concat([r["fulfillment"] for r in optimal], ignore_index=True)
    df_fulfillment_all["run_timestamp"] = run_ts
    df_fulfillment_all["model_status"] = "optimal"

    df_utilization_all = pd.concat([r["overflow"] for r in optimal], ignore_index=True)
    df_utilization_all["run_timestamp"] = run_ts

    df_labor_all = pd.concat([r["labor"] for r in optimal], ignore_index=True)
    df_labor_all["run_timestamp"] = run_ts

    spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.optimization_results")
    spark.createDataFrame(df_fulfillment_all).write.saveAsTable(f"{SCHEMA}.optimization_results")
    print(f"  ✓ optimization_results: {len(df_fulfillment_all)} rows across {len(optimal)} DCs")

    spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.capacity_utilization_results")
    spark.createDataFrame(df_utilization_all).write.saveAsTable(f"{SCHEMA}.capacity_utilization_results")
    print(f"  ✓ capacity_utilization_results: {len(df_utilization_all)} rows")

    spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.labor_allocation_results")
    spark.createDataFrame(df_labor_all).write.saveAsTable(f"{SCHEMA}.labor_allocation_results")
    print(f"  ✓ labor_allocation_results: {len(df_labor_all)} rows")

# COMMAND ----------

# DBTITLE 1,Per-DC and network summary
if optimal:
    summary_rows = []
    for r in optimal:
        summary_rows.append({
            "dc_id": r["dc_id"],
            "facility_name": dc_name_lookup.get(r["dc_id"], "?"),
            "net_profit":     round(r["net_profit"], 2),
            "revenue":        round(r["total_revenue"], 2),
            "holding_cost":   round(r["total_holding_cost"], 2),
            "labor_cost":     round(r["total_labor_cost"], 2),
            "penalty_cost":   round(r["total_penalty_cost"], 2),
            "fill_rate_pct":  round(r["fill_rate"], 1),
        })
    df_summary = pd.DataFrame(summary_rows).sort_values("dc_id")

    print("=" * 115)
    print("  PER-DC SUMMARY")
    print("=" * 115)
    print(f"{'DC':<8} {'Facility':<28} {'Net Profit':>14} {'Revenue':>14} {'Holding':>11} "
          f"{'Labor':>11} {'Penalty':>11} {'Fill%':>7}")
    print("-" * 115)
    for _, row in df_summary.iterrows():
        print(f"{row['dc_id']:<8} {row['facility_name'][:27]:<28} "
              f"${row['net_profit']:>12,.0f}  ${row['revenue']:>12,.0f}  "
              f"${row['holding_cost']:>9,.0f}  ${row['labor_cost']:>9,.0f}  "
              f"${row['penalty_cost']:>9,.0f}  {row['fill_rate_pct']:>6.1f}%")
    print("-" * 115)

    # Network totals
    total_net = df_summary["net_profit"].sum()
    total_rev = df_summary["revenue"].sum()
    total_hold = df_summary["holding_cost"].sum()
    total_lab = df_summary["labor_cost"].sum()
    total_pen = df_summary["penalty_cost"].sum()
    net_fill = df_summary["fill_rate_pct"].mean()

    print(f"{'NETWORK':<8} {'':<28} ${total_net:>12,.0f}  ${total_rev:>12,.0f}  "
          f"${total_hold:>9,.0f}  ${total_lab:>9,.0f}  "
          f"${total_pen:>9,.0f}  {net_fill:>6.1f}%")
    print("=" * 115)

    # Flag DCs with penalties (capacity bottlenecks)
    stressed = df_summary[df_summary["penalty_cost"] > 0].sort_values("penalty_cost", ascending=False)
    if not stressed.empty:
        print("\n⚠️  DCs incurring overflow penalties (capacity bottlenecks):")
        for _, row in stressed.iterrows():
            print(f"    {row['dc_id']} ({row['facility_name']}): "
                  f"${row['penalty_cost']:,.0f} in penalties → "
                  f"investigate via the Streamlit app's overflow tab")
    else:
        print("\n✓ No DC exceeded capacity — all operations within limits.")
else:
    print("⚠️  No optimal results to summarize.")

# COMMAND ----------

# DBTITLE 1,Forecast vs Demand Accuracy (history-window analytics)
# MAGIC %md
# MAGIC Computes forecast-vs-actual metrics from the history window of
# MAGIC `demand_forecast` (rows where `actual_units IS NOT NULL`) and writes
# MAGIC them to two Delta tables for downstream consumption (Lakeview, Genie,
# MAGIC the Streamlit app):
# MAGIC
# MAGIC - `forecast_accuracy_network` — one row per DC plus a `NETWORK` rollup row.
# MAGIC - `forecast_accuracy_daily`   — one row per `(dc_id, forecast_date)` for trend charts.
# MAGIC
# MAGIC Formulas mirror `app.py:load_accuracy_network()` so the app and these
# MAGIC tables agree. Source data is untouched — these are pure derived tables.

# COMMAND ----------

# DBTITLE 1,forecast_accuracy_network
# At-risk threshold matches app.py:_ACC_RISK_THRESHOLD. Keep them in lockstep.
ACC_RISK_THRESHOLD = 0.30

network_sql = f"""
WITH hist AS (
  SELECT
    f.dc_id,
    f.sku_id,
    f.forecast_date,
    f.demand_units,
    f.actual_units,
    ABS(f.demand_units - f.actual_units)           AS abs_err,
    (f.demand_units - f.actual_units)              AS signed_err,
    LEAST(f.demand_units, f.actual_units)          AS fill_units
  FROM {SCHEMA}.demand_forecast f
  WHERE f.actual_units IS NOT NULL
),
per_sku AS (
  SELECT
    dc_id,
    sku_id,
    SUM(abs_err) / NULLIF(SUM(actual_units), 0) AS sku_mape
  FROM hist
  GROUP BY dc_id, sku_id
),
per_dc AS (
  SELECT
    d.dc_id,
    d.facility_name,
    SUM(h.abs_err)    / NULLIF(SUM(h.actual_units), 0) AS mape,
    SUM(h.signed_err) / NULLIF(SUM(h.actual_units), 0) AS bias_pct,
    SUM(h.fill_units) / NULLIF(SUM(h.actual_units), 0) AS fill_rate_accuracy,
    SUM(h.actual_units)        AS total_actual_units,
    SUM(h.demand_units)        AS total_forecast_units,
    COUNT(DISTINCT h.sku_id)   AS n_skus,
    COUNT(DISTINCT CASE WHEN s.sku_mape > {ACC_RISK_THRESHOLD} THEN s.sku_id END)
                               AS n_at_risk_skus,
    COUNT(DISTINCT h.forecast_date) AS history_days
  FROM {SCHEMA}.distribution_centers d
  JOIN hist h        USING (dc_id)
  LEFT JOIN per_sku s USING (dc_id, sku_id)
  WHERE d.facility_type = '{FACILITY_FILTER}'
  GROUP BY d.dc_id, d.facility_name
),
net AS (
  SELECT
    'NETWORK' AS dc_id,
    'All Wholesale DCs' AS facility_name,
    SUM(h.abs_err)    / NULLIF(SUM(h.actual_units), 0) AS mape,
    SUM(h.signed_err) / NULLIF(SUM(h.actual_units), 0) AS bias_pct,
    SUM(h.fill_units) / NULLIF(SUM(h.actual_units), 0) AS fill_rate_accuracy,
    SUM(h.actual_units)        AS total_actual_units,
    SUM(h.demand_units)        AS total_forecast_units,
    COUNT(DISTINCT h.sku_id)   AS n_skus,
    COUNT(DISTINCT CASE WHEN s.sku_mape > {ACC_RISK_THRESHOLD} THEN s.sku_id END)
                               AS n_at_risk_skus,
    COUNT(DISTINCT h.forecast_date) AS history_days
  FROM hist h
  LEFT JOIN per_sku s USING (dc_id, sku_id)
)
SELECT *, current_timestamp() AS computed_at FROM per_dc
UNION ALL
SELECT *, current_timestamp() AS computed_at FROM net
"""

spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.forecast_accuracy_network")
spark.sql(network_sql).write.saveAsTable(f"{SCHEMA}.forecast_accuracy_network")

# Documentation + informational PK so the table is self-describing in
# Catalog Explorer / Genie, matching the rest of the schema.
spark.sql(f"""
COMMENT ON TABLE {SCHEMA}.forecast_accuracy_network IS
  'Forecast-vs-actual accuracy aggregated over the history window of demand_forecast. One row per Wholesale DC plus a NETWORK rollup row. Refreshed each time the LP notebook runs.'
""")
for col, comment in {
    "dc_id":                "Distribution center id, or the literal NETWORK for the network rollup row.",
    "facility_name":        "DC facility name; 'All Wholesale DCs' for the network row.",
    "mape":                 "Weighted mean absolute percent error = SUM(|forecast - actual|) / SUM(actual).",
    "bias_pct":             "Weighted bias = SUM(forecast - actual) / SUM(actual). Positive = over-forecast.",
    "fill_rate_accuracy":   "SUM(min(forecast, actual)) / SUM(actual). Caps at 1.0.",
    "total_actual_units":   "Total actual demand over the history window.",
    "total_forecast_units": "Total forecasted demand over the history window.",
    "n_skus":               "Distinct SKUs observed at this DC.",
    "n_at_risk_skus":       f"Distinct SKUs with per-SKU MAPE above {ACC_RISK_THRESHOLD} (the same threshold the app uses).",
    "history_days":         "Distinct forecast_date values in the history window.",
    "computed_at":          "Snapshot timestamp when this row was written.",
}.items():
    spark.sql(f"""
        COMMENT ON COLUMN {SCHEMA}.forecast_accuracy_network.{col}
        IS '{comment.replace("'", "''")}'
    """)

spark.sql(f"ALTER TABLE {SCHEMA}.forecast_accuracy_network ALTER COLUMN dc_id SET NOT NULL")
spark.sql(f"""
ALTER TABLE {SCHEMA}.forecast_accuracy_network
ADD CONSTRAINT pk_forecast_accuracy_network PRIMARY KEY (dc_id) RELY
""")

print(f"  ✓ forecast_accuracy_network: "
      f"{spark.table(f'{SCHEMA}.forecast_accuracy_network').count()} rows")

# COMMAND ----------

# DBTITLE 1,forecast_accuracy_daily
daily_sql = f"""
SELECT
  dc_id,
  forecast_date,
  SUM(ABS(demand_units - actual_units))   / NULLIF(SUM(actual_units), 0) AS mape,
  SUM(demand_units - actual_units)        / NULLIF(SUM(actual_units), 0) AS bias_pct,
  SUM(actual_units)   AS actual_units,
  SUM(demand_units)   AS forecast_units,
  current_timestamp() AS computed_at
FROM {SCHEMA}.demand_forecast
WHERE actual_units IS NOT NULL
GROUP BY dc_id, forecast_date
"""

spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.forecast_accuracy_daily")
spark.sql(daily_sql).write.saveAsTable(f"{SCHEMA}.forecast_accuracy_daily")

spark.sql(f"""
COMMENT ON TABLE {SCHEMA}.forecast_accuracy_daily IS
  'Daily forecast-vs-actual rollup per DC across the history window of demand_forecast. Powers the MAPE trend chart in the Streamlit app and any time-series dashboards.'
""")
for col, comment in {
    "dc_id":          "Distribution center id.",
    "forecast_date":  "Calendar day in the history window.",
    "mape":           "Daily weighted MAPE for this DC.",
    "bias_pct":       "Daily weighted bias % (positive = over-forecast).",
    "actual_units":   "Sum of actual demand for this DC on this day.",
    "forecast_units": "Sum of forecasted demand for this DC on this day.",
    "computed_at":    "Snapshot timestamp when this row was written.",
}.items():
    spark.sql(f"""
        COMMENT ON COLUMN {SCHEMA}.forecast_accuracy_daily.{col}
        IS '{comment.replace("'", "''")}'
    """)

spark.sql(f"ALTER TABLE {SCHEMA}.forecast_accuracy_daily ALTER COLUMN dc_id SET NOT NULL")
spark.sql(f"ALTER TABLE {SCHEMA}.forecast_accuracy_daily ALTER COLUMN forecast_date SET NOT NULL")
spark.sql(f"""
ALTER TABLE {SCHEMA}.forecast_accuracy_daily
ADD CONSTRAINT pk_forecast_accuracy_daily PRIMARY KEY (dc_id, forecast_date) RELY
""")

print(f"  ✓ forecast_accuracy_daily: "
      f"{spark.table(f'{SCHEMA}.forecast_accuracy_daily').count()} rows")

# COMMAND ----------

# DBTITLE 1,Display accuracy summary
df_acc_net = (
    spark.table(f"{SCHEMA}.forecast_accuracy_network")
         .orderBy("mape", ascending=False)
         .toPandas()
)
display(df_acc_net)

net_row = df_acc_net[df_acc_net["dc_id"] == "NETWORK"].iloc[0]
print("=" * 80)
print("  FORECAST ACCURACY — NETWORK")
print("=" * 80)
print(f"  MAPE:               {net_row['mape'] * 100:>6.1f}%")
print(f"  Bias %:             {net_row['bias_pct'] * 100:>+6.1f}%  "
      f"({'over-forecasting' if net_row['bias_pct'] > 0 else 'under-forecasting'})")
print(f"  Fill-rate accuracy: {net_row['fill_rate_accuracy'] * 100:>6.1f}%")
print(f"  History days:       {int(net_row['history_days'])}")
print(f"  At-risk SKUs:       {int(net_row['n_at_risk_skus'])} "
      f"(MAPE > {ACC_RISK_THRESHOLD * 100:.0f}%)")
print("=" * 80)

at_risk_dcs = df_acc_net[
    (df_acc_net["dc_id"] != "NETWORK") & (df_acc_net["mape"] > ACC_RISK_THRESHOLD)
].sort_values("mape", ascending=False)
if not at_risk_dcs.empty:
    print(f"\n⚠️  DCs with MAPE above {ACC_RISK_THRESHOLD * 100:.0f}%:")
    for _, row in at_risk_dcs.iterrows():
        print(f"    {row['dc_id']} ({row['facility_name']}): "
              f"MAPE {row['mape'] * 100:.1f}%, bias {row['bias_pct'] * 100:+.1f}% "
              f"→ investigate variance drivers in the Streamlit app")
else:
    print(f"\n✓ All DCs within the {ACC_RISK_THRESHOLD * 100:.0f}% MAPE threshold.")

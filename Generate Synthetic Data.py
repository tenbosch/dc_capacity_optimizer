# Databricks notebook source
# MAGIC %md
# MAGIC # Generate Synthetic Data
# MAGIC
# MAGIC Populates the input tables in `tenbosch.scmo_poc` consumed by the
# MAGIC **DC Capacity Optimization Model** notebook and the `dc-capacity-optimizer`
# MAGIC Databricks app.
# MAGIC
# MAGIC **All tunables live in `config.yaml`** (same folder as this notebook).
# MAGIC To change the number of DCs, SKUs, planning days, random ranges, etc.,
# MAGIC edit the YAML — no code changes required.
# MAGIC
# MAGIC **Tables produced (multi-DC):**
# MAGIC - `product_master` — SKU catalog (no `dc_id`)
# MAGIC - `dc_capacity` — one row per DC
# MAGIC - `demand_forecast`, `inventory_levels`, `labor_availability`,
# MAGIC   `throughput_rates`, `inbound_plan`, `outbound_plan`,
# MAGIC   `optimization_parameters` — all carry `dc_id`
# MAGIC
# MAGIC The DC list is read from `tenbosch.scmo_poc.distribution_centers` and
# MAGIC filtered to rows where `facility_type = 'Wholesale DC'` — that table is
# MAGIC the source of truth for which DCs exist and is **not** written by this
# MAGIC notebook.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip install pyyaml --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load config.yaml
import yaml

CONFIG_PATH = "/Workspace/Users/jeff.tenbosch@databricks.com/dc-capacity-optimizer/config.yaml"

with open(CONFIG_PATH) as _f:
    cfg = yaml.safe_load(_f)

print(f"Loaded config from {CONFIG_PATH}")
print(f"  schema:               {cfg['schema']}")
print(f"  DC source:            {cfg['schema']}.distribution_centers "
      f"WHERE facility_type='{cfg['synthetic_data']['distribution_centers']['facility_type']}'")
print(f"  n_skus:               {cfg['synthetic_data']['skus']['n_skus']}")
print(f"  n_periods:            {cfg['synthetic_data']['planning']['n_periods']} days")
print(f"  random_seed:          {cfg['synthetic_data']['random_seed']}")

# COMMAND ----------

# DBTITLE 1,Imports and global setup
import uuid
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType,
    IntegerType, DoubleType, TimestampType, BooleanType, LongType
)

spark = SparkSession.builder.getOrCreate()

SCHEMA = cfg["schema"]
SD = cfg["synthetic_data"]
np.random.seed(SD["random_seed"])


def _to_date(v):
    """Accept either a date object or YYYY-MM-DD string from YAML."""
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    return datetime.strptime(str(v), "%Y-%m-%d").date()


def runif(rng, size=None):
    """Uniform float in [min, max]."""
    return np.random.uniform(rng["min"], rng["max"], size=size)


def rint(rng, size=None):
    """Inclusive integer in [min, max]."""
    return np.random.randint(rng["min"], rng["max"] + 1, size=size)


def rchoice(options, size=None):
    return np.random.choice(options, size=size)


NOW = datetime.now()

# COMMAND ----------

# DBTITLE 1,Helpers — comments, constraints, and distribution_centers PK

def _esc(s):
    """Escape single quotes for SQL string literals."""
    return s.replace("'", "''")


def add_table_metadata(table, table_comment, column_comments):
    """Attach a table-level COMMENT and per-column COMMENTs to an existing table."""
    full = f"{SCHEMA}.{table}"
    spark.sql(f"COMMENT ON TABLE {full} IS '{_esc(table_comment)}'")
    for col, comment in column_comments.items():
        spark.sql(f"ALTER TABLE {full} ALTER COLUMN {col} COMMENT '{_esc(comment)}'")


def add_constraints(table, pk_cols, fks=()):
    """Declare an informational PRIMARY KEY plus zero-or-more FOREIGN KEYs.

    fks entries: (constraint_name, [local_cols], parent_table, [parent_cols]).
    Parent tables must already exist with matching PKs. PK columns are set
    NOT NULL first (a UC requirement).
    """
    full = f"{SCHEMA}.{table}"
    for col in pk_cols:
        spark.sql(f"ALTER TABLE {full} ALTER COLUMN {col} SET NOT NULL")
    pk_cols_sql = ", ".join(pk_cols)
    spark.sql(f"ALTER TABLE {full} ADD CONSTRAINT pk_{table} PRIMARY KEY ({pk_cols_sql})")
    for cname, local_cols, parent, parent_cols in fks:
        local_sql = ", ".join(local_cols)
        parent_sql = ", ".join(parent_cols)
        spark.sql(
            f"ALTER TABLE {full} ADD CONSTRAINT {cname} "
            f"FOREIGN KEY ({local_sql}) REFERENCES {SCHEMA}.{parent} ({parent_sql})"
        )


# Ensure distribution_centers has a PK on dc_id so child tables can FK to it.
# Idempotent: skip if the PK already exists.
_catalog, _schema_name = SCHEMA.split(".", 1)
_existing_pk = spark.sql(f"""
    SELECT constraint_name
    FROM {_catalog}.information_schema.table_constraints
    WHERE table_schema = '{_schema_name}'
      AND table_name = 'distribution_centers'
      AND constraint_type = 'PRIMARY KEY'
""").count()
if _existing_pk == 0:
    spark.sql(f"ALTER TABLE {SCHEMA}.distribution_centers ALTER COLUMN dc_id SET NOT NULL")
    spark.sql(
        f"ALTER TABLE {SCHEMA}.distribution_centers "
        f"ADD CONSTRAINT pk_distribution_centers PRIMARY KEY (dc_id)"
    )
    print(f"  ✓ Added pk_distribution_centers on {SCHEMA}.distribution_centers")
else:
    print(f"  - {SCHEMA}.distribution_centers already has a PRIMARY KEY; skipping.")

# COMMAND ----------

# DBTITLE 1,Build DC, SKU, and date dimensions
# --- DCs (sourced from distribution_centers table, filtered to Wholesale DCs) ---
dc_cfg = SD["distribution_centers"]
facility_filter = dc_cfg["facility_type"]
df_dc_meta = (
    spark.table(f"{SCHEMA}.distribution_centers")
         .where(f"facility_type = '{facility_filter}'")
         .toPandas()
         .sort_values("dc_id")
         .reset_index(drop=True)
)
if df_dc_meta.empty:
    raise RuntimeError(
        f"No rows in {SCHEMA}.distribution_centers with facility_type='{facility_filter}'. "
        "Populate that table first."
    )
dc_ids = df_dc_meta["dc_id"].tolist()
n_dcs = len(dc_ids)

# --- SKUs ---
sku_cfg = SD["skus"]
n_skus = sku_cfg["n_skus"]
sku_ids = [f"{sku_cfg['id_prefix']}{str(i + 1).zfill(sku_cfg['id_padding'])}"
           for i in range(n_skus)]

# --- Planning horizon ---
plan_cfg = SD["planning"]
start_date = _to_date(plan_cfg["start_date"])
n_periods = plan_cfg["n_periods"]
planning_dates = [start_date + timedelta(days=d) for d in range(n_periods)]

print(f"DCs:      {n_dcs} ({dc_ids[0]} … {dc_ids[-1]})")
print(f"SKUs:     {n_skus} ({sku_ids[0]} … {sku_ids[-1]})")
print(f"Periods:  {n_periods} days ({planning_dates[0]} → {planning_dates[-1]})")

# COMMAND ----------

# DBTITLE 1,product_master
# Each SKU picks one category uniformly, then draws its dimensional and
# economic params from that category's ranges. np.random.choice doesn't
# play well with a list of dicts, so we sample by index.
categories = sku_cfg["categories"]
products = []
for sku in sku_ids:
    cat = categories[int(np.random.randint(0, len(categories)))]
    cat_name = cat["name"]
    products.append({
        "sku_id": sku,
        "sku_description": f"{cat_name.title()} {sku}",
        "product_category": cat_name,
        "unit_cube_ft3":   round(float(runif(cat["unit_cube_ft3"])), 2),
        "unit_weight_lbs": round(float(runif(cat["unit_weight_lbs"])), 2),
        "units_per_case":   int(rchoice(sku_cfg["units_per_case_choices"])),
        "cases_per_pallet": int(rchoice(sku_cfg["cases_per_pallet_choices"])),
        "revenue_per_unit":              round(float(runif(cat["revenue_per_unit"])), 2),
        "cogs_per_unit":                 round(float(runif(cat["cogs_per_unit"])), 2),
        "holding_cost_per_unit_per_day": round(float(runif(cat["holding_cost_per_unit_per_day"])), 3),
        "is_active": True,
        "created_at": NOW,
        "updated_at": NOW,
    })

df_products_pd = pd.DataFrame(products)
products_by_sku = {p["sku_id"]: p for p in products}

spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.product_master")
spark.createDataFrame(df_products_pd).write.saveAsTable(f"{SCHEMA}.product_master")

add_table_metadata(
    "product_master",
    "SKU master: one row per stock-keeping unit with its pharma category, "
    "physical dimensions, and unit economics.",
    {
        "sku_id": "Stock-keeping unit identifier (e.g. SKU-0042). Primary key.",
        "sku_description": "Human-readable product description, prefixed with the pharma category.",
        "product_category": "Pharma product category: brand-name pharmaceuticals, generic pharmaceuticals, vaccines, over-the-counter medicines, health and beauty products, medical supplies, or medical equipment.",
        "unit_cube_ft3": "Volume per single unit in cubic feet.",
        "unit_weight_lbs": "Weight per single unit in pounds.",
        "units_per_case": "Eaches per case for this SKU.",
        "cases_per_pallet": "Cases per pallet for this SKU.",
        "revenue_per_unit": "Revenue earned per fulfilled unit in USD.",
        "cogs_per_unit": "Cost of goods sold per unit in USD.",
        "holding_cost_per_unit_per_day": "Daily holding cost per unit on-hand in USD.",
        "is_active": "True if the SKU is currently active in the catalog.",
        "created_at": "Row creation timestamp.",
        "updated_at": "Row last-updated timestamp.",
    },
)
add_constraints("product_master", ["sku_id"])

print(f"  ✓ product_master: {len(products)} rows")

# COMMAND ----------

# DBTITLE 1,dc_capacity
cap_cfg = SD["dc_capacity"]
capacity_rows = []
for dc in dc_ids:
    storage_positions = int(rint(cap_cfg["storage_positions"]))
    rack_positions = int(storage_positions * cap_cfg["rack_ratio"])
    floor_positions = storage_positions - rack_positions
    capacity_rows.append({
        "dc_id": dc,
        "effective_date": date(2025, 1, 1),
        "total_storage_positions": storage_positions,
        "total_storage_cube_ft3": round(float(runif(cap_cfg["storage_cube_ft3"])), 2),
        "rack_positions": rack_positions,
        "floor_positions": floor_positions,
        "pick_locations": int(rint(cap_cfg["pick_locations"])),
        "inbound_dock_doors": int(rint(cap_cfg["inbound_dock_doors"])),
        "outbound_dock_doors": int(rint(cap_cfg["outbound_dock_doors"])),
        "max_daily_inbound_pallets": int(rint(cap_cfg["max_daily_inbound_pallets"])),
        "max_daily_outbound_pallets": int(rint(cap_cfg["max_daily_outbound_pallets"])),
        "max_daily_inbound_cases": int(rint(cap_cfg["max_daily_inbound_cases"])),
        "max_daily_outbound_cases": int(rint(cap_cfg["max_daily_outbound_cases"])),
        "max_daily_throughput_units": int(rint(cap_cfg["max_daily_throughput_units"])),
        "max_concurrent_trailers": int(rint(cap_cfg["max_concurrent_trailers"])),
        "utilization_target_pct": cap_cfg["utilization_target_pct"],
        "created_at": NOW,
    })

spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.dc_capacity")
spark.createDataFrame(pd.DataFrame(capacity_rows)).write.saveAsTable(f"{SCHEMA}.dc_capacity")

add_table_metadata(
    "dc_capacity",
    "Physical capacity envelope for each Wholesale DC: storage positions, "
    "dock doors, and daily throughput ceilings. One row per DC per effective date.",
    {
        "dc_id": "Distribution center identifier; FK to distribution_centers.",
        "effective_date": "Date this capacity row becomes effective.",
        "total_storage_positions": "Total pallet positions across rack and floor.",
        "total_storage_cube_ft3": "Total storage volume in cubic feet.",
        "rack_positions": "Pallet positions in racks.",
        "floor_positions": "Pallet positions on the floor.",
        "pick_locations": "Number of forward-pick locations.",
        "inbound_dock_doors": "Inbound dock door count.",
        "outbound_dock_doors": "Outbound dock door count.",
        "max_daily_inbound_pallets": "Maximum pallets that can be received per day.",
        "max_daily_outbound_pallets": "Maximum pallets that can ship per day.",
        "max_daily_inbound_cases": "Maximum cases that can be received per day.",
        "max_daily_outbound_cases": "Maximum cases that can ship per day.",
        "max_daily_throughput_units": "Maximum units that can be processed end-to-end per day.",
        "max_concurrent_trailers": "Maximum trailers parked on-site at one time.",
        "utilization_target_pct": "Planning target for sustained utilization (0-1).",
        "created_at": "Row creation timestamp.",
    },
)
add_constraints(
    "dc_capacity",
    ["dc_id", "effective_date"],
    fks=[
        ("fk_dc_capacity_dc", ["dc_id"], "distribution_centers", ["dc_id"]),
    ],
)

print(f"  ✓ dc_capacity: {len(capacity_rows)} rows")

# COMMAND ----------

# DBTITLE 1,demand_forecast
demand_cfg = SD["demand"]
demand_rows = []
# Each (DC, SKU) gets its own baseline demand level; daily values vary around it.
for dc in dc_ids:
    for sku in sku_ids:
        prod = products_by_sku[sku]
        base = int(rint(demand_cfg["base_units_per_sku"]))
        for d in planning_dates:
            qty = int(base * float(runif(demand_cfg["daily_variance"])))
            cases = qty // prod["units_per_case"]
            pallets = cases / prod["cases_per_pallet"]
            demand_rows.append({
                "forecast_id": str(uuid.uuid4()),
                "dc_id": dc,
                "sku_id": sku,
                "forecast_date": d,
                "forecast_period": "DAILY",
                "demand_units": qty,
                "demand_cases": cases,
                "demand_pallets": round(pallets, 2),
                "demand_cube_ft3": round(qty * prod["unit_cube_ft3"], 2),
                "demand_weight_lbs": round(qty * prod["unit_weight_lbs"], 2),
                "confidence_level": round(float(runif(demand_cfg["confidence"])), 2),
                "forecast_source": demand_cfg["forecast_source"],
                "created_at": NOW,
            })

spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.demand_forecast")
spark.createDataFrame(pd.DataFrame(demand_rows)).write.saveAsTable(f"{SCHEMA}.demand_forecast")

add_table_metadata(
    "demand_forecast",
    "Daily demand forecast per (DC, SKU). Drives the LP fulfillment and "
    "the inventory / inbound sizing logic.",
    {
        "forecast_id": "Surrogate UUID primary key for this forecast row.",
        "dc_id": "Distribution center receiving the demand; FK to distribution_centers.",
        "sku_id": "SKU being forecasted; FK to product_master.",
        "forecast_date": "Day this forecast applies to.",
        "forecast_period": "Forecast bucket size; currently always DAILY.",
        "demand_units": "Forecast demand in eaches.",
        "demand_cases": "Forecast demand in cases (demand_units / units_per_case).",
        "demand_pallets": "Forecast demand in pallets.",
        "demand_cube_ft3": "Forecast demand volume in cubic feet.",
        "demand_weight_lbs": "Forecast demand weight in pounds.",
        "confidence_level": "Forecast confidence in [0, 1].",
        "forecast_source": "Method that produced the forecast (e.g. STATISTICAL_MODEL).",
        "created_at": "Row creation timestamp.",
    },
)
add_constraints(
    "demand_forecast",
    ["forecast_id"],
    fks=[
        ("fk_demand_forecast_dc",  ["dc_id"],  "distribution_centers", ["dc_id"]),
        ("fk_demand_forecast_sku", ["sku_id"], "product_master",       ["sku_id"]),
    ],
)

print(f"  ✓ demand_forecast: {len(demand_rows)} rows")

# COMMAND ----------

# DBTITLE 1,inventory_levels
# On-hand inventory is sized as `days_of_supply × avg_daily_demand` for each
# (DC, SKU). This gives a realistic position that bridges to the next inbound
# receipt without artificially over- or under-stocking. The previous
# random + rescale approach caused per-SKU stockouts that the LP couldn't
# fulfill, forcing inventory to accumulate later from continuing inbound.
inv_cfg = SD["inventory"]
snapshot_date = _to_date(inv_cfg["snapshot_date"])

demand_total_by_key = {}
for r in demand_rows:
    k = (r["dc_id"], r["sku_id"])
    demand_total_by_key[k] = demand_total_by_key.get(k, 0) + r["demand_units"]

inv_rows = []
for dc in dc_ids:
    for sku in sku_ids:
        prod = products_by_sku[sku]
        avg_daily_demand = demand_total_by_key.get((dc, sku), 0) / max(n_periods, 1)
        dos = float(runif(inv_cfg["days_of_supply"]))
        on_hand = max(1, int(round(dos * avg_daily_demand)))
        allocated = int(on_hand * inv_cfg["allocated_ratio"])
        available = on_hand - allocated
        cases = on_hand // prod["units_per_case"]
        pallets = cases / prod["cases_per_pallet"]
        inv_rows.append({
            "inventory_id": str(uuid.uuid4()),
            "dc_id": dc,
            "sku_id": sku,
            "snapshot_date": snapshot_date,
            "on_hand_units": on_hand,
            "on_hand_cases": cases,
            "on_hand_pallets": round(pallets, 2),
            "allocated_units": allocated,
            "available_units": available,
            "in_transit_units": int(rint(inv_cfg["in_transit_units"])),
            "backorder_units": 0,
            "total_cube_ft3": round(on_hand * prod["unit_cube_ft3"], 2),
            "total_weight_lbs": round(on_hand * prod["unit_weight_lbs"], 2),
            "days_of_supply": round(dos, 1),
            "inventory_value": round(on_hand * prod["cogs_per_unit"], 2),
            "created_at": NOW,
        })

spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.inventory_levels")
spark.createDataFrame(pd.DataFrame(inv_rows)).write.saveAsTable(f"{SCHEMA}.inventory_levels")

add_table_metadata(
    "inventory_levels",
    "Point-in-time inventory snapshot per (DC, SKU). Seeds the LP's "
    "starting inventory balance for the planning horizon.",
    {
        "inventory_id": "Surrogate UUID primary key for this snapshot row.",
        "dc_id": "Distribution center holding the inventory; FK to distribution_centers.",
        "sku_id": "SKU being measured; FK to product_master.",
        "snapshot_date": "As-of date of this inventory snapshot.",
        "on_hand_units": "Total physical units on hand.",
        "on_hand_cases": "On-hand units expressed as cases.",
        "on_hand_pallets": "On-hand units expressed as pallets.",
        "allocated_units": "Units reserved against existing orders.",
        "available_units": "Units free to allocate (on_hand minus allocated).",
        "in_transit_units": "Units in transit toward this DC.",
        "backorder_units": "Units on backorder; zero in synthetic data.",
        "total_cube_ft3": "On-hand inventory volume in cubic feet.",
        "total_weight_lbs": "On-hand inventory weight in pounds.",
        "days_of_supply": "Days the on-hand position covers at average daily demand.",
        "inventory_value": "On-hand inventory value in USD (on_hand × cogs_per_unit).",
        "created_at": "Row creation timestamp.",
    },
)
add_constraints(
    "inventory_levels",
    ["inventory_id"],
    fks=[
        ("fk_inventory_levels_dc",  ["dc_id"],  "distribution_centers", ["dc_id"]),
        ("fk_inventory_levels_sku", ["sku_id"], "product_master",       ["sku_id"]),
    ],
)

print(f"  ✓ inventory_levels: {len(inv_rows)} rows")

# COMMAND ----------

# DBTITLE 1,labor_availability
lab_cfg = SD["labor"]
labor_rows = []
for dc in dc_ids:
    for d in planning_dates:
        for func in lab_cfg["functions"]:
            headcount = int(rint(lab_cfg["headcount_per_function"]))
            planned_hrs = headcount * lab_cfg["hours_per_shift"]
            labor_rows.append({
                "labor_id": str(uuid.uuid4()),
                "dc_id": dc,
                "work_date": d,
                "shift_id": lab_cfg["shift_id"],
                "labor_function": func,
                "planned_headcount": headcount,
                "planned_hours": planned_hrs,
                "available_hours": round(planned_hrs * float(runif(lab_cfg["available_factor"])), 1),
                "overtime_hours_available": round(planned_hrs * lab_cfg["overtime_factor"], 1),
                "labor_cost_per_hour": lab_cfg["labor_cost_per_hour"],
                "overtime_cost_per_hour": lab_cfg["overtime_cost_per_hour"],
                "temp_labor_available_hours": round(float(runif(lab_cfg["temp_labor_hours"])), 1),
                "temp_labor_cost_per_hour": lab_cfg["temp_labor_cost_per_hour"],
                "created_at": NOW,
            })

spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.labor_availability")
spark.createDataFrame(pd.DataFrame(labor_rows)).write.saveAsTable(f"{SCHEMA}.labor_availability")

add_table_metadata(
    "labor_availability",
    "Daily labor availability per (DC, shift, labor function): planned "
    "headcount, regular hours, overtime, and temp labor.",
    {
        "labor_id": "Surrogate UUID primary key for this labor row.",
        "dc_id": "Distribution center this labor is staffed at; FK to distribution_centers.",
        "work_date": "Date the labor is available.",
        "shift_id": "Shift identifier (e.g. 1ST).",
        "labor_function": "Labor function: RECEIVING, PICKING, PACKING, or SHIPPING.",
        "planned_headcount": "Headcount planned for this function on this date.",
        "planned_hours": "Total planned hours (headcount × hours_per_shift).",
        "available_hours": "Effectively available hours after attrition / time-off factor.",
        "overtime_hours_available": "Additional overtime hours that can be tapped.",
        "labor_cost_per_hour": "Fully loaded regular labor cost per hour in USD.",
        "overtime_cost_per_hour": "Overtime labor cost per hour in USD.",
        "temp_labor_available_hours": "Available temp labor hours for this day.",
        "temp_labor_cost_per_hour": "Temp labor cost per hour in USD.",
        "created_at": "Row creation timestamp.",
    },
)
add_constraints(
    "labor_availability",
    ["labor_id"],
    fks=[
        ("fk_labor_availability_dc", ["dc_id"], "distribution_centers", ["dc_id"]),
    ],
)

print(f"  ✓ labor_availability: {len(labor_rows)} rows")

# COMMAND ----------

# DBTITLE 1,throughput_rates (explicit schema for nullable typed columns)
throughput_schema = StructType([
    StructField("rate_id", StringType(), False),
    StructField("dc_id", StringType(), False),
    StructField("activity_type", StringType(), False),
    StructField("product_category", StringType(), True),
    StructField("sku_id", StringType(), True),
    StructField("units_per_labor_hour", DoubleType(), True),
    StructField("cases_per_labor_hour", DoubleType(), True),
    StructField("pallets_per_labor_hour", DoubleType(), True),
    StructField("effective_date", DateType(), False),
    StructField("expiration_date", DateType(), True),
    StructField("rate_source", StringType(), True),
    StructField("created_at", TimestampType(), True),
])

tp_cfg = SD["throughput"]
throughput_rows = []
for dc in dc_ids:
    for sku in sku_ids:
        prod = products_by_sku[sku]
        uph = round(float(runif(tp_cfg["units_per_labor_hour"])), 1)
        throughput_rows.append((
            str(uuid.uuid4()),
            dc,
            "PICKING",
            prod["product_category"],
            sku,
            uph,
            round(uph / prod["units_per_case"], 1),
            round(uph / (prod["units_per_case"] * prod["cases_per_pallet"]), 2),
            date(2025, 1, 1),
            None,
            tp_cfg["rate_source"],
            NOW,
        ))

spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.throughput_rates")
spark.createDataFrame(throughput_rows, schema=throughput_schema) \
     .write.saveAsTable(f"{SCHEMA}.throughput_rates")

add_table_metadata(
    "throughput_rates",
    "Picking throughput rates per (DC, SKU): units / cases / pallets per "
    "labor hour. Feeds the LP's labor capacity constraint.",
    {
        "rate_id": "Surrogate UUID primary key for this rate row.",
        "dc_id": "Distribution center the rate applies to; FK to distribution_centers.",
        "activity_type": "Activity the rate measures (currently PICKING).",
        "product_category": "Pharma product category this rate applies to (denormalized from product_master).",
        "sku_id": "SKU the rate applies to; FK to product_master.",
        "units_per_labor_hour": "Eaches a single labor hour can process for this SKU.",
        "cases_per_labor_hour": "Cases per labor hour (units / units_per_case).",
        "pallets_per_labor_hour": "Pallets per labor hour.",
        "effective_date": "Date this rate becomes effective.",
        "expiration_date": "Date this rate stops applying (null = open-ended).",
        "rate_source": "How the rate was determined (e.g. ENGINEERED_STANDARD).",
        "created_at": "Row creation timestamp.",
    },
)
add_constraints(
    "throughput_rates",
    ["rate_id"],
    fks=[
        ("fk_throughput_rates_dc",  ["dc_id"],  "distribution_centers", ["dc_id"]),
        ("fk_throughput_rates_sku", ["sku_id"], "product_master",       ["sku_id"]),
    ],
)

print(f"  ✓ throughput_rates: {len(throughput_rows)} rows")

# COMMAND ----------

# DBTITLE 1,inbound_plan (explicit schema for nullable actual_date)
inbound_schema = StructType([
    StructField("inbound_id", StringType(), False),
    StructField("dc_id", StringType(), False),
    StructField("sku_id", StringType(), False),
    StructField("supplier_id", StringType(), True),
    StructField("po_number", StringType(), True),
    StructField("scheduled_date", DateType(), False),
    StructField("actual_date", DateType(), True),
    StructField("inbound_units", IntegerType(), True),
    StructField("inbound_cases", IntegerType(), True),
    StructField("inbound_pallets", DoubleType(), True),
    StructField("inbound_cube_ft3", DoubleType(), True),
    StructField("inbound_weight_lbs", DoubleType(), True),
    StructField("receipt_status", StringType(), True),
    StructField("trailer_count", IntegerType(), True),
    StructField("created_at", TimestampType(), True),
])

inb_cfg = SD["inbound"]
# Demand-aware: total receipts per (DC, SKU) ≈ total_demand × coverage_ratio,
# split across `n_receipts_per_horizon` EVENLY-SPACED arrival days. Even
# spacing prevents 10+ day gaps where per-SKU inventory hits zero (which
# causes the LP to under-fulfill and leftover inbound to accumulate).
coverage_rng = inb_cfg.get("coverage_ratio", {"min": 0.95, "max": 1.05})
n_receipts_rng = inb_cfg.get("n_receipts_per_horizon", {"min": 4, "max": 12})

inbound_rows = []
n_dates = len(planning_dates)
for dc in dc_ids:
    for sku in sku_ids:
        prod = products_by_sku[sku]
        units_per_pallet = prod["units_per_case"] * prod["cases_per_pallet"]
        total_demand_units = demand_total_by_key.get((dc, sku), 0)
        coverage = float(runif(coverage_rng))
        target_pallets = int(round(total_demand_units * coverage / units_per_pallet))
        if target_pallets <= 0:
            continue
        n_receipts = int(rint(n_receipts_rng))
        n_receipts = max(1, min(n_receipts, target_pallets, n_dates))
        # Evenly-spaced receipt day indices: e.g. 6 receipts in 30 days ->
        # days 2, 7, 12, 17, 22, 27.
        receipt_day_idxs = [
            int((i + 0.5) * n_dates / n_receipts) for i in range(n_receipts)
        ]
        base = target_pallets // n_receipts
        remainder = target_pallets - base * n_receipts
        pallets_per_receipt = [base + (1 if i < remainder else 0) for i in range(n_receipts)]
        np.random.shuffle(pallets_per_receipt)
        for day_idx, pallets in zip(receipt_day_idxs, pallets_per_receipt):
            if pallets < 1:
                continue
            d = planning_dates[day_idx]
            units = pallets * units_per_pallet
            cases = units // prod["units_per_case"]
            inbound_rows.append((
                str(uuid.uuid4()),
                dc,
                sku,
                f"SUP-{int(rint(inb_cfg['supplier_id']))}",
                f"PO-{np.random.randint(10000, 100000)}",
                d,
                None,
                int(units),
                int(cases),
                float(pallets),
                round(units * prod["unit_cube_ft3"], 2),
                round(units * prod["unit_weight_lbs"], 2),
                "PLANNED",
                max(1, pallets // 20),
                NOW,
            ))

spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.inbound_plan")
spark.createDataFrame(inbound_rows, schema=inbound_schema) \
     .write.saveAsTable(f"{SCHEMA}.inbound_plan")

add_table_metadata(
    "inbound_plan",
    "Planned inbound receipts at each WDC. Demand-aware sizing keeps the "
    "horizon's receipts ≈ horizon demand so storage stays roughly stable.",
    {
        "inbound_id": "Surrogate UUID primary key for this receipt.",
        "dc_id": "Destination distribution center; FK to distribution_centers.",
        "sku_id": "SKU being received; FK to product_master.",
        "supplier_id": "Originating supplier identifier (synthetic SUP-NNN).",
        "po_number": "Purchase order number (synthetic PO-NNNNN).",
        "scheduled_date": "Planned receipt date.",
        "actual_date": "Actual receipt date; null in the plan, populated post-receipt.",
        "inbound_units": "Receipt size in eaches.",
        "inbound_cases": "Receipt size in cases.",
        "inbound_pallets": "Receipt size in pallets.",
        "inbound_cube_ft3": "Receipt volume in cubic feet.",
        "inbound_weight_lbs": "Receipt weight in pounds.",
        "receipt_status": "Receipt lifecycle status (PLANNED, RECEIVED, ...).",
        "trailer_count": "Trailers needed for this receipt.",
        "created_at": "Row creation timestamp.",
    },
)
add_constraints(
    "inbound_plan",
    ["inbound_id"],
    fks=[
        ("fk_inbound_plan_dc",  ["dc_id"],  "distribution_centers", ["dc_id"]),
        ("fk_inbound_plan_sku", ["sku_id"], "product_master",       ["sku_id"]),
    ],
)

print(f"  ✓ inbound_plan: {len(inbound_rows)} rows")

# COMMAND ----------

# DBTITLE 1,outbound_plan (explicit schema for nullable actual_ship_date)
outbound_schema = StructType([
    StructField("outbound_id", StringType(), False),
    StructField("dc_id", StringType(), False),
    StructField("sku_id", StringType(), False),
    StructField("customer_id", StringType(), True),
    StructField("order_number", StringType(), True),
    StructField("scheduled_ship_date", DateType(), False),
    StructField("actual_ship_date", DateType(), True),
    StructField("outbound_units", IntegerType(), True),
    StructField("outbound_cases", IntegerType(), True),
    StructField("outbound_pallets", DoubleType(), True),
    StructField("outbound_cube_ft3", DoubleType(), True),
    StructField("outbound_weight_lbs", DoubleType(), True),
    StructField("ship_status", StringType(), True),
    StructField("order_priority", StringType(), True),
    StructField("carrier_id", StringType(), True),
    StructField("trailer_count", IntegerType(), True),
    StructField("created_at", TimestampType(), True),
])

out_cfg = SD["outbound"]
outbound_rows = []
for dc in dc_ids:
    for sku in sku_ids:
        prod = products_by_sku[sku]
        for d in planning_dates:
            if np.random.random() < out_cfg["probability_per_sku_day"]:
                units = int(rint(out_cfg["units"]))
                cases = units // prod["units_per_case"]
                pallets = cases / prod["cases_per_pallet"]
                outbound_rows.append((
                    str(uuid.uuid4()),
                    dc,
                    sku,
                    f"CUST-{np.random.randint(1000, 10000)}",
                    f"SO-{np.random.randint(100000, 1000000)}",
                    d,
                    None,
                    int(units),
                    int(cases),
                    round(pallets, 2),
                    round(units * prod["unit_cube_ft3"], 2),
                    round(units * prod["unit_weight_lbs"], 2),
                    "PLANNED",
                    str(rchoice(out_cfg["priorities"])),
                    str(rchoice(out_cfg["carriers"])),
                    1,
                    NOW,
                ))

spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.outbound_plan")
spark.createDataFrame(outbound_rows, schema=outbound_schema) \
     .write.saveAsTable(f"{SCHEMA}.outbound_plan")

add_table_metadata(
    "outbound_plan",
    "Planned outbound shipments leaving each WDC. Carries priority and "
    "carrier so the LP can reason about expediting / detention.",
    {
        "outbound_id": "Surrogate UUID primary key for this shipment.",
        "dc_id": "Origin distribution center; FK to distribution_centers.",
        "sku_id": "SKU being shipped; FK to product_master.",
        "customer_id": "Destination customer identifier (synthetic CUS-NNN).",
        "order_number": "Customer order number (synthetic ORD-NNNNN).",
        "scheduled_ship_date": "Planned ship date.",
        "actual_ship_date": "Actual ship date; null in the plan, populated post-ship.",
        "outbound_units": "Shipment size in eaches.",
        "outbound_cases": "Shipment size in cases.",
        "outbound_pallets": "Shipment size in pallets.",
        "outbound_cube_ft3": "Shipment volume in cubic feet.",
        "outbound_weight_lbs": "Shipment weight in pounds.",
        "ship_status": "Shipment lifecycle status (PLANNED, SHIPPED, ...).",
        "order_priority": "Order priority (CRITICAL, HIGH, MEDIUM, LOW).",
        "carrier_id": "Carrier handling the shipment.",
        "trailer_count": "Trailers needed for this shipment.",
        "created_at": "Row creation timestamp.",
    },
)
add_constraints(
    "outbound_plan",
    ["outbound_id"],
    fks=[
        ("fk_outbound_plan_dc",  ["dc_id"],  "distribution_centers", ["dc_id"]),
        ("fk_outbound_plan_sku", ["sku_id"], "product_master",       ["sku_id"]),
    ],
)

print(f"  ✓ outbound_plan: {len(outbound_rows)} rows")

# COMMAND ----------

# DBTITLE 1,optimization_parameters (per-DC, sourced from config)
params_schema = StructType([
    StructField("param_id", StringType(), False),
    StructField("dc_id", StringType(), False),
    StructField("param_category", StringType(), False),
    StructField("param_name", StringType(), False),
    StructField("param_value", DoubleType(), False),
    StructField("param_unit", StringType(), True),
    StructField("description", StringType(), True),
    StructField("effective_date", DateType(), False),
    StructField("expiration_date", DateType(), True),
    StructField("created_at", TimestampType(), True),
])

opt_cfg = cfg["optimization"]
param_specs = [
    ("OBJECTIVE",  "regular_labor_cost_per_hour",  opt_cfg["default_regular_labor_cost_per_hour"],  "USD",  "Cost per regular labor hour"),
    ("OBJECTIVE",  "overtime_labor_cost_per_hour", opt_cfg["default_overtime_labor_cost_per_hour"], "USD",  "Cost per overtime hour (1.5x)"),
    ("CONSTRAINT", "planning_horizon_days",        float(n_periods),                                 "DAYS", "Days in planning horizon"),
    ("CONSTRAINT", "min_fill_rate_target",         0.95,                                             "PCT",  "Target minimum fill rate"),
    ("WEIGHT",     "max_overtime_ratio",           SD["labor"]["overtime_factor"],                   "PCT",  "Max OT as fraction of regular hours"),
]
params_rows = []
for dc in dc_ids:
    for cat, name, value, unit, desc in param_specs:
        params_rows.append((
            str(uuid.uuid4()), dc, cat, name, float(value), unit, desc,
            date(2025, 1, 1), None, NOW,
        ))

spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.optimization_parameters")
spark.createDataFrame(params_rows, schema=params_schema) \
     .write.saveAsTable(f"{SCHEMA}.optimization_parameters")

add_table_metadata(
    "optimization_parameters",
    "Per-DC LP tunables: objective coefficients, hard constraints, and "
    "weights. Read by the optimization notebook and Streamlit app.",
    {
        "param_id": "Surrogate UUID primary key for this parameter row.",
        "dc_id": "Distribution center the parameter applies to; FK to distribution_centers.",
        "param_category": "Parameter category: OBJECTIVE, CONSTRAINT, or WEIGHT.",
        "param_name": "Parameter name (e.g. regular_labor_cost_per_hour).",
        "param_value": "Parameter numeric value.",
        "param_unit": "Unit of measure (USD, DAYS, PCT, ...).",
        "description": "Human-readable description of the parameter.",
        "effective_date": "Date this parameter becomes effective.",
        "expiration_date": "Date this parameter stops applying (null = open-ended).",
        "created_at": "Row creation timestamp.",
    },
)
add_constraints(
    "optimization_parameters",
    ["param_id"],
    fks=[
        ("fk_optimization_parameters_dc", ["dc_id"], "distribution_centers", ["dc_id"]),
    ],
)

print(f"  ✓ optimization_parameters: {len(params_rows)} rows")

# COMMAND ----------

# DBTITLE 1,NDC capacity + cross-dock simulation
# Pharma manufacturers ship into a National Distribution Center (NDC), which
# operates as a 24-hour cross-dock — no storage; everything must depart for a
# Wholesale DC inside `sla_dwell_hours`. We emit three tables:
#   * ndc_capacity  — per-NDC throughput/dock ceilings
#   * ndc_inbound   — one row per arriving pallet, with arrival timestamp
#   * ndc_outbound  — one row per departing pallet, with dwell hours + SLA flag
NDC_CFG = cfg.get("ndc")
ndc_inbound_rows = []
ndc_outbound_rows = []
ndc_capacity_rows = []

if NDC_CFG and NDC_CFG.get("active_dc_ids"):
    active_ndc_ids = list(NDC_CFG["active_dc_ids"])
    df_ndc_meta = (
        spark.table(f"{SCHEMA}.distribution_centers")
             .where("facility_type = 'NDC'")
             .toPandas()
             .sort_values("dc_id")
             .reset_index(drop=True)
    )
    df_ndc_meta = df_ndc_meta[df_ndc_meta["dc_id"].isin(active_ndc_ids)].reset_index(drop=True)

    if df_ndc_meta.empty:
        print(f"  ⚠ no NDCs match active_dc_ids={active_ndc_ids}; skipping NDC tables")
    else:
        cap_cfg = NDC_CFG["capacity"]
        sim_cfg = NDC_CFG["simulation"]
        sla_hours = float(cap_cfg["sla_dwell_hours"])

        ndc_capacity_schema = StructType([
            StructField("dc_id", StringType(), False),
            StructField("effective_date", DateType(), False),
            StructField("max_hourly_inbound_pallets", IntegerType(), False),
            StructField("max_hourly_outbound_pallets", IntegerType(), False),
            StructField("max_hourly_throughput_pallets", IntegerType(), False),
            StructField("inbound_dock_doors", IntegerType(), False),
            StructField("outbound_dock_doors", IntegerType(), False),
            StructField("sla_dwell_hours", DoubleType(), False),
            StructField("created_at", TimestampType(), True),
        ])
        for ndc_id in df_ndc_meta["dc_id"]:
            ndc_capacity_rows.append((
                ndc_id,
                date(2025, 1, 1),
                int(rint(cap_cfg["max_hourly_inbound_pallets"])),
                int(rint(cap_cfg["max_hourly_outbound_pallets"])),
                int(rint(cap_cfg["max_hourly_throughput_pallets"])),
                int(rint(cap_cfg["inbound_dock_doors"])),
                int(rint(cap_cfg["outbound_dock_doors"])),
                sla_hours,
                NOW,
            ))
        spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.ndc_capacity")
        spark.createDataFrame(ndc_capacity_rows, schema=ndc_capacity_schema) \
             .write.saveAsTable(f"{SCHEMA}.ndc_capacity")

        add_table_metadata(
            "ndc_capacity",
            "Per-NDC hourly throughput and dock ceilings, plus the 24h "
            "cross-dock SLA. One row per NDC per effective date.",
            {
                "dc_id": "NDC identifier; FK to distribution_centers.",
                "effective_date": "Date this capacity row becomes effective.",
                "max_hourly_inbound_pallets": "Maximum inbound pallets per hour.",
                "max_hourly_outbound_pallets": "Maximum outbound pallets per hour.",
                "max_hourly_throughput_pallets": "Maximum end-to-end pallets per hour.",
                "inbound_dock_doors": "Inbound dock door count.",
                "outbound_dock_doors": "Outbound dock door count.",
                "sla_dwell_hours": "Maximum dwell time allowed at the NDC (cross-dock SLA).",
                "created_at": "Row creation timestamp.",
            },
        )
        add_constraints(
            "ndc_capacity",
            ["dc_id", "effective_date"],
            fks=[
                ("fk_ndc_capacity_dc", ["dc_id"], "distribution_centers", ["dc_id"]),
            ],
        )

        print(f"  ✓ ndc_capacity: {len(ndc_capacity_rows)} rows")

        # Dest-DC weights — busier WDCs receive more pallets.
        dest_dcs = list(dc_ids)
        dem_by_dc = (
            pd.DataFrame(demand_rows)
              .groupby("dc_id")["demand_units"].sum()
              .reindex(dest_dcs).fillna(0)
        )
        if dem_by_dc.sum() > 0:
            wdc_weights = (dem_by_dc / dem_by_dc.sum()).values
        else:
            wdc_weights = np.full(len(dest_dcs), 1.0 / len(dest_dcs))

        hour_weights = np.array(sim_cfg["arrival_pattern_hour_weights"], dtype=float)
        hour_probs = hour_weights / hour_weights.sum()
        dwell_mean = float(sim_cfg["dwell_hours_mean"])
        dwell_sd = float(sim_cfg["dwell_hours_stddev"])
        dwell_max = float(sim_cfg["dwell_hours_max"])

        suppliers = list(sim_cfg["suppliers"])
        carriers = list(sim_cfg["manufacturer_carriers"])

        for ndc_id in df_ndc_meta["dc_id"]:
            for d in planning_dates:
                daily_pallets = int(rint(sim_cfg["daily_inbound_pallets"]))
                arrival_hour = np.random.choice(24, size=daily_pallets, p=hour_probs)
                dwell_h = np.clip(
                    np.random.normal(dwell_mean, dwell_sd, size=daily_pallets),
                    0.5, dwell_max,
                )
                pallet_skus = np.random.choice(sku_ids, size=daily_pallets)
                pallet_suppliers = np.random.choice(suppliers, size=daily_pallets)
                pallet_carriers = np.random.choice(carriers, size=daily_pallets)
                pallet_dests = np.random.choice(dest_dcs, size=daily_pallets, p=wdc_weights)

                day_midnight = datetime.combine(d, datetime.min.time())
                for p_idx in range(daily_pallets):
                    sku = pallet_skus[p_idx]
                    prod = products_by_sku[sku]
                    units = int(prod["units_per_case"] * prod["cases_per_pallet"])
                    cube = round(units * prod["unit_cube_ft3"], 2)
                    arr_ts = day_midnight + timedelta(hours=int(arrival_hour[p_idx]))
                    dwell = float(dwell_h[p_idx])
                    dep_ts = arr_ts + timedelta(hours=dwell)
                    inb_id = str(uuid.uuid4())

                    ndc_inbound_rows.append((
                        inb_id, ndc_id, str(pallet_suppliers[p_idx]), sku,
                        arr_ts, 1, units, cube, str(pallet_carriers[p_idx]), NOW,
                    ))
                    ndc_outbound_rows.append((
                        str(uuid.uuid4()), ndc_id, str(pallet_dests[p_idx]), sku,
                        dep_ts, 1, units, cube, inb_id,
                        round(dwell, 2), bool(dwell <= sla_hours), NOW,
                    ))

        ndc_inbound_schema = StructType([
            StructField("ndc_inbound_id", StringType(), False),
            StructField("dc_id", StringType(), False),
            StructField("supplier_id", StringType(), False),
            StructField("sku_id", StringType(), False),
            StructField("arrival_hour_utc", TimestampType(), False),
            StructField("inbound_pallets", IntegerType(), False),
            StructField("inbound_units", IntegerType(), False),
            StructField("inbound_cube_ft3", DoubleType(), False),
            StructField("manufacturer_carrier", StringType(), True),
            StructField("created_at", TimestampType(), True),
        ])
        ndc_outbound_schema = StructType([
            StructField("ndc_outbound_id", StringType(), False),
            StructField("dc_id", StringType(), False),
            StructField("dest_dc_id", StringType(), False),
            StructField("sku_id", StringType(), False),
            StructField("departure_hour_utc", TimestampType(), False),
            StructField("outbound_pallets", IntegerType(), False),
            StructField("outbound_units", IntegerType(), False),
            StructField("outbound_cube_ft3", DoubleType(), False),
            StructField("linked_inbound_id", StringType(), False),
            StructField("dwell_hours", DoubleType(), False),
            StructField("met_sla", BooleanType(), False),
            StructField("created_at", TimestampType(), True),
        ])

        spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.ndc_inbound")
        spark.createDataFrame(ndc_inbound_rows, schema=ndc_inbound_schema) \
             .write.saveAsTable(f"{SCHEMA}.ndc_inbound")

        add_table_metadata(
            "ndc_inbound",
            "One row per pallet arriving at the NDC from a pharma "
            "manufacturer, with arrival timestamp and carrier.",
            {
                "ndc_inbound_id": "Surrogate UUID primary key for this arriving pallet.",
                "dc_id": "NDC receiving the pallet; FK to distribution_centers.",
                "supplier_id": "Pharma manufacturer that shipped the pallet.",
                "sku_id": "SKU on the pallet; FK to product_master.",
                "arrival_hour_utc": "UTC timestamp when the pallet arrived (truncated to the hour).",
                "inbound_pallets": "Pallet count for this row (always 1 in current synthetic data).",
                "inbound_units": "Units on the pallet.",
                "inbound_cube_ft3": "Pallet volume in cubic feet.",
                "manufacturer_carrier": "Carrier that delivered the pallet (e.g. PharmaExpress).",
                "created_at": "Row creation timestamp.",
            },
        )
        add_constraints(
            "ndc_inbound",
            ["ndc_inbound_id"],
            fks=[
                ("fk_ndc_inbound_dc",  ["dc_id"],  "distribution_centers", ["dc_id"]),
                ("fk_ndc_inbound_sku", ["sku_id"], "product_master",       ["sku_id"]),
            ],
        )

        spark.sql(f"DROP TABLE IF EXISTS {SCHEMA}.ndc_outbound")
        spark.createDataFrame(ndc_outbound_rows, schema=ndc_outbound_schema) \
             .write.saveAsTable(f"{SCHEMA}.ndc_outbound")

        add_table_metadata(
            "ndc_outbound",
            "One row per pallet departing the NDC for a downstream WDC, "
            "with dwell time and SLA attainment flag.",
            {
                "ndc_outbound_id": "Surrogate UUID primary key for this departing pallet.",
                "dc_id": "NDC the pallet is departing from; FK to distribution_centers.",
                "dest_dc_id": "Destination WDC; FK to distribution_centers.",
                "sku_id": "SKU on the pallet; FK to product_master.",
                "departure_hour_utc": "UTC timestamp when the pallet departed (truncated to the hour).",
                "outbound_pallets": "Pallet count for this row (always 1 in current synthetic data).",
                "outbound_units": "Units on the pallet.",
                "outbound_cube_ft3": "Pallet volume in cubic feet.",
                "linked_inbound_id": "Matching ndc_inbound.ndc_inbound_id for the same pallet; FK to ndc_inbound.",
                "dwell_hours": "Hours the pallet sat at the NDC before departure.",
                "met_sla": "True if dwell_hours ≤ sla_dwell_hours.",
                "created_at": "Row creation timestamp.",
            },
        )
        add_constraints(
            "ndc_outbound",
            ["ndc_outbound_id"],
            fks=[
                ("fk_ndc_outbound_dc",      ["dc_id"],            "distribution_centers", ["dc_id"]),
                ("fk_ndc_outbound_dest_dc", ["dest_dc_id"],       "distribution_centers", ["dc_id"]),
                ("fk_ndc_outbound_sku",     ["sku_id"],           "product_master",       ["sku_id"]),
                ("fk_ndc_outbound_inbound", ["linked_inbound_id"],"ndc_inbound",          ["ndc_inbound_id"]),
            ],
        )

        n_out = len(ndc_outbound_rows)
        sla_pct = (100.0 * sum(r[10] for r in ndc_outbound_rows) / n_out) if n_out > 0 else 0.0
        avg_dwell = (sum(r[9] for r in ndc_outbound_rows) / n_out) if n_out > 0 else 0.0
        print(f"  ✓ ndc_inbound:  {len(ndc_inbound_rows):,} rows")
        print(f"  ✓ ndc_outbound: {n_out:,} rows  ·  "
              f"SLA attainment: {sla_pct:.1f}%  ·  avg dwell: {avg_dwell:.1f}h")
else:
    print("  ⚠ ndc: section missing from config.yaml; skipping NDC tables")

# COMMAND ----------

# DBTITLE 1,Summary
print("\n" + "=" * 70)
print("  DATA GENERATION COMPLETE")
print("=" * 70)
counts = {
    "product_master":          spark.table(f"{SCHEMA}.product_master").count(),
    "dc_capacity":             spark.table(f"{SCHEMA}.dc_capacity").count(),
    "demand_forecast":         spark.table(f"{SCHEMA}.demand_forecast").count(),
    "inventory_levels":        spark.table(f"{SCHEMA}.inventory_levels").count(),
    "labor_availability":      spark.table(f"{SCHEMA}.labor_availability").count(),
    "throughput_rates":        spark.table(f"{SCHEMA}.throughput_rates").count(),
    "inbound_plan":            spark.table(f"{SCHEMA}.inbound_plan").count(),
    "outbound_plan":           spark.table(f"{SCHEMA}.outbound_plan").count(),
    "optimization_parameters": spark.table(f"{SCHEMA}.optimization_parameters").count(),
}
for _ndc_tbl in ("ndc_capacity", "ndc_inbound", "ndc_outbound"):
    try:
        counts[_ndc_tbl] = spark.table(f"{SCHEMA}.{_ndc_tbl}").count()
    except Exception:
        pass
for tbl, n in counts.items():
    print(f"  {tbl:30s} {n:>10,} rows")
print("=" * 70)
print(f"  DCs:     {n_dcs}")
print(f"  SKUs:    {n_skus}")
print(f"  Periods: {n_periods} days")
print("=" * 70)
print("\nNext: run the 'DC Capacity Optimization Model' notebook to solve.")
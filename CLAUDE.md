# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Databricks App + companion notebooks that solve a multi-DC capacity optimization problem as a linear program. The app (`app.py`) is a Streamlit UI that reads synthetic data from Unity Catalog (`tenbosch.scmo_poc.*`), runs a per-DC LP with PuLP/HiGHS, and visualizes utilization, slack, and what-ifs across the network. The two `.py` files that look like scripts (`Generate Synthetic Data.py`, `DC Capacity Optimization Model.py`) are Databricks notebooks in source form (`# Databricks notebook source` + `# COMMAND ----------` cell delimiters) — do not execute them locally with `python`; they're run inside the workspace.

## Pipeline order

1. **`Generate Synthetic Data.py`** (notebook) — populates `tenbosch.scmo_poc.{product_master, dc_capacity, demand_forecast, inventory_levels, labor_availability, throughput_rates, inbound_plan, outbound_plan, optimization_parameters, ndc_capacity, ndc_inbound, ndc_outbound}`. The DC list is **read from** `tenbosch.scmo_poc.distribution_centers` filtered to `facility_type='Wholesale DC'` — that table is the source of truth for which DCs exist and is **not** written by this notebook.
2. **`DC Capacity Optimization Model.py`** (notebook) — loads the input tables, solves the LP once per DC, writes results back to Delta, and produces a network summary. Reference implementation for the math; the app re-implements the same LP in `run_optimization()` (app.py:848) so it can solve interactively.
3. **`app.py`** (Streamlit, deployed as a Databricks App) — reads the same tables, lets users adjust penalties / capacity multipliers in the sidebar, and re-solves on demand.

## Config is the contract

`config.yaml` is the single source of truth for everything tunable: schema name, synthetic data ranges, LP penalties, fallback capacities, solver preference, and Streamlit UI sliders. The notebooks and `app.py` all `yaml.safe_load` it on startup. **Never hard-code values that already live in `config.yaml`** — add a key and read it instead. The notebook config path is absolute (`/Workspace/Users/jeff.tenbosch@databricks.com/dc-capacity-optimizer/config.yaml`); the app uses a relative `config.yaml`.

Two things to know about the schema:
- `schema: tenbosch.scmo_poc` — all tables are referenced as `{SCHEMA}.{table}` throughout.
- `synthetic_data.distribution_centers.facility_type` filters the source DC table; the legacy `n_dcs` knob does **not** exist.

## LP model shape

Per-DC objective: maximize revenue − holding cost − labor cost − overflow penalties. Hard constraints: inventory balance, demand ceiling, labor capacity (regular + overtime). Soft constraints (with slack + per-unit penalty): storage cu-ft, throughput units, inbound pallets, outbound pallets. A non-zero slack value is the actionable alert — the LP is intentionally always feasible. Solver preference in config (`["highs", "cbc"]`) is tried in order; `_make_solver()` (app.py:816) picks the first one that imports.

## Common commands

### Local app dev
```bash
# Install deps (no venv set up by default; create one if you want isolation)
pip install -r requirements.txt

# Run the Streamlit app locally — needs Databricks auth + a SQL warehouse
export DATABRICKS_WAREHOUSE_ID=<warehouse_id>
# Plus DATABRICKS_HOST / DATABRICKS_TOKEN (or a configured CLI profile)
streamlit run app.py
```

### Deploy / sync to Databricks
```bash
# Bundle deploy (databricks.yml defines the `dc-capacity-optimizer-react` app)
databricks bundle deploy -t dev

# Or sync local files into the workspace folder the notebooks expect
databricks sync . /Workspace/Users/jeff.tenbosch@databricks.com/dc-capacity-optimizer
```

`app.yaml` declares the app's entry point (`streamlit run app.py`) and binds the `sql-warehouse` resource to the `DATABRICKS_WAREHOUSE_ID` env var — when running locally you must export it yourself.

## Editing gotchas

- **The two notebooks are Python source files with magic comments**, not regular modules. Edits must preserve `# COMMAND ----------` cell boundaries and `# MAGIC %md` / `# MAGIC %pip` prefixes. Don't import from them.
- **The app re-implements the LP** rather than importing from the notebook — if you change the objective or constraints in one place, mirror the change in the other (`run_optimization()` in app.py:848 vs the per-DC function in `DC Capacity Optimization Model.py` around line 178).
- **`run_query()` (app.py:85) auto-reconnects** on stale-connection keywords. Wrap new SQL through it, not `_run_query()` directly, so the cache-clear-and-retry path stays consistent.
- **Streamlit caching**: data loaders use `@st.cache_data(ttl=APP_CFG["cache_ttl_seconds"])`; `get_connection()` uses `@st.cache_resource`. Clearing the connection cache on auth/SSL errors is part of the reconnect contract — don't remove it.
- **NDC vs WDC**: the National Distribution Center is a separate cross-dock tier (pharma, 24h SLA, no storage) with its own tables (`ndc_*`) and a pure-Python what-if simulator in the app (`simulate_ndc_dispatch()`, app.py:524). It is **not** part of the LP — keep them separate.

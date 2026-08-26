DC Capacity Forecasting Tool
Built by: Supply Chain Management & Optimization Team
Target Users: Replenishment Operations, Engineering, Supply Chain Planning

Overview
A Streamlit application for forecasting and analyzing distribution center storage capacity across the cold chain network. Helps teams understand capacity drivers, identify inefficiencies, and take data-driven action to optimize warehouse space.

Features
Tab	Purpose
Network Overview	KPIs, stock status distribution, DC utilization, DOH heatmap
Capacity Drivers	DOH breakdown by DC, replenishment policy vs actual, inventory composition
Inefficiency & Outliers	IQR-based outlier detection for DOH and volume, dead stock analysis
Forecasting & ML	Linear regression / exponential smoothing projections for DOH and on-hand qty
Recommendations	Auto-generated, prioritized action items with efficiency scorecards
ML Forecasting
Two forecasting methods are available (configurable via sidebar):

Linear Regression - Projects trend forward with 95% confidence interval based on residual std
Exponential Smoothing (alpha=0.3) - Weights recent observations more heavily; CI expands with horizon
Forecast horizon is adjustable from 2-12 weeks.

Setup
Prerequisites
Databricks workspace with SQL Warehouse access
Access to adh_genpro_use2_prd catalog tables
Python 3.10+
Environment Variables
DATABRICKS_WAREHOUSE_ID=<your-sql-warehouse-id>
Installation
pip install -r requirements.txt
Run Locally
streamlit run app.py
Deploy as Databricks App
This app is configured for deployment via app.yaml and manifest.yaml.

Architecture
app.py
├── Page Config & Theme
├── Data Connection (Statement Execution API)
├── ML Forecasting Engine (numpy-based)
│   ├── linear_forecast()
│   ├── exponential_smoothing()
│   ├── detect_outliers_iqr()
│   └── compute_capacity_score()
├── SQL Queries
│   ├── QUERY_CORE_WEEKLY (main inventory + replenishment)
│   ├── QUERY_VOLUME_DATA (location/volume metrics)
│   ├── QUERY_BILLING (demand/sales)
│   └── QUERY_DC_UTILIZATION (current utilization)
├── Sidebar Filters & Settings
└── 5-Tab UI Layout
Data Sources
adh_genpro_use2_prd.s_warehouse_ops.custom_mat_fcst_position_hhd - Inventory positions
adh_genpro_use2_prd.s_warehouse_ops.replenish_material_hist_hhd - Replenishment policies
adh_genpro_use2_prd.s_products.material_details_consolidated_hhd - Material master
adh_genpro_use2_prd.s_distribution.a_dim_locations_hhd - Warehouse locations
adh_genpro_use2_prd.s_distribution.inventory_locations_hhd - SKU-location mapping
adh_genpro_use2_prd.g_order360.billing_transactions_hhd - Demand/billing
Key Metrics
DOH (Days on Hand) = On-Hand Qty / (Deseasonalized Forecast / 28)
PEL (Pallet Equivalent Locations) = Volume / 75,000 cu in
WMAPE = Sum(|Demand - Forecast|) / Sum(|Demand|) x 100
Efficiency Score = 0-100 composite based on overstock %, dead stock %, and median DOH
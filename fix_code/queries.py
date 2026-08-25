"""SQL query strings for the SCMO dashboard."""

QUERY_CORE_WEEKLY = """
WITH mat_dedup AS (
  SELECT mat_nbr, mat_grp
  FROM (
    SELECT mat_nbr, mat_grp, ROW_NUMBER() OVER (PARTITION BY mat_nbr ORDER BY mat_nbr) AS rn
    FROM adh_genpro_use2_prd.s_products.material_details_consolidated_hhd
    WHERE temp_cond_desc IN ('Keep Frozen', 'Refrge/Do Not Freeze')
  ) WHERE rn = 1
),
item_key_lookup AS (
  SELECT DISTINCT di.item_key, mm.mat_nbr8 AS mat
  FROM adh_genpro_use2_prd.s_products.material_details_consolidated_hhd mm
  INNER JOIN adh_genpro_use2_prd.s_distribution.a_dim_items_hhd di ON di.item_number_nk = mm.mat_nbr8
  WHERE mm.temp_cond_desc IN ('Keep Frozen', 'Refrge/Do Not Freeze')
),
reserve_items AS (
  SELECT DISTINCT il.division_lookup AS plant, ik.mat AS item
  FROM adh_genpro_use2_prd.s_distribution.inventory_locations_hhd il
  INNER JOIN item_key_lookup ik ON il.item_key = ik.item_key
  INNER JOIN adh_genpro_use2_prd.s_distribution.a_dim_locations_hhd loc ON loc.loc_key = il.location_key
  WHERE loc.active_reserve_casepick IN ('Reserve')
    AND YEAR(DATE(il.snapshot_date_key)) >= 2026
),
dc_lookup AS (
  SELECT DISTINCT division_number, division_name AS DC_Name
  FROM adh_genpro_use2_prd.s_distribution.a_dim_divisions_hhd
),
rep_dedup AS (
  SELECT dc_formatted, itm_formatted, order_up_to_level_qty, order_up_to_level_days,
    safety_stock_qty, safety_stock_days, repln_invtry, repln_days, deseasonalized_fcst_for_period
  FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY dc_formatted, itm_formatted ORDER BY adh_updated DESC) AS rn
    FROM adh_genpro_use2_prd.s_warehouse_ops.replenish_material_hist_hhd
  ) WHERE rn = 1
),
inventory_weekly AS (
  SELECT
    CONCAT('WK-', LPAD(WEEKOFYEAR(inv.invtry_date), 2, '0')) AS week_label,
    YEAR(inv.invtry_date) AS yr,
    WEEKOFYEAR(inv.invtry_date) AS wk,
    inv.plant,
    inv.plant || '-' || LTRIM('0', TRIM(inv.mat_nbr)) AS SKU,
    AVG(inv.on_hand_qty) AS Avg_OH_Qty,
    AVG(inv.deseasonalized_fcst_qty) AS Avg_Deseasoned_Fcst_Qty,
    TRY_DIVIDE(AVG(inv.on_hand_qty), AVG(inv.deseasonalized_fcst_qty) / 28) AS DOH,
    CASE
      WHEN AVG(inv.on_hand_qty) > AVG(rep.order_up_to_level_qty) THEN 'Overstock'
      WHEN AVG(inv.on_hand_qty) < AVG(rep.safety_stock_qty) THEN 'Understock'
      ELSE 'Within_Range'
    END AS Stock_Status,
    AVG(rep.order_up_to_level_qty) AS Avg_Order_Up_To_Level_Qty,
    AVG(rep.order_up_to_level_days) AS Avg_Order_Up_To_Level_Days,
    AVG(rep.safety_stock_qty) AS Avg_Safety_Stock_Qty,
    AVG(rep.safety_stock_days) AS Avg_Safety_Stock_Days,
    AVG(rep.repln_invtry) AS Avg_Repln_Invtry,
    AVG(rep.repln_days) AS Avg_Repln_Days,
    AVG(rep.deseasonalized_fcst_for_period) AS Forecast_Demand,
    mat.mat_grp
  FROM adh_genpro_use2_prd.s_warehouse_ops.custom_mat_fcst_position_hhd inv
  INNER JOIN mat_dedup mat ON inv.mat_nbr = mat.mat_nbr
  INNER JOIN rep_dedup rep ON rep.itm_formatted = LTRIM('0', TRIM(inv.mat_nbr)) AND inv.plant = rep.dc_formatted
  INNER JOIN reserve_items ri ON ri.plant = inv.plant AND ri.item = LTRIM('0', TRIM(inv.mat_nbr))
  WHERE inv.invtry_date >= '2026-01-01'
    AND (inv.on_hand_qty + inv.on_ord_qty) > 0
    AND inv.plant NOT IN ('087', '203', '204', '200', '220', '230')
  GROUP BY WEEKOFYEAR(inv.invtry_date), YEAR(inv.invtry_date), inv.plant,
    inv.plant || '-' || LTRIM('0', TRIM(inv.mat_nbr)), mat.mat_grp
)
SELECT iw.week_label, iw.yr, iw.wk, iw.plant, iw.SKU, iw.Avg_OH_Qty,
  iw.Avg_Deseasoned_Fcst_Qty, iw.DOH, iw.Stock_Status,
  iw.Avg_Order_Up_To_Level_Qty, iw.Avg_Order_Up_To_Level_Days,
  iw.Avg_Safety_Stock_Qty, iw.Avg_Safety_Stock_Days,
  iw.Avg_Repln_Invtry, iw.Avg_Repln_Days, iw.Forecast_Demand,
  dc.DC_Name, iw.mat_grp
FROM inventory_weekly iw
INNER JOIN dc_lookup dc ON iw.plant = dc.division_number
WHERE (iw.DOH <= 365 OR iw.DOH IS NULL)
"""

QUERY_VOLUME_DATA = """
WITH item_key_lookup AS (
  SELECT DISTINCT di.item_key, mm.mat_nbr8 AS mat
  FROM adh_genpro_use2_prd.s_products.material_details_consolidated_hhd mm
  INNER JOIN adh_genpro_use2_prd.s_distribution.a_dim_items_hhd di ON di.item_number_nk = mm.mat_nbr8
  WHERE mm.temp_cond_desc IN ('Keep Frozen', 'Refrge/Do Not Freeze')
),
sku_location AS (
  SELECT DISTINCT il.location_key, ik.mat AS item, il.division_lookup AS division,
    YEAR(DATE(il.snapshot_date_key)) AS yr, WEEKOFYEAR(DATE(il.snapshot_date_key)) AS wk
  FROM adh_genpro_use2_prd.s_distribution.inventory_locations_hhd il
  INNER JOIN item_key_lookup ik ON il.item_key = ik.item_key
  WHERE YEAR(DATE(il.snapshot_date_key)) >= 2026
),
location_data AS (
  SELECT
    loc.division_nbr_nk || '-' || sw.item AS SKU,
    loc.division_nbr_nk AS plant,
    sw.yr, sw.wk,
    CONCAT('WK-', LPAD(sw.wk, 2, '0')) AS week_label,
    AVG(loc.height * loc.width * loc.length) AS Total_Volume_Cubic_Inches,
    AVG(loc.height * loc.width * loc.length) / 75000 AS PEL,
    SUM(CASE WHEN loc.active_reserve_casepick = 'Active' THEN 1 ELSE 0 END) AS active_locations,
    SUM(CASE WHEN loc.active_reserve_casepick = 'Reserve' THEN 1 ELSE 0 END) AS reserve_locations
  FROM adh_genpro_use2_prd.s_distribution.a_dim_locations_hhd loc
  INNER JOIN sku_location sw ON loc.loc_key = sw.location_key
  WHERE loc.active_reserve_casepick IN ('Reserve')
    AND TRIM(loc.class) = 'Refrig'
    AND loc.height < 9999 AND loc.width < 9999 AND loc.length < 9999
    
  GROUP BY loc.division_nbr_nk || '-' || sw.item, loc.division_nbr_nk, sw.yr, sw.wk
)
SELECT * FROM location_data
WHERE Total_Volume_Cubic_Inches <= 1000000
  AND plant NOT IN ('087', '203', '204', '200', '220', '230')
"""

QUERY_BILLING = """
WITH mat_dedup AS (
  SELECT mat_nbr, mat_grp
  FROM (
    SELECT mat_nbr, mat_grp, ROW_NUMBER() OVER (PARTITION BY mat_nbr ORDER BY mat_nbr) AS rn
    FROM adh_genpro_use2_prd.s_products.material_details_consolidated_hhd
    WHERE temp_cond_desc IN ('Keep Frozen', 'Refrge/Do Not Freeze')
  ) WHERE rn = 1
),
item_key_lookup AS (
  SELECT DISTINCT di.item_key, mm.mat_nbr8 AS mat
  FROM adh_genpro_use2_prd.s_products.material_details_consolidated_hhd mm
  INNER JOIN adh_genpro_use2_prd.s_distribution.a_dim_items_hhd di ON di.item_number_nk = mm.mat_nbr8
  WHERE mm.temp_cond_desc IN ('Keep Frozen', 'Refrge/Do Not Freeze')
),
reserve_items AS (
  SELECT DISTINCT il.division_lookup AS plant, ik.mat AS item
  FROM adh_genpro_use2_prd.s_distribution.inventory_locations_hhd il
  INNER JOIN item_key_lookup ik ON il.item_key = ik.item_key
  INNER JOIN adh_genpro_use2_prd.s_distribution.a_dim_locations_hhd loc ON loc.loc_key = il.location_key
  WHERE loc.active_reserve_casepick IN ('Reserve')
    AND YEAR(DATE(il.snapshot_date_key)) >= 2026
)
SELECT
  bt.ship_plant || '-' || LTRIM('0', TRIM(bt.mat_nbr)) AS SKU,
  bt.ship_plant AS plant,
  YEAR(bt.invc_created_date) AS yr,
  WEEKOFYEAR(bt.invc_created_date) AS wk,
  CONCAT('WK-', LPAD(WEEKOFYEAR(bt.invc_created_date), 2, '0')) AS week_label,
  SUM(bt.invc_qty) AS Total_Billing_Qty,
  SUM(bt.sales_qty) AS Total_Sales_Qty,
  mat.mat_grp
FROM adh_genpro_use2_prd.g_order360.billing_transactions_hhd bt
INNER JOIN mat_dedup mat ON TRIM(bt.mat_nbr) = TRIM(mat.mat_nbr)
INNER JOIN reserve_items ri ON ri.plant = bt.ship_plant AND ri.item = LTRIM('0', TRIM(bt.mat_nbr))
WHERE bt.invc_created_date >= '2026-01-01'
  AND bt.sales_qty > 0
  AND bt.ship_plant NOT IN ('087', '203', '204', '200', '220', '230')
GROUP BY bt.ship_plant || '-' || LTRIM('0', TRIM(bt.mat_nbr)),
  bt.ship_plant, YEAR(bt.invc_created_date), WEEKOFYEAR(bt.invc_created_date), mat.mat_grp
"""

QUERY_DC_UTILIZATION = """
SELECT dc_name, utilization_pct, lat, lon
FROM VALUES
  ('Mansfield', 90, 42.7588, -71.21),
  ('Corona', 89, 33.8753, -117.5664),
  ('Seattle', 83, 47.6062, -122.3321),
  ('Puerto Rico', 82, 18.2208, -66.5901),
  ('Whitestown', 80, 39.9970, -86.3458),
  ('Chicago', 80, 41.8781, -87.6298),
  ('Richmond', 79, 37.5407, -77.4360),
  ('Newburgh', 76, 41.5034, -74.0104),
  ('Buford', 75, 34.1207, -83.9810),
  ('MONTCLAIR D&S', 74, 40.8259, -74.2090),
  ('Sacramento', 72, 38.5816, -121.4944),
  ('Olive Branch', 72, 34.9618, -89.8295),
  ('Bethlehem', 70, 40.6259, -75.3705),
  ('Columbus', 69, 39.9612, -82.9988),
  ('Kansas City', 68, 39.0997, -94.5786),
  ('Phoenix-B', 68, 33.4484, -112.0740),
  ('BROOKS D&S', 67, 31.7541, -83.5434),
  ('Denver', 66, 39.7392, -104.9903),
  ('Dallas', 65, 32.7767, -96.7970),
  ('Orlando Vista', 64, 28.5383, -81.3792),
  ('Williamston', 62, 42.6890, -84.2830),
  ('Raleigh', 61, 35.7796, -78.6382),
  ('Amityville', 60, 40.6790, -73.4171),
  ('Louisville', 57, 38.2527, -85.7585),
  ('Honolulu', 57, 21.3069, -157.8583),
  ('Shakopee', 55, 44.7974, -93.5272),
  ('Houston', 49, 29.7604, -95.3698),
  ('Salt Lake City', 45, 40.7608, -111.8910),
  ('DOTHAN D&S', 39, 31.2232, -85.3905)
  AS t(dc_name, utilization_pct, lat, lon)
"""

QUERY_FILTER_OPTIONS = """
WITH dc_lookup AS (
  SELECT DISTINCT division_number AS plant, division_name AS DC_Name
  FROM adh_genpro_use2_prd.s_distribution.a_dim_divisions_hhd
  WHERE division_number IN ( '003','004','008','010','012','014','015','016','017','018',
            '019','021','022','023','026','027','028','029','030','037',
            '038','039','041','049','055','063')
    AND adh_delete_flag = FALSE
    AND end_date IS NULL
),
date_range AS (
  SELECT DISTINCT YEAR(invtry_date) AS yr, MONTH(invtry_date) AS mn, WEEKOFYEAR(invtry_date) AS wk
  FROM adh_genpro_use2_prd.s_warehouse_ops.custom_mat_fcst_position_hhd
  WHERE invtry_date >= '2026-01-01'
)
SELECT 'dc' AS filter_type, plant AS val, DC_Name AS label FROM dc_lookup
UNION ALL
SELECT 'year', CAST(yr AS STRING), CAST(yr AS STRING) FROM (SELECT DISTINCT yr FROM date_range)
UNION ALL
SELECT 'month', CAST(yr AS STRING) || '-' || LPAD(CAST(mn AS STRING), 2, '0'), CAST(yr AS STRING) || '-' || LPAD(CAST(mn AS STRING), 2, '0') FROM (SELECT DISTINCT yr, mn FROM date_range)
UNION ALL
SELECT 'month_week', CAST(yr AS STRING) || '-' || LPAD(CAST(mn AS STRING), 2, '0'), CAST(wk AS STRING) FROM date_range
UNION ALL
SELECT 'week', CAST(wk AS STRING), CAST(wk AS STRING) FROM (SELECT DISTINCT wk FROM date_range)
UNION ALL
SELECT 'mat_grp', mat_grp,
  CASE mat_grp WHEN 'BRX' THEN 'Brand Rx' WHEN 'GRX' THEN 'Generic Rx'
  WHEN 'OTC' THEN 'OTC Brand' WHEN 'OTG' THEN 'OTC Generic'
  WHEN 'MSU' THEN 'Medical/Surgical' WHEN 'GMR' THEN 'Generic Medical'
  WHEN 'HBC' THEN 'Health & Beauty' WHEN 'HBG' THEN 'H&B Generic'
  WHEN 'SSU' THEN 'Surgical Supply' WHEN 'HHC' THEN 'Home Health'
  ELSE mat_grp END
FROM (SELECT DISTINCT mat_grp FROM adh_genpro_use2_prd.s_products.material_details_consolidated_hhd
      WHERE temp_cond_desc IN ('Keep Frozen', 'Refrge/Do Not Freeze') AND mat_grp IS NOT NULL)
"""

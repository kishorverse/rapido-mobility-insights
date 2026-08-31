-- Rapido Intelligent Mobility Insights - analysis queries
--
-- The queries backing the dashboard, in standalone runnable form against the
-- schema in schema.sql. Run the whole file, or copy any one of them:
--
--     mysql -u root -p rapido_mobility < sql/analysis_queries.sql
--
-- src/feature_engineering.py holds the same queries with a parameterised WHERE
-- clause injected by build_where_clause(), so the dashboard can filter on city,
-- vehicle, traffic, weather, status, date range and hour. Filter values are
-- always passed to the driver as parameters and never formatted into the SQL
-- string. The versions below are the unfiltered equivalents, for reading and
-- for running by hand.

USE rapido_mobility;

-- ===========================================================================
-- 1. Headline KPIs
-- ===========================================================================

-- q_kpi_summary: totals, outcome rates, revenue and averages in one row.
SELECT
    COUNT(*)                                            AS total_bookings,
    SUM(b.booking_status = 'Completed')                 AS completed,
    SUM(b.booking_status = 'Cancelled')                 AS cancelled,
    SUM(b.booking_status = 'Incomplete')                AS incomplete,
    ROUND(100 * AVG(b.booking_status = 'Completed'), 2) AS completion_rate,
    ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate,
    ROUND(SUM(CASE WHEN b.booking_status = 'Completed'
                   THEN b.booking_value ELSE 0 END), 2) AS revenue,
    ROUND(AVG(b.booking_value), 2)                      AS avg_fare,
    ROUND(AVG(b.ride_distance_km), 2)                   AS avg_distance,
    ROUND(AVG(b.surge_multiplier), 3)                   AS avg_surge,
    COUNT(DISTINCT b.customer_id)                       AS active_customers,
    COUNT(DISTINCT b.driver_id)                         AS active_drivers
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id;


-- ===========================================================================
-- 2. Volume and demand
-- ===========================================================================

-- q_rides_by_hour: volume and cancellation rate across the 24-hour clock.
SELECT HOUR(b.booking_ts) AS hour_of_day,
       COUNT(*)           AS rides,
       ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- q_rides_by_weekday: volume by day of week, in calendar order.
SELECT DAYNAME(b.booking_ts) AS day_of_week,
       COUNT(*)              AS rides,
       ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
GROUP BY day_of_week, DAYOFWEEK(b.booking_ts)
ORDER BY DAYOFWEEK(b.booking_ts);

-- q_rides_by_city: volume, cancellation rate and revenue per city.
SELECT c.city_name AS city,
       COUNT(*)    AS rides,
       ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate,
       ROUND(AVG(b.booking_value), 2)                      AS avg_fare,
       ROUND(SUM(b.booking_value), 2)                      AS revenue
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
GROUP BY c.city_name
ORDER BY rides DESC;

-- q_monthly_trend: monthly bookings, cancellation rate and completed revenue.
SELECT DATE_FORMAT(b.booking_ts, '%Y-%m') AS month_label,
       COUNT(*)                           AS rides,
       ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate,
       ROUND(SUM(CASE WHEN b.booking_status = 'Completed'
                      THEN b.booking_value ELSE 0 END), 2) AS revenue
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
GROUP BY month_label
ORDER BY month_label;

-- q_demand_by_day_hour: counts for the weekday-by-hour heatmap.
SELECT DAYNAME(b.booking_ts) AS day_of_week,
       HOUR(b.booking_ts)    AS hour_of_day,
       COUNT(*)              AS rides
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
GROUP BY day_of_week, hour_of_day;


-- ===========================================================================
-- 3. Cancellations
-- ===========================================================================

-- q_cancellation_rate_by_city_hour: rate for every city-hour cell.
SELECT c.city_name        AS city,
       HOUR(b.booking_ts) AS hour_of_day,
       COUNT(*)           AS rides,
       ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
GROUP BY city, hour_of_day
ORDER BY city, hour_of_day;

-- q_peak_cancellation_windows: worst city-hour windows with a real sample size.
SELECT c.city_name        AS city,
       HOUR(b.booking_ts) AS hour_of_day,
       COUNT(*)           AS rides,
       ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
GROUP BY city, hour_of_day
HAVING rides >= 50
ORDER BY cancel_rate DESC
LIMIT 15;

-- q_cancellation_by_category: rate by traffic level. Swap b.traffic_level for
-- b.weather_condition, v.vehicle_name or c.city_name for the other three cuts.
SELECT b.traffic_level AS traffic_level,
       COUNT(*)        AS rides,
       ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2)  AS cancel_rate,
       ROUND(100 * AVG(b.booking_status = 'Incomplete'), 2) AS incomplete_rate
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
GROUP BY traffic_level
ORDER BY cancel_rate DESC;

-- q_status_split_by_category: outcome share per level, for 100% stacked bars.
SELECT b.traffic_level  AS traffic_level,
       b.booking_status AS booking_status,
       COUNT(*)         AS rides,
       ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (
           PARTITION BY b.traffic_level), 2) AS share_pct
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
GROUP BY traffic_level, b.booking_status
ORDER BY traffic_level, b.booking_status;

-- q_cancellation_reasons: distribution of stated incomplete-ride reasons.
-- Recorded only for rides that ended Incomplete; Cancelled rows carry no code.
SELECT b.incomplete_ride_reason AS incomplete_ride_reason,
       COUNT(*)                 AS rides
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
WHERE b.incomplete_ride_reason IS NOT NULL
GROUP BY b.incomplete_ride_reason
ORDER BY rides DESC;

-- q_cancellation_reasons_by_party: reasons attributed to the accountable party.
-- The source records what went wrong but not who is answerable, so this mapping
-- is an operational interpretation, kept in one place here and in Python.
SELECT b.incomplete_ride_reason AS incomplete_ride_reason,
       CASE
           WHEN b.incomplete_ride_reason = 'Customer No-show' THEN 'Customer'
           WHEN b.incomplete_ride_reason = 'Driver Delay'     THEN 'Driver'
           WHEN b.incomplete_ride_reason = 'Vehicle Issue'    THEN 'Driver'
           WHEN b.incomplete_ride_reason = 'App Issue'        THEN 'Platform'
           ELSE 'Unknown'
       END      AS responsible_party,
       COUNT(*) AS rides,
       ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS share_pct
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
WHERE b.incomplete_ride_reason IS NOT NULL
GROUP BY b.incomplete_ride_reason
ORDER BY rides DESC;

-- q_cancellation_by_surge: rate across surge bands. Surge is the strongest
-- single lever here - cancellations run ~5% at 1.0x and ~35% above 2.0x.
SELECT CASE
           WHEN b.surge_multiplier <= 1.0 THEN 'None (1.0)'
           WHEN b.surge_multiplier <= 1.5 THEN 'Low (1.0-1.5)'
           WHEN b.surge_multiplier <= 2.0 THEN 'Medium (1.5-2.0)'
           ELSE 'High (>2.0)'
       END      AS surge_band,
       COUNT(*) AS rides,
       ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate,
       ROUND(AVG(b.booking_value), 2)                      AS avg_fare
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
GROUP BY surge_band
ORDER BY cancel_rate DESC;


-- ===========================================================================
-- 4. Fares and revenue
-- ===========================================================================

-- q_distance_vs_fare: sampled pairs for the scatter plot. The dashboard never
-- pulls all 100k rows to the browser.
SELECT b.ride_distance_km, b.booking_value,
       v.vehicle_name AS vehicle_type, b.surge_multiplier
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
ORDER BY RAND()
LIMIT 4000;

-- q_fare_by_vehicle_city: average fare and fare per km by city and vehicle.
SELECT c.city_name    AS city,
       v.vehicle_name AS vehicle_type,
       COUNT(*)       AS rides,
       ROUND(AVG(b.booking_value), 2)                      AS avg_fare,
       ROUND(AVG(b.booking_value / b.ride_distance_km), 2) AS avg_fare_per_km,
       ROUND(SUM(b.booking_value), 2)                      AS revenue
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
GROUP BY city, vehicle_type
ORDER BY city, vehicle_type;

-- q_surge_by_hour: average surge multiplier and fare per hour.
SELECT HOUR(b.booking_ts)                AS hour_of_day,
       ROUND(AVG(b.surge_multiplier), 3) AS avg_surge,
       ROUND(AVG(b.booking_value), 2)    AS avg_fare,
       COUNT(*)                          AS rides
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- q_revenue_by_city_vehicle: completed-ride revenue only. Cancelled bookings
-- carry a quoted value in the source data but never convert.
SELECT c.city_name    AS city,
       v.vehicle_name AS vehicle_type,
       ROUND(SUM(b.booking_value), 2) AS revenue,
       COUNT(*)                       AS rides
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
WHERE b.booking_status = 'Completed'
GROUP BY city, vehicle_type
ORDER BY revenue DESC;

-- q_fare_by_conditions: fare and surge across traffic x weather combinations.
SELECT b.traffic_level, b.weather_condition,
       COUNT(*)                          AS rides,
       ROUND(AVG(b.booking_value), 2)    AS avg_fare,
       ROUND(AVG(b.surge_multiplier), 3) AS avg_surge,
       ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
GROUP BY b.traffic_level, b.weather_condition
ORDER BY cancel_rate DESC;


-- ===========================================================================
-- 5. Locations and zone demand
-- ===========================================================================

-- q_top_pickup_locations: busiest pickup zones. Location codes repeat across
-- cities, so the zone label is city-qualified.
SELECT CONCAT(c.city_name, ' / ', l.location_code) AS zone,
       COUNT(*)                                    AS rides,
       ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate,
       ROUND(AVG(b.booking_value), 2)                      AS avg_fare
FROM bookings b
JOIN cities        c ON c.city_id         = b.city_id
JOIN vehicle_types v ON v.vehicle_type_id = b.vehicle_type_id
JOIN locations     l ON l.location_id     = b.pickup_location_id
GROUP BY zone
ORDER BY rides DESC
LIMIT 20;

-- q_busiest_routes: busiest city-qualified pickup-to-drop routes.
SELECT CONCAT(c.city_name, ': ', pl.location_code,
              ' -> ', dl.location_code)              AS route,
       COUNT(*)                                      AS rides,
       ROUND(AVG(b.ride_distance_km), 2)             AS avg_distance,
       ROUND(AVG(b.booking_value), 2)                AS avg_fare,
       ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate
FROM bookings b
JOIN cities        c  ON c.city_id         = b.city_id
JOIN vehicle_types v  ON v.vehicle_type_id = b.vehicle_type_id
JOIN locations     pl ON pl.location_id    = b.pickup_location_id
JOIN locations     dl ON dl.location_id    = b.drop_location_id
GROUP BY route
ORDER BY rides DESC
LIMIT 20;

-- q_demand_level_distribution: zone aggregates by demand level and vehicle.
-- The source contains only Low and Medium levels - no High is present.
SELECT ld.demand_level, v.vehicle_name AS vehicle_type,
       COUNT(*)                               AS zone_slots,
       ROUND(AVG(ld.avg_wait_time_min), 2)    AS avg_wait_min,
       ROUND(AVG(ld.avg_surge_multiplier), 3) AS avg_surge,
       SUM(ld.total_requests)                 AS total_requests
FROM location_demand ld
JOIN vehicle_types v ON v.vehicle_type_id = ld.vehicle_type_id
GROUP BY ld.demand_level, vehicle_type
ORDER BY ld.demand_level, vehicle_type;

-- q_wait_time_by_hour: average zone wait time and surge by hour.
SELECT ld.hour_of_day,
       ROUND(AVG(ld.avg_wait_time_min), 2)    AS avg_wait_min,
       ROUND(AVG(ld.avg_surge_multiplier), 3) AS avg_surge,
       SUM(ld.total_requests)                 AS total_requests
FROM location_demand ld
GROUP BY ld.hour_of_day
ORDER BY ld.hour_of_day;


-- ===========================================================================
-- 6. Customers and drivers
-- ===========================================================================

-- q_high_risk_customers: highest observed cancellation rate. This is history,
-- not a prediction - the cancellation model scores a specific future booking.
SELECT cu.customer_id,
       ci.city_name           AS city,
       cu.customer_age        AS age,
       cu.total_bookings,
       cu.cancelled_rides,
       ROUND(100 * cu.cancellation_rate, 2) AS cancel_rate,
       cu.avg_customer_rating AS rating
FROM customers cu
JOIN cities ci ON ci.city_id = cu.city_id
WHERE cu.total_bookings >= 5
ORDER BY cu.cancellation_rate DESC, cu.total_bookings DESC
LIMIT 50;

-- q_unreliable_drivers: highest observed delay rate.
SELECT d.driver_id,
       ci.city_name   AS city,
       v.vehicle_name AS vehicle_type,
       d.total_assigned_rides,
       d.delay_count,
       ROUND(100 * d.delay_rate, 2)      AS delay_rate,
       ROUND(100 * d.acceptance_rate, 2) AS acceptance_rate,
       d.avg_driver_rating               AS rating,
       d.avg_pickup_delay_min
FROM drivers d
JOIN cities        ci ON ci.city_id        = d.city_id
JOIN vehicle_types v  ON v.vehicle_type_id = d.vehicle_type_id
WHERE d.total_assigned_rides >= 5
ORDER BY d.delay_rate DESC, d.total_assigned_rides DESC
LIMIT 50;

-- q_top_drivers: reliability = 35% acceptance + 35% punctuality + 30% rating.
SELECT d.driver_id,
       ci.city_name   AS city,
       v.vehicle_name AS vehicle_type,
       d.total_assigned_rides,
       ROUND(100 * d.acceptance_rate, 2) AS acceptance_rate,
       ROUND(100 * d.delay_rate, 2)      AS delay_rate,
       d.avg_driver_rating               AS rating,
       ROUND(100 * (0.35 * d.acceptance_rate
                  + 0.35 * (1 - d.delay_rate)
                  + 0.30 * ((d.avg_driver_rating - 1) / 4)), 2) AS reliability_score
FROM drivers d
JOIN cities        ci ON ci.city_id        = d.city_id
JOIN vehicle_types v  ON v.vehicle_type_id = d.vehicle_type_id
WHERE d.total_assigned_rides >= 5
ORDER BY reliability_score DESC
LIMIT 50;

-- q_customer_demographics: counts and behaviour by gender and age band.
SELECT cu.customer_gender AS gender,
       CASE
           WHEN cu.customer_age < 25 THEN '18-24'
           WHEN cu.customer_age < 35 THEN '25-34'
           WHEN cu.customer_age < 45 THEN '35-44'
           WHEN cu.customer_age < 60 THEN '45-59'
           ELSE '60+'
       END AS age_band,
       COUNT(*)                                  AS customers,
       ROUND(100 * AVG(cu.cancellation_rate), 2) AS avg_cancel_rate,
       ROUND(AVG(cu.avg_customer_rating), 2)     AS avg_rating
FROM customers cu
GROUP BY gender, age_band
ORDER BY gender, age_band;

-- q_customer_vs_driver_ratings: side-by-side rating distributions.
SELECT 'Customer' AS party,
       ROUND(cu.avg_customer_rating, 1) AS rating,
       COUNT(*) AS people
FROM customers cu
GROUP BY rating
UNION ALL
SELECT 'Driver' AS party,
       ROUND(d.avg_driver_rating, 1) AS rating,
       COUNT(*) AS people
FROM drivers d
GROUP BY rating
ORDER BY party, rating;

-- q_driver_scatter: driver-level metrics for the reliability scatter plot.
SELECT d.driver_id,
       ROUND(100 * (0.35 * d.acceptance_rate
                  + 0.35 * (1 - d.delay_rate)
                  + 0.30 * ((d.avg_driver_rating - 1) / 4)), 2)
           AS driver_reliability_score,
       d.avg_pickup_delay_min,
       d.avg_driver_rating,
       d.total_assigned_rides
FROM drivers d
LIMIT 3000;


-- ===========================================================================
-- 7. Data explorer
-- ===========================================================================

-- q_bookings_page: one page of booking records. Paging happens in SQL, served
-- by idx_bookings_ts, so the browser never receives the full table.
SELECT b.booking_id, b.booking_ts, c.city_name AS city,
       v.vehicle_name   AS vehicle_type,
       pl.location_code AS pickup, dl.location_code AS drop_zone,
       b.ride_distance_km, b.traffic_level, b.weather_condition,
       b.surge_multiplier, b.booking_value, b.booking_status
FROM bookings b
JOIN cities        c  ON c.city_id         = b.city_id
JOIN vehicle_types v  ON v.vehicle_type_id = b.vehicle_type_id
JOIN locations     pl ON pl.location_id    = b.pickup_location_id
JOIN locations     dl ON dl.location_id    = b.drop_location_id
ORDER BY b.booking_ts DESC
LIMIT 50 OFFSET 0;

-- q_filter_options: distinct values for the sidebar, so nothing is hard-coded.
SELECT city_name FROM cities ORDER BY city_name;
SELECT vehicle_name FROM vehicle_types ORDER BY vehicle_name;
SELECT MIN(booking_ts) AS min_ts, MAX(booking_ts) AS max_ts FROM bookings;

"""Named, parameterised SQL queries backing the dashboard.

Every query is parameterised through the shared filter builder, so no user
input is ever concatenated into SQL. Each function returns a DataFrame shaped
for a specific chart in :mod:`rapido.charts`.
"""

from __future__ import annotations

import logging

import pandas as pd

from rapido import db

logger = logging.getLogger(__name__)

#: Reusable join from bookings to its dimensions.
_BASE_JOIN = """
    FROM bookings b
    JOIN cities        c  ON c.city_id         = b.city_id
    JOIN vehicle_types v  ON v.vehicle_type_id = b.vehicle_type_id
"""


def build_where_clause(filters: dict | None) -> tuple[str, list]:
    """Translate a filter dict into a SQL WHERE clause and parameter list.

    Supported keys: ``cities``, ``vehicle_types``, ``traffic_levels``,
    ``weather_conditions``, ``statuses``, ``date_from``, ``date_to``,
    ``hour_from``, ``hour_to``.

    Returns:
        The clause (starting with ``WHERE``, or empty) and its parameters.
    """
    filters = filters or {}
    clauses: list[str] = []
    params: list = []

    list_filters = [
        ("cities", "c.city_name"),
        ("vehicle_types", "v.vehicle_name"),
        ("traffic_levels", "b.traffic_level"),
        ("weather_conditions", "b.weather_condition"),
        ("statuses", "b.booking_status"),
    ]
    for key, column in list_filters:
        values = filters.get(key)
        if values:
            placeholders = ", ".join(["%s"] * len(values))
            clauses.append(f"{column} IN ({placeholders})")
            params.extend(values)

    if filters.get("date_from"):
        clauses.append("b.booking_ts >= %s")
        params.append(str(filters["date_from"]))
    if filters.get("date_to"):
        clauses.append("b.booking_ts < DATE_ADD(%s, INTERVAL 1 DAY)")
        params.append(str(filters["date_to"]))
    if filters.get("hour_from") is not None:
        clauses.append("HOUR(b.booking_ts) >= %s")
        params.append(int(filters["hour_from"]))
    if filters.get("hour_to") is not None:
        clauses.append("HOUR(b.booking_ts) <= %s")
        params.append(int(filters["hour_to"]))

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def _run(sql: str, params: list) -> pd.DataFrame:
    """Execute a query, returning an empty frame if it fails."""
    try:
        return db.read_sql(sql, tuple(params))
    except db.DatabaseError as exc:
        logger.error("Query failed: %s", exc)
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Headline KPIs
# --------------------------------------------------------------------------- #


def q_kpi_summary(filters: dict | None = None) -> pd.DataFrame:
    """Total bookings, outcome rates, revenue and averages."""
    where, params = build_where_clause(filters)
    sql = f"""
        SELECT
            COUNT(*)                                              AS total_bookings,
            SUM(b.booking_status = 'Completed')                   AS completed,
            SUM(b.booking_status = 'Cancelled')                   AS cancelled,
            SUM(b.booking_status = 'Incomplete')                  AS incomplete,
            ROUND(100 * AVG(b.booking_status = 'Completed'), 2)   AS completion_rate,
            ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2)   AS cancel_rate,
            ROUND(SUM(CASE WHEN b.booking_status = 'Completed'
                           THEN b.booking_value ELSE 0 END), 2)   AS revenue,
            ROUND(AVG(b.booking_value), 2)                        AS avg_fare,
            ROUND(AVG(b.ride_distance_km), 2)                     AS avg_distance,
            ROUND(AVG(b.surge_multiplier), 3)                     AS avg_surge,
            COUNT(DISTINCT b.customer_id)                         AS active_customers,
            COUNT(DISTINCT b.driver_id)                           AS active_drivers
        {_BASE_JOIN}
        {where}
    """
    return _run(sql, params)


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #


def q_rides_by_hour(filters: dict | None = None) -> pd.DataFrame:
    """Booking volume and cancellation rate per hour."""
    where, params = build_where_clause(filters)
    sql = f"""
        SELECT HOUR(b.booking_ts) AS hour_of_day,
               COUNT(*)           AS rides,
               ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate
        {_BASE_JOIN}
        {where}
        GROUP BY hour_of_day
        ORDER BY hour_of_day
    """
    return _run(sql, params)


def q_rides_by_weekday(filters: dict | None = None) -> pd.DataFrame:
    """Booking volume by day of week."""
    where, params = build_where_clause(filters)
    sql = f"""
        SELECT DAYNAME(b.booking_ts) AS day_of_week,
               COUNT(*)              AS rides,
               ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate
        {_BASE_JOIN}
        {where}
        GROUP BY day_of_week, DAYOFWEEK(b.booking_ts)
        ORDER BY DAYOFWEEK(b.booking_ts)
    """
    return _run(sql, params)


def q_rides_by_city(filters: dict | None = None) -> pd.DataFrame:
    """Booking volume, cancellation rate and revenue per city."""
    where, params = build_where_clause(filters)
    sql = f"""
        SELECT c.city_name AS city,
               COUNT(*)    AS rides,
               ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate,
               ROUND(AVG(b.booking_value), 2)                      AS avg_fare,
               ROUND(SUM(b.booking_value), 2)                      AS revenue
        {_BASE_JOIN}
        {where}
        GROUP BY c.city_name
        ORDER BY rides DESC
    """
    return _run(sql, params)


def q_monthly_trend(filters: dict | None = None) -> pd.DataFrame:
    """Monthly bookings, cancellations and completed revenue."""
    where, params = build_where_clause(filters)
    sql = f"""
        SELECT DATE_FORMAT(b.booking_ts, '%Y-%m') AS month_label,
               COUNT(*)                           AS rides,
               ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate,
               ROUND(SUM(CASE WHEN b.booking_status = 'Completed'
                              THEN b.booking_value ELSE 0 END), 2) AS revenue
        {_BASE_JOIN}
        {where}
        GROUP BY month_label
        ORDER BY month_label
    """
    return _run(sql, params)


def q_demand_by_day_hour(filters: dict | None = None) -> pd.DataFrame:
    """Booking counts for the weekday-by-hour heatmap."""
    where, params = build_where_clause(filters)
    sql = f"""
        SELECT DAYNAME(b.booking_ts) AS day_of_week,
               HOUR(b.booking_ts)    AS hour_of_day,
               COUNT(*)              AS rides
        {_BASE_JOIN}
        {where}
        GROUP BY day_of_week, hour_of_day
    """
    return _run(sql, params)


# --------------------------------------------------------------------------- #
# Cancellations
# --------------------------------------------------------------------------- #


def q_cancellation_rate_by_city_hour(filters: dict | None = None) -> pd.DataFrame:
    """Cancellation rate for every city-hour combination."""
    where, params = build_where_clause(filters)
    sql = f"""
        SELECT c.city_name        AS city,
               HOUR(b.booking_ts) AS hour_of_day,
               COUNT(*)           AS rides,
               ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate
        {_BASE_JOIN}
        {where}
        GROUP BY city, hour_of_day
        ORDER BY city, hour_of_day
    """
    return _run(sql, params)


def q_peak_cancellation_windows(
    filters: dict | None = None, limit: int = 15, min_rides: int = 50
) -> pd.DataFrame:
    """The worst city-hour windows, filtered for a meaningful sample size."""
    where, params = build_where_clause(filters)
    sql = f"""
        SELECT c.city_name        AS city,
               HOUR(b.booking_ts) AS hour_of_day,
               COUNT(*)           AS rides,
               ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate
        {_BASE_JOIN}
        {where}
        GROUP BY city, hour_of_day
        HAVING rides >= %s
        ORDER BY cancel_rate DESC
        LIMIT %s
    """
    return _run(sql, params + [min_rides, limit])


def q_cancellation_by_category(
    filters: dict | None = None, category: str = "traffic_level"
) -> pd.DataFrame:
    """Cancellation rate by traffic, weather, vehicle type or city."""
    columns = {
        "traffic_level": "b.traffic_level",
        "weather_condition": "b.weather_condition",
        "vehicle_type": "v.vehicle_name",
        "city": "c.city_name",
    }
    if category not in columns:
        raise ValueError(f"Unsupported category {category!r}.")

    where, params = build_where_clause(filters)
    sql = f"""
        SELECT {columns[category]} AS {category},
               COUNT(*)            AS rides,
               ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2)  AS cancel_rate,
               ROUND(100 * AVG(b.booking_status = 'Incomplete'), 2) AS incomplete_rate
        {_BASE_JOIN}
        {where}
        GROUP BY {category}
        ORDER BY cancel_rate DESC
    """
    return _run(sql, params)


def q_status_split_by_category(
    filters: dict | None = None, category: str = "traffic_level"
) -> pd.DataFrame:
    """Outcome share per level of a categorical driver, for stacked bars."""
    columns = {
        "traffic_level": "b.traffic_level",
        "weather_condition": "b.weather_condition",
        "vehicle_type": "v.vehicle_name",
        "city": "c.city_name",
    }
    if category not in columns:
        raise ValueError(f"Unsupported category {category!r}.")

    where, params = build_where_clause(filters)
    sql = f"""
        SELECT {columns[category]}  AS {category},
               b.booking_status     AS booking_status,
               COUNT(*)             AS rides,
               ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (
                   PARTITION BY {columns[category]}), 2) AS share_pct
        {_BASE_JOIN}
        {where}
        GROUP BY {category}, b.booking_status
        ORDER BY {category}, b.booking_status
    """
    return _run(sql, params)


def q_cancellation_reasons(filters: dict | None = None) -> pd.DataFrame:
    """Distribution of stated incomplete-ride reasons."""
    where, params = build_where_clause(filters)
    joiner = "AND" if where else "WHERE"
    sql = f"""
        SELECT b.incomplete_ride_reason AS incomplete_ride_reason,
               COUNT(*)                 AS rides
        {_BASE_JOIN}
        {where}
        {joiner} b.incomplete_ride_reason IS NOT NULL
        GROUP BY b.incomplete_ride_reason
        ORDER BY rides DESC
    """
    return _run(sql, params)


#: Attribution of each stated reason to the party that caused it.
#: The source data records *what* went wrong but not *who* is accountable,
#: so this mapping is our operational interpretation, kept in one place.
REASON_PARTY = {
    "Customer No-show": "Customer",
    "Driver Delay": "Driver",
    "Vehicle Issue": "Driver",
    "App Issue": "Platform",
}


def q_cancellation_reasons_by_party(filters: dict | None = None) -> pd.DataFrame:
    """Stated reasons attributed to the responsible party.

    Answers the brief's "customer vs driver cancellation reasons" question.
    The CASE expression uses only literal keys from :data:`REASON_PARTY`,
    never user input, so it stays injection-safe.
    """
    where, params = build_where_clause(filters)
    joiner = "AND" if where else "WHERE"
    branches = '\n'.join(
        f"                   WHEN b.incomplete_ride_reason = '{reason}' THEN '{party}'"
        for reason, party in REASON_PARTY.items()
    )
    sql = f"""
        SELECT b.incomplete_ride_reason AS incomplete_ride_reason,
               CASE
{branches}
                   ELSE 'Unknown'
               END      AS responsible_party,
               COUNT(*) AS rides,
               ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS share_pct
        {_BASE_JOIN}
        {where}
        {joiner} b.incomplete_ride_reason IS NOT NULL
        GROUP BY b.incomplete_ride_reason
        ORDER BY rides DESC
    """
    return _run(sql, params)


def q_cancellation_by_surge(filters: dict | None = None) -> pd.DataFrame:
    """Cancellation rate across surge bands."""
    where, params = build_where_clause(filters)
    sql = f"""
        SELECT CASE
                   WHEN b.surge_multiplier <= 1.0 THEN 'None (1.0)'
                   WHEN b.surge_multiplier <= 1.5 THEN 'Low (1.0-1.5)'
                   WHEN b.surge_multiplier <= 2.0 THEN 'Medium (1.5-2.0)'
                   ELSE 'High (>2.0)'
               END      AS surge_band,
               COUNT(*) AS rides,
               ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate,
               ROUND(AVG(b.booking_value), 2)                      AS avg_fare
        {_BASE_JOIN}
        {where}
        GROUP BY surge_band
        ORDER BY cancel_rate DESC
    """
    return _run(sql, params)


# --------------------------------------------------------------------------- #
# Fares
# --------------------------------------------------------------------------- #


def q_distance_vs_fare(
    filters: dict | None = None, sample: int = 4000
) -> pd.DataFrame:
    """Sampled distance-fare pairs for the scatter plot.

    Sampling keeps the payload small; the dashboard never pulls 100k rows to
    the browser.
    """
    where, params = build_where_clause(filters)
    sql = f"""
        SELECT b.ride_distance_km, b.booking_value,
               v.vehicle_name AS vehicle_type, b.surge_multiplier
        {_BASE_JOIN}
        {where}
        ORDER BY RAND()
        LIMIT %s
    """
    return _run(sql, params + [sample])


def q_fare_by_vehicle_city(filters: dict | None = None) -> pd.DataFrame:
    """Average fare and fare per kilometre by city and vehicle type."""
    where, params = build_where_clause(filters)
    sql = f"""
        SELECT c.city_name    AS city,
               v.vehicle_name AS vehicle_type,
               COUNT(*)       AS rides,
               ROUND(AVG(b.booking_value), 2)                        AS avg_fare,
               ROUND(AVG(b.booking_value / b.ride_distance_km), 2)   AS avg_fare_per_km,
               ROUND(SUM(b.booking_value), 2)                        AS revenue
        {_BASE_JOIN}
        {where}
        GROUP BY city, vehicle_type
        ORDER BY city, vehicle_type
    """
    return _run(sql, params)


def q_surge_by_hour(filters: dict | None = None) -> pd.DataFrame:
    """Average surge multiplier and fare per hour."""
    where, params = build_where_clause(filters)
    sql = f"""
        SELECT HOUR(b.booking_ts)              AS hour_of_day,
               ROUND(AVG(b.surge_multiplier), 3) AS avg_surge,
               ROUND(AVG(b.booking_value), 2)    AS avg_fare,
               COUNT(*)                          AS rides
        {_BASE_JOIN}
        {where}
        GROUP BY hour_of_day
        ORDER BY hour_of_day
    """
    return _run(sql, params)


def q_revenue_by_city_vehicle(filters: dict | None = None) -> pd.DataFrame:
    """Completed-ride revenue by city and vehicle type."""
    where, params = build_where_clause(filters)
    joiner = "AND" if where else "WHERE"
    sql = f"""
        SELECT c.city_name    AS city,
               v.vehicle_name AS vehicle_type,
               ROUND(SUM(b.booking_value), 2) AS revenue,
               COUNT(*)                       AS rides
        {_BASE_JOIN}
        {where}
        {joiner} b.booking_status = 'Completed'
        GROUP BY city, vehicle_type
        ORDER BY revenue DESC
    """
    return _run(sql, params)


def q_fare_by_conditions(filters: dict | None = None) -> pd.DataFrame:
    """Average fare and surge across traffic and weather combinations."""
    where, params = build_where_clause(filters)
    sql = f"""
        SELECT b.traffic_level, b.weather_condition,
               COUNT(*)                          AS rides,
               ROUND(AVG(b.booking_value), 2)    AS avg_fare,
               ROUND(AVG(b.surge_multiplier), 3) AS avg_surge,
               ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate
        {_BASE_JOIN}
        {where}
        GROUP BY b.traffic_level, b.weather_condition
        ORDER BY cancel_rate DESC
    """
    return _run(sql, params)


# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #


def q_top_pickup_locations(
    filters: dict | None = None, limit: int = 20
) -> pd.DataFrame:
    """Busiest pickup zones, city-qualified."""
    where, params = build_where_clause(filters)
    sql = f"""
        SELECT CONCAT(c.city_name, ' / ', l.location_code) AS zone,
               COUNT(*)                                    AS rides,
               ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate,
               ROUND(AVG(b.booking_value), 2)                      AS avg_fare
        {_BASE_JOIN}
        JOIN locations l ON l.location_id = b.pickup_location_id
        {where}
        GROUP BY zone
        ORDER BY rides DESC
        LIMIT %s
    """
    return _run(sql, params + [limit])


def q_busiest_routes(filters: dict | None = None, limit: int = 20) -> pd.DataFrame:
    """Busiest city-qualified pickup-to-drop routes."""
    where, params = build_where_clause(filters)
    sql = f"""
        SELECT CONCAT(c.city_name, ': ', pl.location_code,
                      ' -> ', dl.location_code)             AS route,
               COUNT(*)                                     AS rides,
               ROUND(AVG(b.ride_distance_km), 2)            AS avg_distance,
               ROUND(AVG(b.booking_value), 2)               AS avg_fare,
               ROUND(100 * AVG(b.booking_status = 'Cancelled'), 2) AS cancel_rate
        {_BASE_JOIN}
        JOIN locations pl ON pl.location_id = b.pickup_location_id
        JOIN locations dl ON dl.location_id = b.drop_location_id
        {where}
        GROUP BY route
        ORDER BY rides DESC
        LIMIT %s
    """
    return _run(sql, params + [limit])


def q_demand_level_distribution(filters: dict | None = None) -> pd.DataFrame:
    """Zone-demand aggregates by demand level and vehicle type."""
    sql = """
        SELECT ld.demand_level, v.vehicle_name AS vehicle_type,
               COUNT(*)                            AS zone_slots,
               ROUND(AVG(ld.avg_wait_time_min), 2) AS avg_wait_min,
               ROUND(AVG(ld.avg_surge_multiplier), 3) AS avg_surge,
               SUM(ld.total_requests)              AS total_requests
        FROM location_demand ld
        JOIN vehicle_types v ON v.vehicle_type_id = ld.vehicle_type_id
        GROUP BY ld.demand_level, vehicle_type
        ORDER BY ld.demand_level, vehicle_type
    """
    return _run(sql, [])


def q_wait_time_by_hour(filters: dict | None = None) -> pd.DataFrame:
    """Average zone wait time and surge by hour, from the demand table."""
    sql = """
        SELECT ld.hour_of_day,
               ROUND(AVG(ld.avg_wait_time_min), 2)    AS avg_wait_min,
               ROUND(AVG(ld.avg_surge_multiplier), 3) AS avg_surge,
               SUM(ld.total_requests)                 AS total_requests
        FROM location_demand ld
        GROUP BY ld.hour_of_day
        ORDER BY ld.hour_of_day
    """
    return _run(sql, [])


# --------------------------------------------------------------------------- #
# People
# --------------------------------------------------------------------------- #


def q_high_risk_customers(limit: int = 50, min_bookings: int = 5) -> pd.DataFrame:
    """Customers with the highest observed cancellation rate."""
    sql = """
        SELECT cu.customer_id,
               ci.city_name           AS city,
               cu.customer_age        AS age,
               cu.total_bookings,
               cu.cancelled_rides,
               ROUND(100 * cu.cancellation_rate, 2) AS cancel_rate,
               cu.avg_customer_rating AS rating
        FROM customers cu
        JOIN cities ci ON ci.city_id = cu.city_id
        WHERE cu.total_bookings >= %s
        ORDER BY cu.cancellation_rate DESC, cu.total_bookings DESC
        LIMIT %s
    """
    return _run(sql, [min_bookings, limit])


def q_unreliable_drivers(limit: int = 50, min_rides: int = 5) -> pd.DataFrame:
    """Drivers with the highest observed delay rate."""
    sql = """
        SELECT d.driver_id,
               ci.city_name AS city,
               v.vehicle_name AS vehicle_type,
               d.total_assigned_rides,
               d.delay_count,
               ROUND(100 * d.delay_rate, 2)      AS delay_rate,
               ROUND(100 * d.acceptance_rate, 2) AS acceptance_rate,
               d.avg_driver_rating               AS rating,
               d.avg_pickup_delay_min
        FROM drivers d
        JOIN cities ci       ON ci.city_id = d.city_id
        JOIN vehicle_types v ON v.vehicle_type_id = d.vehicle_type_id
        WHERE d.total_assigned_rides >= %s
        ORDER BY d.delay_rate DESC, d.total_assigned_rides DESC
        LIMIT %s
    """
    return _run(sql, [min_rides, limit])


def q_top_drivers(limit: int = 50, min_rides: int = 5) -> pd.DataFrame:
    """Best drivers by acceptance rate, punctuality and rating."""
    sql = """
        SELECT d.driver_id,
               ci.city_name   AS city,
               v.vehicle_name AS vehicle_type,
               d.total_assigned_rides,
               ROUND(100 * d.acceptance_rate, 2) AS acceptance_rate,
               ROUND(100 * d.delay_rate, 2)      AS delay_rate,
               d.avg_driver_rating               AS rating,
               ROUND(100 * (0.35 * d.acceptance_rate
                          + 0.35 * (1 - d.delay_rate)
                          + 0.30 * ((d.avg_driver_rating - 1) / 4)), 2)
                   AS reliability_score
        FROM drivers d
        JOIN cities ci       ON ci.city_id = d.city_id
        JOIN vehicle_types v ON v.vehicle_type_id = d.vehicle_type_id
        WHERE d.total_assigned_rides >= %s
        ORDER BY reliability_score DESC
        LIMIT %s
    """
    return _run(sql, [min_rides, limit])


def q_customer_demographics(filters: dict | None = None) -> pd.DataFrame:
    """Customer counts and behaviour by gender and age band."""
    sql = """
        SELECT cu.customer_gender AS gender,
               CASE
                   WHEN cu.customer_age < 25 THEN '18-24'
                   WHEN cu.customer_age < 35 THEN '25-34'
                   WHEN cu.customer_age < 45 THEN '35-44'
                   WHEN cu.customer_age < 60 THEN '45-59'
                   ELSE '60+'
               END AS age_band,
               COUNT(*)                                AS customers,
               ROUND(100 * AVG(cu.cancellation_rate), 2) AS avg_cancel_rate,
               ROUND(AVG(cu.avg_customer_rating), 2)     AS avg_rating
        FROM customers cu
        GROUP BY gender, age_band
        ORDER BY gender, age_band
    """
    return _run(sql, [])


def q_customer_vs_driver_ratings(filters: dict | None = None) -> pd.DataFrame:
    """Side-by-side rating distributions for customers and drivers."""
    sql = """
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
        ORDER BY party, rating
    """
    return _run(sql, [])


def q_driver_scatter(limit: int = 3000) -> pd.DataFrame:
    """Driver-level metrics for the reliability scatter plot."""
    sql = """
        SELECT d.driver_id,
               ROUND(100 * (0.35 * d.acceptance_rate
                          + 0.35 * (1 - d.delay_rate)
                          + 0.30 * ((d.avg_driver_rating - 1) / 4)), 2)
                   AS driver_reliability_score,
               d.avg_pickup_delay_min,
               d.avg_driver_rating,
               d.total_assigned_rides
        FROM drivers d
        LIMIT %s
    """
    return _run(sql, [limit])


def q_bookings_page(
    filters: dict | None = None, page: int = 1, page_size: int = 50
) -> pd.DataFrame:
    """One page of booking records, for the paginated data explorer."""
    where, params = build_where_clause(filters)
    offset = max(0, (page - 1) * page_size)
    sql = f"""
        SELECT b.booking_id, b.booking_ts, c.city_name AS city,
               v.vehicle_name AS vehicle_type,
               pl.location_code AS pickup, dl.location_code AS drop_zone,
               b.ride_distance_km, b.traffic_level, b.weather_condition,
               b.surge_multiplier, b.booking_value, b.booking_status
        {_BASE_JOIN}
        JOIN locations pl ON pl.location_id = b.pickup_location_id
        JOIN locations dl ON dl.location_id = b.drop_location_id
        {where}
        ORDER BY b.booking_ts DESC
        LIMIT %s OFFSET %s
    """
    return _run(sql, params + [page_size, offset])


def q_filter_options() -> dict:
    """Fetch distinct filter values so the sidebar never hard-codes them."""
    cities = _run("SELECT city_name FROM cities ORDER BY city_name", [])
    vehicles = _run("SELECT vehicle_name FROM vehicle_types ORDER BY vehicle_name", [])
    date_range = _run(
        "SELECT MIN(booking_ts) AS min_ts, MAX(booking_ts) AS max_ts FROM bookings", []
    )
    return {
        "cities": cities["city_name"].tolist() if not cities.empty else [],
        "vehicle_types": vehicles["vehicle_name"].tolist()
        if not vehicles.empty
        else [],
        "date_min": date_range["min_ts"].iloc[0] if not date_range.empty else None,
        "date_max": date_range["max_ts"].iloc[0] if not date_range.empty else None,
    }

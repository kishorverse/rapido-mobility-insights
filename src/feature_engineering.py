"""Feature engineering, SQL analysis queries and significance testing.

Three things that all sit between cleaned data and a result:

1. **Features** - the master analytical table used by the notebooks, the models
   and the dashboard. Two families live here: *context* features derived from
   the booking row itself (time, distance, route, conditions), which are safe
   everywhere; and *history* features describing the customer or driver, built
   as an expanding window over strictly earlier bookings so a booking never
   contributes to its own predictors. The static rate columns shipped in
   ``customers.csv`` / ``drivers.csv`` are whole-period aggregates that already
   include the row being predicted, which is why they are replaced rather than
   used directly.
2. **Queries** - the named, parameterised SQL backing every dashboard chart.
   Filters go through :func:`build_where_clause`, so no user input is ever
   concatenated into SQL. ``sql/analysis_queries.sql`` holds the same queries in
   standalone runnable form.
3. **Statistics** - the significance tests backing the EDA claims. Each returns
   a plain dict so the result can be rendered or dropped into a report without
   further shaping.

Usage:
    python src/feature_engineering.py build      # rebuild data/processed/model_data.csv
    python src/feature_engineering.py tests      # run the significance tests
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_preprocessing as dp

logger = logging.getLogger(__name__)


# =========================================================================== #
# 1. Feature engineering
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Time features
# --------------------------------------------------------------------------- #


def add_time_parts(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar parts from ``booking_ts``."""
    frame = frame.copy()
    timestamp = frame["booking_ts"]
    frame["hour_of_day"] = timestamp.dt.hour
    frame["day_of_week"] = timestamp.dt.day_name()
    frame["day_of_month"] = timestamp.dt.day
    frame["month"] = timestamp.dt.month
    frame["week_of_year"] = timestamp.dt.isocalendar().week.astype(int)
    frame["is_weekend"] = timestamp.dt.dayofweek.isin([5, 6]).astype(int)
    return frame


def add_rush_hour_flag(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag bookings made during morning or evening rush hours."""
    frame = frame.copy()
    frame["rush_hour_flag"] = frame["hour_of_day"].isin(dp.RUSH_HOURS).astype(int)
    return frame


def add_night_ride_flag(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag late-night and early-morning bookings."""
    frame = frame.copy()
    frame["is_night_ride"] = frame["hour_of_day"].isin(dp.NIGHT_HOURS).astype(int)
    return frame


def add_time_of_day_band(frame: pd.DataFrame) -> pd.DataFrame:
    """Bucket the hour into a coarse, human-readable band."""
    frame = frame.copy()
    frame["time_of_day_band"] = pd.cut(
        frame["hour_of_day"],
        bins=[-1, 5, 11, 16, 20, 23],
        labels=["Night", "Morning", "Afternoon", "Evening", "Late Evening"],
    )
    return frame


# --------------------------------------------------------------------------- #
# Trip context features
# --------------------------------------------------------------------------- #


def add_long_distance_flag(
    frame: pd.DataFrame, threshold: float | None = None
) -> pd.DataFrame:
    """Flag rides longer than ``threshold`` kilometres."""
    frame = frame.copy()
    limit = dp.LONG_DISTANCE_KM if threshold is None else threshold
    frame["long_distance_flag"] = (frame["ride_distance_km"] > limit).astype(int)
    return frame


def add_distance_band(frame: pd.DataFrame) -> pd.DataFrame:
    """Bucket ride distance into short, medium and long bands."""
    frame = frame.copy()
    frame["distance_band"] = pd.cut(
        frame["ride_distance_km"],
        bins=[0, 5, 15, np.inf],
        labels=["Short", "Medium", "Long"],
    )
    return frame


def add_city_route_pair(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the route identifier.

    The spec asks for ``Pickup City + Drop City``, but every booking is
    intra-city and ``Loc_1..Loc_50`` repeat in all five cities. The meaningful
    equivalent is the city-qualified pickup-to-drop route.
    """
    frame = frame.copy()
    city = frame["city"].astype(str)
    pickup = frame["pickup_location"].astype(str)
    drop = frame["drop_location"].astype(str)
    frame["city_route_pair"] = city + ": " + pickup + " -> " + drop
    frame["location_pair"] = pickup + " -> " + drop
    return frame


def add_same_zone_flag(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag rides that start and end in the same zone."""
    frame = frame.copy()
    frame["is_same_zone"] = (
        frame["pickup_location"].astype(str) == frame["drop_location"].astype(str)
    ).astype(int)
    return frame


def add_expected_speed(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute the implied average speed from distance and estimated time."""
    frame = frame.copy()
    frame["expected_speed_kmph"] = frame["ride_distance_km"] / (
        frame["estimated_ride_time_min"].replace(0, np.nan) / 60
    )
    return frame


def add_surge_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    """Bucket the surge multiplier into interpretable bands."""
    frame = frame.copy()
    frame["surge_bucket"] = pd.cut(
        frame["surge_multiplier"],
        bins=[0, 1.0, 1.5, 2.0, np.inf],
        labels=["None", "Low", "Medium", "High"],
        include_lowest=True,
    )
    return frame


def add_adverse_conditions_flag(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag bookings facing both heavy rain and high traffic."""
    frame = frame.copy()
    frame["adverse_conditions_flag"] = (
        (frame["weather_condition"].astype(str) == "Heavy Rain")
        & (frame["traffic_level"].astype(str) == "High")
    ).astype(int)
    frame["bad_weather_flag"] = (
        frame["weather_condition"].astype(str).isin(["Rain", "Heavy Rain"]).astype(int)
    )
    frame["high_traffic_flag"] = (
        frame["traffic_level"].astype(str) == "High"
    ).astype(int)
    return frame


def add_fare_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute ``fare_per_km`` and ``fare_per_min``.

    Warning:
        Both derive from ``booking_value``. They are dashboard and EDA
        features only and are blocked from the fare model's inputs by
        ``dp.FARE_DERIVED_COLUMNS``.
    """
    frame = frame.copy()
    frame["fare_per_km"] = frame["booking_value"] / frame["ride_distance_km"].replace(
        0, np.nan
    )
    frame["fare_per_min"] = frame["booking_value"] / frame[
        "estimated_ride_time_min"
    ].replace(0, np.nan)
    return frame


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #


def compute_driver_reliability_score(drivers: pd.DataFrame) -> pd.Series:
    """Score driver reliability from 0 to 100.

    Blends acceptance rate (35%), on-time behaviour (35%) and rating (30%).
    Higher is better.
    """
    acceptance = drivers["acceptance_rate"].clip(0, 1)
    punctuality = 1 - drivers["delay_rate"].clip(0, 1)
    rating = ((drivers["avg_driver_rating"] - 1) / 4).clip(0, 1)
    score = 100 * (0.35 * acceptance + 0.35 * punctuality + 0.30 * rating)
    return score.round(2).rename("driver_reliability_score")


def compute_customer_loyalty_score(customers: pd.DataFrame) -> pd.Series:
    """Score customer loyalty from 0 to 100.

    Blends booking volume (40%, log-scaled), completion rate (40%) and rating
    (20%). Higher is better.
    """
    volume = np.log1p(customers["total_bookings"])
    volume = (volume / volume.max()).clip(0, 1)
    completion = (
        customers["completed_rides"] / customers["total_bookings"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)
    rating = ((customers["avg_customer_rating"] - 1) / 4).clip(0, 1)
    score = 100 * (0.40 * volume + 0.40 * completion + 0.20 * rating)
    return score.round(2).rename("customer_loyalty_score")


def add_customer_tenure_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    """Bucket customers by how long ago they signed up."""
    frame = frame.copy()
    frame["customer_tenure_bucket"] = pd.cut(
        frame["customer_signup_days_ago"],
        bins=[-1, 90, 365, 730, np.inf],
        labels=["New", "Growing", "Established", "Veteran"],
    )
    return frame


# --------------------------------------------------------------------------- #
# Prior-history features (temporal, leakage-safe)
# --------------------------------------------------------------------------- #


def _prior_history(frame: pd.DataFrame, entity: str, prefix: str) -> pd.DataFrame:
    """Build expanding prior-history features for one entity column.

    For each booking, counts only that entity's *strictly earlier* bookings.
    The first booking of an entity gets zero priors and a NaN rate, which the
    model pipeline imputes -- an honest "no history yet" signal.

    Args:
        frame: Bookings sorted arbitrarily; sorted internally by timestamp.
        entity: ``"customer_id"`` or ``"driver_id"``.
        prefix: Column-name prefix, e.g. ``"cust"`` or ``"drv"``.
    """
    working = frame[[entity, "booking_ts", "booking_status"]].copy()
    working["_order"] = np.arange(len(working))
    working = working.sort_values([entity, "booking_ts", "_order"])

    status = working["booking_status"].astype(str)
    is_cancelled = (status == "Cancelled").astype(int)
    is_incomplete = (status == "Incomplete").astype(int)
    is_completed = (status == "Completed").astype(int)

    grouped = working.groupby(entity, observed=True)
    prior_rides = grouped.cumcount()

    def _prior_sum(series: pd.Series) -> pd.Series:
        """Cumulative sum shifted by one, so the current row is excluded."""
        return series.groupby(working[entity], observed=True).cumsum().sub(series)

    prior_cancelled = _prior_sum(is_cancelled)
    prior_incomplete = _prior_sum(is_incomplete)
    prior_completed = _prior_sum(is_completed)

    denominator = prior_rides.replace(0, np.nan)
    result = pd.DataFrame(
        {
            f"{prefix}_prior_rides": prior_rides,
            f"{prefix}_prior_cancelled": prior_cancelled,
            f"{prefix}_prior_incomplete": prior_incomplete,
            f"{prefix}_prior_cancel_rate": prior_cancelled / denominator,
            f"{prefix}_prior_incomplete_rate": prior_incomplete / denominator,
            f"{prefix}_prior_completion_rate": prior_completed / denominator,
            f"{prefix}_is_first_ride": (prior_rides == 0).astype(int),
        },
        index=working.index,
    )
    result["_order"] = working["_order"].to_numpy()
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def add_prior_customer_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach leakage-safe prior-booking history for each customer."""
    history = _prior_history(frame, "customer_id", "cust")
    return pd.concat([frame.reset_index(drop=True), history], axis=1)


def add_prior_driver_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach leakage-safe prior-booking history for each driver."""
    history = _prior_history(frame, "driver_id", "drv")
    return pd.concat([frame.reset_index(drop=True), history], axis=1)


# --------------------------------------------------------------------------- #
# Merges
# --------------------------------------------------------------------------- #


def merge_customer_features(
    bookings: pd.DataFrame, customers: pd.DataFrame
) -> pd.DataFrame:
    """Join customer profile attributes onto bookings.

    Excludes the whole-period outcome aggregates (``cancellation_rate``,
    ``cancelled_rides`` and friends), which are computed over the same rows
    being predicted. The prior-history columns replace them.
    """
    profile_columns = [
        "customer_id",
        "customer_gender",
        "customer_age",
        "customer_signup_days_ago",
        "preferred_vehicle_type",
        "avg_customer_rating",
        "customer_loyalty_score",
    ]
    enriched = customers.copy()
    enriched["customer_loyalty_score"] = compute_customer_loyalty_score(enriched)
    return bookings.merge(
        enriched[profile_columns], on="customer_id", how="left", validate="m:1"
    )


def merge_driver_features(
    bookings: pd.DataFrame, drivers: pd.DataFrame
) -> pd.DataFrame:
    """Join driver profile attributes onto bookings.

    Excludes ``delay_rate`` and ``incomplete_rides``, which are whole-period
    aggregates of the target.
    """
    profile_columns = [
        "driver_id",
        "driver_age",
        "driver_experience_years",
        "acceptance_rate",
        "avg_driver_rating",
        "avg_pickup_delay_min",
        "driver_reliability_score",
    ]
    enriched = drivers.copy()
    enriched["driver_reliability_score"] = compute_driver_reliability_score(enriched)
    return bookings.merge(
        enriched[profile_columns], on="driver_id", how="left", validate="m:1"
    )


def merge_time_features(
    bookings: pd.DataFrame, time_features: pd.DataFrame
) -> pd.DataFrame:
    """Join the hourly calendar dimension onto bookings.

    ``is_holiday`` is dropped: it is 0 for every hour of 2025.
    """
    calendar = time_features[["datetime", "peak_time_flag", "season"]].copy()
    frame = bookings.copy()
    frame["_hour_slot"] = frame["booking_ts"].dt.floor("h")
    merged = frame.merge(
        calendar, left_on="_hour_slot", right_on="datetime", how="left", validate="m:1"
    )
    return merged.drop(columns=["_hour_slot", "datetime"])


def merge_demand_features(
    bookings: pd.DataFrame, location_demand: pd.DataFrame
) -> pd.DataFrame:
    """Join pickup-zone demand for the matching city, hour and vehicle type."""
    demand = location_demand[
        [
            "city",
            "pickup_location",
            "hour_of_day",
            "vehicle_type",
            "total_requests",
            "avg_wait_time_min",
            "avg_surge_multiplier",
            "demand_level",
        ]
    ].copy()
    demand = demand.rename(
        columns={
            "total_requests": "zone_total_requests",
            "avg_wait_time_min": "zone_avg_wait_min",
            "avg_surge_multiplier": "zone_avg_surge",
            "demand_level": "zone_demand_level",
        }
    )
    for column in ("city", "pickup_location", "vehicle_type"):
        demand[column] = demand[column].astype(str)

    frame = bookings.copy()
    keys = ["city", "pickup_location", "hour_of_day", "vehicle_type"]
    for column in ("city", "pickup_location", "vehicle_type"):
        frame[column] = frame[column].astype(str)

    return frame.merge(demand, on=keys, how="left", validate="m:1")


def add_demand_supply_ratio(frame: pd.DataFrame) -> pd.DataFrame:
    """Express this booking's surge relative to its zone's typical surge."""
    frame = frame.copy()
    frame["surge_vs_zone_ratio"] = frame["surge_multiplier"] / frame[
        "zone_avg_surge"
    ].replace(0, np.nan)
    frame["zone_demand_pressure"] = frame["zone_total_requests"] * frame[
        "zone_avg_wait_min"
    ]
    return frame


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def build_feature_table(cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the master analytical table used by EDA, models and dashboard.

    Args:
        cleaned: Output of :func:`data_preprocessing.clean_all`.

    Returns:
        One row per booking with context, profile, demand and prior-history
        features attached.
    """
    frame = cleaned["bookings"].copy()

    frame = add_time_parts(frame)
    frame = add_rush_hour_flag(frame)
    frame = add_night_ride_flag(frame)
    frame = add_time_of_day_band(frame)
    frame = add_long_distance_flag(frame)
    frame = add_distance_band(frame)
    frame = add_city_route_pair(frame)
    frame = add_same_zone_flag(frame)
    frame = add_expected_speed(frame)
    frame = add_surge_bucket(frame)
    frame = add_adverse_conditions_flag(frame)
    frame = add_fare_ratios(frame)

    frame = merge_customer_features(frame, cleaned["customers"])
    frame = merge_driver_features(frame, cleaned["drivers"])
    frame = merge_time_features(frame, cleaned["time_features"])
    frame = merge_demand_features(frame, cleaned["location_demand"])

    frame = add_customer_tenure_bucket(frame)
    frame = add_demand_supply_ratio(frame)

    frame = add_prior_customer_history(frame)
    frame = add_prior_driver_history(frame)

    logger.info("Feature table built: %d rows x %d columns", *frame.shape)
    return frame


def build_and_cache(rebuild: bool = False) -> pd.DataFrame:
    """Return the model-ready feature table, building and caching it if needed.

    Args:
        rebuild: Ignore any existing cache and rebuild from the raw CSVs.
    """
    if not rebuild and dp.model_data_exists():
        logger.info("Loading cached model data")
        return dp.load_model_data()

    logger.info("Building feature table from raw data")
    frame = build_feature_table(dp.load_and_clean())
    dp.save_model_data(frame)
    return frame


# =========================================================================== #
# 2. SQL analysis queries
# =========================================================================== #

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
        return dp.read_sql(sql, tuple(params))
    except dp.DatabaseError as exc:
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
    branches = "\n".join(
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


def q_distance_vs_fare(filters: dict | None = None, sample: int = 4000) -> pd.DataFrame:
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


# =========================================================================== #
# 3. Statistical significance tests
# =========================================================================== #

ALPHA = 0.05


def _interpret(p_value: float, alpha: float = ALPHA) -> str:
    """Return a plain-language verdict for a p-value."""
    return (
        "significant (reject H0)" if p_value < alpha else "not significant (retain H0)"
    )


def cramers_v(frame: pd.DataFrame, col_a: str, col_b: str) -> float:
    """Cramer's V effect size for two categorical columns.

    Chi-square p-values collapse to zero on 100k rows, so effect size is what
    actually distinguishes a real driver from a trivial one.
    """
    table = pd.crosstab(frame[col_a], frame[col_b])
    chi2 = scipy_stats.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    min_dim = min(table.shape) - 1
    if n == 0 or min_dim == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def chi_square_independence(
    frame: pd.DataFrame, col_a: str, col_b: str, alpha: float = ALPHA
) -> dict:
    """Test whether two categorical variables are independent.

    Chosen because both variables are categorical and the question is one of
    association, not of mean difference.
    """
    table = pd.crosstab(frame[col_a], frame[col_b])
    chi2, p_value, dof, _expected = scipy_stats.chi2_contingency(table)
    return {
        "test": "Chi-square test of independence",
        "variables": f"{col_a} vs {col_b}",
        "hypothesis": f"H0: {col_a} and {col_b} are independent",
        "statistic": round(float(chi2), 4),
        "p_value": float(p_value),
        "dof": int(dof),
        "effect_size_cramers_v": round(cramers_v(frame, col_a, col_b), 4),
        "alpha": alpha,
        "conclusion": _interpret(p_value, alpha),
    }


def anova_by_group(
    frame: pd.DataFrame, value_col: str, group_col: str, alpha: float = ALPHA
) -> dict:
    """One-way ANOVA of a numeric column across three or more groups.

    Chosen over repeated t-tests because it compares all group means in a
    single test and avoids inflating the family-wise error rate.
    """
    groups = [
        group[value_col].dropna().to_numpy()
        for _, group in frame.groupby(group_col, observed=True)
        if len(group) > 1
    ]
    if len(groups) < 2:
        raise ValueError(f"Need at least two groups in {group_col!r}.")

    f_stat, p_value = scipy_stats.f_oneway(*groups)
    means = frame.groupby(group_col, observed=True)[value_col].mean().round(2)
    return {
        "test": "One-way ANOVA",
        "variables": f"{value_col} across {group_col}",
        "hypothesis": f"H0: mean {value_col} is equal across all {group_col} groups",
        "statistic": round(float(f_stat), 4),
        "p_value": float(p_value),
        "groups": int(len(groups)),
        "group_means": means.to_dict(),
        "alpha": alpha,
        "conclusion": _interpret(p_value, alpha),
    }


def ttest_two_groups(
    frame: pd.DataFrame, value_col: str, group_col: str, alpha: float = ALPHA
) -> dict:
    """Welch's t-test between exactly two groups.

    Welch's variant is used because it does not assume equal variances.
    """
    levels = frame[group_col].dropna().unique()
    if len(levels) != 2:
        raise ValueError(
            f"{group_col!r} must have exactly two levels, found {len(levels)}."
        )

    first = frame.loc[frame[group_col] == levels[0], value_col].dropna()
    second = frame.loc[frame[group_col] == levels[1], value_col].dropna()
    t_stat, p_value = scipy_stats.ttest_ind(first, second, equal_var=False)

    pooled_sd = np.sqrt((first.var() + second.var()) / 2)
    cohens_d = (first.mean() - second.mean()) / pooled_sd if pooled_sd else 0.0

    return {
        "test": "Welch's two-sample t-test",
        "variables": f"{value_col} by {group_col}",
        "hypothesis": f"H0: mean {value_col} is equal for {levels[0]} and {levels[1]}",
        "statistic": round(float(t_stat), 4),
        "p_value": float(p_value),
        "group_means": {
            str(levels[0]): round(float(first.mean()), 2),
            str(levels[1]): round(float(second.mean()), 2),
        },
        "effect_size_cohens_d": round(float(cohens_d), 4),
        "alpha": alpha,
        "conclusion": _interpret(p_value, alpha),
    }


def correlation_matrix(
    frame: pd.DataFrame, columns: list[str], method: str = "pearson"
) -> pd.DataFrame:
    """Correlation matrix for the given numeric columns."""
    available = [column for column in columns if column in frame.columns]
    return frame[available].corr(method=method).round(3)


def correlation_with_target(
    frame: pd.DataFrame, columns: list[str], target: str
) -> pd.DataFrame:
    """Rank numeric columns by absolute correlation with a numeric target."""
    available = [
        column for column in columns if column in frame.columns and column != target
    ]
    rows = []
    for column in available:
        subset = frame[[column, target]].dropna()
        if len(subset) < 2:
            continue
        pearson = subset[column].corr(subset[target])
        spearman = subset[column].corr(subset[target], method="spearman")
        rows.append(
            {
                "feature": column,
                "pearson": round(float(pearson), 4),
                "spearman": round(float(spearman), 4),
                "abs_pearson": round(abs(float(pearson)), 4),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("abs_pearson", ascending=False)
        .reset_index(drop=True)
    )


def cancellation_rate_by(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Cancellation and incompletion rate per level of a categorical column."""
    status = frame["booking_status"].astype(str)
    grouped = (
        frame.assign(
            _cancelled=(status == "Cancelled").astype(int),
            _incomplete=(status == "Incomplete").astype(int),
        )
        .groupby(group_col, observed=True)
        .agg(
            rides=("booking_id", "count"),
            cancel_rate=("_cancelled", "mean"),
            incomplete_rate=("_incomplete", "mean"),
        )
        .reset_index()
    )
    grouped["cancel_rate"] = (100 * grouped["cancel_rate"]).round(2)
    grouped["incomplete_rate"] = (100 * grouped["incomplete_rate"]).round(2)
    return grouped.sort_values("cancel_rate", ascending=False)


def run_standard_tests(frame: pd.DataFrame) -> list[dict]:
    """Run the project's headline significance tests.

    Returns:
        One dict per test, ready for :func:`summarise_tests`.
    """
    results = []

    for column in ("traffic_level", "weather_condition", "vehicle_type", "city"):
        if column in frame.columns:
            results.append(chi_square_independence(frame, column, "booking_status"))

    if {"booking_value", "vehicle_type"} <= set(frame.columns):
        results.append(anova_by_group(frame, "booking_value", "vehicle_type"))

    if {"ride_distance_km", "traffic_level"} <= set(frame.columns):
        results.append(anova_by_group(frame, "surge_multiplier", "traffic_level"))

    if {"booking_value", "is_weekend"} <= set(frame.columns):
        weekend = frame.assign(
            weekend_label=np.where(frame["is_weekend"] == 1, "Weekend", "Weekday")
        )
        results.append(ttest_two_groups(weekend, "booking_value", "weekend_label"))

    return results


def summarise_tests(results: list[dict]) -> pd.DataFrame:
    """Flatten a list of test results into a comparison table."""
    rows = []
    for result in results:
        rows.append(
            {
                "test": result["test"],
                "variables": result["variables"],
                "statistic": result["statistic"],
                "p_value": (
                    "< 0.0001"
                    if result["p_value"] < 1e-4
                    else round(result["p_value"], 4)
                ),
                "effect_size": result.get(
                    "effect_size_cramers_v", result.get("effect_size_cohens_d", "-")
                ),
                "conclusion": result["conclusion"],
            }
        )
    return pd.DataFrame(rows)


# =========================================================================== #
# 4. Command line
# =========================================================================== #


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Rapido feature engineering and significance testing."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="build data/processed/model_data.csv"
    )
    build_parser.add_argument(
        "--rebuild", action="store_true", help="ignore the cache and rebuild"
    )

    subparsers.add_parser("tests", help="run the headline significance tests")

    args = parser.parse_args(argv)

    if args.command == "build":
        frame = build_and_cache(rebuild=args.rebuild)
        print(f"Feature table: {frame.shape[0]:,} rows x {frame.shape[1]} columns")
        print(f"Written to {dp.MODEL_DATA_FILE}")
        return 0

    frame = build_and_cache()
    print(summarise_tests(run_standard_tests(frame)).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

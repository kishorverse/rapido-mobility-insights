"""Feature engineering for the Rapido models and dashboard.

Two families of features live here:

* **Context features** derived from the booking row itself (time, distance,
  route, conditions). Safe everywhere.
* **History features** describing the customer or driver. These are built as
  *prior* history -- an expanding window over strictly earlier bookings -- so a
  booking never contributes to its own predictors. The static rate columns
  shipped in ``customers.csv`` / ``drivers.csv`` are whole-period aggregates
  that already include the row being predicted, which is why they are replaced
  rather than used directly. See ``docs/PROJECT_PLAN.md`` section 1.2.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


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
    frame["rush_hour_flag"] = frame["hour_of_day"].isin(config.RUSH_HOURS).astype(int)
    return frame


def add_night_ride_flag(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag late-night and early-morning bookings."""
    frame = frame.copy()
    frame["is_night_ride"] = frame["hour_of_day"].isin(config.NIGHT_HOURS).astype(int)
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
    limit = config.LONG_DISTANCE_KM if threshold is None else threshold
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
        ``config.FARE_DERIVED_COLUMNS``.
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


def _prior_history(
    frame: pd.DataFrame, entity: str, prefix: str
) -> pd.DataFrame:
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
        return (
            series.groupby(working[entity], observed=True)
            .cumsum()
            .sub(series)
        )

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
        cleaned: Output of :func:`rapido.cleaning.clean_all`.

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

    logger.info(
        "Feature table built: %d rows x %d columns", *frame.shape
    )
    return frame

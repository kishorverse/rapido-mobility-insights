"""Cleaning and validation for the five Rapido source files.

Design rule: cleaning never invents data. Structural nulls (the post-outcome
columns) are preserved as ``NaN`` rather than imputed, because imputing them
would fabricate the very signal the models are supposed to predict. See
``docs/PROJECT_PLAN.md`` section 1.1.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

import config

logger = logging.getLogger(__name__)

#: Plausible ranges used by :func:`validate_value_ranges`.
BOOKING_RANGES = {
    "ride_distance_km": (0.1, 100.0),
    "estimated_ride_time_min": (1.0, 300.0),
    "actual_ride_time_min": (1.0, 300.0),
    "base_fare": (10.0, 5000.0),
    "surge_multiplier": (1.0, 3.0),
    "booking_value": (10.0, 10000.0),
    "hour_of_day": (0, 23),
}

CUSTOMER_RANGES = {
    "customer_age": (16, 100),
    "cancellation_rate": (0.0, 1.0),
    "avg_customer_rating": (1.0, 5.0),
    "customer_signup_days_ago": (0, 5000),
}

DRIVER_RANGES = {
    "driver_age": (18, 80),
    "acceptance_rate": (0.0, 1.0),
    "delay_rate": (0.0, 1.0),
    "avg_driver_rating": (1.0, 5.0),
    "avg_pickup_delay_min": (0.0, 120.0),
    "driver_experience_years": (0, 40),
}


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #


def standardise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to lower snake_case and strip whitespace."""
    frame = frame.copy()
    frame.columns = [
        re.sub(r"[^0-9a-zA-Z]+", "_", str(column)).strip("_").lower()
        for column in frame.columns
    ]
    return frame


def strip_string_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim leading and trailing whitespace from object/string columns."""
    frame = frame.copy()
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        frame[column] = frame[column].str.strip()
    return frame


def coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce the given columns to numeric, turning unparseable values to NaN."""
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def coerce_categorical(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cast the given columns to pandas ``category`` dtype."""
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("category")
    return frame


def drop_duplicates_by_key(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """Drop duplicate rows on ``key``, keeping the first occurrence."""
    before = len(frame)
    frame = frame.drop_duplicates(subset=[key], keep="first")
    removed = before - len(frame)
    if removed:
        logger.warning("Dropped %d duplicate rows on %s", removed, key)
    return frame


def detect_outliers_iqr(
    frame: pd.DataFrame, column: str, factor: float = 1.5
) -> pd.Series:
    """Return a boolean mask of IQR outliers for ``column``.

    Args:
        frame: Source frame.
        column: Numeric column to inspect.
        factor: IQR multiplier; 1.5 is the conventional fence.
    """
    series = frame[column]
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return (series < lower) | (series > upper)


def summarise_outliers(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Count IQR outliers per column without modifying the data."""
    rows = []
    for column in columns:
        if column not in frame.columns:
            continue
        mask = detect_outliers_iqr(frame, column)
        rows.append(
            {
                "column": column,
                "outliers": int(mask.sum()),
                "outlier_pct": round(100 * float(mask.mean()), 3),
                "min": frame[column].min(),
                "max": frame[column].max(),
            }
        )
    return pd.DataFrame(rows)


def cap_outliers(
    frame: pd.DataFrame, columns: list[str], factor: float = 1.5
) -> pd.DataFrame:
    """Winsorise the given columns at their IQR fences.

    Used only where an extreme value would distort a chart axis or a linear
    model. Tree models are unaffected, so this stays opt-in.
    """
    frame = frame.copy()
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        frame[column] = series.clip(q1 - factor * iqr, q3 + factor * iqr)
    return frame


def validate_value_ranges(frame: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Report values falling outside the expected range for each column."""
    rows = []
    for column, (low, high) in rules.items():
        if column not in frame.columns:
            continue
        series = frame[column]
        violations = int(((series < low) | (series > high)).sum())
        rows.append(
            {
                "column": column,
                "expected_min": low,
                "expected_max": high,
                "actual_min": series.min(),
                "actual_max": series.max(),
                "violations": violations,
                "status": "PASS" if violations == 0 else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def validate_referential_integrity(
    bookings: pd.DataFrame,
    customers: pd.DataFrame,
    drivers: pd.DataFrame,
) -> dict:
    """Return orphan foreign-key counts for the booking fact table."""
    return {
        "orphan_customers": int(
            (~bookings["customer_id"].isin(customers["customer_id"])).sum()
        ),
        "orphan_drivers": int(
            (~bookings["driver_id"].isin(drivers["driver_id"])).sum()
        ),
    }


# --------------------------------------------------------------------------- #
# Booking-specific cleaning
# --------------------------------------------------------------------------- #


def parse_booking_datetime(frame: pd.DataFrame) -> pd.DataFrame:
    """Combine ``booking_date`` and ``booking_time`` into a ``booking_ts``.

    The original date and time columns are dropped; every downstream time
    feature derives from the single timestamp.
    """
    frame = frame.copy()
    frame["booking_ts"] = pd.to_datetime(
        frame["booking_date"].astype(str) + " " + frame["booking_time"].astype(str),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    unparsed = int(frame["booking_ts"].isna().sum())
    if unparsed:
        logger.warning("%d booking timestamps could not be parsed", unparsed)

    return frame.drop(columns=["booking_date", "booking_time"])


def namespace_locations(frame: pd.DataFrame) -> pd.DataFrame:
    """Prefix location codes with their city.

    ``Loc_1`` through ``Loc_50`` repeat identically in all five cities, so the
    bare code is not a unique key. Without this, the SQL ``locations``
    dimension would merge Bangalore's Loc_1 with Delhi's.
    """
    frame = frame.copy()
    city = frame["city"].astype(str)
    for column in ("pickup_location", "drop_location"):
        frame[f"{column}_key"] = city + "::" + frame[column].astype(str)
    return frame


def handle_missing_bookings(frame: pd.DataFrame) -> pd.DataFrame:
    """Handle booking nulls, deliberately preserving the structural ones.

    ``actual_ride_time_min`` and ``incomplete_ride_reason`` are null by design
    for unresolved rides. They stay ``NaN`` (the reason becomes the explicit
    label ``"Not Applicable"`` only for display) and never reach a model.
    """
    frame = frame.copy()

    structural = set(config.POST_OUTCOME_COLUMNS)
    for column in frame.columns:
        if column in structural or not frame[column].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            median = frame[column].median()
            frame[column] = frame[column].fillna(median)
            logger.info("Filled %s nulls with median %.3f", column, median)
        else:
            mode = frame[column].mode()
            if len(mode):
                frame[column] = frame[column].fillna(mode.iloc[0])
                logger.info("Filled %s nulls with mode %r", column, mode.iloc[0])

    return frame


def clean_bookings(frame: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline for the booking fact table."""
    frame = standardise_columns(frame)
    frame = strip_string_columns(frame)
    frame = drop_duplicates_by_key(frame, "booking_id")
    frame = parse_booking_datetime(frame)
    frame = coerce_numeric(
        frame,
        [
            "ride_distance_km",
            "estimated_ride_time_min",
            "actual_ride_time_min",
            "base_fare",
            "surge_multiplier",
            "booking_value",
            "hour_of_day",
            "is_weekend",
        ],
    )
    frame = namespace_locations(frame)
    frame = handle_missing_bookings(frame)
    frame = coerce_categorical(
        frame,
        [
            "city",
            "vehicle_type",
            "traffic_level",
            "weather_condition",
            "booking_status",
            "day_of_week",
            "incomplete_ride_reason",
        ],
    )
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Dimension cleaning
# --------------------------------------------------------------------------- #


def handle_missing_customers(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill customer nulls: median for numerics, mode for categoricals."""
    return _fill_generic(frame, "customers")


def handle_missing_drivers(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill driver nulls: median for numerics, mode for categoricals."""
    return _fill_generic(frame, "drivers")


def _fill_generic(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Shared null-filling for dimension tables with no structural nulls."""
    frame = frame.copy()
    for column in frame.columns:
        if not frame[column].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].fillna(frame[column].median())
        else:
            mode = frame[column].mode()
            if len(mode):
                frame[column] = frame[column].fillna(mode.iloc[0])
        logger.info("Filled nulls in %s.%s", label, column)
    return frame


def clean_customers(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean the customer dimension."""
    frame = standardise_columns(frame)
    frame = strip_string_columns(frame)
    frame = drop_duplicates_by_key(frame, "customer_id")
    frame = coerce_numeric(
        frame,
        [
            "customer_age",
            "customer_signup_days_ago",
            "total_bookings",
            "completed_rides",
            "cancelled_rides",
            "incomplete_rides",
            "cancellation_rate",
            "avg_customer_rating",
            "customer_cancel_flag",
        ],
    )
    frame = handle_missing_customers(frame)
    frame = coerce_categorical(
        frame, ["customer_gender", "customer_city", "preferred_vehicle_type"]
    )
    return frame.reset_index(drop=True)


def clean_drivers(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean the driver dimension."""
    frame = standardise_columns(frame)
    frame = strip_string_columns(frame)
    frame = drop_duplicates_by_key(frame, "driver_id")
    frame = coerce_numeric(
        frame,
        [
            "driver_age",
            "driver_experience_years",
            "total_assigned_rides",
            "accepted_rides",
            "incomplete_rides",
            "delay_count",
            "acceptance_rate",
            "delay_rate",
            "avg_driver_rating",
            "avg_pickup_delay_min",
            "driver_delay_flag",
        ],
    )
    frame = handle_missing_drivers(frame)
    frame = coerce_categorical(frame, ["driver_city", "vehicle_type"])
    return frame.reset_index(drop=True)


def clean_location_demand(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean the location-demand aggregate and namespace its location codes."""
    frame = standardise_columns(frame)
    frame = strip_string_columns(frame)
    frame = coerce_numeric(
        frame,
        [
            "hour_of_day",
            "total_requests",
            "completed_rides",
            "cancelled_rides",
            "avg_wait_time_min",
            "avg_surge_multiplier",
        ],
    )
    frame["pickup_location_key"] = (
        frame["city"].astype(str) + "::" + frame["pickup_location"].astype(str)
    )
    frame = _fill_generic(frame, "location_demand")
    frame = coerce_categorical(frame, ["city", "vehicle_type", "demand_level"])
    return frame.reset_index(drop=True)


def clean_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean the hourly calendar dimension.

    ``is_holiday`` is retained here for completeness but is excluded from all
    feature matrices, since it is 0 for every hour of 2025.
    """
    frame = standardise_columns(frame)
    frame = strip_string_columns(frame)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame = drop_duplicates_by_key(frame, "datetime")
    frame = coerce_numeric(
        frame, ["hour_of_day", "is_weekend", "is_holiday", "peak_time_flag"]
    )
    frame = coerce_categorical(frame, ["day_of_week", "season"])
    return frame.reset_index(drop=True)


def clean_all(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Run every per-file cleaner and return the cleaned frames."""
    cleaners = {
        "bookings": clean_bookings,
        "customers": clean_customers,
        "drivers": clean_drivers,
        "location_demand": clean_location_demand,
        "time_features": clean_time_features,
    }
    cleaned = {}
    for name, cleaner in cleaners.items():
        if name not in raw:
            raise KeyError(f"Missing raw frame {name!r} in clean_all input.")
        cleaned[name] = cleaner(raw[name])
        logger.info("Cleaned %s: %d rows", name, len(cleaned[name]))
    return cleaned


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def build_cleaning_summary(
    raw: dict[str, pd.DataFrame], cleaned: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Compare row counts and null totals before and after cleaning."""
    rows = []
    for name in raw:
        before, after = raw[name], cleaned[name]
        rows.append(
            {
                "dataset": name,
                "rows_before": len(before),
                "rows_after": len(after),
                "rows_removed": len(before) - len(after),
                "nulls_before": int(before.isna().sum().sum()),
                "nulls_after": int(after.isna().sum().sum()),
                "columns_after": after.shape[1],
            }
        )
    return pd.DataFrame(rows)

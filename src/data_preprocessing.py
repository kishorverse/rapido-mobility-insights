"""Configuration, raw data loading, cleaning and the MySQL ETL pipeline.

This module is the foundation of the project. Everything that turns the five
source CSVs into queryable, model-ready data lives here, in the order it runs:

1. **Configuration** - paths, database credentials, domain constants and the
   leakage-control lists. No other module hard-codes an environment value.
2. **Loading** - every read of the five source files goes through :func:`read_raw`
   so dtypes and file locations are defined once.
3. **Cleaning** - per-file validation and null handling. Cleaning never invents
   data: the structural nulls (the post-outcome columns) are preserved as NaN
   rather than imputed, because imputing them would fabricate the very signal
   the models are supposed to predict.
4. **Database** - connection handling and the access layer. The DDL itself lives
   in ``sql/schema.sql`` and is read from there, so the schema on disk and the
   schema that runs are always the same statements.
5. **ETL** - extract, transform and load into the normalised star schema:
   text city, location and vehicle values become integer foreign keys drawn
   from generated dimensions.

Every statement is parameterised. Values are passed to the driver, never
formatted into the SQL string.

Usage:
    python src/data_preprocessing.py etl --rebuild      # drop and reload MySQL
    python src/data_preprocessing.py etl --verify-only  # connectivity + counts
    python src/data_preprocessing.py clean              # rebuild the cache only
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from pandas._libs.parsers import STR_NA_VALUES

# src/ holds four flat modules that import each other by bare name. Adding this
# directory to the path keeps that working whichever entry point is used: the
# CLI, the Streamlit app, a notebook, or an interactive shell.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
except ImportError:  # pragma: no cover - dependency missing
    mysql = None
    MySQLError = Exception

logger = logging.getLogger(__name__)


# =========================================================================== #
# 1. Configuration
# =========================================================================== #

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover - optional dependency
    pass

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
SQL_DIR = PROJECT_ROOT / "sql"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_FILES = {
    "bookings": RAW_DIR / "bookings.csv",
    "customers": RAW_DIR / "customers.csv",
    "drivers": RAW_DIR / "drivers.csv",
    "location_demand": RAW_DIR / "location_demand.csv",
    "time_features": RAW_DIR / "time_features.csv",
}

#: The single processed artefact: one model-ready row per booking.
MODEL_DATA_FILE = PROCESSED_DIR / "model_data.csv"

SCHEMA_FILE = SQL_DIR / "schema.sql"

for _directory in (DATA_DIR, PROCESSED_DIR, MODEL_DIR, REPORTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Database credentials
# --------------------------------------------------------------------------- #

DB_NAME = os.getenv("RAPIDO_DB_NAME", "rapido_mobility")


def get_db_config(include_database: bool = True) -> dict:
    """Return MySQL connection settings, overridable through environment vars.

    Args:
        include_database: When ``False`` the ``database`` key is omitted, which
            is required for the ``CREATE DATABASE`` bootstrap connection.
    """
    settings = {
        "host": os.getenv("RAPIDO_DB_HOST", "localhost"),
        "port": int(os.getenv("RAPIDO_DB_PORT", "3306")),
        "user": os.getenv("RAPIDO_DB_USER", "root"),
        "password": os.getenv("RAPIDO_DB_PASSWORD", ""),
    }
    if include_database:
        settings["database"] = DB_NAME
    return settings


# --------------------------------------------------------------------------- #
# Domain constants (verified against the raw files)
# --------------------------------------------------------------------------- #

CITIES = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Mumbai"]
VEHICLE_TYPES = ["Auto", "Bike", "Cab"]
TRAFFIC_LEVELS = ["Low", "Medium", "High"]
WEATHER_CONDITIONS = ["Clear", "Rain", "Heavy Rain"]
BOOKING_STATUSES = ["Completed", "Cancelled", "Incomplete"]
SEASONS = ["Summer", "Monsoon", "Winter"]
GENDERS = ["Female", "Male", "Non-Binary"]
DEMAND_LEVELS = ["Low", "Medium"]  # no "High" present in location_demand.csv

WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

#: Hours treated as rush hour for ``rush_hour_flag``.
RUSH_HOURS = list(range(8, 11)) + list(range(17, 21))

#: Hours treated as night rides for ``is_night_ride``.
NIGHT_HOURS = list(range(22, 24)) + list(range(0, 6))

#: Distance in km above which a ride is flagged long distance.
LONG_DISTANCE_KM = 15.0


# --------------------------------------------------------------------------- #
# Leakage control
# --------------------------------------------------------------------------- #

#: Columns that only exist *after* a ride has resolved. They must never enter a
#: model feature matrix.
POST_OUTCOME_COLUMNS = [
    "actual_ride_time_min",
    "incomplete_ride_reason",
    "booking_status",
]

#: ``is_holiday`` is 0 for all 8,760 hours of 2025 -- zero variance, dropped
#: from every feature matrix.
ZERO_VARIANCE_COLUMNS = ["is_holiday"]

#: Columns derived from ``booking_value`` -- valid for EDA, invalid as fare
#: model inputs.
FARE_DERIVED_COLUMNS = ["fare_per_km", "fare_per_min", "booking_value"]

#: ``base_fare * surge_multiplier`` reproduces ``booking_value`` to within 5%,
#: so ``base_fare`` is excluded from the honest pre-quote fare model.
FARE_FORMULA_COLUMNS = ["base_fare"]


# --------------------------------------------------------------------------- #
# Modelling defaults
# --------------------------------------------------------------------------- #

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

MODEL_NAMES = {
    "outcome": "ride_outcome_model",
    "fare": "fare_prediction_model",
    "customer_risk": "customer_cancellation_model",
    "driver_risk": "driver_delay_model",
}


def get_model_path(name: str) -> Path:
    """Return the on-disk artefact path for a trained model."""
    return MODEL_DIR / f"{name}.pkl"


# =========================================================================== #
# 2. Raw data loading
# =========================================================================== #

#: Explicit dtypes avoid pandas guessing ID columns as numbers and avoid the
#: mixed-type warning on the 100k-row bookings file.
RAW_DTYPES = {
    "bookings": {
        "booking_id": "string",
        "customer_id": "string",
        "driver_id": "string",
        "city": "category",
        "vehicle_type": "category",
        "traffic_level": "category",
        "weather_condition": "category",
        "booking_status": "category",
        "day_of_week": "category",
    },
    "customers": {
        "customer_id": "string",
        "customer_gender": "category",
        "customer_city": "category",
        "preferred_vehicle_type": "category",
    },
    "drivers": {
        "driver_id": "string",
        "driver_city": "category",
        "vehicle_type": "category",
    },
    "location_demand": {
        "city": "category",
        "vehicle_type": "category",
        "demand_level": "category",
    },
    "time_features": {
        "day_of_week": "category",
        "season": "category",
    },
}


def read_raw(name: str, path: Path | None = None) -> pd.DataFrame:
    """Read one of the five known source files with its declared dtypes.

    Args:
        name: Logical dataset name, a key of :data:`RAW_FILES`.
        path: Optional override, useful for tests and fixtures.

    Raises:
        FileNotFoundError: If the source file is missing.
        ValueError: If ``name`` is not a known dataset.
    """
    if name not in RAW_FILES:
        raise ValueError(
            f"Unknown dataset {name!r}. Expected one of {sorted(RAW_FILES)}."
        )

    source = Path(path) if path is not None else RAW_FILES[name]
    if not source.exists():
        raise FileNotFoundError(
            f"Source file for {name!r} not found at {source}. "
            "Check that the five CSVs sit in data/raw/."
        )

    try:
        frame = pd.read_csv(source, dtype=RAW_DTYPES.get(name))
    except pd.errors.ParserError as exc:  # pragma: no cover - corrupt input
        raise ValueError(f"Could not parse {source}: {exc}") from exc

    logger.info("Loaded %s: %d rows x %d columns", name, *frame.shape)
    return frame


def load_bookings(path: Path | None = None) -> pd.DataFrame:
    """Load the booking fact table (~100,000 rows)."""
    return read_raw("bookings", path)


def load_customers(path: Path | None = None) -> pd.DataFrame:
    """Load the customer dimension (~10,000 rows)."""
    return read_raw("customers", path)


def load_drivers(path: Path | None = None) -> pd.DataFrame:
    """Load the driver dimension (~5,000 rows)."""
    return read_raw("drivers", path)


def load_location_demand(path: Path | None = None) -> pd.DataFrame:
    """Load demand aggregates by city, location, hour and vehicle type."""
    return read_raw("location_demand", path)


def load_time_features(path: Path | None = None) -> pd.DataFrame:
    """Load the hourly calendar dimension for 2025."""
    return read_raw("time_features", path)


def load_all_raw() -> dict[str, pd.DataFrame]:
    """Load all five source files keyed by logical dataset name."""
    return {name: read_raw(name) for name in RAW_FILES}


# --------------------------------------------------------------------------- #
# Processed model data
# --------------------------------------------------------------------------- #


def save_model_data(frame: pd.DataFrame) -> Path:
    """Write the model-ready feature table to ``data/processed/model_data.csv``."""
    frame.to_csv(MODEL_DATA_FILE, index=False)
    logger.info("Saved model data (%d rows) to %s", len(frame), MODEL_DATA_FILE)
    return MODEL_DATA_FILE


#: pandas treats the bare string ``None`` as missing by default, but it is a
#: real category here: ``surge_bucket`` labels an un-surged ride "None". Reading
#: with the default NA set would blank that value on ~8% of rows and the models
#: one-hot encoded it at training time, so it is removed from the NA list.
_NA_VALUES = sorted(STR_NA_VALUES - {"None"})


def load_model_data() -> pd.DataFrame:
    """Read the cached model-ready feature table.

    Raises:
        FileNotFoundError: If the cache has not been built yet.
    """
    if not MODEL_DATA_FILE.exists():
        raise FileNotFoundError(
            f"No model data at {MODEL_DATA_FILE}. "
            "Run: python src/feature_engineering.py build"
        )
    return pd.read_csv(
        MODEL_DATA_FILE,
        parse_dates=["booking_ts"],
        low_memory=False,
        keep_default_na=False,
        na_values=_NA_VALUES,
    )


def model_data_exists() -> bool:
    """Return whether the processed feature table is on disk."""
    return MODEL_DATA_FILE.exists()


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #


def profile_dataframe(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Build a column-level profile: dtype, nulls, uniques, range and sample.

    Args:
        frame: The DataFrame to profile.
        name: Dataset label, carried into the ``dataset`` column.

    Returns:
        One row per column of ``frame``.
    """
    rows = []
    total = len(frame)

    for column in frame.columns:
        series = frame[column]
        null_count = int(series.isna().sum())
        row = {
            "dataset": name,
            "column": column,
            "dtype": str(series.dtype),
            "non_null": total - null_count,
            "nulls": null_count,
            "null_pct": round(100 * null_count / total, 2) if total else 0.0,
            "unique": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            row["min"] = series.min()
            row["max"] = series.max()
            row["mean"] = round(float(series.mean()), 3) if null_count < total else None
        else:
            non_null = series.dropna()
            row["min"] = None
            row["max"] = None
            row["mean"] = None
            row["sample_values"] = ", ".join(
                str(value) for value in non_null.unique()[:5]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def profile_all_raw() -> pd.DataFrame:
    """Profile every raw source file and return one stacked frame."""
    frames = load_all_raw()
    return pd.concat(
        [profile_dataframe(frame, name) for name, frame in frames.items()],
        ignore_index=True,
    )


# =========================================================================== #
# 3. Cleaning
# =========================================================================== #

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
# Generic cleaning helpers
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
    for unresolved rides. They stay ``NaN`` and never reach a model.
    """
    frame = frame.copy()

    structural = set(POST_OUTCOME_COLUMNS)
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


def handle_missing_customers(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill customer nulls: median for numerics, mode for categoricals."""
    return _fill_generic(frame, "customers")


def handle_missing_drivers(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill driver nulls: median for numerics, mode for categoricals."""
    return _fill_generic(frame, "drivers")


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


def load_and_clean() -> dict[str, pd.DataFrame]:
    """Read all five source files and clean them in one call."""
    return clean_all(load_all_raw())


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


# =========================================================================== #
# 4. Database
# =========================================================================== #


class DatabaseError(RuntimeError):
    """Raised when a database operation fails, with actionable context."""


def _split_sql_statements(text: str) -> list[tuple[str, str]]:
    """Split a .sql file into ``(leading_comment, statement)`` pairs.

    Comments immediately preceding a statement are kept with it, which is how
    the index rationales in ``schema.sql`` survive the round trip into
    :func:`get_index_documentation`.
    """
    pairs: list[tuple[str, str]] = []
    comment_lines: list[str] = []
    body_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not body_lines and stripped.startswith("--"):
            comment = stripped.lstrip("-").strip()
            if comment and not set(comment) <= {"="}:
                comment_lines.append(comment)
            continue
        if not body_lines and not stripped:
            comment_lines = []
            continue

        body_lines.append(line)
        if stripped.endswith(";"):
            statement = "\n".join(body_lines).strip().rstrip(";").strip()
            pairs.append((" ".join(comment_lines), statement))
            comment_lines, body_lines = [], []

    return pairs


def _load_schema() -> dict:
    """Parse ``sql/schema.sql`` into the statements the loader needs.

    Keeping the DDL in the .sql file and reading it here means the schema a
    reviewer opens is byte-for-byte the schema that runs.
    """
    if not SCHEMA_FILE.exists():
        raise DatabaseError(f"Schema file missing at {SCHEMA_FILE}.")

    text = SCHEMA_FILE.read_text(encoding="utf-8")
    tables: dict[str, str] = {}
    order: list[str] = []
    indexes: list[tuple[str, str, str, str]] = []

    for comment, statement in _split_sql_statements(text):
        upper = statement.upper()
        if upper.startswith("CREATE TABLE"):
            match = re.search(
                r"CREATE TABLE(?: IF NOT EXISTS)?\s+`?(\w+)`?", statement, re.IGNORECASE
            )
            if match:
                name = match.group(1)
                tables[name] = statement
                order.append(name)
        elif upper.startswith("CREATE INDEX"):
            match = re.search(
                r"CREATE INDEX\s+`?(\w+)`?\s+ON\s+`?(\w+)`?", statement, re.IGNORECASE
            )
            if match:
                indexes.append((match.group(1), match.group(2), statement, comment))

    if not tables:
        raise DatabaseError(f"No CREATE TABLE statements found in {SCHEMA_FILE}.")

    return {"tables": tables, "order": order, "indexes": indexes}


_SCHEMA = _load_schema()

#: Creation order respects foreign-key dependencies, as written in schema.sql.
TABLE_ORDER: list[str] = _SCHEMA["order"]
CREATE_TABLE_STATEMENTS: dict[str, str] = _SCHEMA["tables"]
INDEX_STATEMENTS: list[tuple[str, str, str, str]] = _SCHEMA["indexes"]

#: Dropped in reverse dependency order so foreign keys never block a rebuild.
DROP_STATEMENTS = [f"DROP TABLE IF EXISTS {table}" for table in reversed(TABLE_ORDER)]

#: Column order used when inserting each table. Auto-increment surrogate keys
#: that the database generates itself are omitted.
INSERT_COLUMNS = {
    "cities": ["city_id", "city_name"],
    "vehicle_types": ["vehicle_type_id", "vehicle_name"],
    "locations": ["location_id", "city_id", "location_code"],
    "customers": [
        "customer_id",
        "customer_gender",
        "customer_age",
        "city_id",
        "customer_signup_days_ago",
        "preferred_vehicle_type_id",
        "total_bookings",
        "completed_rides",
        "cancelled_rides",
        "incomplete_rides",
        "cancellation_rate",
        "avg_customer_rating",
        "customer_cancel_flag",
    ],
    "drivers": [
        "driver_id",
        "driver_age",
        "city_id",
        "vehicle_type_id",
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
    "time_features": [
        "slot_datetime",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "peak_time_flag",
        "season",
    ],
    "location_demand": [
        "city_id",
        "location_id",
        "hour_of_day",
        "vehicle_type_id",
        "total_requests",
        "completed_rides",
        "cancelled_rides",
        "avg_wait_time_min",
        "avg_surge_multiplier",
        "demand_level",
    ],
    "bookings": [
        "booking_id",
        "booking_ts",
        "city_id",
        "pickup_location_id",
        "drop_location_id",
        "vehicle_type_id",
        "customer_id",
        "driver_id",
        "ride_distance_km",
        "estimated_ride_time_min",
        "actual_ride_time_min",
        "traffic_level",
        "weather_condition",
        "base_fare",
        "surge_multiplier",
        "booking_value",
        "booking_status",
        "incomplete_ride_reason",
    ],
}


def get_create_statements() -> list[str]:
    """Return CREATE TABLE statements in foreign-key-safe order."""
    return [CREATE_TABLE_STATEMENTS[table] for table in TABLE_ORDER]


def get_index_statements() -> list[str]:
    """Return the CREATE INDEX statements."""
    return [statement for _, _, statement, _ in INDEX_STATEMENTS]


def get_index_documentation() -> list[dict]:
    """Return index name, table and the rationale recorded in schema.sql."""
    return [
        {"index": name, "table": table, "rationale": reason}
        for name, table, _, reason in INDEX_STATEMENTS
    ]


def get_insert_columns(table: str) -> list[str]:
    """Return the insert column order for ``table``."""
    if table not in INSERT_COLUMNS:
        raise ValueError(f"Unknown table {table!r}.")
    return INSERT_COLUMNS[table]


# --------------------------------------------------------------------------- #
# Connection and access layer
# --------------------------------------------------------------------------- #


@contextmanager
def get_connection(include_database: bool = True):
    """Yield a MySQL connection, closing it on exit.

    Args:
        include_database: Pass ``False`` for the bootstrap connection used by
            :func:`create_database`, before the schema exists.

    Raises:
        DatabaseError: If the connection cannot be established.
    """
    if mysql is None:  # pragma: no cover
        raise DatabaseError(
            "mysql-connector-python is not installed. "
            "Run: pip install -r requirements.txt"
        )

    settings = get_db_config(include_database=include_database)
    connection = None
    try:
        connection = mysql.connector.connect(**settings)
        yield connection
    except MySQLError as exc:
        raise DatabaseError(
            f"MySQL connection failed for "
            f"{settings['user']}@{settings['host']}:{settings['port']}: {exc}. "
            "Set RAPIDO_DB_USER / RAPIDO_DB_PASSWORD in your environment."
        ) from exc
    finally:
        if connection is not None and connection.is_connected():
            connection.close()


def execute(sql: str, params: tuple | None = None) -> int:
    """Execute a single statement and return the affected row count."""
    with get_connection() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(sql, params or ())
            connection.commit()
            return cursor.rowcount
        except MySQLError as exc:
            connection.rollback()
            raise DatabaseError(
                f"Statement failed: {sql.strip()[:120]}... ({exc})"
            ) from exc
        finally:
            cursor.close()


def executemany(sql: str, rows: list[tuple]) -> int:
    """Execute a parameterised statement over many rows in one round trip."""
    if not rows:
        return 0
    with get_connection() as connection:
        cursor = connection.cursor()
        try:
            cursor.executemany(sql, rows)
            connection.commit()
            return cursor.rowcount
        except MySQLError as exc:
            connection.rollback()
            raise DatabaseError(f"Bulk statement failed ({exc})") from exc
        finally:
            cursor.close()


def read_sql(sql: str, params: tuple | None = None) -> pd.DataFrame:
    """Run a SELECT and return the result as a DataFrame.

    Uses a dictionary cursor rather than ``pandas.read_sql`` so that the
    connector is driven through its supported API; pandas only officially
    supports SQLAlchemy connectables.
    """
    with get_connection() as connection:
        cursor = connection.cursor(dictionary=True)
        try:
            cursor.execute(sql, params or ())
            rows = cursor.fetchall()
            columns = [column[0] for column in cursor.description]
            return pd.DataFrame(rows, columns=columns)
        except MySQLError as exc:
            raise DatabaseError(
                f"Query failed: {sql.strip()[:120]}... ({exc})"
            ) from exc
        finally:
            cursor.close()


def create_database() -> None:
    """Create the project database if it does not already exist."""
    with get_connection(include_database=False) as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {DB_NAME} "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            connection.commit()
            logger.info("Database %s ready", DB_NAME)
        finally:
            cursor.close()


def create_tables() -> None:
    """Create every table in foreign-key-safe order, using schema.sql."""
    with get_connection() as connection:
        cursor = connection.cursor()
        try:
            for table in TABLE_ORDER:
                cursor.execute(CREATE_TABLE_STATEMENTS[table])
                logger.info("Table ready: %s", table)
            connection.commit()
        finally:
            cursor.close()


def create_indexes() -> None:
    """Create the documented indexes, skipping any that already exist."""
    existing = (
        {row["index"] for _, row in list_indexes().iterrows()}
        if table_exists("bookings")
        else set()
    )

    with get_connection() as connection:
        cursor = connection.cursor()
        try:
            for name, _table, statement, _reason in INDEX_STATEMENTS:
                if name in existing:
                    logger.info("Index already present: %s", name)
                    continue
                cursor.execute(statement)
                logger.info("Created index: %s", name)
            connection.commit()
        except MySQLError as exc:
            connection.rollback()
            raise DatabaseError(f"Index creation failed: {exc}") from exc
        finally:
            cursor.close()


def list_indexes() -> pd.DataFrame:
    """List non-primary indexes defined in the project database."""
    return read_sql(
        """
        SELECT DISTINCT TABLE_NAME AS `table`, INDEX_NAME AS `index`
        FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = %s AND INDEX_NAME <> 'PRIMARY'
        """,
        (DB_NAME,),
    )


def drop_all_tables() -> None:
    """Drop every project table, ignoring foreign keys during the teardown."""
    with get_connection() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
            for statement in DROP_STATEMENTS:
                cursor.execute(statement)
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
            connection.commit()
            logger.info("Dropped %d tables", len(DROP_STATEMENTS))
        finally:
            cursor.close()


def table_exists(name: str) -> bool:
    """Return whether ``name`` exists in the project database."""
    result = read_sql(
        """
        SELECT COUNT(*) AS n
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        """,
        (DB_NAME, name),
    )
    return bool(result["n"].iloc[0])


def row_count(table: str) -> int:
    """Return the row count of ``table``."""
    return int(read_sql(f"SELECT COUNT(*) AS n FROM {table}")["n"].iloc[0])


def truncate_table(table: str) -> None:
    """Empty a table without dropping it."""
    with get_connection() as connection:
        cursor = connection.cursor()
        try:
            cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
            cursor.execute(f"TRUNCATE TABLE {table}")
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
            connection.commit()
        finally:
            cursor.close()


def _to_sql_rows(frame: pd.DataFrame, columns: list[str]) -> list[tuple]:
    """Convert a frame to tuples, turning NaN/NaT into SQL ``NULL``."""
    subset = frame[columns].astype(object).where(pd.notna(frame[columns]), None)
    rows = []
    for record in subset.itertuples(index=False, name=None):
        rows.append(
            tuple(
                value.item() if isinstance(value, np.generic) else value
                for value in record
            )
        )
    return rows


def bulk_insert(table: str, frame: pd.DataFrame, chunk_size: int = 5000) -> int:
    """Insert a DataFrame into ``table`` in chunks.

    Args:
        table: Target table name.
        frame: Source data; must contain every column in the table's insert list.
        chunk_size: Rows per round trip. 5,000 keeps the 100k booking load to
            20 batches without exceeding the default packet size.

    Returns:
        Number of rows inserted.
    """
    columns = get_insert_columns(table)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Frame for {table!r} is missing columns: {missing}")

    placeholders = ", ".join(["%s"] * len(columns))
    column_list = ", ".join(f"`{column}`" for column in columns)
    statement = f"INSERT INTO {table} ({column_list}) VALUES ({placeholders})"

    rows = _to_sql_rows(frame, columns)
    inserted = 0

    with get_connection() as connection:
        cursor = connection.cursor()
        try:
            for start in range(0, len(rows), chunk_size):
                batch = rows[start : start + chunk_size]
                cursor.executemany(statement, batch)
                inserted += len(batch)
            connection.commit()
            logger.info("Inserted %d rows into %s", inserted, table)
        except MySQLError as exc:
            connection.rollback()
            raise DatabaseError(
                f"Bulk insert into {table} failed after {inserted} rows: {exc}"
            ) from exc
        finally:
            cursor.close()

    return inserted


def healthcheck() -> dict:
    """Return connectivity status and per-table row counts."""
    status = {"connected": False, "database": DB_NAME, "tables": {}}
    try:
        with get_connection() as connection:
            status["connected"] = connection.is_connected()
            status["server_version"] = connection.server_info
        for table in TABLE_ORDER:
            status["tables"][table] = row_count(table) if table_exists(table) else None
    except DatabaseError as exc:
        status["error"] = str(exc)
    return status


# =========================================================================== #
# 5. ETL
# =========================================================================== #


def extract() -> dict[str, pd.DataFrame]:
    """Read all five raw source files."""
    logger.info("Extract: reading raw CSVs")
    return load_all_raw()


def build_dimension_tables(cleaned: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Generate the ``cities``, ``vehicle_types`` and ``locations`` dimensions.

    Location codes repeat across cities, so the location dimension is built
    from the distinct ``(city, code)`` pairs observed in bookings *and*
    location demand.
    """
    bookings = cleaned["bookings"]
    demand = cleaned["location_demand"]

    city_names = sorted(
        set(bookings["city"].astype(str))
        | set(demand["city"].astype(str))
        | set(cleaned["customers"]["customer_city"].astype(str))
        | set(cleaned["drivers"]["driver_city"].astype(str))
    )
    cities = pd.DataFrame(
        {"city_id": range(1, len(city_names) + 1), "city_name": city_names}
    )

    vehicle_names = sorted(
        set(bookings["vehicle_type"].astype(str))
        | set(demand["vehicle_type"].astype(str))
        | set(cleaned["drivers"]["vehicle_type"].astype(str))
        | set(cleaned["customers"]["preferred_vehicle_type"].astype(str))
    )
    vehicle_types = pd.DataFrame(
        {
            "vehicle_type_id": range(1, len(vehicle_names) + 1),
            "vehicle_name": vehicle_names,
        }
    )

    pairs = pd.concat(
        [
            bookings[["city", "pickup_location"]].rename(
                columns={"pickup_location": "location_code"}
            ),
            bookings[["city", "drop_location"]].rename(
                columns={"drop_location": "location_code"}
            ),
            demand[["city", "pickup_location"]].rename(
                columns={"pickup_location": "location_code"}
            ),
        ],
        ignore_index=True,
    )
    pairs["city"] = pairs["city"].astype(str)
    pairs["location_code"] = pairs["location_code"].astype(str)
    locations = (
        pairs.drop_duplicates()
        .sort_values(["city", "location_code"])
        .reset_index(drop=True)
    )
    locations = locations.merge(
        cities, left_on="city", right_on="city_name", how="left"
    )
    locations.insert(0, "location_id", range(1, len(locations) + 1))
    locations = locations[["location_id", "city_id", "location_code", "city"]]

    logger.info(
        "Dimensions: %d cities, %d vehicle types, %d locations",
        len(cities),
        len(vehicle_types),
        len(locations),
    )
    return {"cities": cities, "vehicle_types": vehicle_types, "locations": locations}


def _city_lookup(cities: pd.DataFrame) -> dict[str, int]:
    """Map city name to surrogate key."""
    return dict(zip(cities["city_name"], cities["city_id"]))


def _vehicle_lookup(vehicle_types: pd.DataFrame) -> dict[str, int]:
    """Map vehicle name to surrogate key."""
    return dict(zip(vehicle_types["vehicle_name"], vehicle_types["vehicle_type_id"]))


def _location_lookup(locations: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Map ``(city, location_code)`` to surrogate key."""
    return {
        (city, code): location_id
        for location_id, city, code in zip(
            locations["location_id"], locations["city"], locations["location_code"]
        )
    }


def transform_bookings(
    bookings: pd.DataFrame, dimensions: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Replace booking text keys with surrogate foreign keys.

    ``day_of_week``, ``is_weekend`` and ``hour_of_day`` are dropped here: they
    are derivable from ``booking_ts`` and duplicated in ``time_features``.
    """
    cities = _city_lookup(dimensions["cities"])
    vehicles = _vehicle_lookup(dimensions["vehicle_types"])
    locations = _location_lookup(dimensions["locations"])

    frame = bookings.copy()
    city_text = frame["city"].astype(str)
    frame["city_id"] = city_text.map(cities)
    frame["vehicle_type_id"] = frame["vehicle_type"].astype(str).map(vehicles)
    frame["pickup_location_id"] = [
        locations[(city, code)]
        for city, code in zip(city_text, frame["pickup_location"].astype(str))
    ]
    frame["drop_location_id"] = [
        locations[(city, code)]
        for city, code in zip(city_text, frame["drop_location"].astype(str))
    ]
    frame["incomplete_ride_reason"] = frame["incomplete_ride_reason"].astype(object)
    frame["booking_status"] = frame["booking_status"].astype(str)
    frame["traffic_level"] = frame["traffic_level"].astype(str)
    frame["weather_condition"] = frame["weather_condition"].astype(str)

    return frame[get_insert_columns("bookings")]


def transform_customers(
    customers: pd.DataFrame, dimensions: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Map customer city and preferred vehicle to surrogate keys."""
    cities = _city_lookup(dimensions["cities"])
    vehicles = _vehicle_lookup(dimensions["vehicle_types"])

    frame = customers.copy()
    frame["city_id"] = frame["customer_city"].astype(str).map(cities)
    frame["preferred_vehicle_type_id"] = (
        frame["preferred_vehicle_type"].astype(str).map(vehicles)
    )
    frame["customer_gender"] = frame["customer_gender"].astype(str)
    return frame[get_insert_columns("customers")]


def transform_drivers(
    drivers: pd.DataFrame, dimensions: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Map driver city and vehicle type to surrogate keys."""
    cities = _city_lookup(dimensions["cities"])
    vehicles = _vehicle_lookup(dimensions["vehicle_types"])

    frame = drivers.copy()
    frame["city_id"] = frame["driver_city"].astype(str).map(cities)
    frame["vehicle_type_id"] = frame["vehicle_type"].astype(str).map(vehicles)
    return frame[get_insert_columns("drivers")]


def transform_location_demand(
    demand: pd.DataFrame, dimensions: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Map demand rows onto city, location and vehicle surrogate keys."""
    cities = _city_lookup(dimensions["cities"])
    vehicles = _vehicle_lookup(dimensions["vehicle_types"])
    locations = _location_lookup(dimensions["locations"])

    frame = demand.copy()
    city_text = frame["city"].astype(str)
    frame["city_id"] = city_text.map(cities)
    frame["vehicle_type_id"] = frame["vehicle_type"].astype(str).map(vehicles)
    frame["location_id"] = [
        locations[(city, code)]
        for city, code in zip(city_text, frame["pickup_location"].astype(str))
    ]
    frame["demand_level"] = frame["demand_level"].astype(str)
    return frame[get_insert_columns("location_demand")]


def transform_time_features(time_features: pd.DataFrame) -> pd.DataFrame:
    """Rename ``datetime`` to ``slot_datetime`` (a MySQL reserved word)."""
    frame = time_features.rename(columns={"datetime": "slot_datetime"}).copy()
    frame["day_of_week"] = frame["day_of_week"].astype(str)
    frame["season"] = frame["season"].astype(str)
    return frame[get_insert_columns("time_features")]


def transform(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Clean the raw frames and shape them into loadable tables."""
    logger.info("Transform: cleaning and normalising")
    cleaned = clean_all(raw)
    dimensions = build_dimension_tables(cleaned)

    tables = {
        "cities": dimensions["cities"],
        "vehicle_types": dimensions["vehicle_types"],
        "locations": dimensions["locations"][get_insert_columns("locations")],
        "customers": transform_customers(cleaned["customers"], dimensions),
        "drivers": transform_drivers(cleaned["drivers"], dimensions),
        "time_features": transform_time_features(cleaned["time_features"]),
        "location_demand": transform_location_demand(
            cleaned["location_demand"], dimensions
        ),
        "bookings": transform_bookings(cleaned["bookings"], dimensions),
    }
    return {"tables": tables, "cleaned": cleaned, "dimensions": dimensions}


def load(tables: dict[str, pd.DataFrame], rebuild: bool = False) -> dict[str, int]:
    """Create the schema if needed and insert every table.

    Args:
        tables: Table name to frame, already shaped for insertion.
        rebuild: Drop and recreate all tables first.

    Returns:
        Rows inserted per table.
    """
    create_database()
    if rebuild:
        logger.info("Load: rebuilding schema from scratch")
        drop_all_tables()
    create_tables()

    inserted = {}
    for table in TABLE_ORDER:
        if row_count(table) and not rebuild:
            logger.info("Skipping %s: already populated", table)
            inserted[table] = 0
            continue
        inserted[table] = bulk_insert(table, tables[table])

    create_indexes()
    return inserted


def verify_load(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare in-memory row counts against what actually landed in MySQL."""
    rows = []
    for table in TABLE_ORDER:
        expected = len(tables[table])
        actual = row_count(table) if table_exists(table) else 0
        rows.append(
            {
                "table": table,
                "expected": expected,
                "actual": actual,
                "status": "PASS" if expected == actual else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def run_etl(rebuild: bool = False, cache: bool = True) -> dict:
    """Run extract, transform, load and verification end to end.

    Args:
        rebuild: Drop and recreate the schema before loading.
        cache: Also rebuild ``data/processed/model_data.csv`` from the cleaned
            frames, so the models and dashboard have fresh features.
    """
    raw = extract()
    result = transform(raw)
    tables = result["tables"]

    if cache:
        # Imported here rather than at module scope: feature_engineering imports
        # this module, so a top-level import would be circular.
        import feature_engineering

        frame = feature_engineering.build_feature_table(result["cleaned"])
        save_model_data(frame)

    inserted = load(tables, rebuild=rebuild)
    verification = verify_load(tables)

    logger.info("ETL complete: %d rows inserted", sum(inserted.values()))
    return {
        "inserted": inserted,
        "verification": verification,
        "tables": tables,
        "cleaned": result["cleaned"],
    }


# =========================================================================== #
# 6. Command line
# =========================================================================== #


def _run_etl_command(args: argparse.Namespace) -> int:
    """Load the cleaned data into MySQL, or just report what is already there."""
    try:
        if args.verify_only:
            status = healthcheck()
            print(f"Connected: {status['connected']}  ({status['database']})")
            for table, count in status["tables"].items():
                print(f"  {table:<18} {count if count is not None else 'missing'}")
            return 0

        result = run_etl(rebuild=args.rebuild, cache=not args.no_cache)
        print("\nRows inserted:")
        for table, count in result["inserted"].items():
            print(f"  {table:<18} {count:>8,}")
        print("\nVerification:")
        print(result["verification"].to_string(index=False))

        failures = result["verification"]["status"].eq("FAIL").sum()
        return 1 if failures else 0
    except DatabaseError as exc:
        print(f"Database error: {exc}", file=sys.stderr)
        return 1


def _run_clean_command(args: argparse.Namespace) -> int:
    """Clean the raw files and report what changed, without touching MySQL."""
    raw = load_all_raw()
    cleaned = clean_all(raw)
    print(build_cleaning_summary(raw, cleaned).to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Rapido data preprocessing: cleaning and the MySQL ETL."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    etl_parser = subparsers.add_parser("etl", help="load cleaned data into MySQL")
    etl_parser.add_argument(
        "--rebuild", action="store_true", help="drop and recreate all tables"
    )
    etl_parser.add_argument(
        "--verify-only", action="store_true", help="report connectivity and row counts"
    )
    etl_parser.add_argument(
        "--no-cache", action="store_true", help="skip rebuilding model_data.csv"
    )
    etl_parser.set_defaults(handler=_run_etl_command)

    clean_parser = subparsers.add_parser("clean", help="clean the raw files only")
    clean_parser.set_defaults(handler=_run_clean_command)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

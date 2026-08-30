"""Central configuration for the Rapido Intelligent Mobility Insights project.

All paths, database credentials, domain constants and modelling defaults live
here so that no module hard-codes an environment-specific value.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:  # pragma: no cover - optional dependency
    pass

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "Rapido_dataset"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

RAW_FILES = {
    "bookings": RAW_DIR / "bookings.csv",
    "customers": RAW_DIR / "customers.csv",
    "drivers": RAW_DIR / "drivers.csv",
    "location_demand": RAW_DIR / "location_demand.csv",
    "time_features": RAW_DIR / "time_features.csv",
}

for _directory in (DATA_DIR, PROCESSED_DIR, MODEL_DIR, DOCS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #

DB_NAME = os.getenv("RAPIDO_DB_NAME", "rapido_mobility")


def get_db_config(include_database: bool = True) -> dict:
    """Return MySQL connection settings, overridable through environment vars.

    Args:
        include_database: When ``False`` the ``database`` key is omitted, which
            is required for the ``CREATE DATABASE`` bootstrap connection.
    """
    config = {
        "host": os.getenv("RAPIDO_DB_HOST", "localhost"),
        "port": int(os.getenv("RAPIDO_DB_PORT", "3306")),
        "user": os.getenv("RAPIDO_DB_USER", "root"),
        "password": os.getenv("RAPIDO_DB_PASSWORD", ""),
    }
    if include_database:
        config["database"] = DB_NAME
    return config


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
#: model feature matrix. See docs/PROJECT_PLAN.md section 1.2.
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
    "outcome": "ride_outcome",
    "fare": "fare_prediction",
    "customer_risk": "customer_cancellation_risk",
    "driver_risk": "driver_delay_risk",
}


def get_model_path(name: str) -> Path:
    """Return the on-disk artefact path for a trained model."""
    return MODEL_DIR / f"{name}.joblib"

"""Raw data loading and processed-data caching.

Every read of the five source CSVs goes through this module so that dtypes,
error handling and file locations are defined in exactly one place.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)

#: Explicit dtypes avoid pandas guessing ID columns as numbers and avoid the
#: mixed-type warning on the 100k-row bookings file.
_DTYPES = {
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


def _read_csv(name: str, path: Path | None = None) -> pd.DataFrame:
    """Read one of the five known source files with its declared dtypes.

    Args:
        name: Logical dataset name, a key of ``config.RAW_FILES``.
        path: Optional override, useful for tests and fixtures.

    Raises:
        FileNotFoundError: If the source file is missing.
        ValueError: If ``name`` is not a known dataset.
    """
    if name not in config.RAW_FILES:
        raise ValueError(
            f"Unknown dataset {name!r}. Expected one of {sorted(config.RAW_FILES)}."
        )

    source = Path(path) if path is not None else config.RAW_FILES[name]
    if not source.exists():
        raise FileNotFoundError(
            f"Source file for {name!r} not found at {source}. "
            "Check that Rapido_dataset/ sits next to config.py."
        )

    try:
        frame = pd.read_csv(source, dtype=_DTYPES.get(name))
    except pd.errors.ParserError as exc:  # pragma: no cover - corrupt input
        raise ValueError(f"Could not parse {source}: {exc}") from exc

    logger.info("Loaded %s: %d rows x %d columns", name, *frame.shape)
    return frame


def load_bookings(path: Path | None = None) -> pd.DataFrame:
    """Load the booking fact table (~100,000 rows)."""
    return _read_csv("bookings", path)


def load_customers(path: Path | None = None) -> pd.DataFrame:
    """Load the customer dimension (~10,000 rows)."""
    return _read_csv("customers", path)


def load_drivers(path: Path | None = None) -> pd.DataFrame:
    """Load the driver dimension (~5,000 rows)."""
    return _read_csv("drivers", path)


def load_location_demand(path: Path | None = None) -> pd.DataFrame:
    """Load demand aggregates by city, location, hour and vehicle type."""
    return _read_csv("location_demand", path)


def load_time_features(path: Path | None = None) -> pd.DataFrame:
    """Load the hourly calendar dimension for 2025."""
    return _read_csv("time_features", path)


def load_all_raw() -> dict[str, pd.DataFrame]:
    """Load all five source files keyed by logical dataset name."""
    return {name: _read_csv(name) for name in config.RAW_FILES}


def save_processed(frame: pd.DataFrame, name: str) -> Path:
    """Persist a processed frame to Parquet under ``data/processed``."""
    target = config.PROCESSED_DIR / f"{name}.parquet"
    frame.to_parquet(target, index=False)
    logger.info("Saved %s (%d rows) to %s", name, len(frame), target)
    return target


def load_processed(name: str) -> pd.DataFrame:
    """Read a previously cached processed frame from Parquet."""
    source = config.PROCESSED_DIR / f"{name}.parquet"
    if not source.exists():
        raise FileNotFoundError(
            f"No cached frame named {name!r} at {source}. "
            "Run scripts/manage.py etl first."
        )
    return pd.read_parquet(source)


def processed_exists(name: str) -> bool:
    """Return whether a processed Parquet cache exists for ``name``."""
    return (config.PROCESSED_DIR / f"{name}.parquet").exists()


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

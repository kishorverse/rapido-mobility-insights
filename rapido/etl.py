"""Extract - Transform - Load pipeline into the normalised MySQL schema.

The transform step converts the flat CSV frames into the surrogate-keyed star
schema declared in :mod:`rapido.schema`: text city, location and vehicle values
are replaced by integer foreign keys drawn from generated dimensions.
"""

from __future__ import annotations

import logging

import pandas as pd

import config
from rapido import cleaning, db, io, schema

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Extract
# --------------------------------------------------------------------------- #


def extract() -> dict[str, pd.DataFrame]:
    """Read all five raw source files."""
    logger.info("Extract: reading raw CSVs")
    return io.load_all_raw()


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #


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

    return frame[schema.get_insert_columns("bookings")]


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
    return frame[schema.get_insert_columns("customers")]


def transform_drivers(
    drivers: pd.DataFrame, dimensions: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Map driver city and vehicle type to surrogate keys."""
    cities = _city_lookup(dimensions["cities"])
    vehicles = _vehicle_lookup(dimensions["vehicle_types"])

    frame = drivers.copy()
    frame["city_id"] = frame["driver_city"].astype(str).map(cities)
    frame["vehicle_type_id"] = frame["vehicle_type"].astype(str).map(vehicles)
    return frame[schema.get_insert_columns("drivers")]


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
    return frame[schema.get_insert_columns("location_demand")]


def transform_time_features(time_features: pd.DataFrame) -> pd.DataFrame:
    """Rename ``datetime`` to ``slot_datetime`` (a MySQL reserved word)."""
    frame = time_features.rename(columns={"datetime": "slot_datetime"}).copy()
    frame["day_of_week"] = frame["day_of_week"].astype(str)
    frame["season"] = frame["season"].astype(str)
    return frame[schema.get_insert_columns("time_features")]


def transform(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Clean the raw frames and shape them into loadable tables."""
    logger.info("Transform: cleaning and normalising")
    cleaned = cleaning.clean_all(raw)
    dimensions = build_dimension_tables(cleaned)

    tables = {
        "cities": dimensions["cities"],
        "vehicle_types": dimensions["vehicle_types"],
        "locations": dimensions["locations"][
            schema.get_insert_columns("locations")
        ],
        "customers": transform_customers(cleaned["customers"], dimensions),
        "drivers": transform_drivers(cleaned["drivers"], dimensions),
        "time_features": transform_time_features(cleaned["time_features"]),
        "location_demand": transform_location_demand(
            cleaned["location_demand"], dimensions
        ),
        "bookings": transform_bookings(cleaned["bookings"], dimensions),
    }
    return {"tables": tables, "cleaned": cleaned, "dimensions": dimensions}


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #


def load(tables: dict[str, pd.DataFrame], rebuild: bool = False) -> dict[str, int]:
    """Create the schema if needed and insert every table.

    Args:
        tables: Table name to frame, already shaped for insertion.
        rebuild: Drop and recreate all tables first.

    Returns:
        Rows inserted per table.
    """
    db.create_database()
    if rebuild:
        logger.info("Load: rebuilding schema from scratch")
        db.drop_all_tables()
    db.create_tables()

    inserted = {}
    for table in schema.TABLE_ORDER:
        if db.row_count(table) and not rebuild:
            logger.info("Skipping %s: already populated", table)
            inserted[table] = 0
            continue
        inserted[table] = db.bulk_insert(table, tables[table])

    db.create_indexes()
    return inserted


def verify_load(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare in-memory row counts against what actually landed in MySQL."""
    rows = []
    for table in schema.TABLE_ORDER:
        expected = len(tables[table])
        actual = db.row_count(table) if db.table_exists(table) else 0
        rows.append(
            {
                "table": table,
                "expected": expected,
                "actual": actual,
                "status": "PASS" if expected == actual else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def cache_processed(cleaned: dict[str, pd.DataFrame]) -> None:
    """Write the cleaned frames to Parquet for fast model and EDA reloads."""
    for name, frame in cleaned.items():
        io.save_processed(frame, name)


def run_etl(rebuild: bool = False, cache: bool = True) -> dict:
    """Run extract, transform, load and verification end to end."""
    raw = extract()
    result = transform(raw)
    tables = result["tables"]

    if cache:
        cache_processed(result["cleaned"])

    inserted = load(tables, rebuild=rebuild)
    verification = verify_load(tables)

    logger.info("ETL complete: %d rows inserted", sum(inserted.values()))
    return {
        "inserted": inserted,
        "verification": verification,
        "tables": tables,
        "cleaned": result["cleaned"],
    }

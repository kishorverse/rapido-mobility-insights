"""Normalised MySQL schema for the Rapido platform.

Normalisation decisions, each defensible in review:

* ``city``, ``location`` and ``vehicle_type`` are repeated text in the CSVs, so
  they become surrogate-keyed dimensions. Locations are keyed on
  ``(city_id, location_code)`` because ``Loc_1..Loc_50`` repeat in every city.
* ``day_of_week``, ``is_weekend`` and ``hour_of_day`` are dropped from
  ``bookings``: they are transitively dependent on ``booking_ts`` and already
  live in ``time_features``. Storing them again would violate 3NF.
* ``traffic_level``, ``weather_condition``, ``booking_status`` and
  ``demand_level`` stay as ``ENUM`` columns. They are closed low-cardinality
  domains carrying no attributes of their own, so a lookup table would add a
  join without removing redundancy.
"""

from __future__ import annotations

#: Creation order respects foreign-key dependencies.
TABLE_ORDER = [
    "cities",
    "vehicle_types",
    "locations",
    "customers",
    "drivers",
    "time_features",
    "location_demand",
    "bookings",
]

CREATE_TABLE_STATEMENTS = {
    "cities": """
        CREATE TABLE IF NOT EXISTS cities (
            city_id   SMALLINT     NOT NULL AUTO_INCREMENT,
            city_name VARCHAR(50)  NOT NULL,
            PRIMARY KEY (city_id),
            UNIQUE KEY uq_cities_name (city_name)
        ) ENGINE=InnoDB
    """,
    "vehicle_types": """
        CREATE TABLE IF NOT EXISTS vehicle_types (
            vehicle_type_id SMALLINT    NOT NULL AUTO_INCREMENT,
            vehicle_name    VARCHAR(30) NOT NULL,
            PRIMARY KEY (vehicle_type_id),
            UNIQUE KEY uq_vehicle_name (vehicle_name)
        ) ENGINE=InnoDB
    """,
    "locations": """
        CREATE TABLE IF NOT EXISTS locations (
            location_id   INT         NOT NULL AUTO_INCREMENT,
            city_id       SMALLINT    NOT NULL,
            location_code VARCHAR(20) NOT NULL,
            PRIMARY KEY (location_id),
            UNIQUE KEY uq_location_city_code (city_id, location_code),
            CONSTRAINT fk_locations_city FOREIGN KEY (city_id)
                REFERENCES cities (city_id)
        ) ENGINE=InnoDB
    """,
    "customers": """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id               VARCHAR(12)  NOT NULL,
            customer_gender           VARCHAR(20),
            customer_age              TINYINT UNSIGNED,
            city_id                   SMALLINT,
            customer_signup_days_ago  SMALLINT UNSIGNED,
            preferred_vehicle_type_id SMALLINT,
            total_bookings            SMALLINT UNSIGNED,
            completed_rides           SMALLINT UNSIGNED,
            cancelled_rides           SMALLINT UNSIGNED,
            incomplete_rides          SMALLINT UNSIGNED,
            cancellation_rate         DECIMAL(6,4),
            avg_customer_rating       DECIMAL(3,2),
            customer_cancel_flag      TINYINT(1),
            PRIMARY KEY (customer_id),
            CONSTRAINT fk_customers_city FOREIGN KEY (city_id)
                REFERENCES cities (city_id),
            CONSTRAINT fk_customers_vehicle FOREIGN KEY (preferred_vehicle_type_id)
                REFERENCES vehicle_types (vehicle_type_id)
        ) ENGINE=InnoDB
    """,
    "drivers": """
        CREATE TABLE IF NOT EXISTS drivers (
            driver_id               VARCHAR(12) NOT NULL,
            driver_age              TINYINT UNSIGNED,
            city_id                 SMALLINT,
            vehicle_type_id         SMALLINT,
            driver_experience_years TINYINT UNSIGNED,
            total_assigned_rides    SMALLINT UNSIGNED,
            accepted_rides          SMALLINT UNSIGNED,
            incomplete_rides        SMALLINT UNSIGNED,
            delay_count             SMALLINT UNSIGNED,
            acceptance_rate         DECIMAL(6,4),
            delay_rate              DECIMAL(6,4),
            avg_driver_rating       DECIMAL(3,2),
            avg_pickup_delay_min    DECIMAL(6,2),
            driver_delay_flag       TINYINT(1),
            PRIMARY KEY (driver_id),
            CONSTRAINT fk_drivers_city FOREIGN KEY (city_id)
                REFERENCES cities (city_id),
            CONSTRAINT fk_drivers_vehicle FOREIGN KEY (vehicle_type_id)
                REFERENCES vehicle_types (vehicle_type_id)
        ) ENGINE=InnoDB
    """,
    "time_features": """
        CREATE TABLE IF NOT EXISTS time_features (
            slot_datetime  DATETIME    NOT NULL,
            hour_of_day    TINYINT UNSIGNED NOT NULL,
            day_of_week    VARCHAR(10) NOT NULL,
            is_weekend     TINYINT(1)  NOT NULL,
            is_holiday     TINYINT(1)  NOT NULL,
            peak_time_flag TINYINT(1)  NOT NULL,
            season         VARCHAR(20) NOT NULL,
            PRIMARY KEY (slot_datetime)
        ) ENGINE=InnoDB
    """,
    "location_demand": """
        CREATE TABLE IF NOT EXISTS location_demand (
            demand_id            INT      NOT NULL AUTO_INCREMENT,
            city_id              SMALLINT NOT NULL,
            location_id          INT      NOT NULL,
            hour_of_day          TINYINT UNSIGNED NOT NULL,
            vehicle_type_id      SMALLINT NOT NULL,
            total_requests       INT UNSIGNED,
            completed_rides      INT UNSIGNED,
            cancelled_rides      INT UNSIGNED,
            avg_wait_time_min    DECIMAL(8,3),
            avg_surge_multiplier DECIMAL(4,2),
            demand_level         ENUM('Low','Medium','High'),
            PRIMARY KEY (demand_id),
            UNIQUE KEY uq_demand_slot (location_id, hour_of_day, vehicle_type_id),
            CONSTRAINT fk_demand_city FOREIGN KEY (city_id)
                REFERENCES cities (city_id),
            CONSTRAINT fk_demand_location FOREIGN KEY (location_id)
                REFERENCES locations (location_id),
            CONSTRAINT fk_demand_vehicle FOREIGN KEY (vehicle_type_id)
                REFERENCES vehicle_types (vehicle_type_id)
        ) ENGINE=InnoDB
    """,
    "bookings": """
        CREATE TABLE IF NOT EXISTS bookings (
            booking_id              VARCHAR(12) NOT NULL,
            booking_ts              DATETIME    NOT NULL,
            city_id                 SMALLINT    NOT NULL,
            pickup_location_id      INT         NOT NULL,
            drop_location_id        INT         NOT NULL,
            vehicle_type_id         SMALLINT    NOT NULL,
            customer_id             VARCHAR(12) NOT NULL,
            driver_id               VARCHAR(12) NOT NULL,
            ride_distance_km        DECIMAL(6,2),
            estimated_ride_time_min DECIMAL(6,2),
            actual_ride_time_min    DECIMAL(6,2) NULL,
            traffic_level           ENUM('Low','Medium','High'),
            weather_condition       ENUM('Clear','Rain','Heavy Rain'),
            base_fare               DECIMAL(8,2),
            surge_multiplier        DECIMAL(4,2),
            booking_value           DECIMAL(10,2),
            booking_status          ENUM('Completed','Cancelled','Incomplete') NOT NULL,
            incomplete_ride_reason  VARCHAR(50) NULL,
            PRIMARY KEY (booking_id),
            CONSTRAINT fk_bookings_city FOREIGN KEY (city_id)
                REFERENCES cities (city_id),
            CONSTRAINT fk_bookings_pickup FOREIGN KEY (pickup_location_id)
                REFERENCES locations (location_id),
            CONSTRAINT fk_bookings_drop FOREIGN KEY (drop_location_id)
                REFERENCES locations (location_id),
            CONSTRAINT fk_bookings_vehicle FOREIGN KEY (vehicle_type_id)
                REFERENCES vehicle_types (vehicle_type_id),
            CONSTRAINT fk_bookings_customer FOREIGN KEY (customer_id)
                REFERENCES customers (customer_id),
            CONSTRAINT fk_bookings_driver FOREIGN KEY (driver_id)
                REFERENCES drivers (driver_id)
        ) ENGINE=InnoDB
    """,
}

#: Each index is paired with the dashboard query that justifies it.
INDEX_STATEMENTS = [
    (
        "idx_bookings_ts",
        "bookings",
        "CREATE INDEX idx_bookings_ts ON bookings (booking_ts)",
        "Date-range filter applied on every dashboard page.",
    ),
    (
        "idx_bookings_city_status",
        "bookings",
        "CREATE INDEX idx_bookings_city_status ON bookings (city_id, booking_status)",
        "Cancellation rate by city; the composite avoids a full scan per city.",
    ),
    (
        "idx_bookings_customer",
        "bookings",
        "CREATE INDEX idx_bookings_customer ON bookings (customer_id)",
        "High-risk customer drill-down and leave-one-out history aggregates.",
    ),
    (
        "idx_bookings_driver",
        "bookings",
        "CREATE INDEX idx_bookings_driver ON bookings (driver_id)",
        "Driver reliability leaderboard.",
    ),
    (
        "idx_bookings_vehicle_status",
        "bookings",
        "CREATE INDEX idx_bookings_vehicle_status "
        "ON bookings (vehicle_type_id, booking_status)",
        "Cancellation split by vehicle type.",
    ),
    (
        "idx_bookings_pickup",
        "bookings",
        "CREATE INDEX idx_bookings_pickup ON bookings (pickup_location_id)",
        "Top pickup locations and route-pair aggregation.",
    ),
    (
        "idx_demand_city_hour",
        "location_demand",
        "CREATE INDEX idx_demand_city_hour ON location_demand (city_id, hour_of_day)",
        "Hourly demand heatmap.",
    ),
]

#: Dropped in reverse dependency order so foreign keys never block a rebuild.
DROP_STATEMENTS = [f"DROP TABLE IF EXISTS {table}" for table in reversed(TABLE_ORDER)]

#: Column order used when inserting each table, matching the DDL above.
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
    """Return index name, table and rationale for the README and viva."""
    return [
        {"index": name, "table": table, "rationale": reason}
        for name, table, _, reason in INDEX_STATEMENTS
    ]


def get_insert_columns(table: str) -> list[str]:
    """Return the insert column order for ``table``."""
    if table not in INSERT_COLUMNS:
        raise ValueError(f"Unknown table {table!r}.")
    return INSERT_COLUMNS[table]

"""MySQL schema definition and access layer.

One module owns the database: the DDL that defines the normalised tables and
their indexes, and the connection handling that runs statements against them.
Keeping the two together means a schema change and the code that applies it are
never out of step.

Every statement is parameterised. Values are passed to the driver, never
formatted into the SQL string.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import numpy as np
import pandas as pd

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
except ImportError:  # pragma: no cover - dependency missing
    mysql = None
    MySQLError = Exception

import config


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


logger = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    """Raised when a database operation fails, with actionable context."""


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

    settings = config.get_db_config(include_database=include_database)
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
            raise DatabaseError(f"Statement failed: {sql.strip()[:120]}... ({exc})") from exc
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
            raise DatabaseError(f"Query failed: {sql.strip()[:120]}... ({exc})") from exc
        finally:
            cursor.close()


def create_database() -> None:
    """Create the project database if it does not already exist."""
    with get_connection(include_database=False) as connection:
        cursor = connection.cursor()
        try:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {config.DB_NAME} "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            connection.commit()
            logger.info("Database %s ready", config.DB_NAME)
        finally:
            cursor.close()


def create_tables() -> None:
    """Create every table in foreign-key-safe order."""
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
    existing = {row["index"] for _, row in list_indexes().iterrows()} if table_exists(
        "bookings"
    ) else set()

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
        (config.DB_NAME,),
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
        (config.DB_NAME, name),
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
    status = {"connected": False, "database": config.DB_NAME, "tables": {}}
    try:
        with get_connection() as connection:
            status["connected"] = connection.is_connected()
            status["server_version"] = connection.server_info
        for table in TABLE_ORDER:
            status["tables"][table] = row_count(table) if table_exists(table) else None
    except DatabaseError as exc:
        status["error"] = str(exc)
    return status

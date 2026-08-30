"""MySQL access layer.

Every database call in the project goes through this module, so connection
handling and error translation exist in exactly one place.
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
from rapido import schema

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
            for table in schema.TABLE_ORDER:
                cursor.execute(schema.CREATE_TABLE_STATEMENTS[table])
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
            for name, _table, statement, _reason in schema.INDEX_STATEMENTS:
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
            for statement in schema.DROP_STATEMENTS:
                cursor.execute(statement)
            cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
            connection.commit()
            logger.info("Dropped %d tables", len(schema.DROP_STATEMENTS))
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
    columns = schema.get_insert_columns(table)
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
        for table in schema.TABLE_ORDER:
            status["tables"][table] = row_count(table) if table_exists(table) else None
    except DatabaseError as exc:
        status["error"] = str(exc)
    return status

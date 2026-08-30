"""Tests for the SQL query layer.

Skipped automatically when MySQL is unreachable, so the suite still runs on a
machine without the database.
"""

from __future__ import annotations

import inspect

import pytest

from rapido import db, queries


def _database_available() -> bool:
    """Return whether the project database is reachable and populated."""
    try:
        return bool(db.table_exists("bookings") and db.row_count("bookings"))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _database_available(), reason="MySQL not reachable or not loaded"
)

FILTERS = {
    "cities": ["Mumbai", "Delhi"],
    "date_from": "2025-06-01",
    "date_to": "2025-06-30",
}

QUERY_NAMES = [
    name
    for name, function in inspect.getmembers(queries, inspect.isfunction)
    if name.startswith("q_")
]


def test_query_count():
    """The dashboard is backed by a substantial named-query layer."""
    assert len(QUERY_NAMES) >= 20


@pytest.mark.parametrize("name", QUERY_NAMES)
def test_query_returns_data(name):
    """Every query runs and returns rows for a representative filter set."""
    function = getattr(queries, name)
    parameters = inspect.signature(function).parameters
    result = function(FILTERS) if "filters" in parameters else function()

    if isinstance(result, dict):
        assert result
    else:
        assert not result.empty, f"{name} returned no rows"


# --------------------------------------------------------------------------- #
# Filter builder
# --------------------------------------------------------------------------- #


def test_where_clause_empty_for_no_filters():
    """No filters produces no WHERE clause."""
    where, params = queries.build_where_clause(None)
    assert where == ""
    assert params == []


def test_where_clause_is_parameterised():
    """User values never appear inline in the SQL string."""
    where, params = queries.build_where_clause({"cities": ["Mumbai", "Delhi"]})
    assert "%s, %s" in where
    assert "Mumbai" not in where
    assert params == ["Mumbai", "Delhi"]


def test_where_clause_combines_filters():
    """Multiple filters are ANDed together with matching parameters."""
    where, params = queries.build_where_clause(
        {
            "cities": ["Mumbai"],
            "vehicle_types": ["Cab"],
            "date_from": "2025-01-01",
            "hour_from": 8,
        }
    )
    assert where.count("AND") == 3
    assert params == ["Mumbai", "Cab", "2025-01-01", 8]


def test_sql_injection_attempt_is_parameterised():
    """A hostile city name is carried as a parameter, not as SQL."""
    hostile = "Mumbai'; DROP TABLE bookings; --"
    where, params = queries.build_where_clause({"cities": [hostile]})
    assert "DROP TABLE" not in where
    assert params == [hostile]


def test_filters_actually_reduce_results():
    """Filtering to one city returns fewer rows than no filter."""
    unfiltered = queries.q_kpi_summary(None)["total_bookings"].iloc[0]
    filtered = queries.q_kpi_summary({"cities": ["Mumbai"]})["total_bookings"].iloc[0]
    assert 0 < filtered < unfiltered


def test_unsupported_category_rejected():
    """An unknown grouping column is rejected before reaching SQL."""
    with pytest.raises(ValueError, match="Unsupported category"):
        queries.q_cancellation_by_category(None, category="not_a_column")


def test_pagination_returns_page_size():
    """The explorer query honours its page size and offset."""
    first = queries.q_bookings_page(None, page=1, page_size=10)
    second = queries.q_bookings_page(None, page=2, page_size=10)
    assert len(first) == len(second) == 10
    assert set(first["booking_id"]) != set(second["booking_id"])


def test_kpi_summary_matches_known_totals():
    """Unfiltered totals match the documented dataset counts."""
    summary = queries.q_kpi_summary(None).iloc[0]
    assert int(summary["total_bookings"]) == 100_000
    assert int(summary["cancelled"]) == 23_284
    assert int(summary["incomplete"]) == 8_370


def test_reason_party_mapping_covers_source_values():
    """Every reason code in the data maps to a named accountable party."""
    reasons = queries.q_cancellation_reasons(None)["incomplete_ride_reason"]
    assert set(reasons) <= set(queries.REASON_PARTY)


def test_reasons_by_party_totals_match_plain_reasons():
    """Attribution regroups the same rides without gaining or losing any."""
    plain = queries.q_cancellation_reasons(None)["rides"].sum()
    attributed = queries.q_cancellation_reasons_by_party(None)["rides"].sum()
    assert int(plain) == int(attributed) == 8_370


def test_reasons_by_party_has_no_unknown_bucket():
    """No reason falls through to the Unknown fallback."""
    frame = queries.q_cancellation_reasons_by_party(None)
    assert "Unknown" not in set(frame["responsible_party"])

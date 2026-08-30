"""Tests for the cleaning layer."""

from __future__ import annotations

import pandas as pd
import pytest

import config
from rapido import cleaning


@pytest.fixture(scope="module")
def cleaned_bookings(bookings):
    """Bookings put through the full cleaning pipeline."""
    return cleaning.clean_bookings(bookings)


def test_standardise_columns():
    """Mixed-case and spaced headers become snake_case."""
    frame = pd.DataFrame({"Booking ID": [1], "Ride-Distance KM": [2.0]})
    result = cleaning.standardise_columns(frame)
    assert list(result.columns) == ["booking_id", "ride_distance_km"]


def test_booking_timestamp_parsed(cleaned_bookings):
    """booking_date and booking_time collapse into one valid timestamp."""
    assert "booking_ts" in cleaned_bookings.columns
    assert "booking_date" not in cleaned_bookings.columns
    assert "booking_time" not in cleaned_bookings.columns
    assert cleaned_bookings["booking_ts"].isna().sum() == 0
    assert cleaned_bookings["booking_ts"].dt.year.eq(2025).all()


def test_structural_nulls_preserved(cleaned_bookings):
    """Post-outcome columns are never imputed."""
    assert cleaned_bookings["actual_ride_time_min"].isna().sum() == 31_654
    assert cleaned_bookings["incomplete_ride_reason"].isna().sum() == 91_630


def test_no_rows_lost(bookings, cleaned_bookings):
    """Cleaning is non-destructive for this dataset."""
    assert len(cleaned_bookings) == len(bookings)


def test_locations_namespaced(cleaned_bookings):
    """Location keys are city-qualified, giving 250 distinct zones."""
    assert cleaned_bookings["pickup_location_key"].nunique() == 250
    assert cleaned_bookings["pickup_location_key"].str.contains("::").all()


def test_value_ranges_pass(cleaned_bookings):
    """Every numeric column sits inside its documented plausible range."""
    report = cleaning.validate_value_ranges(cleaned_bookings, cleaning.BOOKING_RANGES)
    assert (report["status"] == "PASS").all(), report[report["status"] == "FAIL"]


def test_referential_integrity(cleaned_bookings, customers, drivers):
    """No orphan foreign keys survive cleaning."""
    result = cleaning.validate_referential_integrity(
        cleaned_bookings,
        cleaning.clean_customers(customers),
        cleaning.clean_drivers(drivers),
    )
    assert result == {"orphan_customers": 0, "orphan_drivers": 0}


def test_detect_outliers_iqr():
    """A planted extreme value is detected by the IQR fence."""
    frame = pd.DataFrame({"value": [10, 11, 12, 11, 10, 500]})
    mask = cleaning.detect_outliers_iqr(frame, "value")
    assert mask.sum() == 1
    assert frame.loc[mask, "value"].iloc[0] == 500


def test_cap_outliers_bounds():
    """Capping pulls extremes inside the fence without dropping rows."""
    frame = pd.DataFrame({"value": [10, 11, 12, 11, 10, 500]})
    capped = cleaning.cap_outliers(frame, ["value"])
    assert len(capped) == len(frame)
    assert capped["value"].max() < 500


def test_drop_duplicates_by_key():
    """Duplicate keys collapse to the first occurrence."""
    frame = pd.DataFrame({"id": ["a", "a", "b"], "value": [1, 2, 3]})
    result = cleaning.drop_duplicates_by_key(frame, "id")
    assert len(result) == 2
    assert result.loc[result["id"] == "a", "value"].iloc[0] == 1


def test_clean_all_returns_every_dataset(raw_frames):
    """clean_all covers all five datasets."""
    cleaned = cleaning.clean_all(raw_frames)
    assert set(cleaned) == set(config.RAW_FILES)


def test_clean_all_rejects_missing_input(raw_frames):
    """A missing source frame fails loudly rather than silently."""
    partial = {k: v for k, v in raw_frames.items() if k != "drivers"}
    with pytest.raises(KeyError, match="drivers"):
        cleaning.clean_all(partial)

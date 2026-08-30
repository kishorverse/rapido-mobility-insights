"""Tests for raw loading and the data-quality assumptions the build relies on.

These lock in the findings recorded in docs/data_quality_report.md so that a
change in the source data fails loudly instead of silently corrupting a model.
"""

from __future__ import annotations

import pytest

import config
from rapido import io


def test_all_raw_files_load(raw_frames):
    """All five source files load and are non-empty."""
    assert set(raw_frames) == set(config.RAW_FILES)
    for name, frame in raw_frames.items():
        assert not frame.empty, f"{name} loaded empty"


def test_unknown_dataset_raises():
    """An unknown dataset name is rejected before touching the filesystem."""
    with pytest.raises(ValueError, match="Unknown dataset"):
        io._read_csv("payments")


def test_missing_file_raises(tmp_path):
    """A missing source path raises a helpful FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        io.load_bookings(tmp_path / "nope.csv")


def test_booking_ids_unique(bookings):
    """booking_id is a valid primary key."""
    assert bookings["booking_id"].is_unique


def test_referential_integrity(bookings, customers, drivers):
    """Every booking foreign key resolves to a dimension row."""
    assert bookings["customer_id"].isin(customers["customer_id"]).all()
    assert bookings["driver_id"].isin(drivers["driver_id"]).all()


def test_actual_ride_time_is_post_outcome(bookings):
    """actual_ride_time_min is null exactly for non-Completed rides.

    This is the leakage guard: if it ever stops holding, the outcome model's
    blocklist assumptions need revisiting.
    """
    completed = bookings["booking_status"] == "Completed"
    assert bookings.loc[completed, "actual_ride_time_min"].notna().all()
    assert bookings.loc[~completed, "actual_ride_time_min"].isna().all()


def test_incomplete_reason_only_for_incomplete(bookings):
    """incomplete_ride_reason is populated only for Incomplete rides."""
    incomplete = bookings["booking_status"] == "Incomplete"
    assert bookings.loc[incomplete, "incomplete_ride_reason"].notna().all()
    assert bookings.loc[~incomplete, "incomplete_ride_reason"].isna().all()


def test_fare_is_surge_formula(bookings):
    """booking_value stays within 5% of base_fare * surge_multiplier."""
    ratio = bookings["booking_value"] / (
        bookings["base_fare"] * bookings["surge_multiplier"]
    )
    assert ratio.between(0.94, 1.06).all()


def test_dimension_flags_are_thresholds(customers, drivers):
    """The risk flags are exact cut-offs, so they cannot be model targets."""
    assert customers.loc[customers["customer_cancel_flag"] == 0, "cancellation_rate"].max() <= 0.20
    assert customers.loc[customers["customer_cancel_flag"] == 1, "cancellation_rate"].min() > 0.20
    assert drivers.loc[drivers["driver_delay_flag"] == 0, "delay_rate"].max() <= 0.10
    assert drivers.loc[drivers["driver_delay_flag"] == 1, "delay_rate"].min() > 0.10


def test_categorical_domains(bookings):
    """Categorical columns contain only the documented values."""
    assert set(bookings["city"].cat.categories) == set(config.CITIES)
    assert set(bookings["vehicle_type"].cat.categories) == set(config.VEHICLE_TYPES)
    assert set(bookings["traffic_level"].cat.categories) == set(config.TRAFFIC_LEVELS)
    assert set(bookings["weather_condition"].cat.categories) == set(
        config.WEATHER_CONDITIONS
    )


def test_no_payment_column_anywhere(raw_frames):
    """The spec's payment-method analysis has no supporting column.

    Documented deviation: no source file carries payment information.
    """
    for frame in raw_frames.values():
        assert not [c for c in frame.columns if "payment" in c.lower()]


def test_location_codes_shared_across_cities(bookings):
    """Loc_N codes repeat in every city, so locations must be city-namespaced."""
    per_city = bookings.groupby("city", observed=True)["pickup_location"].nunique()
    assert per_city.nunique() == 1
    assert bookings["pickup_location"].nunique() == per_city.iloc[0]


def test_profile_dataframe_shape(bookings):
    """profile_dataframe returns one row per column."""
    profile = io.profile_dataframe(bookings, "bookings")
    assert len(profile) == bookings.shape[1]
    assert {"dataset", "column", "dtype", "nulls", "unique"} <= set(profile.columns)

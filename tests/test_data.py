"""Tests for the data pipeline: loading, cleaning and feature engineering.

These follow the path a row takes from CSV to model input. The loading tests
double as assertions about the dataset itself - the leakage traps documented in
the README are pinned here, so if the source data ever changes shape, these fail
before any model is trained on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from rapido import cleaning, features, io


# --------------------------------------------------------------------------- #
# Loading and raw-data invariants
#
# What the source files must look like before anything touches them.
# --------------------------------------------------------------------------- #


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


def test_raw_referential_integrity(bookings, customers, drivers):
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


# --------------------------------------------------------------------------- #
# Cleaning
#
# Structure preserved, nulls handled deliberately, no rows lost.
# --------------------------------------------------------------------------- #


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


def test_cleaned_referential_integrity(cleaned_bookings, customers, drivers):
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


# --------------------------------------------------------------------------- #
# Feature engineering
#
# Derived columns, scores, and the temporal guard on prior history.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def feature_table(raw_frames):
    """The master feature table, built once for the module."""
    from rapido import cleaning

    return features.build_feature_table(cleaning.clean_all(raw_frames))


def test_rush_hour_flag():
    """Rush-hour hours are flagged and off-peak hours are not."""
    frame = pd.DataFrame({"hour_of_day": [3, 9, 13, 18, 23]})
    result = features.add_rush_hour_flag(frame)
    assert result["rush_hour_flag"].tolist() == [0, 1, 0, 1, 0]


def test_long_distance_flag():
    """The distance threshold is applied strictly."""
    frame = pd.DataFrame({"ride_distance_km": [1.0, 15.0, 15.1, 30.0]})
    result = features.add_long_distance_flag(frame, threshold=15.0)
    assert result["long_distance_flag"].tolist() == [0, 0, 1, 1]


def test_city_route_pair_is_city_qualified():
    """Identical location codes in different cities yield different routes."""
    frame = pd.DataFrame(
        {
            "city": ["Delhi", "Mumbai"],
            "pickup_location": ["Loc_1", "Loc_1"],
            "drop_location": ["Loc_2", "Loc_2"],
        }
    )
    result = features.add_city_route_pair(frame)
    assert result["city_route_pair"].nunique() == 2
    assert result["location_pair"].nunique() == 1


def test_same_zone_flag():
    """A ride ending where it started is flagged."""
    frame = pd.DataFrame(
        {"pickup_location": ["Loc_1", "Loc_1"], "drop_location": ["Loc_1", "Loc_2"]}
    )
    result = features.add_same_zone_flag(frame)
    assert result["is_same_zone"].tolist() == [1, 0]


def test_driver_reliability_score_bounds():
    """The score stays in [0, 100] and rewards the better driver."""
    drivers = pd.DataFrame(
        {
            "acceptance_rate": [1.0, 0.0],
            "delay_rate": [0.0, 1.0],
            "avg_driver_rating": [5.0, 1.0],
        }
    )
    score = features.compute_driver_reliability_score(drivers)
    assert score.iloc[0] == pytest.approx(100.0)
    assert score.iloc[1] == pytest.approx(0.0)


def test_customer_loyalty_score_bounds():
    """The loyalty score stays within [0, 100]."""
    customers = pd.DataFrame(
        {
            "total_bookings": [50, 1],
            "completed_rides": [50, 0],
            "avg_customer_rating": [5.0, 1.0],
        }
    )
    score = features.compute_customer_loyalty_score(customers)
    assert score.between(0, 100).all()
    assert score.iloc[0] > score.iloc[1]


def test_prior_history_excludes_current_row():
    """Prior counts are strictly backward-looking on a hand-built example."""
    frame = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C1", "C2"],
            "booking_ts": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-01"]
            ),
            "booking_status": ["Cancelled", "Completed", "Cancelled", "Completed"],
        }
    )
    result = features.add_prior_customer_history(frame)

    assert result["cust_prior_rides"].tolist() == [0, 1, 2, 0]
    assert result["cust_prior_cancelled"].tolist() == [0, 1, 1, 0]
    # Row 3 sees one cancellation out of two prior rides; its own cancellation
    # must not be counted.
    assert result["cust_prior_cancel_rate"].iloc[2] == pytest.approx(0.5)
    assert np.isnan(result["cust_prior_cancel_rate"].iloc[0])
    assert result["cust_is_first_ride"].tolist() == [1, 0, 0, 1]


def test_prior_history_preserves_row_order():
    """History columns realign to the original row order after sorting."""
    frame = pd.DataFrame(
        {
            "customer_id": ["C2", "C1", "C1"],
            "booking_ts": pd.to_datetime(["2025-01-05", "2025-01-01", "2025-01-02"]),
            "booking_status": ["Completed", "Cancelled", "Completed"],
            "marker": [10, 20, 30],
        }
    )
    result = features.add_prior_customer_history(frame)
    assert result["marker"].tolist() == [10, 20, 30]
    assert result["cust_prior_rides"].tolist() == [0, 0, 1]


def test_first_ride_null_counts(feature_table):
    """Exactly one first ride per customer and per driver lacks a prior rate."""
    assert feature_table["cust_prior_cancel_rate"].isna().sum() == 10_000
    assert feature_table["drv_prior_cancel_rate"].isna().sum() == 5_000
    assert feature_table["cust_is_first_ride"].sum() == 10_000
    assert feature_table["drv_is_first_ride"].sum() == 5_000


def test_whole_period_aggregates_excluded(feature_table):
    """The leaky whole-period rate columns never reach the feature table."""
    banned = {
        "cancellation_rate",
        "cancelled_rides",
        "delay_rate",
        "delay_count",
        "customer_cancel_flag",
        "driver_delay_flag",
        "total_bookings",
        "total_assigned_rides",
    }
    assert not banned & set(feature_table.columns)


def test_merges_did_not_duplicate_rows(feature_table, bookings):
    """All four joins are many-to-one, so the row count is unchanged."""
    assert len(feature_table) == len(bookings)
    assert feature_table["booking_id"].is_unique


def test_demand_merge_populated(feature_table):
    """The zone-demand join resolves for every booking."""
    assert feature_table["zone_demand_level"].notna().all()
    assert feature_table["zone_avg_wait_min"].notna().all()


def test_holiday_column_dropped(feature_table):
    """The zero-variance is_holiday column is not carried forward."""
    assert "is_holiday" not in feature_table.columns

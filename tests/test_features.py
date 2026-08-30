"""Tests for feature engineering, with emphasis on the leakage-safe history."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rapido import features


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

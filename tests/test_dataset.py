"""Tests for the leakage guard and dataset construction.

These are the highest-value tests in the suite: they are what stops a blocked
column from silently reaching a model fit.
"""

from __future__ import annotations

import pandas as pd
import pytest

import config
from rapido.models import dataset


@pytest.fixture(scope="module")
def feature_table():
    """The cached feature table."""
    from rapido import io

    return io.load_processed("features")


# --------------------------------------------------------------------------- #
# Leakage guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("column", config.POST_OUTCOME_COLUMNS)
@pytest.mark.parametrize("target", list(dataset.LEAKY_BY_TARGET))
def test_post_outcome_columns_blocked_everywhere(column, target):
    """Every post-outcome column is rejected for every target."""
    features = pd.DataFrame({column: [1, 2, 3]})
    with pytest.raises(dataset.LeakageError, match=column):
        dataset.assert_no_leakage(features, target)


@pytest.mark.parametrize(
    "column", ["base_fare", "booking_value", "fare_per_km", "fare_per_min"]
)
def test_fare_columns_blocked_for_fare_target(column):
    """Fare-derived columns are rejected for the fare model."""
    features = pd.DataFrame({column: [1.0, 2.0]})
    with pytest.raises(dataset.LeakageError, match=column):
        dataset.assert_no_leakage(features, "fare")


def test_fare_columns_allowed_for_outcome_target():
    """The quoted fare is legitimate input for outcome models.

    It is known before the trip starts, and price sensitivity is a real
    cancellation driver.
    """
    features = pd.DataFrame({"booking_value": [100.0], "base_fare": [80.0]})
    dataset.assert_no_leakage(features, "outcome")


@pytest.mark.parametrize("column", dataset.IDENTIFIER_COLUMNS)
def test_identifiers_blocked(column):
    """Identifier and high-cardinality key columns are rejected."""
    features = pd.DataFrame({column: ["a", "b"]})
    with pytest.raises(dataset.LeakageError, match="Identifier"):
        dataset.assert_no_leakage(features, "outcome")


def test_unknown_target_rejected():
    """An unrecognised target name fails loudly."""
    with pytest.raises(ValueError, match="Unknown target"):
        dataset.assert_no_leakage(pd.DataFrame({"x": [1]}), "not_a_model")


def test_clean_matrix_passes():
    """A legitimate feature matrix passes the guard."""
    features = pd.DataFrame({"ride_distance_km": [5.0], "city": ["Mumbai"]})
    dataset.assert_no_leakage(features, "outcome")


# --------------------------------------------------------------------------- #
# Dataset builders
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "target", ["outcome", "fare", "customer_risk", "driver_risk"]
)
def test_builders_produce_clean_matrices(feature_table, target):
    """Every builder returns aligned, leakage-free data."""
    features, response = dataset.build_dataset(feature_table, target)
    assert len(features) == len(response) == len(feature_table)
    assert not features.empty
    dataset.assert_no_leakage(features, target)


def test_fare_dataset_excludes_base_fare(feature_table):
    """The deployed fare model never sees base_fare."""
    features, _ = dataset.build_fare_dataset(feature_table)
    assert "base_fare" not in features.columns
    assert "booking_value" not in features.columns


def test_fare_ablation_includes_base_fare(feature_table):
    """The ablation deliberately re-adds base_fare."""
    features, _ = dataset.build_fare_dataset(feature_table, include_base_fare=True)
    assert "base_fare" in features.columns


def test_outcome_target_has_three_classes(feature_table):
    """The outcome target keeps all three classes."""
    _, response = dataset.build_outcome_dataset(feature_table)
    assert set(response.unique()) == {"Completed", "Cancelled", "Incomplete"}


def test_binary_targets_are_binary(feature_table):
    """Both risk targets are 0/1 with the expected prevalence."""
    _, cancelled = dataset.build_customer_risk_dataset(feature_table)
    _, incomplete = dataset.build_driver_risk_dataset(feature_table)

    assert set(cancelled.unique()) <= {0, 1}
    assert set(incomplete.unique()) <= {0, 1}
    assert cancelled.sum() == 23_284
    assert incomplete.sum() == 8_370


def test_split_is_stratified(feature_table):
    """Class balance is preserved across the train/test split."""
    features, response = dataset.build_customer_risk_dataset(feature_table)
    x_train, x_test, y_train, y_test = dataset.split_train_test(features, response)

    assert len(x_train) == 80_000
    assert len(x_test) == 20_000
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_split_is_reproducible(feature_table):
    """The same seed yields the same split."""
    features, response = dataset.build_fare_dataset(feature_table)
    first = dataset.split_train_test(features, response, stratify=False)[1]
    second = dataset.split_train_test(features, response, stratify=False)[1]
    assert first.index.equals(second.index)


def test_feature_types_partition_columns(feature_table):
    """Numeric and categorical lists together cover every column exactly once."""
    features, _ = dataset.build_outcome_dataset(feature_table)
    numeric, categorical = dataset.get_feature_types(features)
    assert set(numeric) | set(categorical) == set(features.columns)
    assert not set(numeric) & set(categorical)


def test_zero_variance_column_absent(feature_table):
    """is_holiday never appears in a feature matrix."""
    features, _ = dataset.build_outcome_dataset(feature_table)
    assert "is_holiday" not in features.columns

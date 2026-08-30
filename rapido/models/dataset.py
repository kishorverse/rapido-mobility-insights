"""Feature-matrix construction with an enforced leakage guard.

Each model declares which columns it must never see. :func:`assert_no_leakage`
raises rather than warns, so a blocked column cannot silently reach a fit.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config

logger = logging.getLogger(__name__)


class LeakageError(ValueError):
    """Raised when a feature matrix contains a column the target forbids."""


#: Identifiers and high-cardinality keys: no predictive value, high overfit risk.
IDENTIFIER_COLUMNS = [
    "booking_id",
    "customer_id",
    "driver_id",
    "booking_ts",
    "pickup_location",
    "drop_location",
    "pickup_location_key",
    "drop_location_key",
    "city_route_pair",
    "location_pair",
]

#: Columns each target must never see.
#:
#: ``booking_value`` and ``base_fare`` are *quoted* before the trip, so they are
#: legitimate inputs to the outcome and risk models -- price sensitivity is a
#: real cancellation driver. They are blocked only for the fare model, where
#: ``booking_value`` is the target and ``base_fare`` reproduces it to within 5%.
LEAKY_BY_TARGET: dict[str, list[str]] = {
    "outcome": list(config.POST_OUTCOME_COLUMNS),
    "fare": list(config.POST_OUTCOME_COLUMNS)
    + ["booking_value", "base_fare", "fare_per_km", "fare_per_min"],
    "customer_risk": list(config.POST_OUTCOME_COLUMNS),
    "driver_risk": list(config.POST_OUTCOME_COLUMNS),
}

#: Numeric features shared by every model.
BASE_NUMERIC = [
    "ride_distance_km",
    "estimated_ride_time_min",
    "surge_multiplier",
    "hour_of_day",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "rush_hour_flag",
    "is_night_ride",
    "long_distance_flag",
    "is_same_zone",
    "expected_speed_kmph",
    "adverse_conditions_flag",
    "bad_weather_flag",
    "high_traffic_flag",
    "peak_time_flag",
    "zone_total_requests",
    "zone_avg_wait_min",
    "zone_avg_surge",
    "surge_vs_zone_ratio",
    "zone_demand_pressure",
    "customer_age",
    "customer_signup_days_ago",
    "avg_customer_rating",
    "customer_loyalty_score",
    "driver_age",
    "driver_experience_years",
    "acceptance_rate",
    "avg_driver_rating",
    "avg_pickup_delay_min",
    "driver_reliability_score",
    "cust_prior_rides",
    "cust_prior_cancelled",
    "cust_prior_incomplete",
    "cust_prior_cancel_rate",
    "cust_prior_incomplete_rate",
    "cust_prior_completion_rate",
    "cust_is_first_ride",
    "drv_prior_rides",
    "drv_prior_cancelled",
    "drv_prior_incomplete",
    "drv_prior_cancel_rate",
    "drv_prior_incomplete_rate",
    "drv_prior_completion_rate",
    "drv_is_first_ride",
]

#: Numeric features available only when fare is not the target.
FARE_NUMERIC = ["base_fare", "booking_value", "fare_per_km", "fare_per_min"]

#: Categorical features shared by every model.
BASE_CATEGORICAL = [
    "city",
    "vehicle_type",
    "traffic_level",
    "weather_condition",
    "day_of_week",
    "time_of_day_band",
    "distance_band",
    "surge_bucket",
    "season",
    "zone_demand_level",
    "customer_gender",
    "preferred_vehicle_type",
    "customer_tenure_bucket",
]


def assert_no_leakage(features: pd.DataFrame, target: str) -> None:
    """Raise if ``features`` contains a column blocked for ``target``.

    Args:
        features: The candidate feature matrix.
        target: One of ``outcome``, ``fare``, ``customer_risk``, ``driver_risk``.

    Raises:
        LeakageError: If any blocked column is present.
        ValueError: If ``target`` is unknown.
    """
    if target not in LEAKY_BY_TARGET:
        raise ValueError(
            f"Unknown target {target!r}. Expected one of {sorted(LEAKY_BY_TARGET)}."
        )

    blocked = set(LEAKY_BY_TARGET[target]) & set(features.columns)
    if blocked:
        raise LeakageError(
            f"Target {target!r} must not see {sorted(blocked)}. "
            "See docs/PROJECT_PLAN.md section 1.2."
        )

    identifiers = set(IDENTIFIER_COLUMNS) & set(features.columns)
    if identifiers:
        raise LeakageError(
            f"Identifier columns leaked into the {target!r} matrix: {sorted(identifiers)}"
        )


def _select_features(
    frame: pd.DataFrame, target: str, include_fare_columns: bool
) -> pd.DataFrame:
    """Assemble and validate the feature matrix for a target."""
    numeric = list(BASE_NUMERIC)
    if include_fare_columns:
        numeric += FARE_NUMERIC

    columns = [column for column in numeric + BASE_CATEGORICAL if column in frame.columns]
    features = frame[columns].copy()

    for column in BASE_CATEGORICAL:
        if column in features.columns:
            features[column] = features[column].astype(str)

    assert_no_leakage(features, target)
    return features


def get_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split a feature matrix into numeric and categorical column names."""
    numeric = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    return numeric, categorical


def build_outcome_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Multi-class dataset predicting ``booking_status``."""
    features = _select_features(frame, "outcome", include_fare_columns=True)
    target = frame["booking_status"].astype(str).rename("booking_status")
    return features, target


def build_fare_dataset(
    frame: pd.DataFrame, include_base_fare: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """Regression dataset predicting ``booking_value``.

    Args:
        include_base_fare: When ``True``, returns the deliberately leaky
            ablation used to demonstrate that fare is a formula. The honest
            pre-quote model uses the default.
    """
    features = _select_features(frame, "fare", include_fare_columns=False)
    if include_base_fare:
        features = features.copy()
        features["base_fare"] = frame["base_fare"]
    target = frame["booking_value"].rename("booking_value")
    return features, target


def build_customer_risk_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Binary dataset predicting whether a booking is cancelled."""
    features = _select_features(frame, "customer_risk", include_fare_columns=True)
    target = (
        (frame["booking_status"].astype(str) == "Cancelled")
        .astype(int)
        .rename("is_cancelled")
    )
    return features, target


def build_driver_risk_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Binary dataset predicting a driver-caused delay or incomplete ride.

    Positive class: the ride ended Incomplete. Roughly 8.4% of bookings, so the
    estimator is trained with balanced class weights.
    """
    features = _select_features(frame, "driver_risk", include_fare_columns=True)
    target = (
        (frame["booking_status"].astype(str) == "Incomplete")
        .astype(int)
        .rename("is_incomplete")
    )
    return features, target


DATASET_BUILDERS = {
    "outcome": build_outcome_dataset,
    "fare": build_fare_dataset,
    "customer_risk": build_customer_risk_dataset,
    "driver_risk": build_driver_risk_dataset,
}


def build_dataset(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Dispatch to the dataset builder for ``target``."""
    if target not in DATASET_BUILDERS:
        raise ValueError(f"Unknown target {target!r}.")
    return DATASET_BUILDERS[target](frame)


def split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
    stratify: bool = True,
    test_size: float | None = None,
):
    """80/20 split, stratified for classification targets."""
    return train_test_split(
        features,
        target,
        test_size=config.TEST_SIZE if test_size is None else test_size,
        random_state=config.RANDOM_STATE,
        stratify=target if stratify else None,
    )


def describe_dataset(features: pd.DataFrame, target: pd.Series) -> dict:
    """Summarise a built dataset for logging and the model report."""
    numeric, categorical = get_feature_types(features)
    summary = {
        "rows": len(features),
        "n_features": features.shape[1],
        "n_numeric": len(numeric),
        "n_categorical": len(categorical),
        "target": target.name,
    }
    if target.dtype == object or target.nunique() <= 10:
        summary["class_balance"] = (
            target.value_counts(normalize=True).round(4).to_dict()
        )
    else:
        summary["target_mean"] = round(float(target.mean()), 2)
        summary["target_std"] = round(float(target.std()), 2)
    return summary

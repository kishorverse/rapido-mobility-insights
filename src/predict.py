"""Prediction serving: turn a dict of user inputs into a scored booking.

The deployment layer. It loads the pipelines that :mod:`train_models` wrote,
fills in whatever the caller did not supply, keeps the engineered fields
internally consistent, and returns a prediction with the context an operator
needs to act on it. Nothing here fits a model.

Usage:
    python src/predict.py --distance 10 --vehicle Cab --surge 1.8 \
        --traffic High --weather "Heavy Rain" --hour 18
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_preprocessing as dp
import train_models as tm

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pricing and risk bands
# --------------------------------------------------------------------------- #

#: Per-vehicle tariff recovered from the data (R^2 = 1.000000).
#: base_fare = flagfall + rate_per_km * distance
TARIFF = {
    "Bike": {"flagfall": 20.0, "per_km": 8.0},
    "Auto": {"flagfall": 40.0, "per_km": 12.0},
    "Cab": {"flagfall": 80.0, "per_km": 18.0},
}

RISK_BANDS = [(0.30, "Low"), (0.60, "Medium"), (1.01, "High")]


def risk_level(probability: float) -> str:
    """Map a probability to a Low / Medium / High band."""
    for threshold, label in RISK_BANDS:
        if probability < threshold:
            return label
    return "High"


def estimate_base_fare(vehicle_type: str, distance_km: float) -> float:
    """Apply the recovered tariff formula."""
    tariff = TARIFF.get(vehicle_type, TARIFF["Auto"])
    return tariff["flagfall"] + tariff["per_km"] * distance_km


# --------------------------------------------------------------------------- #
# Building a single feature row
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=1)
def _feature_defaults() -> dict:
    """Median and modal values for every model feature, computed once."""
    frame = dp.load_model_data()
    defaults: dict = {}

    for column in tm.BASE_NUMERIC + tm.FARE_NUMERIC:
        if column in frame.columns:
            value = frame[column].median()
            defaults[column] = float(value) if pd.notna(value) else 0.0

    for column in tm.BASE_CATEGORICAL:
        if column in frame.columns:
            mode = frame[column].astype(str).mode()
            defaults[column] = mode.iloc[0] if len(mode) else "Unknown"

    return defaults


def get_defaults() -> dict:
    """Public accessor for the cached feature defaults."""
    return dict(_feature_defaults())


def _time_band(hour: int) -> str:
    """Bucket an hour into its display band."""
    if hour <= 5:
        return "Night"
    if hour <= 11:
        return "Morning"
    if hour <= 16:
        return "Afternoon"
    if hour <= 20:
        return "Evening"
    return "Late Evening"


def _distance_band(distance: float) -> str:
    """Bucket a distance into its display band."""
    if distance <= 5:
        return "Short"
    return "Medium" if distance <= 15 else "Long"


def _surge_bucket(surge: float) -> str:
    """Bucket a surge multiplier into its display band."""
    if surge <= 1.0:
        return "None"
    if surge <= 1.5:
        return "Low"
    return "Medium" if surge <= 2.0 else "High"


def _derive_dependent_fields(row: dict) -> dict:
    """Recompute the engineered fields that depend on user inputs.

    Without this, a user changing distance would leave ``long_distance_flag``
    and ``expected_speed_kmph`` stale, and the model would see a contradictory
    row.
    """
    distance = float(row.get("ride_distance_km", 5.0))
    estimated = float(row.get("estimated_ride_time_min", 20.0))
    hour = int(row.get("hour_of_day", 9))
    surge = float(row.get("surge_multiplier", 1.0))
    vehicle = str(row.get("vehicle_type", "Auto"))
    traffic = str(row.get("traffic_level", "Medium"))
    weather = str(row.get("weather_condition", "Clear"))

    base_fare = estimate_base_fare(vehicle, distance)
    booking_value = base_fare * surge

    derived = {
        "rush_hour_flag": int(hour in dp.RUSH_HOURS),
        "is_night_ride": int(hour in dp.NIGHT_HOURS),
        "long_distance_flag": int(distance > dp.LONG_DISTANCE_KM),
        "expected_speed_kmph": distance / (estimated / 60) if estimated else np.nan,
        "high_traffic_flag": int(traffic == "High"),
        "bad_weather_flag": int(weather in {"Rain", "Heavy Rain"}),
        "adverse_conditions_flag": int(traffic == "High" and weather == "Heavy Rain"),
        "peak_time_flag": int(hour in dp.RUSH_HOURS),
        "base_fare": base_fare,
        "booking_value": booking_value,
        "fare_per_km": booking_value / distance if distance else np.nan,
        "fare_per_min": booking_value / estimated if estimated else np.nan,
        "time_of_day_band": _time_band(hour),
        "distance_band": _distance_band(distance),
        "surge_bucket": _surge_bucket(surge),
    }

    zone_surge = row.get("zone_avg_surge") or np.nan
    derived["surge_vs_zone_ratio"] = surge / zone_surge if zone_surge else np.nan
    return derived


def build_feature_row(inputs: dict, target: str) -> pd.DataFrame:
    """Build a single-row feature matrix for a model.

    Args:
        inputs: User-supplied values, using feature column names.
        target: Which model the row is destined for.

    Returns:
        A one-row DataFrame carrying every column that model expects.
    """
    row = get_defaults()
    row.update({key: value for key, value in inputs.items() if value is not None})

    derived = _derive_dependent_fields(row)
    row.update(derived)

    numeric = list(tm.BASE_NUMERIC)
    if target != "fare":
        numeric += tm.FARE_NUMERIC

    columns = numeric + tm.BASE_CATEGORICAL
    frame = pd.DataFrame([{column: row.get(column) for column in columns}])

    for column in tm.BASE_CATEGORICAL:
        frame[column] = frame[column].astype(str)

    return frame


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #


def predict_outcome(inputs: dict) -> dict:
    """Predict the ride outcome with class probabilities."""
    model, metadata = tm.load_model(dp.MODEL_NAMES["outcome"])
    row = build_feature_row(inputs, "outcome")
    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    classes = list(model.named_steps["model"].classes_)
    return {
        "prediction": str(prediction),
        "probabilities": {
            str(label): round(float(value), 4)
            for label, value in zip(classes, probabilities)
        },
        "confidence": round(float(np.max(probabilities)), 4),
        "metadata": metadata,
        "row": row,
    }


def predict_fare(inputs: dict) -> dict:
    """Predict the booking value before confirmation."""
    model, metadata = tm.load_model(dp.MODEL_NAMES["fare"])
    row = build_feature_row(inputs, "fare")
    predicted = float(model.predict(row)[0])
    metrics = metadata.get("metrics", {})
    tolerance = metrics.get("mape_pct", 5.0) / 100
    return {
        "predicted_fare": round(predicted, 2),
        "lower_bound": round(predicted * (1 - tolerance), 2),
        "upper_bound": round(predicted * (1 + tolerance), 2),
        "formula_fare": round(
            estimate_base_fare(
                str(inputs.get("vehicle_type", "Auto")),
                float(inputs.get("ride_distance_km", 5.0)),
            )
            * float(inputs.get("surge_multiplier", 1.0)),
            2,
        ),
        "metadata": metadata,
        "row": row,
    }


def _predict_binary(model_key: str, inputs: dict, label: str) -> dict:
    """Shared binary-risk prediction path."""
    model, metadata = tm.load_model(dp.MODEL_NAMES[model_key])
    row = build_feature_row(inputs, model_key)
    probability = float(model.predict_proba(row)[0][1])
    return {
        "probability": round(probability, 4),
        "risk_level": risk_level(probability),
        "label": label,
        "metadata": metadata,
        "row": row,
    }


def predict_customer_risk(inputs: dict) -> dict:
    """Predict the probability that a booking is cancelled."""
    return _predict_binary("customer_risk", inputs, "Cancellation risk")


def predict_driver_risk(inputs: dict) -> dict:
    """Predict the probability of a driver-caused delay or incomplete ride."""
    return _predict_binary("driver_risk", inputs, "Delay / incomplete risk")


def predict_all(inputs: dict) -> dict:
    """Run every available model against one booking."""
    availability = tm.available_models()
    results: dict = {}
    if availability["outcome"]:
        results["outcome"] = predict_outcome(inputs)
    if availability["fare"]:
        results["fare"] = predict_fare(inputs)
    if availability["customer_risk"]:
        results["customer_risk"] = predict_customer_risk(inputs)
    if availability["driver_risk"]:
        results["driver_risk"] = predict_driver_risk(inputs)
    return results


def explain_prediction(
    metadata: dict, row: pd.DataFrame, top_n: int = 6
) -> pd.DataFrame:
    """Show which of the model's top features this booking carries."""
    top_features = metadata.get("top_features")
    if top_features is None:
        return pd.DataFrame(columns=["feature", "value", "importance"])
    importance = (
        pd.DataFrame(top_features)
        if not isinstance(top_features, pd.DataFrame)
        else top_features
    )
    return tm.top_drivers_for_prediction(None, row, importance, top_n=top_n)


def available_models() -> dict[str, bool]:
    """Report which trained artefacts are present on disk."""
    return tm.available_models()


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """Score one booking described on the command line."""
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Score a hypothetical booking against the trained models."
    )
    parser.add_argument("--city", default="Mumbai", choices=dp.CITIES)
    parser.add_argument("--vehicle", default="Cab", choices=dp.VEHICLE_TYPES)
    parser.add_argument("--distance", type=float, default=10.0, help="km")
    parser.add_argument("--minutes", type=float, default=35.0, help="estimated time")
    parser.add_argument("--surge", type=float, default=1.0)
    parser.add_argument("--hour", type=int, default=18, choices=range(24))
    parser.add_argument("--traffic", default="Medium", choices=dp.TRAFFIC_LEVELS)
    parser.add_argument("--weather", default="Clear", choices=dp.WEATHER_CONDITIONS)
    args = parser.parse_args(argv)

    inputs = {
        "city": args.city,
        "vehicle_type": args.vehicle,
        "ride_distance_km": args.distance,
        "estimated_ride_time_min": args.minutes,
        "surge_multiplier": args.surge,
        "hour_of_day": args.hour,
        "traffic_level": args.traffic,
        "weather_condition": args.weather,
    }

    results = predict_all(inputs)
    if not results:
        print("No trained models found. Run: python src/train_models.py")
        return 1

    report = {}
    if "outcome" in results:
        report["outcome"] = {
            "prediction": results["outcome"]["prediction"],
            "probabilities": results["outcome"]["probabilities"],
        }
    if "fare" in results:
        report["fare"] = {
            "predicted": results["fare"]["predicted_fare"],
            "tariff_formula": results["fare"]["formula_fare"],
        }
    for key in ("customer_risk", "driver_risk"):
        if key in results:
            report[key] = {
                "probability": results[key]["probability"],
                "risk_level": results[key]["risk_level"],
            }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

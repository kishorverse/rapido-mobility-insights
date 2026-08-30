"""Artefact persistence and prediction serving.

The deployment layer: it writes fitted pipelines to disk with the metadata needed
to reproduce them, loads them back, and turns a dict of user inputs into a
prediction. Nothing here fits a model.
"""

from __future__ import annotations

from __future__ import annotations
from datetime import datetime
from pathlib import Path
import functools
import json
import logging

import joblib
import pandas as pd
import numpy as np

import config
from rapido import io
from rapido.models import dataset, evaluate


logger = logging.getLogger(__name__)

METRICS_FILE = config.MODEL_DIR / "metrics.json"


def save_model(pipeline, name: str, metrics: dict, metadata: dict | None = None) -> Path:
    """Persist a fitted pipeline together with its metrics and metadata."""
    path = config.get_model_path(name)
    payload = {
        "pipeline": pipeline,
        "metrics": metrics,
        "metadata": {
            **(metadata or {}),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "model_name": name,
        },
    }
    joblib.dump(payload, path, compress=3)
    logger.info("Saved model %s to %s", name, path)
    save_metrics(name, {"metrics": metrics, **(metadata or {})})
    return path


def load_model(name: str):
    """Load a persisted pipeline and its metadata.

    Returns:
        Tuple of (pipeline, metadata dict including metrics).

    Raises:
        FileNotFoundError: If the artefact does not exist.
    """
    path = config.get_model_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model named {name!r} at {path}. "
            "Run: python scripts/manage.py train"
        )
    payload = joblib.load(path)
    metadata = dict(payload.get("metadata", {}))
    metadata["metrics"] = payload.get("metrics", {})
    return payload["pipeline"], metadata


def model_exists(name: str) -> bool:
    """Return whether a trained artefact is on disk."""
    return config.get_model_path(name).exists()


def list_models() -> pd.DataFrame:
    """List every persisted model with its headline metric."""
    rows = []
    for path in sorted(config.MODEL_DIR.glob("*.joblib")):
        name = path.stem
        row = {
            "model": name,
            "file": path.name,
            "size_mb": round(path.stat().st_size / 1_048_576, 2),
        }
        try:
            _, metadata = load_model(name)
            metrics = metadata.get("metrics", {})
            row["algorithm"] = metadata.get("algorithm", "-")
            row["headline"] = (
                metrics.get("f1_macro")
                or metrics.get("r2")
                or metrics.get("pr_auc")
                or "-"
            )
            row["trained_at"] = metadata.get("saved_at", "-")
        except Exception as exc:  # pragma: no cover - corrupt artefact
            row["algorithm"] = f"unreadable ({exc})"
        rows.append(row)
    return pd.DataFrame(rows)


def save_metrics(name: str, payload: dict) -> Path:
    """Merge one model's metrics into the shared metrics file."""
    existing = load_metrics() if METRICS_FILE.exists() else {}
    existing[name] = _jsonable(payload)
    METRICS_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return METRICS_FILE


def load_metrics(name: str | None = None) -> dict:
    """Read stored metrics for one model, or all of them."""
    if not METRICS_FILE.exists():
        return {}
    data = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    return data.get(name, {}) if name else data


def _jsonable(value):
    """Recursively coerce numpy and pandas values into JSON-safe types."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


logger = logging.getLogger(__name__)

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


@functools.lru_cache(maxsize=1)
def _feature_defaults() -> dict:
    """Median and modal values for every model feature, computed once."""
    frame = io.load_processed("features")
    defaults: dict = {}

    for column in dataset.BASE_NUMERIC + dataset.FARE_NUMERIC:
        if column in frame.columns:
            value = frame[column].median()
            defaults[column] = float(value) if pd.notna(value) else 0.0

    for column in dataset.BASE_CATEGORICAL:
        if column in frame.columns:
            mode = frame[column].astype(str).mode()
            defaults[column] = mode.iloc[0] if len(mode) else "Unknown"

    return defaults


def get_defaults() -> dict:
    """Public accessor for the cached feature defaults."""
    return dict(_feature_defaults())


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

    numeric = list(dataset.BASE_NUMERIC)
    if target != "fare":
        numeric += dataset.FARE_NUMERIC

    columns = numeric + dataset.BASE_CATEGORICAL
    frame = pd.DataFrame([{column: row.get(column) for column in columns}])

    for column in dataset.BASE_CATEGORICAL:
        frame[column] = frame[column].astype(str)

    return frame


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
        "rush_hour_flag": int(hour in config.RUSH_HOURS),
        "is_night_ride": int(hour in config.NIGHT_HOURS),
        "long_distance_flag": int(distance > config.LONG_DISTANCE_KM),
        "expected_speed_kmph": distance / (estimated / 60) if estimated else np.nan,
        "high_traffic_flag": int(traffic == "High"),
        "bad_weather_flag": int(weather in {"Rain", "Heavy Rain"}),
        "adverse_conditions_flag": int(traffic == "High" and weather == "Heavy Rain"),
        "peak_time_flag": int(hour in config.RUSH_HOURS),
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


def predict_outcome(inputs: dict) -> dict:
    """Predict the ride outcome with class probabilities."""
    model, metadata = load_model(config.MODEL_NAMES["outcome"])
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
    model, metadata = load_model(config.MODEL_NAMES["fare"])
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
    model, metadata = load_model(config.MODEL_NAMES[model_key])
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


def explain_prediction(metadata: dict, row: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Show which of the model's top features this booking carries."""
    top_features = metadata.get("top_features")
    if top_features is None:
        return pd.DataFrame(columns=["feature", "value", "importance"])
    importance = (
        pd.DataFrame(top_features)
        if not isinstance(top_features, pd.DataFrame)
        else top_features
    )
    return evaluate.top_drivers_for_prediction(None, row, importance, top_n=top_n)


def available_models() -> dict[str, bool]:
    """Report which trained artefacts are present on disk."""
    return {
        key: model_exists(name) for key, name in config.MODEL_NAMES.items()
    }

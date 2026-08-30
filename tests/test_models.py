"""Tests for trained model artefacts, evaluation maths and the serving layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from rapido.models import evaluate, registry, serve

MODELS_TRAINED = all(
    registry.model_exists(name) for name in config.MODEL_NAMES.values()
)

needs_models = pytest.mark.skipif(
    not MODELS_TRAINED, reason="Models not trained; run scripts/train_all.py"
)


# --------------------------------------------------------------------------- #
# Evaluation maths
# --------------------------------------------------------------------------- #


def test_within_tolerance_rate_exact():
    """A perfect prediction is fully inside any tolerance band."""
    actual = np.array([100.0, 200.0, 300.0])
    assert evaluate.within_tolerance_rate(actual, actual, 0.10) == 1.0


def test_within_tolerance_rate_boundary():
    """The band is inclusive at exactly the tolerance."""
    actual = np.array([100.0, 100.0])
    predicted = np.array([110.0, 111.0])
    assert evaluate.within_tolerance_rate(actual, predicted, 0.10) == 0.5


def test_regression_metrics_perfect_fit():
    """A perfect regression scores R² = 1 and zero error."""
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    metrics = evaluate.regression_metrics(actual, actual)
    assert metrics["r2"] == 1.0
    assert metrics["rmse"] == 0.0
    assert metrics["within_10_pct"] == 1.0


def test_classification_metrics_perfect_fit():
    """A perfect classification scores 1.0 on accuracy and macro-F1."""
    labels = ["Completed", "Cancelled", "Incomplete", "Completed"]
    metrics = evaluate.classification_metrics(labels, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0


def test_confusion_matrix_shape():
    """The confusion matrix is square over the label set."""
    labels = ["A", "B", "C"]
    matrix = evaluate.confusion_matrix_df(
        ["A", "B", "C", "A"], ["A", "B", "A", "A"], labels
    )
    assert matrix.shape == (3, 3)
    assert matrix.to_numpy().sum() == 4


def test_threshold_table_monotonic_recall():
    """Recall never increases as the decision threshold rises."""
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 2, 500)
    scores = rng.random(500)
    table = evaluate.threshold_table(truth, scores)
    assert table["recall"].is_monotonic_decreasing


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@needs_models
@pytest.mark.parametrize("key", list(config.MODEL_NAMES))
def test_model_round_trips(key):
    """Every artefact loads back with its metrics and metadata intact."""
    pipeline, metadata = registry.load_model(config.MODEL_NAMES[key])
    assert hasattr(pipeline, "predict")
    assert metadata["metrics"]
    assert metadata["algorithm"]


def test_missing_model_raises_helpful_error():
    """Loading an unknown model explains how to produce it."""
    with pytest.raises(FileNotFoundError, match="train_all"):
        registry.load_model("no_such_model")


@needs_models
def test_list_models_covers_all_four():
    """The registry lists every trained artefact."""
    listed = registry.list_models()
    assert len(listed) >= 4


# --------------------------------------------------------------------------- #
# Model quality
# --------------------------------------------------------------------------- #


@needs_models
def test_models_beat_their_baselines():
    """Each model outperforms the dummy baseline recorded at training time."""
    for key in ("outcome", "customer_risk", "driver_risk"):
        _, metadata = registry.load_model(config.MODEL_NAMES[key])
        leaderboard = pd.DataFrame(metadata["leaderboard"])
        best = leaderboard.loc[leaderboard["model"] != "dummy", "f1_macro"].max()
        baseline = leaderboard.loc[leaderboard["model"] == "dummy", "f1_macro"].iloc[0]
        assert best > baseline, f"{key} failed to beat its baseline"


@needs_models
def test_fare_model_meets_project_benchmark():
    """The fare model satisfies the brief's ±10% tolerance target."""
    _, metadata = registry.load_model(config.MODEL_NAMES["fare"])
    metrics = metadata["metrics"]
    assert metrics["within_10_pct"] > 0.90
    assert metrics["r2"] > 0.90


@needs_models
def test_fare_model_respects_noise_floor():
    """MAPE cannot beat the 2.5% floor set by the ±5% uniform noise term."""
    _, metadata = registry.load_model(config.MODEL_NAMES["fare"])
    assert metadata["metrics"]["mape_pct"] >= 2.4, (
        "MAPE below the theoretical noise floor implies leakage"
    )


@needs_models
def test_risk_models_discriminate():
    """Both risk models score meaningfully better than random."""
    for key in ("customer_risk", "driver_risk"):
        _, metadata = registry.load_model(config.MODEL_NAMES[key])
        assert metadata["metrics"]["roc_auc"] > 0.65


# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #


def test_tariff_formula():
    """The recovered tariff reproduces known base fares."""
    assert serve.estimate_base_fare("Bike", 10) == pytest.approx(100.0)
    assert serve.estimate_base_fare("Auto", 10) == pytest.approx(160.0)
    assert serve.estimate_base_fare("Cab", 10) == pytest.approx(260.0)


def test_risk_level_bands():
    """Probabilities map to the documented risk bands."""
    assert serve.risk_level(0.10) == "Low"
    assert serve.risk_level(0.45) == "Medium"
    assert serve.risk_level(0.90) == "High"


@needs_models
def test_feature_row_matches_training_columns():
    """A built row carries exactly the columns its model was fitted on."""
    inputs = {"ride_distance_km": 10.0, "vehicle_type": "Cab", "city": "Mumbai"}
    for key in config.MODEL_NAMES:
        row = serve.build_feature_row(inputs, key)
        _, metadata = registry.load_model(config.MODEL_NAMES[key])
        assert set(row.columns) == set(metadata["features"])
        assert len(row) == 1


@needs_models
def test_derived_fields_stay_consistent():
    """Engineered flags are recomputed from user inputs, not left stale."""
    row = serve.build_feature_row(
        {"ride_distance_km": 20.0, "hour_of_day": 18, "vehicle_type": "Cab"},
        "outcome",
    )
    assert row["long_distance_flag"].iloc[0] == 1
    assert row["rush_hour_flag"].iloc[0] == 1
    assert row["distance_band"].iloc[0] == "Long"


@needs_models
def test_predictions_respond_to_conditions():
    """Adverse conditions raise cancellation risk above calm conditions."""
    base = {
        "ride_distance_km": 12.0,
        "estimated_ride_time_min": 35.0,
        "vehicle_type": "Cab",
        "city": "Mumbai",
        "hour_of_day": 18,
    }
    adverse = serve.predict_customer_risk(
        {**base, "traffic_level": "High", "weather_condition": "Heavy Rain", "surge_multiplier": 2.0}
    )
    calm = serve.predict_customer_risk(
        {**base, "traffic_level": "Low", "weather_condition": "Clear", "surge_multiplier": 1.0}
    )
    assert adverse["probability"] > calm["probability"]


@needs_models
def test_outcome_probabilities_sum_to_one():
    """Class probabilities form a valid distribution."""
    result = serve.predict_outcome({"ride_distance_km": 8.0, "vehicle_type": "Auto"})
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert result["prediction"] in config.BOOKING_STATUSES


@needs_models
def test_fare_prediction_near_tariff():
    """The model's fare estimate lands close to the tariff formula."""
    result = serve.predict_fare(
        {"ride_distance_km": 10.0, "vehicle_type": "Cab", "surge_multiplier": 1.5}
    )
    assert abs(result["predicted_fare"] - result["formula_fare"]) / result[
        "formula_fare"
    ] < 0.15


def test_tune_model_rejects_unknown_strategy():
    """Only the two documented search strategies are accepted."""
    from rapido.models import train

    with pytest.raises(ValueError, match="Unknown search_strategy"):
        train.tune_model("rf", "classification", None, None, search_strategy="optuna")

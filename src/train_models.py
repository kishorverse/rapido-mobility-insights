"""Feature matrices, model training, evaluation and artefact persistence.

Everything that turns the feature table into a saved model lives here:

1. **Datasets** - feature-matrix construction with an enforced leakage guard.
   Each model declares which columns it must never see, and
   :func:`assert_no_leakage` raises rather than warns, so a blocked column
   cannot silently reach a fit.
2. **Training** - preprocessing lives inside the sklearn Pipeline rather than
   being applied to the frame beforehand, so every fold of cross-validation
   re-fits the imputer, scaler and encoder on training data only. That is what
   keeps the CV scores honest. Every model follows the same protocol: build the
   matrices, split 80/20 stratified, score a baseline, compare candidate
   estimators, tune the winner, then persist it.
3. **Evaluation** - metrics answer "how well does it score"; feature importance
   answers "on what grounds". Both are diagnostics applied to an already-fitted
   estimator and neither touches the training data.
4. **Persistence** - fitted pipelines are written to ``models/*.pkl`` with the
   metadata needed to reproduce them. :mod:`predict` reads them back.

Usage:
    python src/train_models.py                       # train all four models
    python src/train_models.py --model fare --tune
    python src/train_models.py --tune --search grid  # exhaustive GridSearchCV
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_preprocessing as dp
import feature_engineering as fe

logger = logging.getLogger(__name__)


# =========================================================================== #
# 1. Feature matrices and the leakage guard
# =========================================================================== #


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
    "outcome": list(dp.POST_OUTCOME_COLUMNS),
    "fare": list(dp.POST_OUTCOME_COLUMNS)
    + ["booking_value", "base_fare", "fare_per_km", "fare_per_min"],
    "customer_risk": list(dp.POST_OUTCOME_COLUMNS),
    "driver_risk": list(dp.POST_OUTCOME_COLUMNS),
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
            "See the leakage-control notes in README.md."
        )

    identifiers = set(IDENTIFIER_COLUMNS) & set(features.columns)
    if identifiers:
        raise LeakageError(
            f"Identifier columns leaked into the {target!r} matrix: "
            f"{sorted(identifiers)}"
        )


def _select_features(
    frame: pd.DataFrame, target: str, include_fare_columns: bool
) -> pd.DataFrame:
    """Assemble and validate the feature matrix for a target."""
    numeric = list(BASE_NUMERIC)
    if include_fare_columns:
        numeric += FARE_NUMERIC

    columns = [
        column for column in numeric + BASE_CATEGORICAL if column in frame.columns
    ]
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
        test_size=dp.TEST_SIZE if test_size is None else test_size,
        random_state=dp.RANDOM_STATE,
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
        summary["class_balance"] = target.value_counts(normalize=True).round(4).to_dict()
    else:
        summary["target_mean"] = round(float(target.mean()), 2)
        summary["target_std"] = round(float(target.std()), 2)
    return summary


# =========================================================================== #
# 2. Evaluation
# =========================================================================== #


def get_feature_names(pipeline) -> list[str]:
    """Recover post-encoding feature names from a fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocess"]
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:  # pragma: no cover - very old sklearn
        return [f"feature_{index}" for index in range(preprocessor.transform_shape_[1])]


def tree_feature_importance(pipeline, top_n: int | None = None) -> pd.DataFrame:
    """Impurity-based importance for tree ensembles.

    Returns an empty frame for estimators without ``feature_importances_``
    (for example logistic regression), where permutation importance is used.
    """
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    names = get_feature_names(pipeline)
    values = estimator.feature_importances_
    frame = (
        pd.DataFrame({"feature": names[: len(values)], "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    frame["importance"] = frame["importance"].round(6)
    return frame.head(top_n) if top_n else frame


def coefficient_importance(pipeline, top_n: int | None = None) -> pd.DataFrame:
    """Absolute coefficient magnitude for linear estimators."""
    estimator = pipeline.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        return pd.DataFrame(columns=["feature", "importance"])

    names = get_feature_names(pipeline)
    coefficients = np.asarray(estimator.coef_)
    values = (
        np.abs(coefficients).mean(axis=0)
        if coefficients.ndim > 1
        else np.abs(coefficients)
    )
    frame = (
        pd.DataFrame({"feature": names[: len(values)], "importance": values.round(6)})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return frame.head(top_n) if top_n else frame


def permutation_feature_importance(
    pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    task: str,
    n_repeats: int = 5,
    sample: int = 10_000,
) -> pd.DataFrame:
    """Permutation importance on the original (pre-encoding) columns.

    More trustworthy than impurity importance, which is biased toward
    high-cardinality features. Computed on a sample because permuting 45
    columns over 20k rows is expensive.

    Args:
        pipeline: A fitted pipeline.
        features: Held-out feature matrix.
        target: Held-out target.
        task: ``"classification"`` or ``"regression"``.
        n_repeats: Shuffles per column.
        sample: Maximum rows used.
    """
    if len(features) > sample:
        subset = features.sample(sample, random_state=dp.RANDOM_STATE)
        target = target.loc[subset.index]
        features = subset

    scoring = "f1_macro" if task == "classification" else "r2"
    result = permutation_importance(
        pipeline,
        features,
        target,
        n_repeats=n_repeats,
        random_state=dp.RANDOM_STATE,
        scoring=scoring,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": features.columns,
                "importance": result.importances_mean.round(6),
                "std": result.importances_std.round(6),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def top_drivers_for_prediction(
    pipeline, row: pd.DataFrame, importance: pd.DataFrame, top_n: int = 6
) -> pd.DataFrame:
    """Show the values a single prediction rests on, ranked by importance.

    Not a SHAP decomposition: it reports which of the model's globally most
    important features this booking carries, and at what value. That is enough
    for an operator to see why a booking was flagged.
    """
    if importance.empty:
        return pd.DataFrame(columns=["feature", "value", "importance"])

    original = set(row.columns)
    ranked = [
        feature for feature in importance["feature"].tolist() if feature in original
    ]

    rows = []
    for feature in ranked[:top_n]:
        score = importance.loc[importance["feature"] == feature, "importance"]
        rows.append(
            {
                "feature": feature,
                "value": row[feature].iloc[0],
                "importance": round(float(score.iloc[0]), 6) if len(score) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def importance_for(pipeline, features=None, target=None, task="classification"):
    """Return the best available importance table for a pipeline.

    Prefers permutation importance when held-out data is supplied, because it
    is measured on the original columns rather than one-hot fragments.
    """
    if features is not None and target is not None:
        try:
            return permutation_feature_importance(pipeline, features, target, task)
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning("Permutation importance failed (%s); using impurity", exc)

    tree = tree_feature_importance(pipeline)
    return tree if not tree.empty else coefficient_importance(pipeline)


def classification_metrics(
    y_true, y_pred, y_proba=None, labels: list | None = None
) -> dict:
    """Compute the standard classification metric block."""
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "precision_macro": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "recall_macro": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
    }

    classes = labels if labels is not None else sorted(pd.Series(y_true).unique())
    binary = len(classes) == 2

    if y_proba is not None:
        try:
            if binary:
                positive = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                metrics["roc_auc"] = round(float(roc_auc_score(y_true, positive)), 4)
                metrics["pr_auc"] = round(
                    float(average_precision_score(y_true, positive)), 4
                )
                metrics["positive_rate"] = round(float(np.mean(y_true)), 4)
            else:
                metrics["roc_auc_ovr"] = round(
                    float(
                        roc_auc_score(
                            y_true, y_proba, multi_class="ovr", average="macro"
                        )
                    ),
                    4,
                )
        except ValueError:
            pass

    return metrics


def per_class_metrics(y_true, y_pred, labels: list | None = None) -> pd.DataFrame:
    """Precision, recall, F1 and support for each class."""
    classes = labels if labels is not None else sorted(pd.Series(y_true).unique())
    rows = []
    for _index, label in enumerate(classes):
        actual = np.asarray(y_true) == label
        predicted = np.asarray(y_pred) == label
        rows.append(
            {
                "class": label,
                "precision": round(
                    float(precision_score(actual, predicted, zero_division=0)), 4
                ),
                "recall": round(
                    float(recall_score(actual, predicted, zero_division=0)), 4
                ),
                "f1": round(float(f1_score(actual, predicted, zero_division=0)), 4),
                "support": int(actual.sum()),
            }
        )
    return pd.DataFrame(rows)


def within_tolerance_rate(y_true, y_pred, tolerance: float = 0.10) -> float:
    """Share of predictions falling within ``tolerance`` of the actual value.

    This is the project's stated fare benchmark, expressed directly.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    safe = np.where(y_true == 0, np.nan, y_true)
    relative_error = np.abs((y_pred - y_true) / safe)
    return float(np.nanmean(relative_error <= tolerance))


def regression_metrics(y_true, y_pred) -> dict:
    """Standard regression metric block, including the tolerance band."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {
        "rmse": round(rmse, 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mape_pct": round(
            100 * float(mean_absolute_percentage_error(y_true, y_pred)), 3
        ),
        "rmse_pct_of_mean": round(100 * rmse / float(np.mean(y_true)), 3),
        "within_10_pct": round(within_tolerance_rate(y_true, y_pred, 0.10), 4),
        "within_20_pct": round(within_tolerance_rate(y_true, y_pred, 0.20), 4),
    }


def confusion_matrix_df(y_true, y_pred, labels: list) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )


def roc_data(y_true, y_proba) -> tuple[np.ndarray, np.ndarray, float]:
    """False-positive rate, true-positive rate and AUC for a binary target."""
    positive = y_proba[:, 1] if np.ndim(y_proba) > 1 else y_proba
    fpr, tpr, _ = roc_curve(y_true, positive)
    return fpr, tpr, float(roc_auc_score(y_true, positive))


def pr_data(y_true, y_proba) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Precision, recall, average precision and the prevalence baseline."""
    positive = y_proba[:, 1] if np.ndim(y_proba) > 1 else y_proba
    precision, recall, _ = precision_recall_curve(y_true, positive)
    return (
        precision,
        recall,
        float(average_precision_score(y_true, positive)),
        float(np.mean(y_true)),
    )


def cross_validate_model(
    pipeline, features, target, task: str, cv: int | None = None
) -> dict:
    """Cross-validate a pipeline and summarise fold scores."""
    folds = dp.CV_FOLDS if cv is None else cv
    scoring = "f1_macro" if task == "classification" else "neg_root_mean_squared_error"
    splitter = (
        StratifiedKFold(n_splits=folds, shuffle=True, random_state=dp.RANDOM_STATE)
        if task == "classification"
        else folds
    )
    scores = cross_val_score(
        pipeline, features, target, cv=splitter, scoring=scoring, n_jobs=-1
    )
    if task == "regression":
        scores = -scores
    return {
        "metric": "f1_macro" if task == "classification" else "rmse",
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
        "folds": [round(float(score), 4) for score in scores],
    }


def compare_models(results: dict) -> pd.DataFrame:
    """Turn a name-to-metrics mapping into a sorted leaderboard."""
    rows = []
    for name, metrics in results.items():
        row = {"model": name}
        row.update(metrics)
        rows.append(row)

    frame = pd.DataFrame(rows)
    for column in ("f1_macro", "r2", "pr_auc"):
        if column in frame.columns:
            return frame.sort_values(column, ascending=False).reset_index(drop=True)
    return frame


def threshold_table(
    y_true, y_proba, thresholds: list[float] | None = None
) -> pd.DataFrame:
    """Precision, recall and flag volume across decision thresholds.

    Operations needs this to choose a cut-off: a risk score is only useful once
    someone decides how many bookings they can afford to intervene on.
    """
    positive = y_proba[:, 1] if np.ndim(y_proba) > 1 else y_proba
    steps = thresholds or [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    rows = []
    for threshold in steps:
        predicted = (positive >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "flagged": int(predicted.sum()),
                "flagged_pct": round(100 * float(predicted.mean()), 2),
                "precision": round(
                    float(precision_score(y_true, predicted, zero_division=0)), 4
                ),
                "recall": round(
                    float(recall_score(y_true, predicted, zero_division=0)), 4
                ),
                "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
            }
        )
    return pd.DataFrame(rows)


# =========================================================================== #
# 3. Artefact persistence
# =========================================================================== #

METRICS_FILE = dp.MODEL_DIR / "metrics.json"


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


def save_model(pipeline, name: str, metrics: dict, metadata: dict | None = None) -> Path:
    """Persist a fitted pipeline together with its metrics and metadata."""
    path = dp.get_model_path(name)
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
    path = dp.get_model_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model named {name!r} at {path}. "
            "Run: python src/train_models.py"
        )
    payload = joblib.load(path)
    metadata = dict(payload.get("metadata", {}))
    metadata["metrics"] = payload.get("metrics", {})
    return payload["pipeline"], metadata


def model_exists(name: str) -> bool:
    """Return whether a trained artefact is on disk."""
    return dp.get_model_path(name).exists()


def list_models() -> pd.DataFrame:
    """List every persisted model with its headline metric."""
    rows = []
    for path in sorted(dp.MODEL_DIR.glob("*.pkl")):
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


def available_models() -> dict[str, bool]:
    """Report which trained artefacts are present on disk."""
    return {key: model_exists(name) for key, name in dp.MODEL_NAMES.items()}


# =========================================================================== #
# 4. Model construction
# =========================================================================== #


def build_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str], scale: bool = True
) -> ColumnTransformer:
    """Build the shared preprocessing transformer.

    Numeric columns are median-imputed (the prior-history rates are NaN for an
    entity's first ride, which is meaningful rather than missing-at-random) and
    optionally scaled. Categorical columns are most-frequent imputed and
    one-hot encoded, ignoring unseen categories at predict time.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            ("categorical", Pipeline(categorical_steps), categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _classifier(name: str, class_weight: str | None, **params):
    """Instantiate a classifier by short name."""
    seed = dp.RANDOM_STATE
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        return LogisticRegression(
            max_iter=2000, class_weight=class_weight, random_state=seed, **params
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.pop("n_estimators", 300),
            min_samples_leaf=params.pop("min_samples_leaf", 5),
            n_jobs=-1,
            class_weight=class_weight,
            random_state=seed,
            **params,
        )
    if name == "hist_gb":
        # HistGradientBoosting has no class_weight before the fit call, so
        # balancing is applied through sample weights in _fit().
        return HistGradientBoostingClassifier(random_state=seed, **params)
    raise ValueError(f"Unknown classifier {name!r}.")


def _regressor(name: str, **params):
    """Instantiate a regressor by short name."""
    seed = dp.RANDOM_STATE
    if name == "dummy":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return Ridge(alpha=params.pop("alpha", 1.0), random_state=seed, **params)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=params.pop("n_estimators", 200),
            min_samples_leaf=params.pop("min_samples_leaf", 5),
            n_jobs=-1,
            random_state=seed,
            **params,
        )
    if name == "hist_gb":
        return HistGradientBoostingRegressor(random_state=seed, **params)
    raise ValueError(f"Unknown regressor {name!r}.")


def build_model(
    name: str,
    task: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    class_weight: str | None = None,
    **params,
) -> Pipeline:
    """Compose preprocessing and an estimator into one fitted-together pipeline.

    Args:
        name: Estimator short name.
        task: ``"classification"`` or ``"regression"``.
        numeric_columns: Numeric feature names.
        categorical_columns: Categorical feature names.
        class_weight: Passed to classifiers that support it.
        **params: Extra estimator keyword arguments.
    """
    # Tree ensembles do not need scaling; skipping it keeps values interpretable.
    scale = name in {"logreg", "ridge"}
    preprocessor = build_preprocessor(numeric_columns, categorical_columns, scale=scale)

    if task == "classification":
        estimator = _classifier(name, class_weight, **params)
    elif task == "regression":
        estimator = _regressor(name, **params)
    else:
        raise ValueError(f"Unknown task {task!r}.")

    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


#: Candidate estimators tried for each task, cheapest first.
CANDIDATES = {
    "classification": ["dummy", "logreg", "random_forest", "hist_gb"],
    "regression": ["dummy", "ridge", "random_forest", "hist_gb"],
}


def get_param_grid(name: str, task: str) -> dict:
    """Return the hyperparameter search space for an estimator.

    Grids are deliberately small: the dataset has 100k rows, and a wide search
    would cost hours for marginal gain.
    """
    grids = {
        ("logreg", "classification"): {"model__C": [0.1, 1.0, 5.0]},
        ("random_forest", "classification"): {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 16],
            "model__min_samples_leaf": [2, 10],
        },
        ("hist_gb", "classification"): {
            "model__learning_rate": [0.05, 0.1],
            "model__max_iter": [200, 400],
            "model__max_leaf_nodes": [31, 63],
        },
        ("ridge", "regression"): {"model__alpha": [0.1, 1.0, 10.0]},
        ("random_forest", "regression"): {
            "model__n_estimators": [200, 400],
            "model__min_samples_leaf": [2, 10],
        },
        ("hist_gb", "regression"): {
            "model__learning_rate": [0.05, 0.1],
            "model__max_iter": [200, 400],
            "model__max_leaf_nodes": [31, 63],
        },
    }
    return grids.get((name, task), {})


# =========================================================================== #
# 5. Training
# =========================================================================== #


def _fit(model, features, target, balance: bool = False):
    """Fit a pipeline, applying sample weights where the estimator needs them."""
    estimator = model.named_steps["model"]
    needs_weights = balance and estimator.__class__.__name__.startswith(
        "HistGradientBoosting"
    )
    if needs_weights:
        weights = compute_sample_weight("balanced", target)
        model.fit(features, target, model__sample_weight=weights)
    else:
        model.fit(features, target)
    return model


def _score(model, features, target, task: str, labels=None) -> dict:
    """Score a fitted pipeline on a held-out set."""
    predictions = model.predict(features)
    if task == "regression":
        return regression_metrics(target, predictions)

    probabilities = (
        model.predict_proba(features) if hasattr(model, "predict_proba") else None
    )
    return classification_metrics(target, predictions, probabilities, labels)


def train_and_compare(
    features_train,
    target_train,
    features_test,
    target_test,
    task: str,
    balance: bool = False,
    candidates: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Train every candidate estimator and return a leaderboard.

    Returns:
        The leaderboard and a mapping of estimator name to fitted pipeline.
    """
    numeric, categorical = get_feature_types(features_train)
    names = candidates or CANDIDATES[task]
    class_weight = "balanced" if balance else None

    results, fitted = {}, {}
    for name in names:
        started = time.perf_counter()
        model = build_model(
            name,
            task,
            numeric,
            categorical,
            class_weight=class_weight if name != "dummy" else None,
        )
        try:
            _fit(model, features_train, target_train, balance=balance)
        except Exception as exc:  # pragma: no cover - estimator-specific failure
            logger.error("Training %s failed: %s", name, exc)
            continue

        metrics = _score(model, features_test, target_test, task)
        metrics["fit_seconds"] = round(time.perf_counter() - started, 2)
        results[name] = metrics
        fitted[name] = model
        headline = metrics.get("f1_macro", metrics.get("r2"))
        logger.info(
            "  %-14s %s=%.4f (%.1fs)",
            name,
            "f1_macro" if task == "classification" else "r2",
            headline,
            metrics["fit_seconds"],
        )

    return compare_models(results), fitted


def tune_model(
    name: str,
    task: str,
    features,
    target,
    balance: bool = False,
    n_iter: int = 8,
    search_strategy: str = "random",
):
    """Hyperparameter search over the estimator's grid.

    Args:
        search_strategy: ``"random"`` samples ``n_iter`` combinations via
            RandomizedSearchCV; ``"grid"`` runs the exhaustive GridSearchCV.

    Randomised search is the default because the grids cover up to eight
    combinations on 80,000 rows, and sampling lands within noise of the full
    sweep at a fraction of the cost. Exhaustive search stays available for a
    final confirmation run.
    """
    if search_strategy not in {"random", "grid"}:
        raise ValueError(
            f"Unknown search_strategy {search_strategy!r}; use 'random' or 'grid'"
        )
    numeric, categorical = get_feature_types(features)
    grid = get_param_grid(name, task)
    base = build_model(
        name, task, numeric, categorical, class_weight="balanced" if balance else None
    )
    if not grid:
        return _fit(base, features, target, balance=balance), {}

    scoring = "f1_macro" if task == "classification" else "neg_root_mean_squared_error"
    combinations = int(np.prod([len(v) for v in grid.values()]))

    if search_strategy == "grid":
        search = GridSearchCV(base, grid, scoring=scoring, cv=3, n_jobs=-1, refit=True)
    else:
        search = RandomizedSearchCV(
            base,
            grid,
            n_iter=min(n_iter, combinations),
            scoring=scoring,
            cv=3,
            random_state=dp.RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )

    search.fit(features, target)
    logger.info("  tuned %s (%s) -> %s", name, search_strategy, search.best_params_)
    return search.best_estimator_, {
        "search_strategy": search_strategy,
        "candidates_evaluated": len(search.cv_results_["params"]),
        "grid_size": combinations,
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "best_cv_score": round(float(search.best_score_), 4),
    }


def _train_generic(
    frame: pd.DataFrame,
    target_key: str,
    task: str,
    model_name: str,
    balance: bool,
    tune: bool,
    describe: str,
    search_strategy: str = "random",
) -> dict:
    """Shared training routine for all four models."""
    logger.info("=" * 70)
    logger.info("Training: %s", describe)
    logger.info("=" * 70)

    features, target = build_dataset(frame, target_key)
    summary = describe_dataset(features, target)
    logger.info("  dataset: %s", summary)

    stratify = task == "classification"
    x_train, x_test, y_train, y_test = split_train_test(
        features, target, stratify=stratify
    )

    leaderboard, fitted = train_and_compare(
        x_train, y_train, x_test, y_test, task, balance=balance
    )
    if leaderboard.empty:
        raise RuntimeError(f"No estimator trained successfully for {model_name}.")

    ranked = [name for name in leaderboard["model"] if name != "dummy"]
    best_name = ranked[0] if ranked else leaderboard["model"].iloc[0]

    tuning = {}
    best_model = fitted[best_name]
    if tune:
        best_model, tuning = tune_model(
            best_name,
            task,
            x_train,
            y_train,
            balance=balance,
            search_strategy=search_strategy,
        )

    labels = sorted(pd.Series(y_train).unique()) if task == "classification" else None
    metrics = _score(best_model, x_test, y_test, task, labels)
    baseline = leaderboard.loc[leaderboard["model"] == "dummy"].to_dict("records")
    cv = cross_validate_model(best_model, x_train, y_train, task)

    importance = importance_for(best_model, x_test, y_test, task)

    extras: dict = {}
    if task == "classification":
        predictions = best_model.predict(x_test)
        extras["confusion_matrix"] = confusion_matrix_df(y_test, predictions, labels)
        extras["per_class"] = per_class_metrics(y_test, predictions, labels)
        if len(labels) == 2 and hasattr(best_model, "predict_proba"):
            probabilities = best_model.predict_proba(x_test)
            extras["thresholds"] = threshold_table(y_test, probabilities)

    save_model(
        best_model,
        model_name,
        metrics,
        {
            "algorithm": best_name,
            "task": task,
            "target": target_key,
            "description": describe,
            "n_train": len(x_train),
            "n_test": len(x_test),
            "features": list(features.columns),
            "dataset_summary": summary,
            "leaderboard": leaderboard,
            "baseline": baseline,
            "cross_validation": cv,
            "tuning": tuning,
            "top_features": importance.head(15),
            "balanced": balance,
        },
    )

    logger.info("  best: %s  metrics: %s", best_name, metrics)
    return {
        "name": model_name,
        "algorithm": best_name,
        "metrics": metrics,
        "leaderboard": leaderboard,
        "cross_validation": cv,
        "importance": importance,
        "model": best_model,
        "test_data": (x_test, y_test),
        **extras,
    }


def train_outcome_model(
    frame: pd.DataFrame, tune: bool = False, search_strategy: str = "random"
) -> dict:
    """Model 1: predict Completed / Cancelled / Incomplete before the trip."""
    return _train_generic(
        frame,
        "outcome",
        "classification",
        dp.MODEL_NAMES["outcome"],
        balance=True,
        tune=tune,
        search_strategy=search_strategy,
        describe="Ride outcome (3-class): Completed / Cancelled / Incomplete",
    )


def train_fare_model(
    frame: pd.DataFrame, tune: bool = False, search_strategy: str = "random"
) -> dict:
    """Model 2: predict booking value before confirmation, without base_fare."""
    return _train_generic(
        frame,
        "fare",
        "regression",
        dp.MODEL_NAMES["fare"],
        balance=False,
        tune=tune,
        search_strategy=search_strategy,
        describe="Pre-quote fare regression (base_fare excluded)",
    )


def train_customer_risk_model(
    frame: pd.DataFrame, tune: bool = False, search_strategy: str = "random"
) -> dict:
    """Model 3: probability that a booking is cancelled."""
    return _train_generic(
        frame,
        "customer_risk",
        "classification",
        dp.MODEL_NAMES["customer_risk"],
        balance=True,
        tune=tune,
        search_strategy=search_strategy,
        describe="Customer cancellation risk (binary)",
    )


def train_driver_risk_model(
    frame: pd.DataFrame, tune: bool = False, search_strategy: str = "random"
) -> dict:
    """Model 4: probability of a driver-caused delay or incomplete ride."""
    return _train_generic(
        frame,
        "driver_risk",
        "classification",
        dp.MODEL_NAMES["driver_risk"],
        balance=True,
        tune=tune,
        search_strategy=search_strategy,
        describe="Driver delay / incomplete-ride risk (binary)",
    )


def train_fare_leakage_ablation(frame: pd.DataFrame) -> dict:
    """Ablation: refit the fare model *with* ``base_fare`` to expose the formula.

    Not saved as a deployable artefact. Its only purpose is to quantify the gap
    between the honest pre-quote model and the trivial one, so the write-up can
    show the difference rather than assert it.
    """
    features, target = build_fare_dataset(frame, include_base_fare=True)
    x_train, x_test, y_train, y_test = split_train_test(
        features, target, stratify=False
    )
    numeric, categorical = get_feature_types(x_train)
    model = build_model("hist_gb", "regression", numeric, categorical)
    model.fit(x_train, y_train)
    metrics = regression_metrics(y_test, model.predict(x_test))
    logger.info("  leakage ablation (with base_fare): %s", metrics)
    return {"name": "fare_with_base_fare_ablation", "metrics": metrics}


TRAINERS = {
    "outcome": train_outcome_model,
    "fare": train_fare_model,
    "customer_risk": train_customer_risk_model,
    "driver_risk": train_driver_risk_model,
}


def train_all_models(
    frame: pd.DataFrame, tune: bool = False, search_strategy: str = "random"
) -> dict:
    """Train all four models and record the fare leakage ablation."""
    results = {}
    for key, trainer in TRAINERS.items():
        results[key] = trainer(frame, tune=tune, search_strategy=search_strategy)

    ablation = train_fare_leakage_ablation(frame)
    save_metrics("fare_leakage_ablation", ablation)
    results["fare_leakage_ablation"] = ablation
    return results


# =========================================================================== #
# 6. Command line
# =========================================================================== #


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s"
    )

    parser = argparse.ArgumentParser(description="Train the Rapido models.")
    parser.add_argument(
        "--model",
        choices=["all", *TRAINERS],
        default="all",
        help="which model to train (default: all)",
    )
    parser.add_argument(
        "--tune", action="store_true", help="run a hyperparameter search"
    )
    parser.add_argument(
        "--search",
        choices=["random", "grid"],
        default="random",
        help="search strategy when --tune is set",
    )
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="rebuild model_data.csv before training",
    )
    args = parser.parse_args(argv)

    frame = fe.build_and_cache(rebuild=args.rebuild_features)

    if args.model == "all":
        results = train_all_models(
            frame, tune=args.tune, search_strategy=args.search
        )
    else:
        results = {
            args.model: TRAINERS[args.model](
                frame, tune=args.tune, search_strategy=args.search
            )
        }

    print("\nTrained models:")
    for key, result in results.items():
        if key == "fare_leakage_ablation":
            continue
        print(f"  {key:<16} {result['algorithm']:<16} {result['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

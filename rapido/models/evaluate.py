"""Model evaluation and explanation.

Metrics answer "how well does it score"; feature importance answers "on what
grounds". Both are diagnostics applied to an already-fitted estimator, so they
live together and neither is allowed to touch the training data.
"""

from __future__ import annotations

from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
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
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config


logger = logging.getLogger(__name__)


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
    return (
        pd.DataFrame({"feature": names[: len(values)], "importance": values.round(6)})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
        .head(top_n)
        if top_n
        else pd.DataFrame(
            {"feature": names[: len(values)], "importance": values.round(6)}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
    )


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
        subset = features.sample(sample, random_state=config.RANDOM_STATE)
        target = target.loc[subset.index]
        features = subset

    scoring = "f1_macro" if task == "classification" else "r2"
    result = permutation_importance(
        pipeline,
        features,
        target,
        n_repeats=n_repeats,
        random_state=config.RANDOM_STATE,
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
    for index, label in enumerate(classes):
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


def regression_metrics(y_true, y_pred) -> dict:
    """Compute the standard regression metric block, including the tolerance band."""
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


def within_tolerance_rate(y_true, y_pred, tolerance: float = 0.10) -> float:
    """Share of predictions falling within ``tolerance`` of the actual value.

    This is the project's stated fare benchmark, expressed directly.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    safe = np.where(y_true == 0, np.nan, y_true)
    relative_error = np.abs((y_pred - y_true) / safe)
    return float(np.nanmean(relative_error <= tolerance))


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
    folds = config.CV_FOLDS if cv is None else cv
    scoring = "f1_macro" if task == "classification" else "neg_root_mean_squared_error"
    splitter = (
        StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)
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


def threshold_table(y_true, y_proba, thresholds: list[float] | None = None) -> pd.DataFrame:
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

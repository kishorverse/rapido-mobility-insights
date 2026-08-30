"""Model evaluation metrics and diagnostics.

Metric choices, for the record:

* **Macro-F1** leads the multi-class outcome model. Plain accuracy is inflated
  by the 68% Completed majority; macro-F1 weights the 8% Incomplete class
  equally, which is the class operations actually cares about.
* **PR-AUC** leads the binary risk models alongside ROC-AUC, because ROC-AUC
  looks flattering on imbalanced positives while PR-AUC does not.
* **RMSE with a +/-10% hit rate** leads the fare model, since the project
  benchmark is expressed as a tolerance band rather than a raw error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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

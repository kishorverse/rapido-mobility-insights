"""Feature importance and per-prediction explanation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

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

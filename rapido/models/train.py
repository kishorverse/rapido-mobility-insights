"""Training entry points for the four Rapido models.

Every model follows the same protocol: build a leakage-checked dataset, split
80/20 stratified, score a baseline, compare candidate estimators, tune the
winner, evaluate on the held-out test set, then persist.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight

import config
from rapido.models import dataset, evaluate, explain, pipeline as pipe, registry

logger = logging.getLogger(__name__)


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
        return evaluate.regression_metrics(target, predictions)

    probabilities = (
        model.predict_proba(features) if hasattr(model, "predict_proba") else None
    )
    return evaluate.classification_metrics(target, predictions, probabilities, labels)


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
    numeric, categorical = dataset.get_feature_types(features_train)
    names = candidates or pipe.CANDIDATES[task]
    class_weight = "balanced" if balance else None

    results, fitted = {}, {}
    for name in names:
        started = time.perf_counter()
        model = pipe.build_model(
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
        logger.info("  %-14s %s=%.4f (%.1fs)", name,
                    "f1_macro" if task == "classification" else "r2",
                    headline, metrics["fit_seconds"])

    return evaluate.compare_models(results), fitted


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
    numeric, categorical = dataset.get_feature_types(features)
    grid = pipe.get_param_grid(name, task)
    base = pipe.build_model(
        name, task, numeric, categorical, class_weight="balanced" if balance else None
    )
    if not grid:
        return _fit(base, features, target, balance=balance), {}

    scoring = "f1_macro" if task == "classification" else "neg_root_mean_squared_error"
    combinations = int(np.prod([len(v) for v in grid.values()]))

    if search_strategy == "grid":
        search = GridSearchCV(
            base, grid, scoring=scoring, cv=3, n_jobs=-1, refit=True
        )
    else:
        search = RandomizedSearchCV(
            base,
            grid,
            n_iter=min(n_iter, combinations),
            scoring=scoring,
            cv=3,
            random_state=config.RANDOM_STATE,
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

    features, target = dataset.build_dataset(frame, target_key)
    summary = dataset.describe_dataset(features, target)
    logger.info("  dataset: %s", summary)

    stratify = task == "classification"
    x_train, x_test, y_train, y_test = dataset.split_train_test(
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
    cv = evaluate.cross_validate_model(best_model, x_train, y_train, task)

    importance = explain.importance_for(best_model, x_test, y_test, task)

    extras: dict = {}
    if task == "classification":
        predictions = best_model.predict(x_test)
        extras["confusion_matrix"] = evaluate.confusion_matrix_df(
            y_test, predictions, labels
        )
        extras["per_class"] = evaluate.per_class_metrics(y_test, predictions, labels)
        if len(labels) == 2 and hasattr(best_model, "predict_proba"):
            probabilities = best_model.predict_proba(x_test)
            extras["thresholds"] = evaluate.threshold_table(y_test, probabilities)

    registry.save_model(
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
        config.MODEL_NAMES["outcome"],
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
        config.MODEL_NAMES["fare"],
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
        config.MODEL_NAMES["customer_risk"],
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
        config.MODEL_NAMES["driver_risk"],
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
    features, target = dataset.build_fare_dataset(frame, include_base_fare=True)
    x_train, x_test, y_train, y_test = dataset.split_train_test(
        features, target, stratify=False
    )
    numeric, categorical = dataset.get_feature_types(x_train)
    model = pipe.build_model("hist_gb", "regression", numeric, categorical)
    model.fit(x_train, y_train)
    metrics = evaluate.regression_metrics(y_test, model.predict(x_test))
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
    registry.save_metrics("fare_leakage_ablation", ablation)
    results["fare_leakage_ablation"] = ablation
    return results

"""scikit-learn pipeline construction.

Preprocessing lives inside the pipeline rather than being applied beforehand,
so imputation and encoding are fitted on training folds only. That is what
keeps cross-validation scores honest.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


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
    seed = config.RANDOM_STATE
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
        # balancing is applied through sample weights in train.py.
        return HistGradientBoostingClassifier(random_state=seed, **params)
    raise ValueError(f"Unknown classifier {name!r}.")


def _regressor(name: str, **params):
    """Instantiate a regressor by short name."""
    seed = config.RANDOM_STATE
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

"""Statistical significance tests backing the EDA claims.

Each test answers a specific business question, and each function returns a
plain dict so the result can be rendered in Streamlit or dropped into a report
without further shaping.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

ALPHA = 0.05


def _interpret(p_value: float, alpha: float = ALPHA) -> str:
    """Return a plain-language verdict for a p-value."""
    return (
        "significant (reject H0)" if p_value < alpha else "not significant (retain H0)"
    )


def cramers_v(frame: pd.DataFrame, col_a: str, col_b: str) -> float:
    """Cramer's V effect size for two categorical columns.

    Chi-square p-values collapse to zero on 100k rows, so effect size is what
    actually distinguishes a real driver from a trivial one.
    """
    table = pd.crosstab(frame[col_a], frame[col_b])
    chi2 = stats.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    min_dim = min(table.shape) - 1
    if n == 0 or min_dim == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def chi_square_independence(
    frame: pd.DataFrame, col_a: str, col_b: str, alpha: float = ALPHA
) -> dict:
    """Test whether two categorical variables are independent.

    Chosen because both variables are categorical and the question is one of
    association, not of mean difference.
    """
    table = pd.crosstab(frame[col_a], frame[col_b])
    chi2, p_value, dof, _expected = stats.chi2_contingency(table)
    return {
        "test": "Chi-square test of independence",
        "variables": f"{col_a} vs {col_b}",
        "hypothesis": f"H0: {col_a} and {col_b} are independent",
        "statistic": round(float(chi2), 4),
        "p_value": float(p_value),
        "dof": int(dof),
        "effect_size_cramers_v": round(cramers_v(frame, col_a, col_b), 4),
        "alpha": alpha,
        "conclusion": _interpret(p_value, alpha),
    }


def anova_by_group(
    frame: pd.DataFrame, value_col: str, group_col: str, alpha: float = ALPHA
) -> dict:
    """One-way ANOVA of a numeric column across three or more groups.

    Chosen over repeated t-tests because it compares all group means in a
    single test and avoids inflating the family-wise error rate.
    """
    groups = [
        group[value_col].dropna().to_numpy()
        for _, group in frame.groupby(group_col, observed=True)
        if len(group) > 1
    ]
    if len(groups) < 2:
        raise ValueError(f"Need at least two groups in {group_col!r}.")

    f_stat, p_value = stats.f_oneway(*groups)
    means = frame.groupby(group_col, observed=True)[value_col].mean().round(2)
    return {
        "test": "One-way ANOVA",
        "variables": f"{value_col} across {group_col}",
        "hypothesis": f"H0: mean {value_col} is equal across all {group_col} groups",
        "statistic": round(float(f_stat), 4),
        "p_value": float(p_value),
        "groups": int(len(groups)),
        "group_means": means.to_dict(),
        "alpha": alpha,
        "conclusion": _interpret(p_value, alpha),
    }


def ttest_two_groups(
    frame: pd.DataFrame, value_col: str, group_col: str, alpha: float = ALPHA
) -> dict:
    """Welch's t-test between exactly two groups.

    Welch's variant is used because it does not assume equal variances.
    """
    levels = frame[group_col].dropna().unique()
    if len(levels) != 2:
        raise ValueError(
            f"{group_col!r} must have exactly two levels, found {len(levels)}."
        )

    first = frame.loc[frame[group_col] == levels[0], value_col].dropna()
    second = frame.loc[frame[group_col] == levels[1], value_col].dropna()
    t_stat, p_value = stats.ttest_ind(first, second, equal_var=False)

    pooled_sd = np.sqrt((first.var() + second.var()) / 2)
    cohens_d = (first.mean() - second.mean()) / pooled_sd if pooled_sd else 0.0

    return {
        "test": "Welch's two-sample t-test",
        "variables": f"{value_col} by {group_col}",
        "hypothesis": f"H0: mean {value_col} is equal for {levels[0]} and {levels[1]}",
        "statistic": round(float(t_stat), 4),
        "p_value": float(p_value),
        "group_means": {
            str(levels[0]): round(float(first.mean()), 2),
            str(levels[1]): round(float(second.mean()), 2),
        },
        "effect_size_cohens_d": round(float(cohens_d), 4),
        "alpha": alpha,
        "conclusion": _interpret(p_value, alpha),
    }


def correlation_matrix(
    frame: pd.DataFrame, columns: list[str], method: str = "pearson"
) -> pd.DataFrame:
    """Correlation matrix for the given numeric columns."""
    available = [column for column in columns if column in frame.columns]
    return frame[available].corr(method=method).round(3)


def correlation_with_target(
    frame: pd.DataFrame, columns: list[str], target: str
) -> pd.DataFrame:
    """Rank numeric columns by absolute correlation with a numeric target."""
    available = [
        column for column in columns if column in frame.columns and column != target
    ]
    rows = []
    for column in available:
        subset = frame[[column, target]].dropna()
        if len(subset) < 2:
            continue
        pearson = subset[column].corr(subset[target])
        spearman = subset[column].corr(subset[target], method="spearman")
        rows.append(
            {
                "feature": column,
                "pearson": round(float(pearson), 4),
                "spearman": round(float(spearman), 4),
                "abs_pearson": round(abs(float(pearson)), 4),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("abs_pearson", ascending=False)
        .reset_index(drop=True)
    )


def cancellation_rate_by(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Cancellation and incompletion rate per level of a categorical column."""
    status = frame["booking_status"].astype(str)
    grouped = (
        frame.assign(
            _cancelled=(status == "Cancelled").astype(int),
            _incomplete=(status == "Incomplete").astype(int),
        )
        .groupby(group_col, observed=True)
        .agg(
            rides=("booking_id", "count"),
            cancel_rate=("_cancelled", "mean"),
            incomplete_rate=("_incomplete", "mean"),
        )
        .reset_index()
    )
    grouped["cancel_rate"] = (100 * grouped["cancel_rate"]).round(2)
    grouped["incomplete_rate"] = (100 * grouped["incomplete_rate"]).round(2)
    return grouped.sort_values("cancel_rate", ascending=False)


def run_standard_tests(frame: pd.DataFrame) -> list[dict]:
    """Run the project's headline significance tests.

    Returns:
        One dict per test, ready for :func:`summarise_tests`.
    """
    results = []

    for column in ("traffic_level", "weather_condition", "vehicle_type", "city"):
        if column in frame.columns:
            results.append(
                chi_square_independence(frame, column, "booking_status")
            )

    if {"booking_value", "vehicle_type"} <= set(frame.columns):
        results.append(anova_by_group(frame, "booking_value", "vehicle_type"))

    if {"ride_distance_km", "traffic_level"} <= set(frame.columns):
        results.append(anova_by_group(frame, "surge_multiplier", "traffic_level"))

    if {"booking_value", "is_weekend"} <= set(frame.columns):
        weekend = frame.assign(
            weekend_label=np.where(frame["is_weekend"] == 1, "Weekend", "Weekday")
        )
        results.append(ttest_two_groups(weekend, "booking_value", "weekend_label"))

    return results


def summarise_tests(results: list[dict]) -> pd.DataFrame:
    """Flatten a list of test results into a comparison table."""
    rows = []
    for result in results:
        rows.append(
            {
                "test": result["test"],
                "variables": result["variables"],
                "statistic": result["statistic"],
                "p_value": (
                    "< 0.0001"
                    if result["p_value"] < 1e-4
                    else round(result["p_value"], 4)
                ),
                "effect_size": result.get(
                    "effect_size_cramers_v", result.get("effect_size_cohens_d", "-")
                ),
                "conclusion": result["conclusion"],
            }
        )
    return pd.DataFrame(rows)

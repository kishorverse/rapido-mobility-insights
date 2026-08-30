"""Generate notebooks/01_eda.ipynb from the analysis modules.

Keeping the notebook generated rather than hand-edited means it always calls
the same functions the dashboard and models use - no copy-pasted logic that can
drift out of sync.

Usage:
    python scripts/build_notebook.py [--execute]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        """# Rapido: Intelligent Mobility Insights
## Exploratory Data Analysis

Ride patterns, cancellations and fare forecasting across 100,000 bookings in five cities.

This notebook calls the project package directly (`rapido.*`) rather than reimplementing
logic, so every figure here matches the dashboard and the models.

**Contents**
1. Load and profile
2. Data quality and the leakage checks
3. Univariate analysis
4. Bivariate analysis - what drives cancellations
5. Multivariate analysis
6. Statistical significance
7. Fare structure
8. Feature engineering
9. Conclusions""",
    ),
    (
        "code",
        """import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config
from rapido import cleaning, features, io, stats

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (11, 5)

print("Project:", config.BASE_DIR.name)""",
    ),
    ("markdown", "## 1. Load and Profile"),
    (
        "code",
        """raw = io.load_all_raw()
for name, frame in raw.items():
    print(f"{name:<18} {frame.shape[0]:>7,} rows x {frame.shape[1]:>2} columns")""",
    ),
    ("code", """raw["bookings"].head()"""),
    (
        "code",
        """profile = io.profile_dataframe(raw["bookings"], "bookings")
profile[["column", "dtype", "nulls", "null_pct", "unique"]]""",
    ),
    (
        "markdown",
        """## 2. Data Quality and Leakage Checks

Three checks decide how the whole project is built.""",
    ),
    (
        "markdown",
        """### 2.1 Missingness is structural, not random

`actual_ride_time_min` is null for **exactly** the non-Completed rides. Its null
indicator alone reproduces the target, so it can never be a model feature.""",
    ),
    (
        "code",
        """bookings = raw["bookings"]
check = bookings.groupby("booking_status", observed=True).agg(
    rides=("booking_id", "count"),
    actual_time_null_rate=("actual_ride_time_min", lambda s: s.isna().mean()),
    reason_present_rate=("incomplete_ride_reason", lambda s: s.notna().mean()),
)
check""",
    ),
    (
        "markdown",
        """### 2.2 Fare is a deterministic formula

`booking_value / (base_fare x surge_multiplier)` is confined to a narrow band, so
`base_fare` is excluded from the deployed fare model.""",
    ),
    (
        "code",
        """ratio = bookings["booking_value"] / (bookings["base_fare"] * bookings["surge_multiplier"])
print(ratio.describe())

fig, ax = plt.subplots()
ax.hist(ratio, bins=60, color="#3B7DD8", edgecolor="white")
ax.set_title("booking_value / (base_fare x surge) - uniform noise on [0.95, 1.05]")
ax.set_xlabel("Ratio")
plt.show()""",
    ),
    (
        "markdown",
        """### 2.3 The dimension risk flags are thresholds

`customer_cancel_flag` is exactly `cancellation_rate > 0.20` and `driver_delay_flag`
is exactly `delay_rate > 0.10`. Predicting a flag from its own rate is circular, so
the risk models target booking-level outcomes instead.""",
    ),
    (
        "code",
        """customers, drivers = raw["customers"], raw["drivers"]
print(customers.groupby("customer_cancel_flag")["cancellation_rate"].agg(["min", "max"]))
print()
print(drivers.groupby("driver_delay_flag")["delay_rate"].agg(["min", "max"]))""",
    ),
    ("markdown", "## 3. Clean and Build Features"),
    (
        "code",
        """cleaned = cleaning.clean_all(raw)
summary = cleaning.build_cleaning_summary(raw, cleaned)
summary""",
    ),
    (
        "code",
        """df = features.build_feature_table(cleaned)
print(f"Feature table: {df.shape[0]:,} rows x {df.shape[1]} columns")
df[["booking_id", "booking_ts", "city", "vehicle_type", "ride_distance_km",
    "booking_value", "booking_status"]].head()""",
    ),
    ("markdown", "## 4. Univariate Analysis"),
    (
        "code",
        """fig, axes = plt.subplots(1, 3, figsize=(16, 4))

status_counts = df["booking_status"].value_counts()
axes[0].bar(status_counts.index.astype(str), status_counts.values,
            color=["#2E9E5B", "#D64545", "#E8A33D"])
axes[0].set_title("Booking outcome")

axes[1].hist(df["ride_distance_km"], bins=40, color="#3B7DD8", edgecolor="white")
axes[1].set_title("Ride distance (km)")

axes[2].hist(df["booking_value"], bins=50, color="#3B7DD8", edgecolor="white")
axes[2].set_title("Booking value")

plt.tight_layout()
plt.show()

print(df["booking_status"].value_counts(normalize=True).round(4))""",
    ),
    (
        "code",
        """fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, column in zip(axes, ["traffic_level", "weather_condition", "vehicle_type"]):
    counts = df[column].value_counts()
    ax.bar(counts.index.astype(str), counts.values, color="#3B7DD8")
    ax.set_title(column)
plt.tight_layout()
plt.show()""",
    ),
    ("markdown", "## 5. Bivariate Analysis - What Drives Cancellations"),
    (
        "code",
        """for column in ["traffic_level", "weather_condition", "surge_bucket",
               "city", "vehicle_type", "rush_hour_flag"]:
    print(f"--- {column} ---")
    print(stats.cancellation_rate_by(df, column).to_string(index=False))
    print()""",
    ),
    (
        "code",
        """fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, column in zip(axes, ["traffic_level", "weather_condition", "surge_bucket"]):
    rates = stats.cancellation_rate_by(df, column)
    ax.bar(rates[column].astype(str), rates["cancel_rate"], color="#D64545")
    ax.set_title(f"Cancellation rate by {column}")
    ax.set_ylabel("%")
plt.tight_layout()
plt.show()""",
    ),
    (
        "markdown",
        """**Key finding.** Traffic, weather and surge move the cancellation rate by 15-30
percentage points. City and vehicle type move it by less than one point. The two
failure modes are also distinct: weather drives *cancellations* but leaves the
*incompletion* rate flat, while traffic drives both.""",
    ),
    (
        "code",
        """pivot = df.assign(
    cancelled=(df["booking_status"].astype(str) == "Cancelled").astype(int)
).pivot_table(index="city", columns="hour_of_day", values="cancelled", aggfunc="mean") * 100

fig, ax = plt.subplots(figsize=(15, 3.5))
sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Cancel %"})
ax.set_title("Cancellation rate (%) by city and hour")
plt.tight_layout()
plt.show()""",
    ),
    ("markdown", "## 6. Multivariate Analysis"),
    (
        "code",
        """combo = df.assign(
    cancelled=(df["booking_status"].astype(str) == "Cancelled").astype(int)
).pivot_table(
    index="traffic_level", columns="weather_condition", values="cancelled", aggfunc="mean"
) * 100

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(combo, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
ax.set_title("Cancellation rate (%): traffic x weather")
plt.tight_layout()
plt.show()""",
    ),
    (
        "code",
        """numeric_columns = ["ride_distance_km", "estimated_ride_time_min", "surge_multiplier",
                   "base_fare", "booking_value", "fare_per_km", "zone_avg_wait_min",
                   "avg_customer_rating", "avg_driver_rating"]
correlations = stats.correlation_matrix(df, numeric_columns)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(correlations, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, ax=ax)
ax.set_title("Correlation matrix")
plt.tight_layout()
plt.show()""",
    ),
    ("markdown", "## 7. Statistical Significance"),
    (
        "markdown",
        """**Test choices.**

- *Chi-square test of independence* for categorical driver vs categorical outcome:
  the question is association, not a difference of means.
- *One-way ANOVA* for fare across three vehicle types: compares all group means in
  one test instead of three t-tests, avoiding inflated family-wise error.
- *Welch's t-test* for weekend vs weekday fare: two groups, and it does not assume
  equal variances.
- *Cramer's V* alongside every chi-square, because at n = 100,000 p-values collapse
  toward zero and effect size is what actually separates a real driver from a
  trivial one.""",
    ),
    (
        "code",
        """results = stats.run_standard_tests(df)
stats.summarise_tests(results)""",
    ),
    ("markdown", "## 8. Fare Structure"),
    (
        "code",
        """for vehicle, group in df.groupby("vehicle_type", observed=True):
    slope, intercept = np.polyfit(group["ride_distance_km"], group["base_fare"], 1)
    predicted = slope * group["ride_distance_km"] + intercept
    ss_res = ((group["base_fare"] - predicted) ** 2).sum()
    ss_tot = ((group["base_fare"] - group["base_fare"].mean()) ** 2).sum()
    print(f"{vehicle:5s}  base_fare = {intercept:6.2f} + {slope:5.2f} * km   "
          f"R2 = {1 - ss_res / ss_tot:.6f}")""",
    ),
    (
        "markdown",
        """The tariff is exact: Bike ₹20 + ₹8/km, Auto ₹40 + ₹12/km, Cab ₹80 + ₹18/km.

Since `booking_value = base_fare x surge x (1 ± 5% uniform)`, and a uniform variable
on [0.95, 1.05] has mean absolute deviation 2.5%, **no model can achieve better than
2.50% MAPE**. The trained model reaches 2.76% - essentially the noise floor.""",
    ),
    (
        "code",
        """sample = df.sample(4000, random_state=42)
fig, ax = plt.subplots(figsize=(11, 5))
for vehicle, group in sample.groupby("vehicle_type", observed=True):
    ax.scatter(group["ride_distance_km"], group["booking_value"], s=8, alpha=0.5, label=vehicle)
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Booking value")
ax.set_title("Distance vs fare - three tariff bands, widened by surge")
ax.legend()
plt.show()""",
    ),
    ("markdown", "## 9. Feature Engineering"),
    (
        "markdown",
        """Engineered features fall into two groups.

**Context features** come from the booking row: `rush_hour_flag`, `long_distance_flag`,
`city_route_pair`, `expected_speed_kmph`, `surge_bucket`, `adverse_conditions_flag`,
`fare_per_km`, `fare_per_min`.

**History features** describe the customer or driver and are built as *prior* history:
an expanding window over strictly earlier bookings, so a booking never contributes to
its own predictors. The static `cancellation_rate` and `delay_rate` columns are
whole-period aggregates that already include the row being predicted, so they are
replaced rather than used.

The proof is in the null counts below: exactly one null per customer and per driver -
their first-ever ride, which has no prior history.""",
    ),
    (
        "code",
        """history_columns = [c for c in df.columns if c.startswith(("cust_prior", "drv_prior"))]
print(df[history_columns].isna().sum())
print()
print(f"Customers: {df['customer_id'].nunique():,}   Drivers: {df['driver_id'].nunique():,}")""",
    ),
    (
        "code",
        """df.groupby(pd.cut(df["cust_prior_cancel_rate"], bins=[0, .1, .2, .3, .5, 1.0]),
           observed=True).apply(
    lambda g: pd.Series({
        "bookings": len(g),
        "actual_cancel_rate": 100 * (g["booking_status"].astype(str) == "Cancelled").mean(),
    })
)""",
    ),
    ("markdown", "## 10. Conclusions"),
    (
        "markdown",
        """### Findings

1. **Conditions drive outcomes, not geography.** High traffic lifts cancellations to
   33.5% (vs ~18%), heavy rain to 33.7% (vs 10.0%). City and vehicle type are not
   statistically significant.
2. **Surge is the strongest single lever.** Cancellations rise from 5.3% at no surge
   to 35.3% above 2.0x - and surge is platform-controlled.
3. **Two distinct failure modes.** Weather drives cancellations but not incompletions;
   traffic drives both. Rider-side and driver-side interventions differ.
4. **Fare is a solved formula**, recovered exactly. The pre-quote model operates at
   the theoretical noise floor.

### Business recommendations

- Cap surge during heavy rain and high traffic, where risk compounds.
- Allocate drivers on live traffic and zone demand rather than city targets.
- Score bookings before dispatch with the cancellation model (ROC-AUC 0.851) and hold
  driver assignment on high-risk requests.
- Route reliable drivers to high-traffic windows: Driver Delay is the largest single
  incomplete-ride reason.

### Next steps

`scripts/manage.py train` trains the four models on this feature table; `app.py` serves
the dashboard and live predictions.""",
    ),
]


def build() -> nbf.NotebookNode:
    """Assemble the notebook from the cell definitions."""
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(source) if kind == "markdown"
        else nbf.v4.new_code_cell(source)
        for kind, source in CELLS
    ]
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    }
    return notebook


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="Run the notebook after writing it."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = Path(__file__).resolve().parents[1]
    target = root / "notebooks" / "01_eda.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)

    notebook = build()

    if args.execute:
        from nbclient import NotebookClient

        logger.info("Executing notebook (this takes a minute)...")
        client = NotebookClient(
            notebook, timeout=900, kernel_name="python3", resources={
                "metadata": {"path": str(target.parent)}
            }
        )
        client.execute()

    nbf.write(notebook, target)
    logger.info("Wrote %s (%d cells)", target, len(notebook.cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

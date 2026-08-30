"""Profile the raw Rapido source files and write a data-quality report.

Usage:
    python scripts/profile_raw.py [--output docs/data_quality_report.md]

The report records structure, missingness, referential integrity, categorical
domains and the three leakage checks documented in docs/PROJECT_PLAN.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from rapido import io  # noqa: E402

logger = logging.getLogger(__name__)


def check_referential_integrity(
    bookings: pd.DataFrame,
    customers: pd.DataFrame,
    drivers: pd.DataFrame,
) -> pd.DataFrame:
    """Count booking rows whose foreign keys are missing from a dimension."""
    checks = [
        ("bookings.customer_id -> customers", "customer_id", customers, "customer_id"),
        ("bookings.driver_id -> drivers", "driver_id", drivers, "driver_id"),
    ]
    rows = []
    for label, fact_key, dimension, dim_key in checks:
        orphans = int((~bookings[fact_key].isin(dimension[dim_key])).sum())
        rows.append(
            {
                "check": label,
                "orphan_rows": orphans,
                "status": "PASS" if orphans == 0 else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def check_duplicates(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count duplicate primary keys in each dimension and the fact table."""
    keys = {
        "bookings": "booking_id",
        "customers": "customer_id",
        "drivers": "driver_id",
        "time_features": "datetime",
    }
    rows = []
    for name, key in keys.items():
        duplicates = int(frames[name][key].duplicated().sum())
        rows.append(
            {
                "dataset": name,
                "key": key,
                "duplicates": duplicates,
                "status": "PASS" if duplicates == 0 else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def check_outcome_leakage(bookings: pd.DataFrame) -> pd.DataFrame:
    """Measure how perfectly post-outcome nulls reveal ``booking_status``."""
    null_rate = (
        bookings.groupby("booking_status", observed=True)["actual_ride_time_min"]
        .apply(lambda column: column.isna().mean())
        .rename("actual_ride_time_null_rate")
        .reset_index()
    )
    reason_rate = (
        bookings.groupby("booking_status", observed=True)["incomplete_ride_reason"]
        .apply(lambda column: column.notna().mean())
        .rename("incomplete_reason_present_rate")
        .reset_index()
    )
    return null_rate.merge(reason_rate, on="booking_status")


def check_fare_formula(bookings: pd.DataFrame) -> dict:
    """Quantify how closely ``base_fare * surge`` reproduces ``booking_value``."""
    ratio = bookings["booking_value"] / (
        bookings["base_fare"] * bookings["surge_multiplier"]
    )
    return {
        "mean_ratio": round(float(ratio.mean()), 5),
        "std_ratio": round(float(ratio.std()), 5),
        "min_ratio": round(float(ratio.min()), 5),
        "max_ratio": round(float(ratio.max()), 5),
    }


def check_flag_thresholds(
    customers: pd.DataFrame, drivers: pd.DataFrame
) -> pd.DataFrame:
    """Show that the dimension risk flags are pure thresholds on a rate column."""
    rows = []
    for label, frame, flag, rate in [
        ("customers", customers, "customer_cancel_flag", "cancellation_rate"),
        ("drivers", drivers, "driver_delay_flag", "delay_rate"),
    ]:
        grouped = frame.groupby(flag)[rate].agg(["min", "max", "count"])
        for flag_value, stats in grouped.iterrows():
            rows.append(
                {
                    "dataset": label,
                    "flag": flag,
                    "flag_value": flag_value,
                    "rate_column": rate,
                    "rate_min": round(float(stats["min"]), 4),
                    "rate_max": round(float(stats["max"]), 4),
                    "rows": int(stats["count"]),
                }
            )
    return pd.DataFrame(rows)


def check_zero_variance(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """List every column carrying exactly one distinct non-null value."""
    rows = []
    for name, frame in frames.items():
        for column in frame.columns:
            if frame[column].nunique(dropna=True) <= 1:
                rows.append(
                    {
                        "dataset": name,
                        "column": column,
                        "distinct_values": int(frame[column].nunique(dropna=True)),
                        "value": str(frame[column].dropna().unique()[:1]),
                    }
                )
    return pd.DataFrame(rows)


def _to_markdown(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table."""
    return frame.to_markdown(index=False)


def build_report(frames: dict[str, pd.DataFrame]) -> str:
    """Assemble the full Markdown data-quality report."""
    bookings = frames["bookings"]
    customers = frames["customers"]
    drivers = frames["drivers"]

    shape_table = pd.DataFrame(
        [
            {"dataset": name, "rows": len(frame), "columns": frame.shape[1]}
            for name, frame in frames.items()
        ]
    )

    profile = pd.concat(
        [io.profile_dataframe(frame, name) for name, frame in frames.items()],
        ignore_index=True,
    )
    missing = profile[profile["nulls"] > 0][
        ["dataset", "column", "nulls", "null_pct"]
    ]

    status_counts = (
        bookings["booking_status"]
        .value_counts()
        .rename_axis("booking_status")
        .reset_index(name="rows")
    )
    status_counts["share_pct"] = (
        100 * status_counts["rows"] / len(bookings)
    ).round(2)

    fare = check_fare_formula(bookings)

    parts = [
        "# Data Quality Report",
        "",
        f"Generated from `{config.RAW_DIR.name}/` "
        f"({len(frames)} source files).",
        "",
        "## 1. Structure",
        "",
        _to_markdown(shape_table),
        "",
        f"Booking date range: **{bookings['booking_date'].min()}** to "
        f"**{bookings['booking_date'].max()}**.",
        "",
        "## 2. Target Distribution",
        "",
        _to_markdown(status_counts),
        "",
        "## 3. Missing Values",
        "",
        _to_markdown(missing) if len(missing) else "_No missing values._",
        "",
        "Missingness here is **structural, not random** - see section 6.",
        "",
        "## 4. Duplicate Keys",
        "",
        _to_markdown(check_duplicates(frames)),
        "",
        "## 5. Referential Integrity",
        "",
        _to_markdown(check_referential_integrity(bookings, customers, drivers)),
        "",
        "## 6. Leakage Check 1 - Post-Outcome Columns",
        "",
        _to_markdown(check_outcome_leakage(bookings)),
        "",
        "`actual_ride_time_min` is null for **every** non-Completed ride, so its "
        "null indicator alone reproduces the target. Both columns are excluded "
        "from all model feature matrices.",
        "",
        "## 7. Leakage Check 2 - Fare Is a Formula",
        "",
        "`booking_value / (base_fare * surge_multiplier)`:",
        "",
        _to_markdown(pd.DataFrame([fare])),
        "",
        f"The ratio sits in [{fare['min_ratio']}, {fare['max_ratio']}] with "
        f"standard deviation {fare['std_ratio']}. Fare is `base_fare * surge` "
        "plus roughly 5% noise, so `base_fare` is excluded from the pre-quote "
        "fare model.",
        "",
        "## 8. Leakage Check 3 - Dimension Flags Are Thresholds",
        "",
        _to_markdown(check_flag_thresholds(customers, drivers)),
        "",
        "The flags are exact cut-offs on their own rate column, so they cannot "
        "serve as model targets. Booking-level outcomes are used instead.",
        "",
        "## 9. Zero-Variance Columns",
        "",
        _to_markdown(check_zero_variance(frames)),
        "",
        "## 10. Full Column Profile",
        "",
        _to_markdown(profile),
        "",
    ]
    return "\n".join(parts)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DOCS_DIR / "data_quality_report.md",
        help="Destination Markdown file.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        frames = io.load_all_raw()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Could not load raw data: %s", exc)
        return 1

    report = build_report(frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    logger.info("Wrote %s (%d characters)", args.output, len(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

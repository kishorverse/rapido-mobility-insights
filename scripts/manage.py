"""Command-line entry point for the Rapido pipeline.

One CLI for the two operations that change project state: loading MySQL and
training models. Everything it does lives in :mod:`rapido`; this module only
parses arguments and prints results.

Usage:
    python scripts/manage.py etl --rebuild        # drop and reload MySQL
    python scripts/manage.py etl --verify-only    # connectivity + row counts
    python scripts/manage.py train                # train all four models
    python scripts/manage.py train --model fare --tune
    python scripts/manage.py train --tune --search grid
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapido import cleaning, db, etl, features, io  # noqa: E402
from rapido.models import train  # noqa: E402

logger = logging.getLogger(__name__)


def load_feature_table(rebuild: bool = False):
    """Load the cached feature table, rebuilding it from raw data if needed."""
    if not rebuild and io.processed_exists("features"):
        logger.info("Loading cached feature table")
        return io.load_processed("features")

    logger.info("Building feature table from raw data")
    cleaned = cleaning.clean_all(io.load_all_raw())
    frame = features.build_feature_table(cleaned)
    io.save_processed(frame, "features")
    return frame


def run_etl_command(args: argparse.Namespace) -> int:
    """Load the cleaned data into MySQL, or just report what is already there."""
    try:
        if args.verify_only:
            status = db.healthcheck()
            print(f"Connected: {status['connected']}  ({status['database']})")
            for table, count in status["tables"].items():
                print(f"  {table:<18} {count if count is not None else 'missing'}")
            return 0

        result = etl.run_etl(rebuild=args.rebuild, cache=not args.no_cache)
        print("\nRows inserted:")
        for table, count in result["inserted"].items():
            print(f"  {table:<18} {count:>8,}")
        print("\nVerification:")
        print(result["verification"].to_string(index=False))

        failures = result["verification"]["status"].eq("FAIL").sum()
        if failures:
            logger.error("%d table(s) failed verification", failures)
            return 1
        return 0

    except db.DatabaseError as exc:
        logger.error("%s", exc)
        return 1


def run_train_command(args: argparse.Namespace) -> int:
    """Train one model or all four, then print the headline metrics."""
    frame = load_feature_table(rebuild=args.rebuild_features)

    if args.model == "all":
        results = train.train_all_models(
            frame, tune=args.tune, search_strategy=args.search
        )
    else:
        results = {
            args.model: train.TRAINERS[args.model](
                frame, tune=args.tune, search_strategy=args.search
            )
        }

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    for key, result in results.items():
        if "metrics" not in result:
            continue
        print(f"\n{key}  ({result.get('algorithm', '-')})")
        for metric, value in result["metrics"].items():
            print(f"   {metric:<22} {value}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Assemble the top-level parser and its subcommands."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    etl_parser = subcommands.add_parser("etl", help="Load the data into MySQL.")
    etl_parser.add_argument(
        "--rebuild", action="store_true", help="Drop and recreate all tables."
    )
    etl_parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only report connectivity and row counts.",
    )
    etl_parser.add_argument(
        "--no-cache", action="store_true", help="Skip writing cleaned Parquet caches."
    )
    etl_parser.set_defaults(handler=run_etl_command)

    train_parser = subcommands.add_parser("train", help="Train the models.")
    train_parser.add_argument(
        "--model",
        choices=["all", "outcome", "fare", "customer_risk", "driver_risk"],
        default="all",
        help="Which model to train.",
    )
    train_parser.add_argument(
        "--tune", action="store_true", help="Run a hyperparameter search."
    )
    train_parser.add_argument(
        "--search",
        choices=["random", "grid"],
        default="random",
        help="Search strategy used with --tune: sampled or exhaustive.",
    )
    train_parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Rebuild the feature table instead of using the Parquet cache.",
    )
    train_parser.set_defaults(handler=run_train_command)

    return parser


def main() -> int:
    """Parse arguments and dispatch to the chosen subcommand."""
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

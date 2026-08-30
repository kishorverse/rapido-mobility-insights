"""Train the Rapido models and persist them to models/.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --model fare --tune
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapido import cleaning, features, io  # noqa: E402
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


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["all", "outcome", "fare", "customer_risk", "driver_risk"],
        default="all",
        help="Which model to train.",
    )
    parser.add_argument(
        "--tune", action="store_true", help="Run a hyperparameter search."
    )
    parser.add_argument(
        "--search",
        choices=["random", "grid"],
        default="random",
        help="Search strategy used with --tune: sampled or exhaustive.",
    )
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Rebuild the feature table instead of using the Parquet cache.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(message)s", stream=sys.stdout
    )

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


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the Rapido ETL pipeline into MySQL.

Usage:
    python scripts/run_etl.py --rebuild      # drop and reload everything
    python scripts/run_etl.py --verify-only  # report current row counts
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapido import db, etl  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild", action="store_true", help="Drop and recreate all tables."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only report connectivity and row counts.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip writing cleaned Parquet caches.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

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


if __name__ == "__main__":
    raise SystemExit(main())

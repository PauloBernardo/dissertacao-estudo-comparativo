"""Download and validate all datasets for Tier 1 (and optionally Tier 2+).

Usage:
    python scripts/download_data.py --tier 1
    python scripts/download_data.py --tier 2
    python scripts/download_data.py --all
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import DatasetLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TIER1 = ["BCW", "PID", "HAB", "VCP", "GCR", "AUS"]
TIER2 = ["ADULT", "BANK", "CREDIT", "TELCO", "SHOPPERS", "HIGGS50K"]
TIER3 = ["COVER", "KDD99"]
SYNTHETIC = ["TWS", "TWM", "TWC"]


def download_and_validate(names: list[str]) -> dict[str, bool]:
    results = {}
    for name in names:
        try:
            X, y, meta = DatasetLoader.load(name)
            assert X.ndim == 2, "X must be 2D"
            assert y.ndim == 1, "y must be 1D"
            assert len(X) == len(y), "X/y length mismatch"
            assert not any(map(lambda c: c != c, X.flat)), "NaN in X"
            assert set(y).issubset({0, 1}), f"Labels not in {{0,1}}: {set(y)}"
            logger.info(
                "✓ %-10s  n=%6d  p=%3d  class_ratio=%.3f",
                name, len(y), X.shape[1], float(y.mean()),
            )
            results[name] = True
        except Exception as e:
            logger.error("✗ %-10s  ERROR: %s", name, e)
            results[name] = False
    return results


def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tier", type=int, choices=[1, 2, 3], help="Download a specific tier")
    group.add_argument("--all", action="store_true", help="Download all tiers + synthetic")
    group.add_argument(
        "--datasets",
        nargs="+",
        help="Download only the explicitly listed datasets (e.g. ADULT BANK)",
    )
    args = parser.parse_args()

    if args.all:
        names = TIER1 + TIER2 + TIER3 + SYNTHETIC
    elif args.datasets:
        names = args.datasets
    elif args.tier == 1:
        names = TIER1 + SYNTHETIC
    elif args.tier == 2:
        names = TIER1 + TIER2 + SYNTHETIC
    elif args.tier == 3:
        names = TIER1 + TIER2 + TIER3 + SYNTHETIC

    logger.info("Downloading %d datasets: %s", len(names), names)
    results = download_and_validate(names)

    n_ok = sum(results.values())
    n_fail = len(results) - n_ok
    logger.info("Done: %d OK, %d FAILED", n_ok, n_fail)

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

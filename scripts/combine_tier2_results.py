#!/usr/bin/env python3
"""Combine the official Tier 2 CPU and Transformer result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=Path, default=ROOT / "results" / "tier2_gridcv.json")
    parser.add_argument(
        "--transformers",
        type=Path,
        default=ROOT / "results" / "tier2_transformers.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "tier2_combined.json")
    args = parser.parse_args()

    cpu = json.loads(args.cpu.read_text())
    transformers = json.loads(args.transformers.read_text())
    combined = cpu + transformers

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n")

    print(f"cpu records: {len(cpu)}")
    print(f"transformer records: {len(transformers)}")
    print(f"combined records: {len(combined)}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()

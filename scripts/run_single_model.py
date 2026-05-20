#!/usr/bin/env python3
"""Run a single model × dataset × seed experiment and print the result.

Usage
-----
    python scripts/run_single_model.py \
        --model StandardLSSVM \
        --dataset BCW \
        --seed 0 \
        --params sigma=3.0 tau=1.0 \
        --output results/single.json

Model names
-----------
    StandardLSSVM, ADMMNesterovLSSVM, PCPLSSVm, FSALSSVm,
    PruningLSSVM, IPLSSVm, OppositeMapsLSSVM, FTTransformer
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _parse_params(items: list[str]) -> dict:
    """Convert ["k=v", ...] to typed dict."""
    out = {}
    for item in items:
        k, _, v = item.partition("=")
        # Try int, then float, then leave as string
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        out[k] = v
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single experiment.")
    parser.add_argument("--model", required=True, help="Model class name")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. BCW)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--test-size", type=float, default=0.30,
                        help="Test fraction (default: 0.30)")
    parser.add_argument("--params", nargs="*", default=[],
                        metavar="KEY=VALUE",
                        help="Model hyperparameters (e.g. sigma=3.0 tau=1.0)")
    parser.add_argument("--output", default=None,
                        help="Path to save JSON result (optional)")
    args = parser.parse_args()

    model_params = _parse_params(args.params)

    from src.experiments.runner import run_single_experiment

    result = run_single_experiment(
        model_name=args.model,
        dataset_name=args.dataset,
        seed=args.seed,
        model_params=model_params,
        test_size=args.test_size,
    )

    print(json.dumps(result, indent=2, default=str))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str))
        logging.getLogger(__name__).info("Result saved to %s", out_path)

    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

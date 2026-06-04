#!/usr/bin/env python3
"""Tune a model on a dataset using Bayesian Optimisation (Optuna).

Usage
-----
    python scripts/tune_model.py \
        --model StandardLSSVM \
        --dataset BCW \
        --trials 100 \
        --folds 5 \
        --metric f1_macro \
        --seed 0 \
        --output results/tuning/StandardLSSVM_BCW.json
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune a model with Optuna.")
    parser.add_argument("--model", required=True, help="Model class name or config key")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. BCW)")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--folds", type=int, default=5, help="CV folds")
    parser.add_argument("--metric", default="f1_macro",
                        choices=["f1_macro", "accuracy", "auc_roc"],
                        help="Optimisation metric")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=None,
                        help="Time budget in seconds (optional)")
    parser.add_argument("--output", default=None, help="Path to save JSON result")
    args = parser.parse_args()

    from src.tuning.bayesian import tune_model

    result = tune_model(
        model_name=args.model,
        dataset_name=args.dataset,
        n_trials=args.trials,
        cv_folds=args.folds,
        metric=args.metric,
        seed=args.seed,
        timeout=args.timeout,
    )

    print(json.dumps(result, indent=2, default=str))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str))
        logging.getLogger(__name__).info("Result saved to %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

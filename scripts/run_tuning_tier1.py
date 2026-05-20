#!/usr/bin/env python3
"""Tune hyperparameters for all models on Tier 1 datasets.

Saves best params to results/tuning/best_params.json.
Skips combinations that are already saved (resumable).

Usage
-----
    python scripts/run_tuning_tier1.py [--trials-lssvm 100] [--trials-transformer 30]
                                       [--folds 5] [--seed 0] [--jobs 1]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/tuning/tuning.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

TIER1_DATASETS = ["BCW", "PID", "HAB", "VCP", "GCR", "AUS", "TWS", "TWM", "TWC"]

# (config_key, class_name_for_runner, is_transformer)
MODELS = [
    ("lssvm_standard",       "StandardLSSVM",      False),
    ("lssvm_pcp",            "PCPLSSVm",           False),
    ("lssvm_fsa",            "FSALSSVm",           False),
    ("lssvm_pruning",        "PruningLSSVM",        False),
    ("lssvm_ip",             "IPLSSVm",            False),
    ("lssvm_opposite_maps",  "OppositeMapsLSSVM",  False),
    ("ft_transformer",       "FTTransformer_softmax",  True),
    ("ft_transformer_topk",  "FTTransformer_topk",     True),
    ("ft_transformer_entmax","FTTransformer_entmax",   True),
    ("ft_transformer_sparsemax","FTTransformer_sparsemax", True),
]

OUT_FILE = Path("results/tuning/best_params.json")


def _load_existing() -> dict:
    if OUT_FILE.exists():
        return json.loads(OUT_FILE.read_text())
    return {}


def _save(data: dict) -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(data, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-lssvm", type=int, default=100)
    parser.add_argument("--trials-transformer", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-transformer", type=float, default=600.0,
                        help="Max seconds per transformer (model, dataset) combo")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.tuning.bayesian import tune_model

    existing = _load_existing()
    total = len(MODELS) * len(TIER1_DATASETS)
    done = 0
    errors = 0

    log.info("=== Tier 1 Tuning ===")
    log.info("Models: %d | Datasets: %d | Total combos: %d", len(MODELS), len(TIER1_DATASETS), total)
    log.info("LSSVM trials: %d | Transformer trials: %d (timeout: %.0fs)",
             args.trials_lssvm, args.trials_transformer, args.timeout_transformer)

    for config_key, runner_name, is_transformer in MODELS:
        for dataset in TIER1_DATASETS:
            key = f"{runner_name}__{dataset}"
            if key in existing:
                log.info("[SKIP] %s / %s (already tuned)", runner_name, dataset)
                done += 1
                continue

            n_trials = args.trials_transformer if is_transformer else args.trials_lssvm
            timeout = args.timeout_transformer if is_transformer else None
            # Use fewer epochs during transformer tuning to save time
            fixed_override = {"max_epochs": 50, "patience": 10} if is_transformer else None

            log.info("[RUN ] %s / %s (%d trials)...", runner_name, dataset, n_trials)
            t0 = time.perf_counter()
            try:
                result = tune_model(
                    model_name=config_key,
                    dataset_name=dataset,
                    n_trials=n_trials,
                    cv_folds=args.folds,
                    metric="f1_macro",
                    seed=args.seed,
                    timeout=timeout,
                    fixed_params_override=fixed_override,
                )
                elapsed = time.perf_counter() - t0

                # For transformers, restore full training epochs in saved params
                if is_transformer:
                    result["best_params"]["max_epochs"] = 200
                    result["best_params"]["patience"] = 20

                existing[key] = result
                _save(existing)
                done += 1
                log.info("[OK  ] %s / %s — %s=%.4f in %.1fs",
                         runner_name, dataset, result["metric"],
                         result["best_value"], elapsed)

            except Exception as exc:
                errors += 1
                elapsed = time.perf_counter() - t0
                log.error("[ERR ] %s / %s — %s (%.1fs)", runner_name, dataset, exc, elapsed)
                existing[key] = {"error": str(exc), "model": runner_name, "dataset": dataset}
                _save(existing)

    log.info("=== Tuning complete: %d/%d ok, %d errors ===", done - errors, total, errors)
    log.info("Best params saved to %s", OUT_FILE)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tune hyperparameters for all models on MK5 datasets.

Saves to results/tuning/best_params_mk5.json.

Usage
-----
    python scripts/run_tuning_mk5.py [--trials-lssvm 100] [--trials-transformer 30]
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
        logging.FileHandler("results/tuning/tuning_mk5.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

MK5_DATASETS = ["MKE", "MKM", "MKH"]

MODELS = [
    ("lssvm_standard",           "StandardLSSVM",          False),
    ("lssvm_pcp",                "PCPLSSVm",               False),
    ("lssvm_fsa",                "FSALSSVm",               False),
    ("lssvm_pruning",            "PruningLSSVM",           False),
    ("lssvm_ip",                 "IPLSSVm",                False),
    ("lssvm_opposite_maps",      "OppositeMapsLSSVM",      False),
    ("lssvm_admm_nesterov",      "ADMMNesterovLSSVM",      False),
    ("ft_transformer",           "FTTransformer_softmax",  True),
    ("ft_transformer_topk",      "FTTransformer_topk",     True),
    ("ft_transformer_entmax",    "FTTransformer_entmax",   True),
    ("ft_transformer_sparsemax", "FTTransformer_sparsemax",True),
]

OUT_FILE = Path("results/tuning/best_params_mk5.json")


def _load_existing() -> dict:
    return json.loads(OUT_FILE.read_text()) if OUT_FILE.exists() else {}


def _save(data: dict) -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(data, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-lssvm", type=int, default=100)
    parser.add_argument("--trials-transformer", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-transformer", type=float, default=600.0)
    args = parser.parse_args()

    from src.tuning.bayesian import tune_model

    existing = _load_existing()
    total = len(MODELS) * len(MK5_DATASETS)
    done = errors = 0

    log.info("=== MK5 Tuning (5 informative features) ===")
    log.info("Models: %d | Datasets: %s | Total: %d", len(MODELS), MK5_DATASETS, total)

    for config_key, runner_name, is_transformer in MODELS:
        for dataset in MK5_DATASETS:
            key = f"{runner_name}__{dataset}"
            if key in existing:
                log.info("[SKIP] %s / %s", runner_name, dataset)
                done += 1
                continue

            n_trials = args.trials_transformer if is_transformer else args.trials_lssvm
            timeout = args.timeout_transformer if is_transformer else None
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
                if is_transformer:
                    result["best_params"]["max_epochs"] = 200
                    result["best_params"]["patience"] = 20
                existing[key] = result
                _save(existing)
                done += 1
                log.info("[OK  ] %s / %s — %.4f in %.1fs",
                         runner_name, dataset, result["best_value"], elapsed)
            except Exception as exc:
                errors += 1
                log.error("[ERR ] %s / %s — %s", runner_name, dataset, exc)
                existing[key] = {"error": str(exc)}
                _save(existing)

    log.info("=== Done: %d/%d ok, %d errors ===", done - errors, total, errors)


if __name__ == "__main__":
    main()

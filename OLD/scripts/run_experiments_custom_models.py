#!/usr/bin/env python3
"""Run NystromLSSVMColnorm and FTTransformerCURColnorm on Tier 1 datasets.

Saves to results/tier1_custom_models.json — does NOT modify tier1_results.json.

Usage
-----
    python scripts/run_experiments_custom_models.py [--seeds 30]
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
        logging.FileHandler("results/tier1_custom_experiment.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

TIER1_DATASETS = ["BCW", "PID", "HAB", "VCP", "GCR", "AUS", "TWS", "TWM", "TWC"]

MODELS = {
    "NystromLSSVMColnorm":    "NystromLSSVMColnorm",
    "FTTransformerCURColnorm": "FTTransformerCURColnorm",
}

DEFAULT_PARAMS = {
    "NystromLSSVMColnorm":     {"sigma": 1.0, "gamma": 1.0, "m_ratio": 0.20},
    "FTTransformerCURColnorm": {"d_model": 32, "n_heads": 4, "n_layers": 2,
                                "m_ratio": 0.10, "lr": 5e-4,
                                "epochs": 200, "patience": 20},
}


def _load_best_params(params_file: Path) -> dict:
    tuned = {}
    if params_file.exists():
        raw = json.loads(params_file.read_text())
        for key, val in raw.items():
            if "best_params" in val:
                tuned[key] = val["best_params"]
    return tuned


def _resolve_params(runner_name: str, dataset: str, tuned: dict) -> dict:
    key = f"{runner_name}__{dataset}"
    if key in tuned:
        return dict(tuned[key])
    log.warning("No tuned params for %s / %s — using defaults", runner_name, dataset)
    return dict(DEFAULT_PARAMS.get(runner_name, {}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--params", type=Path,
                        default=Path("results/tuning/best_params_custom.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/tier1_custom_models.json"))
    args = parser.parse_args()

    from src.experiments.runner import run_single_experiment

    tuned = _load_best_params(args.params)
    log.info("Loaded tuned params for %d combos", len(tuned))

    seeds = list(range(args.seeds))
    total = len(MODELS) * len(TIER1_DATASETS) * len(seeds)

    log.info("=== Custom Models Tier 1 Experiment ===")
    log.info("Models: %s | Datasets: %d | Seeds: %d | Total: %d",
             list(MODELS.keys()), len(TIER1_DATASETS), len(seeds), total)

    existing_results = []
    existing_keys: set[str] = set()
    if args.output.exists():
        existing_results = json.loads(args.output.read_text())
        for r in existing_results:
            variant = r.get("model_variant") or r.get("model")
            k = f"{variant}__{r.get('dataset')}__{r.get('seed')}"
            existing_keys.add(k)
        log.info("Resuming: %d results already saved", len(existing_results))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_results = list(existing_results)
    completed = len(existing_keys)
    errors = 0
    t_start = time.perf_counter()

    for runner_name in MODELS:
        for dataset in TIER1_DATASETS:
            params = _resolve_params(runner_name, dataset, tuned)

            for seed in seeds:
                run_key = f"{runner_name}__{dataset}__{seed}"
                if run_key in existing_keys:
                    continue

                result = run_single_experiment(
                    model_name=runner_name,
                    dataset_name=dataset,
                    seed=seed,
                    model_params=params,
                )
                result["model_variant"] = runner_name

                all_results.append(result)
                existing_keys.add(run_key)
                completed += 1

                if result["status"] != "ok":
                    errors += 1

                if completed % 10 == 0:
                    args.output.write_text(json.dumps(all_results, indent=2, default=str))

                elapsed = time.perf_counter() - t_start
                remaining = total - completed
                eta = (elapsed / completed * remaining) if completed > 0 else 0
                log.info("[%d/%d] %s / %s / seed=%d — %s | ETA %.0fm",
                         completed, total, runner_name, dataset, seed,
                         result["status"], eta / 60)

    args.output.write_text(json.dumps(all_results, indent=2, default=str))
    log.info("=== Done: %d/%d ok, %d errors ===", completed - errors, total, errors)

    # Quick summary vs Tier 1 best
    import numpy as np
    from collections import defaultdict
    scores: dict = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        mv = r.get("model_variant") or r.get("model")
        ds = r.get("dataset", "")
        scores[mv][ds].append(r.get("f1_macro", float("nan")))

    print(f"\n=== F1-macro summary (mean ± std across 9 datasets × 30 seeds) ===")
    for rn in MODELS:
        all_f1 = [v for ds_vals in scores[rn].values() for v in ds_vals]
        if all_f1:
            print(f"  {rn:<28} {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")


if __name__ == "__main__":
    main()

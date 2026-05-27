#!/usr/bin/env python3
"""MK5 experiment: 5 truly informative features, N=400, 3 difficulty levels.

Datasets generated via sklearn.make_classification:
  MKE — Easy   (class_sep=2.0, 1 cluster/class,  flip_y=0.01)
  MKM — Medium (class_sep=1.0, 2 clusters/class, flip_y=0.05)
  MKH — Hard   (class_sep=0.5, 3 clusters/class, flip_y=0.10)

Uses best_params tuned for MK5 datasets (results/tuning/best_params_mk5.json).
Falls back to Tier 1 params if tuning not yet run.

Saves to results/synthetic_mk5.json.

Usage
-----
    # 1. Tune (optional but recommended)
    python scripts/run_tuning_mk5.py

    # 2. Run experiment
    python scripts/run_synthetic_mk5.py [--seeds 30]
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
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

MK5_DATASETS = ["MKE", "MKM", "MKH"]

RUNNER_TO_TUNING_KEY = {
    "StandardLSSVM":           "StandardLSSVM",
    "PCPLSSVm":                "PCPLSSVm",
    "FSALSSVm":                "FSALSSVm",
    "PruningLSSVM":            "PruningLSSVM",
    "IPLSSVm":                 "IPLSSVm",
    "OppositeMapsLSSVM":       "OppositeMapsLSSVM",
    "ADMMNesterovLSSVM":       "ADMMNesterovLSSVM",
    "FTTransformer_softmax":   "FTTransformer_softmax",
    "FTTransformer_topk":      "FTTransformer_topk",
    "FTTransformer_entmax":    "FTTransformer_entmax",
    "FTTransformer_sparsemax": "FTTransformer_sparsemax",
}

TRANSFORMER_VARIANTS = {
    "FTTransformer_softmax", "FTTransformer_topk",
    "FTTransformer_entmax", "FTTransformer_sparsemax",
}

DEFAULT_PARAMS = {
    "StandardLSSVM":        {"sigma": 1.0, "tau": 1.0},
    "PCPLSSVm":             {"sigma": 1.0, "tau": 1.0, "rank": 50},
    "FSALSSVm":             {"sigma": 1.0, "tau": 1.0, "n_atoms": 50},
    "PruningLSSVM":         {"sigma": 1.0, "tau": 1.0},
    "IPLSSVm":              {"sigma": 1.0, "tau": 1.0},
    "OppositeMapsLSSVM":    {"sigma": 1.0, "tau": 1.0, "n_prototypes": 10},
    "ADMMNesterovLSSVM":    {"sigma": 1.0, "tau": 1.0},
    "FTTransformer_softmax":   {"embedding_dim": 64, "num_blocks": 3, "num_heads": 4,
                                "max_epochs": 200, "patience": 20, "attention_type": "softmax"},
    "FTTransformer_topk":      {"embedding_dim": 64, "num_blocks": 3, "num_heads": 4,
                                "max_epochs": 200, "patience": 20, "attention_type": "topk",
                                "topk_ratio": 0.25},
    "FTTransformer_entmax":    {"embedding_dim": 64, "num_blocks": 3, "num_heads": 4,
                                "max_epochs": 200, "patience": 20, "attention_type": "entmax",
                                "alpha": 1.5},
    "FTTransformer_sparsemax": {"embedding_dim": 64, "num_blocks": 3, "num_heads": 4,
                                "max_epochs": 200, "patience": 20, "attention_type": "sparsemax"},
}


def _load_best_params(params_file: Path, fallback: Path | None = None) -> dict:
    tuned = {}
    for f in [params_file, fallback]:
        if f and f.exists():
            raw = json.loads(f.read_text())
            for key, val in raw.items():
                if "best_params" in val and key not in tuned:
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
                        default=Path("results/tuning/best_params_mk5.json"))
    parser.add_argument("--fallback-params", type=Path,
                        default=Path("results/tuning/best_params.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/synthetic_mk5.json"))
    args = parser.parse_args()

    from src.experiments.runner import run_single_experiment

    tuned = _load_best_params(args.params, args.fallback_params)
    log.info("Loaded tuned params for %d combos", len(tuned))

    seeds = list(range(args.seeds))
    runner_names = list(RUNNER_TO_TUNING_KEY.keys())
    total = len(runner_names) * len(MK5_DATASETS) * len(seeds)

    log.info("=== MK5 Experiment (N=400, 5 informative features) ===")
    log.info("Datasets: %s | Models: %d | Seeds: %d | Total: %d",
             MK5_DATASETS, len(runner_names), len(seeds), total)

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

    for runner_name in runner_names:
        actual_class = "FTTransformer" if runner_name in TRANSFORMER_VARIANTS else runner_name

        for dataset in MK5_DATASETS:
            params = _resolve_params(runner_name, dataset, tuned)

            for seed in seeds:
                run_key = f"{runner_name}__{dataset}__{seed}"
                if run_key in existing_keys:
                    continue

                result = run_single_experiment(
                    model_name=actual_class,
                    dataset_name=dataset,
                    seed=seed,
                    model_params=params,
                )
                result["model_variant"] = runner_name
                result["n_features_informative"] = 5

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

    import numpy as np
    from collections import defaultdict
    scores: dict = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        mv = r.get("model_variant") or r.get("model")
        ds = r.get("dataset", "")
        scores[mv][ds].append(r.get("f1_macro", float("nan")))

    models_order = list(RUNNER_TO_TUNING_KEY.keys())
    labels = [
        "LSSVM-Standard","LSSVM-PCP","LSSVM-FSA","LSSVM-Pruning",
        "LSSVM-IP","LSSVM-OppMaps","LSSVM-ADMM",
        "FT-Softmax","FT-TopK","FT-Entmax","FT-Sparsemax",
    ]
    print(f"\n=== F1-macro (mean over 30 seeds) ===")
    print(f"{'Model':<18} {'MKE (easy)':>12} {'MKM (medium)':>14} {'MKH (hard)':>12}")
    print("-" * 60)
    for rn, lbl in zip(models_order, labels):
        row = f"{lbl:<18}"
        for ds in MK5_DATASETS:
            vals = scores[rn].get(ds, [])
            row += f"  {np.mean(vals):>10.3f}" if vals else f"  {'N/A':>10}"
        print(row)


if __name__ == "__main__":
    main()

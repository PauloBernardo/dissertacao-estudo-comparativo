#!/usr/bin/env python3
"""Synthetic dataset scaling experiment: N=2000 vs N=400 baseline.

Runs all 11 models on TWS_2k, TWM_2k, TWC_2k (N=2000) using existing
best_params from Tier 1 tuning. Saves to results/synthetic_scaling_n2000.json.

Usage
-----
    python scripts/run_synthetic_scaling.py [--seeds 30]
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

SYNTHETIC_DATASETS = ["TWS_2k", "TWM_2k", "TWC_2k"]

RUNNER_TO_TUNING_KEY = {
    "StandardLSSVM":          "StandardLSSVM",
    "PCPLSSVm":               "PCPLSSVm",
    "FSALSSVm":               "FSALSSVm",
    "PruningLSSVM":           "PruningLSSVM",
    "IPLSSVm":                "IPLSSVm",
    "OppositeMapsLSSVM":      "OppositeMapsLSSVM",
    "ADMMNesterovLSSVM":      "ADMMNesterovLSSVM",
    "FTTransformer_softmax":  "FTTransformer_softmax",
    "FTTransformer_topk":     "FTTransformer_topk",
    "FTTransformer_entmax":   "FTTransformer_entmax",
    "FTTransformer_sparsemax": "FTTransformer_sparsemax",
}

TRANSFORMER_VARIANTS = {
    "FTTransformer_softmax":   "softmax",
    "FTTransformer_topk":      "topk",
    "FTTransformer_entmax":    "entmax",
    "FTTransformer_sparsemax": "sparsemax",
}

DEFAULT_PARAMS = {
    "StandardLSSVM":        {"sigma": 1.0, "tau": 1.0},
    "PCPLSSVm":             {"sigma": 1.0, "tau": 1.0, "rank": 50},
    "FSALSSVm":             {"sigma": 1.0, "tau": 1.0, "n_atoms": 50},
    "PruningLSSVM":         {"sigma": 1.0, "tau": 1.0},
    "IPLSSVm":              {"sigma": 1.0, "tau": 1.0},
    "OppositeMapsLSSVM":    {"sigma": 1.0, "tau": 1.0, "n_prototypes": 10},
    "ADMMNesterovLSSVM":    {"sigma": 1.0, "tau": 1.0},
    "FTTransformer_softmax":  {"embedding_dim": 64, "num_blocks": 3, "num_heads": 4,
                               "max_epochs": 200, "patience": 20, "attention_type": "softmax"},
    "FTTransformer_topk":     {"embedding_dim": 64, "num_blocks": 3, "num_heads": 4,
                               "max_epochs": 200, "patience": 20, "attention_type": "topk",
                               "topk_ratio": 0.25},
    "FTTransformer_entmax":   {"embedding_dim": 64, "num_blocks": 3, "num_heads": 4,
                               "max_epochs": 200, "patience": 20, "attention_type": "entmax",
                               "alpha": 1.5},
    "FTTransformer_sparsemax": {"embedding_dim": 64, "num_blocks": 3, "num_heads": 4,
                                "max_epochs": 200, "patience": 20, "attention_type": "sparsemax"},
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
    # For scaled datasets, fall back to params tuned on the N=400 version
    base_dataset = dataset.replace("_2k", "")
    key = f"{runner_name}__{base_dataset}"
    if key in tuned:
        return dict(tuned[key])
    log.warning("No tuned params for %s / %s — using defaults", runner_name, dataset)
    return dict(DEFAULT_PARAMS.get(runner_name, {}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--params", type=Path,
                        default=Path("results/tuning/best_params.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/synthetic_scaling_n2000.json"))
    args = parser.parse_args()

    from src.experiments.runner import run_single_experiment

    tuned = _load_best_params(args.params)
    log.info("Loaded tuned params for %d combos", len(tuned))

    seeds = list(range(args.seeds))
    runner_names = list(RUNNER_TO_TUNING_KEY.keys())
    total = len(runner_names) * len(SYNTHETIC_DATASETS) * len(seeds)

    log.info("=== Synthetic Scaling Experiment (N=2000) ===")
    log.info("Models: %d | Datasets: %s | Seeds: %d | Total: %d",
             len(runner_names), SYNTHETIC_DATASETS, len(seeds), total)

    # Resumability
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

        for dataset in SYNTHETIC_DATASETS:
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
                result["n_samples_total"] = 2000

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

    # Quick comparison table
    import pandas as pd
    from collections import defaultdict
    import numpy as np

    # Load N=400 baseline
    baseline_file = Path("results/tier1_results.json")
    baseline_scores: dict = defaultdict(lambda: defaultdict(list))
    if baseline_file.exists():
        baseline = json.loads(baseline_file.read_text())
        for r in baseline:
            ds = r.get("dataset")
            if ds in ("TWS", "TWM", "TWC"):
                mv = r.get("model_variant") or r.get("model")
                baseline_scores[mv][ds].append(r.get("f1_macro", float("nan")))

    # N=2000 results
    scaled_scores: dict = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        mv = r.get("model_variant") or r.get("model")
        ds = r.get("dataset", "").replace("_2k", "")
        scaled_scores[mv][ds].append(r.get("f1_macro", float("nan")))

    print("\n=== F1-macro: N=400 → N=2000 (mean over 30 seeds) ===")
    print(f"{'Model':<25} {'TWS':>14} {'TWM':>14} {'TWC':>14}")
    print("-" * 70)
    for rn in runner_names:
        row = f"{rn:<25}"
        for ds in ("TWS", "TWM", "TWC"):
            b = baseline_scores[rn].get(ds, [])
            s = scaled_scores[rn].get(ds, [])
            b_mean = np.mean(b) if b else float("nan")
            s_mean = np.mean(s) if s else float("nan")
            diff = s_mean - b_mean
            sign = "+" if diff >= 0 else ""
            row += f"  {b_mean:.3f}→{s_mean:.3f}({sign}{diff:.3f})"
        print(row)


if __name__ == "__main__":
    main()

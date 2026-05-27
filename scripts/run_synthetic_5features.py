#!/usr/bin/env python3
"""Synthetic 5-feature experiment: N=400, 2D structure embedded in 5D.

Each synthetic dataset (TWS, TWM, TWC) is extended with 3 Gaussian noise
features (mean=0, std=1). Label depends only on the first 2 features.
Uses existing best_params from Tier 1 tuning (tuned on 2-feature versions).

Saves to results/synthetic_5features.json.

Usage
-----
    python scripts/run_synthetic_5features.py [--seeds 30]
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

SYNTHETIC_DATASETS = ["TWS_5f", "TWM_5f", "TWC_5f"]

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
    "FTTransformer_softmax",
    "FTTransformer_topk",
    "FTTransformer_entmax",
    "FTTransformer_sparsemax",
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


def _load_best_params(params_file: Path) -> dict:
    tuned = {}
    if params_file.exists():
        raw = json.loads(params_file.read_text())
        for key, val in raw.items():
            if "best_params" in val:
                tuned[key] = val["best_params"]
    return tuned


def _resolve_params(runner_name: str, dataset: str, tuned: dict) -> dict:
    # Try exact key first (params tuned for 5f datasets)
    key_5f = f"{runner_name}__{dataset}"
    if key_5f in tuned:
        return dict(tuned[key_5f])
    # Fall back to params tuned on the original 2-feature version
    base_dataset = dataset.replace("_5f", "")
    key_2f = f"{runner_name}__{base_dataset}"
    if key_2f in tuned:
        log.warning("Using 2f params for %s / %s (5f params not found)", runner_name, dataset)
        return dict(tuned[key_2f])
    log.warning("No tuned params for %s / %s — using defaults", runner_name, dataset)
    return dict(DEFAULT_PARAMS.get(runner_name, {}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--params", type=Path,
                        default=Path("results/tuning/best_params_5f.json"),
                        help="Params file. Use best_params_5f.json for tuned run, "
                             "best_params.json for fixed-sigma baseline.")
    parser.add_argument("--output", type=Path,
                        default=Path("results/synthetic_5features_tuned.json"))
    args = parser.parse_args()

    from src.experiments.runner import run_single_experiment

    tuned = _load_best_params(args.params)
    log.info("Loaded tuned params for %d combos", len(tuned))

    seeds = list(range(args.seeds))
    runner_names = list(RUNNER_TO_TUNING_KEY.keys())
    total = len(runner_names) * len(SYNTHETIC_DATASETS) * len(seeds)

    log.info("=== Synthetic 5-Feature Experiment (N=400, 5 features) ===")
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
                result["n_features_total"] = 5
                result["n_features_informative"] = 2

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

    # Comparison table: 2f baseline vs 5f
    import numpy as np
    from collections import defaultdict

    baseline_file = Path("results/tier1_results.json")
    base: dict = defaultdict(lambda: defaultdict(list))
    if baseline_file.exists():
        baseline = json.loads(baseline_file.read_text())
        for r in baseline:
            ds = r.get("dataset")
            if ds in ("TWS", "TWM", "TWC"):
                mv = r.get("model_variant") or r.get("model")
                base[mv][ds].append(r.get("f1_macro", float("nan")))

    scaled: dict = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        mv = r.get("model_variant") or r.get("model")
        ds = r.get("dataset", "").replace("_5f", "")
        scaled[mv][ds].append(r.get("f1_macro", float("nan")))

    print("\n=== F1-macro: 2 features → 5 features (N=400, mean over 30 seeds) ===")
    print(f"{'Model':<25} {'TWS':>14} {'TWM':>14} {'TWC':>14}")
    print("-" * 70)
    for rn in runner_names:
        row = f"{rn:<25}"
        for ds in ("TWS", "TWM", "TWC"):
            b = base[rn].get(ds, [])
            s = scaled[rn].get(ds, [])
            b_mean = np.mean(b) if b else float("nan")
            s_mean = np.mean(s) if s else float("nan")
            diff = s_mean - b_mean
            row += f"  {b_mean:.3f}→{s_mean:.3f}({diff:+.3f})"
        print(row)


if __name__ == "__main__":
    main()

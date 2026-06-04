#!/usr/bin/env python3
"""Run the Tier 2 comparative experiment (11 models × 6 datasets × 30 seeds).

Reads hyperparameters from results/tuning/best_params_tier2.json.
Falls back to best_params.json (Tier 1) for any missing combo.
Saves to results/tier2_results.json (separate from tier1_results.json).

Usage
-----
    python scripts/run_experiments_tier2.py [--seeds 30]
                                            [--params results/tuning/best_params_tier2.json]
                                            [--output results/tier2_results.json]
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
        logging.FileHandler("results/tier2_experiment.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

TIER2_DATASETS = ["ADULT", "BANK", "CREDIT", "TELCO", "SHOPPERS", "HIGGS50K"]

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


def _load_best_params(params_file: Path, fallback_file: Path | None = None) -> dict:
    tuned = {}
    for f in [params_file, fallback_file]:
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
                        default=Path("results/tuning/best_params_tier2.json"))
    parser.add_argument("--fallback-params", type=Path,
                        default=Path("results/tuning/best_params.json"),
                        help="Tier 1 params used as fallback for missing combos")
    parser.add_argument("--output", type=Path,
                        default=Path("results/tier2_results.json"))
    args = parser.parse_args()

    from src.experiments.runner import run_single_experiment

    tuned = _load_best_params(args.params, args.fallback_params)
    log.info("Loaded tuned params for %d (model, dataset) combos", len(tuned))

    seeds = list(range(args.seeds))
    runner_names = list(RUNNER_TO_TUNING_KEY.keys())
    total = len(runner_names) * len(TIER2_DATASETS) * len(seeds)

    log.info("=== Tier 2 Experiment ===")
    log.info("Models: %d | Datasets: %d | Seeds: %d | Total runs: %d",
             len(runner_names), len(TIER2_DATASETS), len(seeds), total)

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

        for dataset in TIER2_DATASETS:
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
    log.info("Results saved to %s", args.output)

    import pandas as pd
    df = pd.DataFrame(all_results)
    if "f1_macro" in df.columns and "model_variant" in df.columns:
        summary = df.groupby("model_variant")["f1_macro"].agg(["mean", "std"]).round(4)
        print("\n=== F1-macro summary (mean ± std across datasets × seeds) ===")
        print(summary.sort_values("mean", ascending=False).to_string())


if __name__ == "__main__":
    main()

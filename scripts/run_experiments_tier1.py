#!/usr/bin/env python3
"""Run the full Tier 1 comparative experiment (10 models × 9 datasets × 30 seeds).

Reads hyperparameters from results/tuning/best_params.json (produced by
run_tuning_tier1.py). Falls back to sensible defaults if tuning was not run.

Usage
-----
    python scripts/run_experiments_tier1.py [--seeds 30] [--jobs 1]
                                            [--params results/tuning/best_params.json]
                                            [--output results/tier1_results.json]
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
        logging.FileHandler("results/tier1_experiment.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

TIER1_DATASETS = ["BCW", "PID", "HAB", "VCP", "GCR", "AUS", "TWS", "TWM", "TWC"]

# Maps runner model names → (config_key used in tuning JSON)
RUNNER_TO_TUNING_KEY = {
    "StandardLSSVM":         "StandardLSSVM",
    "PCPLSSVm":              "PCPLSSVm",
    "FSALSSVm":              "FSALSSVm",
    "PruningLSSVM":          "PruningLSSVM",
    "IPLSSVm":               "IPLSSVm",
    "OppositeMapsLSSVM":     "OppositeMapsLSSVM",
    "FTTransformer_softmax": "FTTransformer_softmax",
    "FTTransformer_topk":    "FTTransformer_topk",
    "FTTransformer_entmax":  "FTTransformer_entmax",
    "FTTransformer_sparsemax": "FTTransformer_sparsemax",
}

# All 4 transformer variants use "FTTransformer" as class in the runner
TRANSFORMER_VARIANTS = {
    "FTTransformer_softmax":    "softmax",
    "FTTransformer_topk":       "topk",
    "FTTransformer_entmax":     "entmax",
    "FTTransformer_sparsemax":  "sparsemax",
}

# Sensible defaults if tuning wasn't run
DEFAULT_PARAMS = {
    "StandardLSSVM":        {"sigma": 1.0, "tau": 1.0},
    "PCPLSSVm":             {"sigma": 1.0, "tau": 1.0, "rank": 50},
    "FSALSSVm":             {"sigma": 1.0, "tau": 1.0, "n_atoms": 50},
    "PruningLSSVM":         {"sigma": 1.0, "tau": 1.0},
    "IPLSSVm":              {"sigma": 1.0, "tau": 1.0},
    "OppositeMapsLSSVM":    {"sigma": 1.0, "tau": 1.0, "n_prototypes": 10},
    "FTTransformer_softmax":  {"embedding_dim": 64, "num_blocks": 3, "num_heads": 4,
                               "max_epochs": 200, "patience": 20, "attention_type": "softmax"},
    "FTTransformer_topk":     {"embedding_dim": 64, "num_blocks": 3, "num_heads": 4,
                               "max_epochs": 200, "patience": 20, "attention_type": "topk",
                               "topk_ratio": 0.25},
    "FTTransformer_entmax":   {"embedding_dim": 64, "num_blocks": 3, "num_heads": 4,
                               "max_epochs": 200, "patience": 20, "attention_type": "entmax",
                               "alpha": 1.5},
    "FTTransformer_sparsemax":{"embedding_dim": 64, "num_blocks": 3, "num_heads": 4,
                               "max_epochs": 200, "patience": 20, "attention_type": "sparsemax"},
}


def _load_best_params(params_file: Path) -> dict:
    """Load tuned params from JSON; fall back to defaults for missing combos."""
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
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel jobs (-1 = all CPUs via joblib)")
    parser.add_argument("--params", type=Path,
                        default=Path("results/tuning/best_params.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/tier1_results.json"))
    args = parser.parse_args()

    from src.experiments.runner import run_single_experiment

    tuned = _load_best_params(args.params)
    if tuned:
        log.info("Loaded tuned params for %d (model, dataset) combos", len(tuned))
    else:
        log.warning("No tuned params found — using defaults for all models")

    seeds = list(range(args.seeds))
    runner_names = list(RUNNER_TO_TUNING_KEY.keys())
    total = len(runner_names) * len(TIER1_DATASETS) * len(seeds)

    log.info("=== Tier 1 Experiment ===")
    log.info("Models: %d | Datasets: %d | Seeds: %d | Total runs: %d",
             len(runner_names), len(TIER1_DATASETS), len(seeds), total)

    # Load existing results for resumability
    existing_results = []
    existing_keys: set[str] = set()
    if args.output.exists():
        existing_results = json.loads(args.output.read_text())
        for r in existing_results:
            k = f"{r.get('model')}__{r.get('dataset')}__{r.get('seed')}"
            existing_keys.add(k)
        log.info("Resuming: %d results already saved", len(existing_results))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_results = list(existing_results)
    completed = len(existing_keys)
    errors = 0
    t_start = time.perf_counter()

    for runner_name in runner_names:
        # FT-Transformer variants map to class "FTTransformer" in runner
        actual_class = "FTTransformer" if runner_name in TRANSFORMER_VARIANTS else runner_name

        for dataset in TIER1_DATASETS:
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
                # Tag with the variant name so we can distinguish transformer variants
                result["model_variant"] = runner_name

                all_results.append(result)
                existing_keys.add(run_key)
                completed += 1

                if result["status"] != "ok":
                    errors += 1

                # Save every 10 runs
                if completed % 10 == 0:
                    args.output.write_text(json.dumps(all_results, indent=2, default=str))

                elapsed = time.perf_counter() - t_start
                remaining = total - completed
                eta = (elapsed / completed * remaining) if completed > 0 else 0
                log.info("[%d/%d] %s / %s / seed=%d — status=%s | ETA %.0fm",
                         completed, total, runner_name, dataset, seed,
                         result["status"], eta / 60)

    # Final save
    args.output.write_text(json.dumps(all_results, indent=2, default=str))
    log.info("=== Done: %d/%d ok, %d errors ===", completed - errors, total, errors)
    log.info("Results saved to %s", args.output)

    # Quick summary
    import pandas as pd
    df = pd.DataFrame(all_results)
    if "f1_macro" in df.columns and "model_variant" in df.columns:
        summary = df.groupby("model_variant")["f1_macro"].agg(["mean", "std"]).round(4)
        print("\n=== F1-macro summary (mean ± std across datasets × seeds) ===")
        print(summary.sort_values("mean", ascending=False).to_string())


if __name__ == "__main__":
    main()

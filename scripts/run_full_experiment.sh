#!/usr/bin/env bash
# DEPRECATED — Este script está desatualizado e não deve ser usado.
#
# Use os scripts específicos em vez deste:
#   python scripts/run_tuning_tier1.py     # tuning Bayesian (Optuna)
#   python scripts/run_experiments_tier1.py --seeds 30   # experimentos Tier 1
#   python scripts/generate_analysis.py    # tabelas e figuras
#
# Este arquivo é mantido apenas para referência histórica.
# -----------------------------------------------------------------------
# Run the full 12 × 18 × 30 comparative experiment.
#
# Usage:
#   bash scripts/run_full_experiment.sh [--tier 1] [--seeds 5] [--jobs 4]
#
# Options:
#   --tier N     Only run datasets of tier N (1, 2, or 3); default: all
#   --seeds N    Number of seeds 0..(N-1) to run; default: 30
#   --jobs N     Parallel jobs via joblib; default: 1 (sequential)
#   --output-dir Output directory for results; default: results/

set -euo pipefail

TIER="all"
N_SEEDS=30
N_JOBS=1
OUTPUT_DIR="results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)   TIER="$2";       shift 2 ;;
        --seeds)  N_SEEDS="$2";    shift 2 ;;
        --jobs)   N_JOBS="$2";     shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PYTHON=".venv/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

echo "=== Comparative Study Experiment Runner ==="
echo "Tier:      $TIER"
echo "Seeds:     0..$((N_SEEDS - 1))"
echo "Jobs:      $N_JOBS"
echo "Output:    $OUTPUT_DIR"
echo ""

# Run via Python orchestrator
"$PYTHON" - <<EOF
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.experiments.runner import run_all
from src.data.loaders import DatasetLoader

# ── Dataset selection ──────────────────────────────────────────────────────────
tier_filter = "$TIER"
all_datasets = DatasetLoader.available()

# Tier assignment (mirrors config/datasets.yaml)
tier_map = {
    "HAB": 1, "PID": 1, "BCW": 1, "VCP": 1, "GCR": 1, "AUS": 1,
    "TWS": 1, "TWM": 1, "TWC": 1,
    "ADULT": 2, "BANK": 2, "CREDIT": 2, "TELCO": 2, "SHOPPERS": 2, "HIGGS50K": 2,
    "HIGGS500K": 3, "COVER": 3, "KDD99": 3,
}

if tier_filter == "all":
    datasets = [d for d in all_datasets if d in tier_map]
else:
    t = int(tier_filter)
    datasets = [d for d in all_datasets if tier_map.get(d) == t]

# ── Model list ────────────────────────────────────────────────────────────────
models = [
    "StandardLSSVM",
    "ADMMNesterovLSSVM",
    "PCPLSSVm",
    "FSALSSVm",
    "PruningLSSVM",
    "IPLSSVm",
    "OppositeMapsLSSVM",
    "FTTransformer",
]

# ── Default hyperparameters (pre-tuned placeholders; replace with Optuna best) ─
model_params = {
    "StandardLSSVM":    {"sigma": 1.0, "tau": 1.0},
    "ADMMNesterovLSSVM":{"sigma": 1.0, "tau": 0.01, "lam": 0.01},
    "PCPLSSVm":         {"sigma": 1.0, "tau": 1.0, "rank": 50},
    "FSALSSVm":         {"sigma": 1.0, "tau": 1.0, "n_atoms": 50},
    "PruningLSSVM":     {"sigma": 1.0, "tau": 1.0},
    "IPLSSVm":          {"sigma": 1.0, "tau": 1.0},
    "OppositeMapsLSSVM":{"sigma": 1.0, "tau": 1.0, "n_prototypes": 10},
    "FTTransformer":    {"embedding_dim": 64, "num_blocks": 3, "num_heads": 4,
                         "max_epochs": 200, "patience": 20, "val_fraction": 0.10,
                         "batch_size": 256, "lr": 1e-4},
}

seeds = list(range(int("$N_SEEDS")))
n_jobs = int("$N_JOBS")

print(f"Running {len(models)} models × {len(datasets)} datasets × {len(seeds)} seeds "
      f"= {len(models)*len(datasets)*len(seeds)} experiments")
print(f"Datasets: {datasets}")
print()

results = run_all(
    model_names=models,
    dataset_names=datasets,
    seeds=seeds,
    model_params_map=model_params,
    n_jobs=n_jobs,
)

# ── Save results ───────────────────────────────────────────────────────────────
out_dir = Path("$OUTPUT_DIR")
out_dir.mkdir(parents=True, exist_ok=True)

results_path = out_dir / "results.json"
results_path.write_text(json.dumps(results, indent=2, default=str))
print(f"Saved {len(results)} results to {results_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
ok = sum(1 for r in results if r.get("status") == "ok")
err = len(results) - ok
print(f"Status: {ok} ok / {err} errors")

if err > 0:
    for r in results:
        if r.get("status") != "ok":
            print(f"  ERROR: {r['model']} / {r['dataset']} / seed={r['seed']}: {r.get('error_message')}")
EOF

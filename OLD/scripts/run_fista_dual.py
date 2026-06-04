#!/usr/bin/env python3
"""Tune and run DualFISTA (FISTA on dual LSSVM formulation) on all benchmarks.

Solves the dual LSSVM LASSO:
    min_α  ½αᵀΩα - 1ᵀα + λ‖α‖₁
where Ω = YKY + (1/τ)I.  L1 is placed directly on Lagrange multipliers
(not on Cholesky-space coefficients), giving exact support-vector sparsity.

Appends results to existing JSON files — never overwrites them.

Phases
------
  1. Tune on Tier 1 (9 datasets)      → best_params_dual_fista.json
  2. Run Tier 1 × 30 seeds            → tier1_custom_models.json
  3. Scaling ablation (2k)            → synthetic_scaling_n2000.json
  4. 5f ablation (fixed params)       → synthetic_5features.json
  5. Tune on 5f datasets              → best_params_dual_fista.json
  6. 5f retuned                       → synthetic_5features_tuned.json
  7. Tune on MK5 datasets             → best_params_dual_fista.json
  8. Run MK5 × 30 seeds               → synthetic_mk5.json

Usage
-----
    python scripts/run_fista_dual.py [--seeds 30] [--trials 100]
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
        logging.FileHandler("results/dual_fista.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

RUNNER_NAME  = "DualFISTALSSVM"
VARIANT_NAME = "DualFISTA"

TIER1_DATASETS = ["BCW", "PID", "HAB", "VCP", "GCR", "AUS", "TWS", "TWM", "TWC"]
MK5_DATASETS   = ["MKE", "MKM", "MKH"]
SCALING_DS     = ["TWS_2k", "TWM_2k", "TWC_2k"]
FIVEFEATURE_DS = ["TWS_5f", "TWM_5f", "TWC_5f"]

PARAMS_FILE = Path("results/tuning/best_params_dual_fista.json")

DEFAULT_PARAMS = {
    "sigma": 1.0, "tau": 1.0, "lambda_": 0.1,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> list:
    return json.loads(path.read_text()) if path.exists() else []


def _load_params(path: Path) -> dict:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {k: v["best_params"] for k, v in raw.items() if "best_params" in v}


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _existing_keys(results: list) -> set[str]:
    keys = set()
    for r in results:
        mv = r.get("model_variant") or r.get("model")
        keys.add(f"{mv}__{r.get('dataset')}__{r.get('seed')}")
    return keys


def _resolve_params(dataset: str, tuned: dict) -> dict:
    """Exact key → strip suffix fallback → defaults."""
    for key in [f"{VARIANT_NAME}__{dataset}",
                f"{VARIANT_NAME}__{dataset.split('_')[0]}"]:
        if key in tuned:
            return dict(tuned[key])
    log.warning("No tuned params for %s — using defaults", dataset)
    return dict(DEFAULT_PARAMS)


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def _tune_datasets(datasets: list[str], n_trials: int, folds: int, seed: int) -> None:
    """Tune DualFISTA on each dataset; append to PARAMS_FILE."""
    import optuna
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from src.data.loaders import DatasetLoader
    from src.models.lssvm.dual.fista_dual_lssvm import DualFISTALSSVM

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    existing = json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else {}

    for dataset in datasets:
        key = f"{VARIANT_NAME}__{dataset}"
        if key in existing:
            log.info("[SKIP ] %s", key)
            continue

        log.info("[TUNE ] %s (%d trials)...", key, n_trials)
        X, y, _ = DatasetLoader.load(dataset)
        y_signed = (y * 2 - 1).astype(int)

        def objective(trial):
            sigma   = trial.suggest_float("sigma",   0.01, 100.0, log=True)
            tau     = trial.suggest_float("tau",     0.01, 100.0, log=True)
            # In dual, λ_max = 1 always (scale-invariant).
            # Search in (0, 1) — above 1 drives all alphas to 0.
            lambda_ = trial.suggest_float("lambda_", 0.001, 0.999, log=True)

            model = DualFISTALSSVM(
                sigma=sigma, tau=tau, lambda_=lambda_,
                max_iter=2000,
            )
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            scores = []
            for tr, val in cv.split(X, y_signed):
                scaler = StandardScaler()
                Xtr  = scaler.fit_transform(X[tr])
                Xval = scaler.transform(X[val])
                try:
                    model.fit(Xtr, y_signed[tr])
                    pred = model.predict(Xval)
                    yb   = ((y_signed[val] + 1) // 2).astype(int)
                    pb   = ((pred           + 1) // 2).astype(int)
                    scores.append(f1_score(yb, pb, average="macro", zero_division=0))
                except Exception:
                    scores.append(0.0)
            return float(sum(scores) / len(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        existing[key] = {
            "best_params": study.best_params,
            "best_value":  study.best_value,
            "metric":      "f1_macro",
        }
        _save_json(PARAMS_FILE, existing)
        log.info("[OK   ] %s — f1=%.4f", key, study.best_value)


# ── Experiment phase ───────────────────────────────────────────────────────────

def _run_phase(
    phase_name: str,
    datasets: list[str],
    output_file: Path,
    seeds: list[int],
    run_single_experiment,
    tuned: dict,
) -> None:
    existing = _load_json(output_file)
    done_keys = _existing_keys(existing)
    all_results = list(existing)

    total = len(datasets) * len(seeds)
    completed = errors = 0
    t0 = time.perf_counter()

    log.info("── %s: %d experiments → %s ──", phase_name, total, output_file.name)

    for dataset in datasets:
        params = _resolve_params(dataset, tuned)

        for seed in seeds:
            run_key = f"{VARIANT_NAME}__{dataset}__{seed}"
            if run_key in done_keys:
                continue

            result = run_single_experiment(
                model_name=RUNNER_NAME,
                dataset_name=dataset,
                seed=seed,
                model_params=params,
            )
            result["model_variant"] = VARIANT_NAME

            all_results.append(result)
            done_keys.add(run_key)
            completed += 1
            if result["status"] != "ok":
                errors += 1

            if completed % 10 == 0:
                _save_json(output_file, all_results)

            elapsed = time.perf_counter() - t0
            eta = (elapsed / completed * (total - completed)) if completed else 0
            log.info("[%d/%d] %s / seed=%d — %s | ETA %.0fm",
                     completed, total, dataset, seed, result["status"], eta / 60)

    _save_json(output_file, all_results)
    log.info("── %s done: %d ok, %d errors ──",
             phase_name, completed - errors, errors)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",      type=int, default=30)
    parser.add_argument("--trials",     type=int, default=100,
                        help="Optuna trials per dataset")
    parser.add_argument("--folds",      type=int, default=5)
    parser.add_argument("--seed-tune",  type=int, default=0)
    args = parser.parse_args()

    from src.experiments.runner import run_single_experiment

    seeds = list(range(args.seeds))
    base  = Path("results")

    log.info("=== DualFISTA LSSVM: full benchmark ===")

    # ── Phase 1: Tune on Tier 1 ─────────────────────────────────────────────
    log.info("── Phase 1: Tuning on Tier 1 ──")
    _tune_datasets(TIER1_DATASETS, args.trials, args.folds, args.seed_tune)

    # ── Phase 2: Run Tier 1 ─────────────────────────────────────────────────
    log.info("── Phase 2: Tier 1 experiments ──")
    tuned = _load_params(PARAMS_FILE)
    _run_phase("TIER1", TIER1_DATASETS,
               base / "tier1_custom_models.json", seeds,
               run_single_experiment, tuned)

    # ── Phase 3: Scaling ablation (use Tier 1 params) ───────────────────────
    log.info("── Phase 3: Scaling ablation (2k) ──")
    tuned = _load_params(PARAMS_FILE)
    _run_phase("SCALING", SCALING_DS,
               base / "synthetic_scaling_n2000.json", seeds,
               run_single_experiment, tuned)

    # ── Phase 4: 5f ablation (fixed Tier 1 params) ──────────────────────────
    log.info("── Phase 4: 5f ablation (Tier 1 params) ──")
    _run_phase("5F_FIXED", FIVEFEATURE_DS,
               base / "synthetic_5features.json", seeds,
               run_single_experiment, tuned)

    # ── Phase 5: Tune on 5f datasets ────────────────────────────────────────
    log.info("── Phase 5: Tuning on 5f datasets ──")
    _tune_datasets(FIVEFEATURE_DS, args.trials, args.folds, args.seed_tune)

    # ── Phase 6: 5f retuned ─────────────────────────────────────────────────
    log.info("── Phase 6: 5f retuned ──")
    tuned = _load_params(PARAMS_FILE)
    _run_phase("5F_TUNED", FIVEFEATURE_DS,
               base / "synthetic_5features_tuned.json", seeds,
               run_single_experiment, tuned)

    # ── Phase 7: Tune on MK5 ────────────────────────────────────────────────
    log.info("── Phase 7: Tuning on MK5 ──")
    _tune_datasets(MK5_DATASETS, args.trials, args.folds, args.seed_tune)

    # ── Phase 8: Run MK5 ────────────────────────────────────────────────────
    log.info("── Phase 8: MK5 experiments ──")
    tuned = _load_params(PARAMS_FILE)
    _run_phase("MK5", MK5_DATASETS,
               base / "synthetic_mk5.json", seeds,
               run_single_experiment, tuned)

    log.info("=== DualFISTA LSSVM: all phases complete ===")


if __name__ == "__main__":
    main()

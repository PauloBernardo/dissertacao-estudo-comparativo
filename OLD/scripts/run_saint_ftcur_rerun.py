#!/usr/bin/env python3
"""Rerun FTTransformerCURColnorm (Nyströmformer) and add SAINTColnorm.

Steps:
  1. Strip old FTTransformerCURColnorm entries from all 4 result JSONs
  2. Tune SAINTColnorm on Tier 1, 5f, and MK5 datasets
  3. Run Tier 1  (9 datasets × 30 seeds) for both new models
  4. Run Scaling (3 datasets × 30 seeds) for both
  5. Run 5f fixed + retune + 5f retuned for both
  6. Run MK5 (3 datasets × 30 seeds) for both

FTTransformerCURColnorm tuning params are reused from existing files
(the Nyströmformer hyperparameter space is identical to the old CUR).

Usage
-----
    python scripts/run_saint_ftcur_rerun.py [--seeds 30] [--trials-saint 20]
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
        logging.FileHandler("results/saint_ftcur_rerun.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Constants ─────────────────────────────────────────────────────────────────

TIER1_DATASETS  = ["BCW", "PID", "HAB", "VCP", "GCR", "AUS", "TWS", "TWM", "TWC"]
SCALING_DS      = ["TWS_2k", "TWM_2k", "TWC_2k"]
FEATURES5_DS    = ["TWS_5f", "TWM_5f", "TWC_5f"]
MK5_DS          = ["MKE", "MKM", "MKH"]

NEW_MODELS = ["FTTransformerCURColnorm", "SAINTColnorm"]

DEFAULT_PARAMS = {
    "FTTransformerCURColnorm": {
        "d_model": 32, "n_heads": 4, "n_layers": 2,
        "m_ratio": 0.10, "lr": 5e-4, "epochs": 200, "patience": 20,
    },
    "SAINTColnorm": {
        "d_model": 32, "n_heads": 4, "n_layers": 2,
        "lr": 5e-4, "epochs": 200, "patience": 20,
    },
}

# Result files that may contain old FTTransformerCURColnorm entries to strip
RESULT_FILES = [
    Path("results/tier1_custom_models.json"),
    Path("results/synthetic_scaling_n2000.json"),
    Path("results/synthetic_5features.json"),
    Path("results/synthetic_5features_tuned.json"),
    Path("results/synthetic_mk5.json"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> list:
    return json.loads(path.read_text()) if path.exists() else []


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _load_params(path: Path) -> dict:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {k: v["best_params"] for k, v in raw.items() if "best_params" in v}


def _resolve_params(runner_name: str, dataset: str, tuned: dict) -> dict:
    for key in [f"{runner_name}__{dataset}",
                f"{runner_name}__{dataset.split('_')[0]}"]:
        if key in tuned:
            return dict(tuned[key])
    log.warning("No tuned params for %s / %s — using defaults", runner_name, dataset)
    return dict(DEFAULT_PARAMS.get(runner_name, {}))


def _existing_keys(results: list) -> set[str]:
    keys = set()
    for r in results:
        mv = r.get("model_variant") or r.get("model")
        keys.add(f"{mv}__{r.get('dataset')}__{r.get('seed')}")
    return keys


# ── Step 1: Strip old FTTransformerCURColnorm ─────────────────────────────────

def strip_old_ftcur() -> None:
    log.info("=== Step 1: Stripping old FTTransformerCURColnorm from result files ===")
    for path in RESULT_FILES:
        if not path.exists():
            log.info("  %s — not found, skip", path)
            continue
        data = json.loads(path.read_text())
        before = len(data)
        data = [r for r in data
                if (r.get("model_variant") or r.get("model")) != "FTTransformerCURColnorm"]
        removed = before - len(data)
        _save_json(path, data)
        log.info("  %s — removed %d old entries (%d remain)", path.name, removed, len(data))


# ── Step 2: Tune SAINTColnorm ─────────────────────────────────────────────────

def _tune_saint_on_datasets(
    datasets: list[str],
    params_file: Path,
    n_trials: int,
    folds: int,
    seed: int,
    timeout: float,
    extra_params_file: Path | None = None,
) -> None:
    """Optuna tuning for SAINTColnorm on the given datasets.

    Tuned params are merged into params_file.
    If extra_params_file is given, those keys are also saved there (for FT-CUR).
    """
    import optuna
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from src.data.loaders import DatasetLoader
    from src.models.ft_transformer_saint_wrapper import SAINTColnorm

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    existing = json.loads(params_file.read_text()) if params_file.exists() else {}

    for dataset in datasets:
        key = f"SAINTColnorm__{dataset}"
        if key in existing:
            log.info("[SKIP ] SAINTColnorm / %s", dataset)
            continue

        log.info("[TUNE ] SAINTColnorm / %s (%d trials)...", dataset, n_trials)
        X, y, _ = DatasetLoader.load(dataset)

        def objective(trial):
            d_model = trial.suggest_categorical("d_model", [16, 32, 64])
            n_heads = trial.suggest_categorical("n_heads", [2, 4])
            n_layers = trial.suggest_int("n_layers", 1, 3)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            if d_model % n_heads != 0:
                return 0.0
            model = SAINTColnorm(
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                lr=lr, epochs=50, patience=10, random_state=seed)
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            scores = []
            for tr, val in cv.split(X, y):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X[tr])
                X_val = scaler.transform(X[val])
                try:
                    model.fit(X_tr, y[tr])
                    pred = model.predict(X_val)
                    scores.append(f1_score(y[val], pred,
                                           average="macro", zero_division=0))
                except Exception:
                    scores.append(0.0)
            return float(sum(scores) / len(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, timeout=timeout,
                       show_progress_bar=False)
        best = dict(study.best_params)
        best["epochs"] = 200
        best["patience"] = 20
        existing[key] = {
            "best_params": best,
            "best_value": study.best_value,
            "metric": "f1_macro",
            "n_trials": len(study.trials),
        }
        _save_json(params_file, existing)
        log.info("[OK   ] SAINTColnorm / %s — f1=%.4f", dataset, study.best_value)


# ── Step 3-6: Run experiments ─────────────────────────────────────────────────

def _run_phase(
    phase_name: str,
    models: list[str],
    datasets: list[str],
    params_file: Path,
    output_file: Path,
    seeds: list[int],
    run_single_experiment,
    extra_field: dict | None = None,
) -> None:
    tuned = _load_params(params_file)
    log.info("Loaded %d tuned combos from %s", len(tuned), params_file)

    existing = _load_json(output_file)
    existing_keys = _existing_keys(existing)
    log.info("Resuming %s: %d entries in %s", phase_name, len(existing), output_file)

    all_results = list(existing)
    total = len(models) * len(datasets) * len(seeds)
    completed = errors = 0
    t_start = time.perf_counter()

    for runner_name in models:
        for dataset in datasets:
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
                if extra_field:
                    result.update(extra_field)

                all_results.append(result)
                existing_keys.add(run_key)
                completed += 1
                if result["status"] != "ok":
                    errors += 1

                if completed % 10 == 0:
                    _save_json(output_file, all_results)

                elapsed = time.perf_counter() - t_start
                remaining = total - completed
                eta = (elapsed / completed * remaining) if completed > 0 else 0
                log.info("[%s %d/%d] %s / %s / seed=%d — %s | ETA %.0fm",
                         phase_name, completed, total,
                         runner_name, dataset, seed, result["status"], eta / 60)

    _save_json(output_file, all_results)
    log.info("=== %s done: %d errors / %d runs ===", phase_name, errors, total)


def _tune_ftcur_if_missing(
    datasets: list[str],
    params_file: Path,
    n_trials: int,
    folds: int,
    seed: int,
    timeout: float,
) -> None:
    """Fill missing FTTransformerCURColnorm entries in params_file."""
    import optuna
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from src.data.loaders import DatasetLoader
    from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    existing = json.loads(params_file.read_text()) if params_file.exists() else {}

    for dataset in datasets:
        key = f"FTTransformerCURColnorm__{dataset}"
        if key in existing:
            log.info("[SKIP ] FTTransformerCURColnorm / %s", dataset)
            continue

        log.info("[TUNE ] FTTransformerCURColnorm / %s (%d trials)...", dataset, n_trials)
        X, y, _ = DatasetLoader.load(dataset)

        def obj_ft(trial):
            d_model = trial.suggest_categorical("d_model", [16, 32, 64])
            n_heads = trial.suggest_categorical("n_heads", [2, 4])
            n_layers = trial.suggest_int("n_layers", 1, 3)
            m_ratio = trial.suggest_float("m_ratio", 0.05, 0.30)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            if d_model % n_heads != 0:
                return 0.0
            model = FTTransformerCURColnorm(
                d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                m_ratio=m_ratio, lr=lr, epochs=50, patience=10,
                random_state=seed)
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            sc = []
            for tr, val in cv.split(X, y):
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X[tr])
                Xval = scaler.transform(X[val])
                try:
                    model.fit(Xtr, y[tr])
                    pred = model.predict(Xval)
                    sc.append(f1_score(y[val], pred, average="macro", zero_division=0))
                except Exception:
                    sc.append(0.0)
            return float(sum(sc) / len(sc))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(obj_ft, n_trials=n_trials, timeout=timeout,
                       show_progress_bar=False)
        best = dict(study.best_params)
        best["epochs"] = 200
        best["patience"] = 20
        existing[key] = {
            "best_params": best,
            "best_value": study.best_value,
            "metric": "f1_macro",
            "n_trials": len(study.trials),
        }
        _save_json(params_file, existing)
        log.info("[OK   ] FTTransformerCURColnorm / %s — f1=%.4f", dataset, study.best_value)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",          type=int,   default=30)
    parser.add_argument("--trials-saint",   type=int,   default=20)
    parser.add_argument("--trials-ftcur",   type=int,   default=20)
    parser.add_argument("--folds",          type=int,   default=5)
    parser.add_argument("--seed-tune",      type=int,   default=0)
    parser.add_argument("--timeout",        type=float, default=300.0)
    parser.add_argument("--skip-strip",     action="store_true",
                        help="Skip stripping old FTTransformerCURColnorm results")
    args = parser.parse_args()

    from src.experiments.runner import run_single_experiment

    seeds      = list(range(args.seeds))
    base       = Path("results")
    params_t1  = base / "tuning/best_params_custom.json"
    params_5f  = base / "tuning/best_params_custom_5f.json"
    params_mk5 = base / "tuning/best_params_custom_mk5.json"
    # Unified params file for SAINT (all experiment contexts)
    params_saint_t1  = base / "tuning/best_params_saint.json"
    params_saint_5f  = base / "tuning/best_params_saint_5f.json"
    params_saint_mk5 = base / "tuning/best_params_saint_mk5.json"

    log.info("=== SAINT + FT-CUR (Nyströmformer) Full Rerun ===")
    log.info("Seeds: %d | Trials SAINT: %d | Trials FT-CUR: %d",
             args.seeds, args.trials_saint, args.trials_ftcur)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Strip old CUR entries
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_strip:
        strip_old_ftcur()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Tune SAINTColnorm — Tier 1 datasets
    # ─────────────────────────────────────────────────────────────────────────
    log.info("=== Step 2a: Tune SAINTColnorm on Tier 1 datasets ===")
    _tune_saint_on_datasets(
        datasets=TIER1_DATASETS,
        params_file=params_saint_t1,
        n_trials=args.trials_saint,
        folds=args.folds,
        seed=args.seed_tune,
        timeout=args.timeout,
    )
    # Fill any missing FT-CUR entries (usually already present from old runs)
    log.info("=== Step 2b: Fill missing FT-CUR tuning on Tier 1 ===")
    _tune_ftcur_if_missing(
        datasets=TIER1_DATASETS,
        params_file=params_t1,
        n_trials=args.trials_ftcur,
        folds=args.folds,
        seed=args.seed_tune,
        timeout=args.timeout,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Tier 1 experiment
    # ─────────────────────────────────────────────────────────────────────────
    log.info("=== Step 3: Tier 1 (9 datasets × %d seeds) ===", args.seeds)

    # Build a combined params map: FT-CUR from params_t1, SAINT from params_saint_t1
    tuned_t1 = _load_params(params_t1)
    tuned_saint_t1 = _load_params(params_saint_t1)
    combined_t1 = {**tuned_t1, **tuned_saint_t1}
    _save_json(base / "tuning/_combined_t1_tmp.json",
               {k: {"best_params": v} for k, v in combined_t1.items()})

    _run_phase(
        phase_name="TIER1-FTCUR",
        models=["FTTransformerCURColnorm"],
        datasets=TIER1_DATASETS,
        params_file=params_t1,
        output_file=base / "tier1_custom_models.json",
        seeds=seeds,
        run_single_experiment=run_single_experiment,
    )
    _run_phase(
        phase_name="TIER1-SAINT",
        models=["SAINTColnorm"],
        datasets=TIER1_DATASETS,
        params_file=params_saint_t1,
        output_file=base / "tier1_custom_models.json",
        seeds=seeds,
        run_single_experiment=run_single_experiment,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Scaling N=2000
    # ─────────────────────────────────────────────────────────────────────────
    log.info("=== Step 4: Scaling N=2000 (3 datasets × %d seeds) ===", args.seeds)
    _run_phase(
        phase_name="SCALING-FTCUR",
        models=["FTTransformerCURColnorm"],
        datasets=SCALING_DS,
        params_file=params_t1,
        output_file=base / "synthetic_scaling_n2000.json",
        seeds=seeds,
        run_single_experiment=run_single_experiment,
    )
    _run_phase(
        phase_name="SCALING-SAINT",
        models=["SAINTColnorm"],
        datasets=SCALING_DS,
        params_file=params_saint_t1,
        output_file=base / "synthetic_scaling_n2000.json",
        seeds=seeds,
        run_single_experiment=run_single_experiment,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5a: 5-features with Tier 1 params (fixed σ transfer)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("=== Step 5a: 5-features (fixed params from Tier 1) ===")
    _run_phase(
        phase_name="5F-FIXED-FTCUR",
        models=["FTTransformerCURColnorm"],
        datasets=FEATURES5_DS,
        params_file=params_t1,
        output_file=base / "synthetic_5features.json",
        seeds=seeds,
        run_single_experiment=run_single_experiment,
    )
    _run_phase(
        phase_name="5F-FIXED-SAINT",
        models=["SAINTColnorm"],
        datasets=FEATURES5_DS,
        params_file=params_saint_t1,
        output_file=base / "synthetic_5features.json",
        seeds=seeds,
        run_single_experiment=run_single_experiment,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5b: Tune on 5f datasets
    # ─────────────────────────────────────────────────────────────────────────
    log.info("=== Step 5b: Tuning on 5f datasets ===")
    _tune_saint_on_datasets(
        datasets=FEATURES5_DS,
        params_file=params_saint_5f,
        n_trials=args.trials_saint,
        folds=args.folds,
        seed=args.seed_tune,
        timeout=args.timeout,
    )
    _tune_ftcur_if_missing(
        datasets=FEATURES5_DS,
        params_file=params_5f,
        n_trials=args.trials_ftcur,
        folds=args.folds,
        seed=args.seed_tune,
        timeout=args.timeout,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5c: 5-features retuned
    # ─────────────────────────────────────────────────────────────────────────
    log.info("=== Step 5c: 5-features (retuned params) ===")
    _run_phase(
        phase_name="5F-RETUNED-FTCUR",
        models=["FTTransformerCURColnorm"],
        datasets=FEATURES5_DS,
        params_file=params_5f,
        output_file=base / "synthetic_5features_tuned.json",
        seeds=seeds,
        run_single_experiment=run_single_experiment,
    )
    _run_phase(
        phase_name="5F-RETUNED-SAINT",
        models=["SAINTColnorm"],
        datasets=FEATURES5_DS,
        params_file=params_saint_5f,
        output_file=base / "synthetic_5features_tuned.json",
        seeds=seeds,
        run_single_experiment=run_single_experiment,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: MK5 (tune + run)
    # ─────────────────────────────────────────────────────────────────────────
    log.info("=== Step 6: MK5 (3 datasets × %d seeds) ===", args.seeds)
    _tune_saint_on_datasets(
        datasets=MK5_DS,
        params_file=params_saint_mk5,
        n_trials=args.trials_saint,
        folds=args.folds,
        seed=args.seed_tune,
        timeout=args.timeout,
    )
    _tune_ftcur_if_missing(
        datasets=MK5_DS,
        params_file=params_mk5,
        n_trials=args.trials_ftcur,
        folds=args.folds,
        seed=args.seed_tune,
        timeout=args.timeout,
    )
    _run_phase(
        phase_name="MK5-FTCUR",
        models=["FTTransformerCURColnorm"],
        datasets=MK5_DS,
        params_file=params_mk5,
        output_file=base / "synthetic_mk5.json",
        seeds=seeds,
        run_single_experiment=run_single_experiment,
        extra_field={"n_features_informative": 5},
    )
    _run_phase(
        phase_name="MK5-SAINT",
        models=["SAINTColnorm"],
        datasets=MK5_DS,
        params_file=params_saint_mk5,
        output_file=base / "synthetic_mk5.json",
        seeds=seeds,
        run_single_experiment=run_single_experiment,
        extra_field={"n_features_informative": 5},
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    import numpy as np
    from collections import defaultdict

    log.info("=== All phases complete ===")
    print("\n" + "=" * 72)
    print("TIER 1 F1-macro summary (mean across 9 datasets × 30 seeds)")
    print("=" * 72)

    tier1_data = _load_json(base / "tier1_custom_models.json")
    scores: dict = defaultdict(lambda: defaultdict(list))
    for r in tier1_data:
        mv = r.get("model_variant") or r.get("model")
        scores[mv][r.get("dataset", "")].append(r.get("f1_macro", float("nan")))

    for m in NEW_MODELS:
        all_f1 = [v for ds_vals in scores[m].values() for v in ds_vals]
        if all_f1:
            print(f"  {m:<28} {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")


if __name__ == "__main__":
    main()

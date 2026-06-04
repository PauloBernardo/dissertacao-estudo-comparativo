#!/usr/bin/env python3
"""Tune and run NystromLSSVMColnorm and FTTransformerCURColnorm on MK5 datasets.

Appends to results/synthetic_mk5.json (does NOT overwrite existing results).
Saves tuning to results/tuning/best_params_custom_mk5.json.

Usage
-----
    python scripts/run_mk5_custom_models.py [--seeds 30] [--trials-lssvm 50]
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
PARAMS_FILE  = Path("results/tuning/best_params_custom_mk5.json")
OUTPUT_FILE  = Path("results/synthetic_mk5.json")

DEFAULT_PARAMS = {
    "NystromLSSVMColnorm":     {"sigma": 1.0, "gamma": 1.0, "m_ratio": 0.20},
    "FTTransformerCURColnorm": {"d_model": 32, "n_heads": 4, "n_layers": 2,
                                "m_ratio": 0.10, "lr": 5e-4,
                                "epochs": 200, "patience": 20},
}


def _load_params(path: Path) -> dict:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {k: v["best_params"] for k, v in raw.items() if "best_params" in v}


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _tune(datasets: list[str], n_trials_lssvm: int, n_trials_ft: int,
          folds: int, seed: int, timeout_ft: float) -> None:
    import optuna
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from src.data.loaders import DatasetLoader

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    existing = json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else {}

    for dataset in datasets:
        X, y, _ = DatasetLoader.load(dataset)

        # ── NystromLSSVMColnorm ─────────────────────────────────────────────
        key = f"NystromLSSVMColnorm__{dataset}"
        if key not in existing:
            log.info("[TUNE ] NystromLSSVMColnorm / %s (%d trials)...", dataset, n_trials_lssvm)
            from src.models.nystrom_lssvm_wrapper import NystromLSSVMColnorm
            y_signed = (y * 2 - 1).astype(int)

            def obj_ny(trial):
                sigma   = trial.suggest_float("sigma",   0.01, 100.0, log=True)
                gamma   = trial.suggest_float("gamma",   0.01, 1000.0, log=True)
                m_ratio = trial.suggest_float("m_ratio", 0.05, 0.50)
                model = NystromLSSVMColnorm(sigma=sigma, gamma=gamma,
                                            m_ratio=m_ratio, random_state=seed)
                cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
                sc = []
                for tr, val in cv.split(X, y_signed):
                    scaler = StandardScaler()
                    Xtr = scaler.fit_transform(X[tr])
                    Xval = scaler.transform(X[val])
                    try:
                        model.fit(Xtr, y_signed[tr])
                        pred = model.predict(Xval)
                        pb = ((pred + 1) // 2).astype(int)
                        yb = ((y_signed[val] + 1) // 2).astype(int)
                        sc.append(f1_score(yb, pb, average="macro", zero_division=0))
                    except Exception:
                        sc.append(0.0)
                return float(sum(sc) / len(sc))

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(obj_ny, n_trials=n_trials_lssvm, show_progress_bar=False)
            existing[key] = {"best_params": study.best_params,
                             "best_value": study.best_value, "metric": "f1_macro"}
            _save_json(PARAMS_FILE, existing)
            log.info("[OK   ] NystromLSSVMColnorm / %s — f1=%.4f", dataset, study.best_value)
        else:
            log.info("[SKIP ] NystromLSSVMColnorm / %s", dataset)

        # ── FTTransformerCURColnorm ─────────────────────────────────────────
        key = f"FTTransformerCURColnorm__{dataset}"
        if key not in existing:
            log.info("[TUNE ] FTTransformerCURColnorm / %s (%d trials)...", dataset, n_trials_ft)
            from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm

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

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(obj_ft, n_trials=n_trials_ft, timeout=timeout_ft,
                           show_progress_bar=False)
            best = dict(study.best_params)
            best["epochs"] = 200
            best["patience"] = 20
            existing[key] = {"best_params": best,
                             "best_value": study.best_value, "metric": "f1_macro"}
            _save_json(PARAMS_FILE, existing)
            log.info("[OK   ] FTTransformerCURColnorm / %s — f1=%.4f", dataset, study.best_value)
        else:
            log.info("[SKIP ] FTTransformerCURColnorm / %s", dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--trials-lssvm", type=int, default=50)
    parser.add_argument("--trials-transformer", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed-tune", type=int, default=0)
    parser.add_argument("--timeout-transformer", type=float, default=300.0)
    args = parser.parse_args()

    from src.experiments.runner import run_single_experiment

    log.info("=== MK5 Custom Models: Tuning + Experiment ===")

    # ── Phase 1: Tuning ──────────────────────────────────────────────────────
    log.info("── Tuning on MK5 datasets ──")
    _tune(MK5_DATASETS, args.trials_lssvm, args.trials_transformer,
          args.folds, args.seed_tune, args.timeout_transformer)

    # ── Phase 2: Experiment ───────────────────────────────────────────────────
    tuned = _load_params(PARAMS_FILE)
    log.info("Loaded %d tuned combos", len(tuned))

    existing_results = json.loads(OUTPUT_FILE.read_text()) if OUTPUT_FILE.exists() else []
    existing_keys = {
        f"{(r.get('model_variant') or r.get('model'))}__{r.get('dataset')}__{r.get('seed')}"
        for r in existing_results
    }
    log.info("Resuming: %d results already in %s", len(existing_results), OUTPUT_FILE)

    all_results = list(existing_results)
    seeds = list(range(args.seeds))
    custom_models = ["NystromLSSVMColnorm", "FTTransformerCURColnorm"]
    total = len(custom_models) * len(MK5_DATASETS) * len(seeds)
    completed = errors = 0
    t_start = time.perf_counter()

    log.info("── Running %d experiments ──", total)
    for runner_name in custom_models:
        for dataset in MK5_DATASETS:
            key_p = f"{runner_name}__{dataset}"
            params = dict(tuned.get(key_p, DEFAULT_PARAMS.get(runner_name, {})))
            if key_p not in tuned:
                log.warning("No tuned params for %s — using defaults", key_p)

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
                result["n_features_informative"] = 5

                all_results.append(result)
                existing_keys.add(run_key)
                completed += 1
                if result["status"] != "ok":
                    errors += 1

                if completed % 10 == 0:
                    _save_json(OUTPUT_FILE, all_results)

                elapsed = time.perf_counter() - t_start
                remaining = total - completed
                eta = (elapsed / completed * remaining) if completed > 0 else 0
                log.info("[%d/%d] %s / %s / seed=%d — %s | ETA %.0fm",
                         completed, total, runner_name, dataset, seed,
                         result["status"], eta / 60)

    _save_json(OUTPUT_FILE, all_results)
    log.info("=== Done: %d/%d ok, %d errors ===", completed - errors, total, errors)


if __name__ == "__main__":
    main()

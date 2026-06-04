#!/usr/bin/env python3
"""Tune e roda XGBoost em todos os 4 experimentos (Tier1, Scaling, 5f, MK5).

XGBoost serve como baseline tree-based geral para contextualizar a comparação
LSSVM vs Transformer no estudo.

Usage
-----
    python scripts/run_xgboost_full.py [--seeds 30] [--trials 30]
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
        logging.FileHandler("results/xgboost_full.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

TIER1_DS   = ["BCW", "PID", "HAB", "VCP", "GCR", "AUS", "TWS", "TWM", "TWC"]
SCALING_DS = ["TWS_2k", "TWM_2k", "TWC_2k"]
F5_DS      = ["TWS_5f", "TWM_5f", "TWC_5f"]
MK5_DS     = ["MKE", "MKM", "MKH"]

PARAMS_T1  = Path("results/tuning/best_params_xgboost.json")
PARAMS_5F  = Path("results/tuning/best_params_xgboost_5f.json")
PARAMS_MK5 = Path("results/tuning/best_params_xgboost_mk5.json")

DEFAULT = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
           "subsample": 1.0, "colsample_bytree": 1.0,
           "reg_lambda": 1.0, "reg_alpha": 0.0}


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _load_params(path):
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {k: v["best_params"] for k, v in raw.items() if "best_params" in v}


def _resolve(dataset, tuned):
    key = f"XGBoost__{dataset}"
    stripped = dataset.split("_")[0]
    for k in [key, f"XGBoost__{stripped}"]:
        if k in tuned:
            return dict(tuned[k])
    log.warning("No tuned params for XGBoost / %s — using defaults", dataset)
    return dict(DEFAULT)


def _existing_keys(results):
    keys = set()
    for r in results:
        mv = r.get("model_variant") or r.get("model")
        keys.add(f"{mv}__{r.get('dataset')}__{r.get('seed')}")
    return keys


def _tune(datasets, params_file, n_trials, folds, seed):
    import optuna
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from src.data.loaders import DatasetLoader
    from src.models.xgboost_wrapper import XGBoostBaseline

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    existing = json.loads(params_file.read_text()) if params_file.exists() else {}

    for dataset in datasets:
        key = f"XGBoost__{dataset}"
        if key in existing:
            log.info("[SKIP ] %s", key)
            continue

        log.info("[TUNE ] %s (%d trials)...", key, n_trials)
        X, y, _ = DatasetLoader.load(dataset)

        def obj(trial):
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 50, 500),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                random_state=seed,
            )
            model = XGBoostBaseline(**params)
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            scores = []
            for tr, val in cv.split(X, y):
                sc = StandardScaler()
                Xt = sc.fit_transform(X[tr])
                Xv = sc.transform(X[val])
                try:
                    model.fit(Xt, y[tr])
                    pred = model.predict(Xv)
                    scores.append(f1_score(y[val], pred, average="macro", zero_division=0))
                except Exception:
                    scores.append(0.0)
            return float(sum(scores) / len(scores))

        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        existing[key] = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "metric": "f1_macro_cv",
            "n_trials": len(study.trials),
        }
        _save_json(params_file, existing)
        log.info("[OK   ] %s  f1_cv=%.4f", key, study.best_value)


def _run_phase(phase_name, datasets, params_file, output_file, seeds,
               run_single_experiment, extra_field=None):
    tuned = _load_params(params_file)
    existing = json.loads(output_file.read_text()) if output_file.exists() else []
    existing_keys = _existing_keys(existing)
    log.info("Resuming %s: %d in %s", phase_name, len(existing), output_file)

    all_results = list(existing)
    total = len(datasets) * len(seeds)
    completed = errors = 0
    t_start = time.perf_counter()

    for dataset in datasets:
        params = _resolve(dataset, tuned)
        for seed in seeds:
            run_key = f"XGBoost__{dataset}__{seed}"
            if run_key in existing_keys:
                continue
            result = run_single_experiment(
                model_name="XGBoost",
                dataset_name=dataset,
                seed=seed,
                model_params=params,
            )
            result["model_variant"] = "XGBoost"
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
            f1 = result.get("f1_macro", float("nan"))
            log.info("[%s %d/%d] %s / seed=%d — %s  f1=%.4f  ETA %.0fs",
                     phase_name, completed, total, dataset, seed,
                     result["status"], f1, eta)

    _save_json(output_file, all_results)
    log.info("=== %s done: %d errors / %d runs ===", phase_name, errors, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",     type=int, default=30)
    parser.add_argument("--trials",    type=int, default=30)
    parser.add_argument("--folds",     type=int, default=5)
    parser.add_argument("--seed-tune", type=int, default=0)
    args = parser.parse_args()

    from src.experiments.runner import run_single_experiment

    seeds = list(range(args.seeds))
    base  = Path("results")

    log.info("=== XGBoost — Tier 1 + Ablações (seeds=%d, trials=%d) ===",
             args.seeds, args.trials)

    # ── Tuning ────────────────────────────────────────────────────────────────
    log.info("─── Tuning: Tier 1 ───")
    _tune(TIER1_DS, PARAMS_T1, args.trials, args.folds, args.seed_tune)
    log.info("─── Tuning: 5f ───")
    _tune(F5_DS, PARAMS_5F, args.trials, args.folds, args.seed_tune)
    log.info("─── Tuning: MK5 ───")
    _tune(MK5_DS, PARAMS_MK5, args.trials, args.folds, args.seed_tune)

    # ── Experimentos ─────────────────────────────────────────────────────────
    log.info("─── Tier 1 ───")
    _run_phase("TIER1", TIER1_DS, PARAMS_T1,
               base / "tier1_custom_models.json", seeds, run_single_experiment)

    log.info("─── Scaling N=2000 ───")
    _run_phase("SCALING", SCALING_DS, PARAMS_T1,
               base / "synthetic_scaling_n2000.json", seeds, run_single_experiment)

    log.info("─── 5-features (params Tier 1) ───")
    _run_phase("5F-FIXED", F5_DS, PARAMS_T1,
               base / "synthetic_5features.json", seeds, run_single_experiment)

    log.info("─── 5-features (retunado) ───")
    _run_phase("5F-RETUNED", F5_DS, PARAMS_5F,
               base / "synthetic_5features_tuned.json", seeds, run_single_experiment)

    log.info("─── MK5 ───")
    _run_phase("MK5", MK5_DS, PARAMS_MK5,
               base / "synthetic_mk5.json", seeds, run_single_experiment,
               extra_field={"n_features_informative": 5})

    # ── Resumo ───────────────────────────────────────────────────────────────
    import numpy as np
    from collections import defaultdict
    log.info("=== Tudo concluído ===")
    data = json.loads((base / "tier1_custom_models.json").read_text())
    scores = defaultdict(list)
    for r in data:
        if (r.get("model_variant") or r.get("model")) == "XGBoost" and r.get("status") == "ok":
            scores[r["dataset"]].append(r.get("f1_macro", float("nan")))
    print("\n=== XGBoost Tier 1 — F1-macro médio ===")
    all_f1 = []
    for ds in TIER1_DS:
        v = scores.get(ds, [])
        if v:
            print(f"  {ds:<8} {np.mean(v):.4f}")
            all_f1.extend(v)
    if all_f1:
        print(f"  {'Média':<8} {np.mean(all_f1):.4f}")


if __name__ == "__main__":
    main()

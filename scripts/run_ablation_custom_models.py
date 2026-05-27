#!/usr/bin/env python3
"""Run NystromLSSVMColnorm and FTTransformerCURColnorm on ablation datasets.

Appends results to existing ablation JSON files (does NOT overwrite them).
Runs four phases in sequence:

  Phase 1 — Scaling   : TWS_2k, TWM_2k, TWC_2k  → synthetic_scaling_n2000.json
  Phase 2 — 5f fixed  : TWS_5f, TWM_5f, TWC_5f  → synthetic_5features.json
             (params from best_params_custom.json, i.e. tuned on 2D data)
  Phase 3 — Tuning 5f : tune custom models on _5f datasets
             → results/tuning/best_params_custom_5f.json
  Phase 4 — 5f retuned: TWS_5f, TWM_5f, TWC_5f  → synthetic_5features_tuned.json
             (params from best_params_custom_5f.json)

Usage
-----
    python scripts/run_ablation_custom_models.py [--seeds 30]
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
        logging.FileHandler("results/ablation_custom.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

CUSTOM_MODELS = ["NystromLSSVMColnorm", "FTTransformerCURColnorm"]

DEFAULT_PARAMS = {
    "NystromLSSVMColnorm":     {"sigma": 1.0, "gamma": 1.0, "m_ratio": 0.20},
    "FTTransformerCURColnorm": {"d_model": 32, "n_heads": 4, "n_layers": 2,
                                "m_ratio": 0.10, "lr": 5e-4,
                                "epochs": 200, "patience": 20},
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> list:
    return json.loads(path.read_text()) if path.exists() else []


def _load_params(path: Path) -> dict:
    tuned = {}
    if path.exists():
        raw = json.loads(path.read_text())
        for key, val in raw.items():
            if "best_params" in val:
                tuned[key] = val["best_params"]
    return tuned


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _resolve_params(runner_name: str, dataset: str, tuned: dict) -> dict:
    """Look up exact key, then strip suffix fallback, then defaults."""
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


def _run_phase(
    phase_name: str,
    datasets: list[str],
    params_file: Path,
    output_file: Path,
    seeds: list[int],
    run_single_experiment,
) -> None:
    tuned = _load_params(params_file)
    log.info("Loaded %d tuned combos from %s", len(tuned), params_file)

    existing = _load_json(output_file)
    existing_keys = _existing_keys(existing)
    log.info("Resuming %s: %d entries already in %s", phase_name, len(existing), output_file)

    all_results = list(existing)
    total = len(CUSTOM_MODELS) * len(datasets) * len(seeds)
    completed = sum(1 for k in existing_keys if any(
        f"{m}__{ds}__{s}" == k
        for m in CUSTOM_MODELS for ds in datasets for s in seeds
    ))
    errors = 0
    t_start = time.perf_counter()

    for runner_name in CUSTOM_MODELS:
        actual_class = runner_name  # runner.py dispatches by exact name
        for dataset in datasets:
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
                    _save_json(output_file, all_results)

                elapsed = time.perf_counter() - t_start
                remaining = total - completed
                eta = (elapsed / completed * remaining) if completed > 0 else 0
                log.info("[%s %d/%d] %s / %s / seed=%d — %s | ETA %.0fm",
                         phase_name, completed, total,
                         runner_name, dataset, seed, result["status"], eta / 60)

    _save_json(output_file, all_results)
    log.info("=== %s done: %d errors / %d runs ===", phase_name, errors, total)


def _tune_5f(datasets: list[str], n_trials_lssvm: int, n_trials_ft: int,
             folds: int, seed: int, timeout_ft: float,
             out_file: Path) -> None:
    import optuna
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from src.data.loaders import DatasetLoader

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    existing = json.loads(out_file.read_text()) if out_file.exists() else {}

    for dataset in datasets:
        X, y, _ = DatasetLoader.load(dataset)

        # ── NystromLSSVMColnorm ─────────────────────────────────────────────
        key_ny = f"NystromLSSVMColnorm__{dataset}"
        if key_ny not in existing:
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
                scores = []
                for tr, val in cv.split(X, y_signed):
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X[tr])
                    X_val = scaler.transform(X[val])
                    try:
                        model.fit(X_tr, y_signed[tr])
                        pred = model.predict(X_val)
                        pred_bin = ((pred + 1) // 2).astype(int)
                        y_bin = ((y_signed[val] + 1) // 2).astype(int)
                        scores.append(f1_score(y_bin, pred_bin,
                                               average="macro", zero_division=0))
                    except Exception:
                        scores.append(0.0)
                return float(sum(scores) / len(scores))

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(obj_ny, n_trials=n_trials_lssvm, show_progress_bar=False)
            existing[key_ny] = {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "metric": "f1_macro",
                "n_trials": n_trials_lssvm,
            }
            _save_json(out_file, existing)
            log.info("[OK   ] NystromLSSVMColnorm / %s — f1=%.4f", dataset, study.best_value)
        else:
            log.info("[SKIP ] NystromLSSVMColnorm / %s", dataset)

        # ── FTTransformerCURColnorm ─────────────────────────────────────────
        key_ft = f"FTTransformerCURColnorm__{dataset}"
        if key_ft not in existing:
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

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(obj_ft, n_trials=n_trials_ft, timeout=timeout_ft,
                           show_progress_bar=False)
            best = dict(study.best_params)
            best["epochs"] = 200
            best["patience"] = 20
            existing[key_ft] = {
                "best_params": best,
                "best_value": study.best_value,
                "metric": "f1_macro",
                "n_trials": len(study.trials),
            }
            _save_json(out_file, existing)
            log.info("[OK   ] FTTransformerCURColnorm / %s — f1=%.4f", dataset, study.best_value)
        else:
            log.info("[SKIP ] FTTransformerCURColnorm / %s", dataset)


# ── main ─────────────────────────────────────────────────────────────────────

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

    seeds = list(range(args.seeds))

    base = Path("results")
    params_tier1 = Path("results/tuning/best_params_custom.json")
    params_5f    = Path("results/tuning/best_params_custom_5f.json")

    log.info("=== Ablation: Custom Models (%s) ===", CUSTOM_MODELS)
    log.info("Seeds: %d", args.seeds)

    # ── Phase 1: Scaling ─────────────────────────────────────────────────────
    log.info("── Phase 1: Scaling (N=2000) ──")
    _run_phase(
        phase_name="SCALING",
        datasets=["TWS_2k", "TWM_2k", "TWC_2k"],
        params_file=params_tier1,
        output_file=base / "synthetic_scaling_n2000.json",
        seeds=seeds,
        run_single_experiment=run_single_experiment,
    )

    # ── Phase 2: 5f fixed σ ──────────────────────────────────────────────────
    log.info("── Phase 2: 5-features fixed sigma ──")
    _run_phase(
        phase_name="5F-FIXED",
        datasets=["TWS_5f", "TWM_5f", "TWC_5f"],
        params_file=params_tier1,
        output_file=base / "synthetic_5features.json",
        seeds=seeds,
        run_single_experiment=run_single_experiment,
    )

    # ── Phase 3: Tune on 5f ──────────────────────────────────────────────────
    log.info("── Phase 3: Tuning on 5f datasets ──")
    _tune_5f(
        datasets=["TWS_5f", "TWM_5f", "TWC_5f"],
        n_trials_lssvm=args.trials_lssvm,
        n_trials_ft=args.trials_transformer,
        folds=args.folds,
        seed=args.seed_tune,
        timeout_ft=args.timeout_transformer,
        out_file=params_5f,
    )

    # ── Phase 4: 5f retuned ──────────────────────────────────────────────────
    log.info("── Phase 4: 5-features retuned sigma ──")
    _run_phase(
        phase_name="5F-RETUNED",
        datasets=["TWS_5f", "TWM_5f", "TWC_5f"],
        params_file=params_5f,
        output_file=base / "synthetic_5features_tuned.json",
        seeds=seeds,
        run_single_experiment=run_single_experiment,
    )

    log.info("=== All ablation phases complete ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tune NystromLSSVMColnorm and FTTransformerCURColnorm on Tier 1 datasets.

Saves to results/tuning/best_params_custom.json.

Usage
-----
    python scripts/run_tuning_custom_models.py [--trials 50] [--folds 5]
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
        logging.FileHandler("results/tuning/tuning_custom.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

TIER1_DATASETS = ["BCW", "PID", "HAB", "VCP", "GCR", "AUS", "TWS", "TWM", "TWC"]

# (config_key used in models.yaml, runner_name, is_transformer)
MODELS = [
    ("nystrom_lssvm",    "NystromLSSVMColnorm",    False),
    ("ft_cur_colnorm",   "FTTransformerCURColnorm", True),
]

OUT_FILE = Path("results/tuning/best_params_custom.json")


def _load_existing() -> dict:
    return json.loads(OUT_FILE.read_text()) if OUT_FILE.exists() else {}


def _save(data: dict) -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(data, indent=2, default=str))


def _tune_nystrom(dataset: str, n_trials: int, folds: int, seed: int) -> dict:
    """Tune NystromLSSVMColnorm via manual Optuna (not in models.yaml)."""
    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from src.data.loaders import DatasetLoader
    from src.data.preprocessing import preprocess
    from src.models.nystrom_lssvm_wrapper import NystromLSSVMColnorm

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y, _ = DatasetLoader.load(dataset)
    # Convert to signed for LSSVM
    y_signed = (y * 2 - 1).astype(int)

    def objective(trial):
        sigma   = trial.suggest_float("sigma",   0.01, 100.0, log=True)
        gamma   = trial.suggest_float("gamma",   0.01, 1000.0, log=True)
        m_ratio = trial.suggest_float("m_ratio", 0.05, 0.50)

        model = NystromLSSVMColnorm(sigma=sigma, gamma=gamma, m_ratio=m_ratio,
                                    random_state=seed)
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        scores = []
        for tr, val in cv.split(X, y_signed):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_val = scaler.transform(X[val])
            try:
                model.fit(X_tr, y_signed[tr])
                pred = model.predict(X_val)
                # convert signed back to binary for f1_macro
                pred_bin = ((pred + 1) // 2).astype(int)
                y_val_bin = ((y_signed[val] + 1) // 2).astype(int)
                scores.append(f1_score(y_val_bin, pred_bin, average="macro",
                                       zero_division=0))
            except Exception:
                scores.append(0.0)
        return float(sum(scores) / len(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "metric": "f1_macro",
        "n_trials": n_trials,
    }


def _tune_ft_cur(dataset: str, n_trials: int, folds: int, seed: int,
                 timeout: float) -> dict:
    """Tune FTTransformerCURColnorm via manual Optuna."""
    import optuna
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from src.data.loaders import DatasetLoader
    from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X, y, _ = DatasetLoader.load(dataset)

    def objective(trial):
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
            random_state=seed,
        )
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        scores = []
        for tr, val in cv.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_val = scaler.transform(X[val])
            try:
                model.fit(X_tr, y[tr])
                pred = model.predict(X_val)
                scores.append(f1_score(y[val], pred, average="macro",
                                       zero_division=0))
            except Exception:
                scores.append(0.0)
        return float(sum(scores) / len(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   show_progress_bar=False)

    best = dict(study.best_params)
    best["epochs"] = 200
    best["patience"] = 20
    return {
        "best_params": best,
        "best_value": study.best_value,
        "metric": "f1_macro",
        "n_trials": len(study.trials),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-lssvm", type=int, default=50)
    parser.add_argument("--trials-transformer", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-transformer", type=float, default=300.0)
    args = parser.parse_args()

    existing = _load_existing()
    total = len(MODELS) * len(TIER1_DATASETS)
    done = errors = 0

    log.info("=== Custom Models Tuning ===")
    log.info("Models: NystromLSSVMColnorm, FTTransformerCURColnorm")
    log.info("Datasets: %d | Total combos: %d", len(TIER1_DATASETS), total)

    for config_key, runner_name, is_transformer in MODELS:
        for dataset in TIER1_DATASETS:
            key = f"{runner_name}__{dataset}"
            if key in existing:
                log.info("[SKIP] %s / %s", runner_name, dataset)
                done += 1
                continue

            log.info("[RUN ] %s / %s...", runner_name, dataset)
            t0 = time.perf_counter()
            try:
                if is_transformer:
                    result = _tune_ft_cur(dataset, args.trials_transformer,
                                          args.folds, args.seed,
                                          args.timeout_transformer)
                else:
                    result = _tune_nystrom(dataset, args.trials_lssvm,
                                           args.folds, args.seed)
                elapsed = time.perf_counter() - t0
                existing[key] = result
                _save(existing)
                done += 1
                log.info("[OK  ] %s / %s — f1=%.4f in %.1fs",
                         runner_name, dataset, result["best_value"], elapsed)
            except Exception as exc:
                errors += 1
                elapsed = time.perf_counter() - t0
                log.error("[ERR ] %s / %s — %s (%.1fs)", runner_name, dataset, exc, elapsed)
                existing[key] = {"error": str(exc)}
                _save(existing)

    log.info("=== Done: %d/%d ok, %d errors ===", done - errors, total, errors)
    log.info("Saved to %s", OUT_FILE)


if __name__ == "__main__":
    main()

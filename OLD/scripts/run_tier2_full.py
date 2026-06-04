#!/usr/bin/env python3
"""Tier 2 — full-data experiments with scalable models only.

Runs on CREDIT (30k), ADULT (48k) and HIGGS50K with the models that
are architecturally designed for large N:
  - NystromLSSVMColnorm   (Nyström kernel approximation)
  - FTTransformerCURColnorm (CUR inter-instance attention)
  - FTTransformer (softmax / topk / entmax / sparsemax)

Appends to results/tier2_full.json — never overwrites.
GPU is used automatically if available (Colab T4/A100).

Usage
-----
  # Local
  python scripts/run_tier2_full.py

  # Colab (with Drive path for persistence)
  python scripts/run_tier2_full.py --drive-path /content/drive/MyDrive/dissertacao
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
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

DATASETS       = ["CREDIT", "ADULT", "HIGGS50K"]
PARAMS_FILE    = Path("results/tuning/best_params_tier2.json")
OUTPUT_FILE    = Path("results/tier2_full.json")

FT_VARIANTS = [
    ("FTTransformer", "FTTransformer_softmax",  {"attention_type": "softmax"}),
    ("FTTransformer", "FTTransformer_topk",     {"attention_type": "topk",     "topk_ratio": 0.10}),
    ("FTTransformer", "FTTransformer_entmax",   {"attention_type": "entmax",   "alpha": 1.5}),
    ("FTTransformer", "FTTransformer_sparsemax",{"attention_type": "sparsemax"}),
]

SCALABLE_LSSVM = [
    ("NystromLSSVMColnorm",    "NystromLSSVMColnorm"),
    ("FTTransformerCURColnorm","FTTransformerCURColnorm"),
]

DEFAULT_PARAMS = {
    "NystromLSSVMColnorm":     {"sigma": 1.0, "gamma": 1.0, "m_ratio": 0.10},
    "FTTransformerCURColnorm": {"d_model": 32, "n_heads": 4, "n_layers": 2,
                                "m_ratio": 0.10, "lr": 5e-4,
                                "epochs": 200, "patience": 20},
    "FTTransformer_softmax":   {"embedding_dim": 64, "num_blocks": 3,
                                "num_heads": 4, "dropout": 0.1,
                                "lr": 1e-3, "batch_size": 256,
                                "max_epochs": 200, "patience": 20,
                                "attention_type": "softmax"},
    "FTTransformer_topk":      {"embedding_dim": 64, "num_blocks": 3,
                                "num_heads": 4, "dropout": 0.1,
                                "lr": 1e-3, "batch_size": 256,
                                "max_epochs": 200, "patience": 20,
                                "attention_type": "topk", "topk_ratio": 0.10},
    "FTTransformer_entmax":    {"embedding_dim": 64, "num_blocks": 3,
                                "num_heads": 4, "dropout": 0.1,
                                "lr": 1e-3, "batch_size": 256,
                                "max_epochs": 200, "patience": 20,
                                "attention_type": "entmax", "alpha": 1.5},
    "FTTransformer_sparsemax": {"embedding_dim": 64, "num_blocks": 3,
                                "num_heads": 4, "dropout": 0.1,
                                "lr": 1e-3, "batch_size": 256,
                                "max_epochs": 200, "patience": 20,
                                "attention_type": "sparsemax"},
}


# ── helpers ───────────────────────────────────────────────────────────────────

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


def _resolve_params(variant_name: str, dataset: str, tuned: dict,
                    extra_params: dict) -> dict:
    key = f"{variant_name}__{dataset}"
    params = dict(tuned[key]) if key in tuned else dict(DEFAULT_PARAMS.get(variant_name, {}))
    params.update(extra_params)
    return params


# ── Tuning ────────────────────────────────────────────────────────────────────

def _sync_params(drive_path: str | None) -> None:
    """Copia best_params_tier2.json para o Drive (proteção contra crash)."""
    if drive_path and PARAMS_FILE.exists():
        dest = Path(drive_path) / "tuning" / PARAMS_FILE.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(PARAMS_FILE, dest)


def _tune(datasets: list[str], n_trials_lssvm: int, n_trials_ft: int,
          folds: int, seed: int, drive_path: str | None = None) -> None:
    import optuna
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from src.data.loaders import DatasetLoader

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    existing = json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else {}

    for dataset in datasets:
        X, y, _ = DatasetLoader.load(dataset)
        log.info("Dataset %s: N=%d, p=%d", dataset, X.shape[0], X.shape[1])

        # ── NystromLSSVMColnorm ───────────────────────────────────────────────
        key = f"NystromLSSVMColnorm__{dataset}"
        if key not in existing:
            log.info("[TUNE ] %s (%d trials)...", key, n_trials_lssvm)
            from src.models.nystrom_lssvm_wrapper import NystromLSSVMColnorm
            y_signed = (y * 2 - 1).astype(int)

            def obj_ny(trial):
                sigma   = trial.suggest_float("sigma",   0.01, 100.0, log=True)
                gamma   = trial.suggest_float("gamma",   0.01, 1000.0, log=True)
                m_ratio = trial.suggest_float("m_ratio", 0.02, 0.20)
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
            _sync_params(drive_path)
            log.info("[OK   ] %s — f1=%.4f", key, study.best_value)
        else:
            log.info("[SKIP ] %s", key)

        # ── FTTransformerCURColnorm ───────────────────────────────────────────
        key = f"FTTransformerCURColnorm__{dataset}"
        if key not in existing:
            log.info("[TUNE ] %s (%d trials)...", key, n_trials_ft)
            from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm

            def obj_cur(trial):
                d_model = trial.suggest_categorical("d_model", [32, 64])
                n_heads = trial.suggest_categorical("n_heads", [2, 4])
                n_layers = trial.suggest_int("n_layers", 1, 3)
                m_ratio = trial.suggest_float("m_ratio", 0.02, 0.15)
                lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                if d_model % n_heads != 0:
                    return 0.0
                model = FTTransformerCURColnorm(
                    d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                    m_ratio=m_ratio, lr=lr, epochs=30, patience=5,
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
            study.optimize(obj_cur, n_trials=n_trials_ft, show_progress_bar=False)
            best = dict(study.best_params)
            best["epochs"] = 200
            best["patience"] = 20
            existing[key] = {"best_params": best,
                             "best_value": study.best_value, "metric": "f1_macro"}
            _save_json(PARAMS_FILE, existing)
            _sync_params(drive_path)
            log.info("[OK   ] %s — f1=%.4f", key, study.best_value)
        else:
            log.info("[SKIP ] %s", key)

        # ── FTTransformer variants ────────────────────────────────────────────
        for _, variant_name, extra in FT_VARIANTS:
            key = f"{variant_name}__{dataset}"
            if key not in existing:
                log.info("[TUNE ] %s (%d trials)...", key, n_trials_ft)
                from src.models.transformers.ft_transformer import FTTransformer

                def obj_ft(trial, _extra=extra):
                    embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
                    num_blocks = trial.suggest_categorical("num_blocks", [2, 3, 4])
                    num_heads = trial.suggest_categorical("num_heads", [2, 4])
                    dropout = trial.suggest_float("dropout", 0.0, 0.3)
                    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                    batch_size = trial.suggest_categorical("batch_size", [256, 512])
                    if embedding_dim % num_heads != 0:
                        return 0.0
                    params = dict(embedding_dim=embedding_dim, num_blocks=num_blocks,
                                  num_heads=num_heads, dropout=dropout, lr=lr,
                                  batch_size=batch_size, max_epochs=30, patience=5,
                                  **_extra)
                    model = FTTransformer(**params)
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
                study.optimize(obj_ft, n_trials=n_trials_ft, show_progress_bar=False)
                best = dict(study.best_params)
                best["max_epochs"] = 200
                best["patience"] = 20
                best.update(extra)
                existing[key] = {"best_params": best,
                                 "best_value": study.best_value, "metric": "f1_macro"}
                _save_json(PARAMS_FILE, existing)
                log.info("[OK   ] %s — f1=%.4f", key, study.best_value)
            else:
                log.info("[SKIP ] %s", key)


# ── Experiments ───────────────────────────────────────────────────────────────

def _run_experiments(seeds: list[int], drive_path: str | None) -> None:
    from src.experiments.runner import run_single_experiment

    tuned = _load_params(PARAMS_FILE)
    existing = json.loads(OUTPUT_FILE.read_text()) if OUTPUT_FILE.exists() else []
    done_keys = _existing_keys(existing)
    all_results = list(existing)

    # Build full model list: (runner_name, variant_name, extra_params)
    all_models = (
        [(r, v, {}) for r, v in SCALABLE_LSSVM] +
        [(r, v, e) for r, v, e in FT_VARIANTS]
    )

    total = len(all_models) * len(DATASETS) * len(seeds)
    completed = errors = 0
    t0 = time.perf_counter()

    log.info("── Running %d experiments ──", total)

    for runner_name, variant_name, extra_params in all_models:
        for dataset in DATASETS:
            params = _resolve_params(variant_name, dataset, tuned, extra_params)

            for seed in seeds:
                run_key = f"{variant_name}__{dataset}__{seed}"
                if run_key in done_keys:
                    continue

                result = run_single_experiment(
                    model_name=runner_name,
                    dataset_name=dataset,
                    seed=seed,
                    model_params=params,
                )
                result["model_variant"] = variant_name

                all_results.append(result)
                done_keys.add(run_key)
                completed += 1
                if result["status"] != "ok":
                    errors += 1

                if completed % 5 == 0:
                    _save_json(OUTPUT_FILE, all_results)
                    if drive_path:
                        dest = Path(drive_path) / OUTPUT_FILE.name
                        shutil.copy(OUTPUT_FILE, dest)

                elapsed = time.perf_counter() - t0
                eta = (elapsed / completed * (total - completed)) if completed else 0
                log.info("[%d/%d] %s / %s / seed=%d — %s | ETA %.0fm",
                         completed, total, variant_name, dataset, seed,
                         result["status"], eta / 60)

    _save_json(OUTPUT_FILE, all_results)
    if drive_path:
        dest = Path(drive_path) / OUTPUT_FILE.name
        shutil.copy(OUTPUT_FILE, dest)
        log.info("Results synced to Drive: %s", dest)

    log.info("=== Done: %d/%d ok, %d errors ===",
             completed - errors, total, errors)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",         type=int,   default=30)
    parser.add_argument("--trials-lssvm",  type=int,   default=50)
    parser.add_argument("--trials-ft",     type=int,   default=20)
    parser.add_argument("--folds",         type=int,   default=3,
                        help="CV folds for tuning (3 recommended for large datasets)")
    parser.add_argument("--seed-tune",     type=int,   default=0)
    parser.add_argument("--drive-path",    type=str,   default=None,
                        help="Google Drive path to sync results (Colab only)")
    parser.add_argument("--skip-tuning",   action="store_true",
                        help="Skip tuning phase and use existing/default params")
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("=== Tier 2 Full-Data Experiments ===")
    log.info("Device: %s | datasets: %s", device.upper(), DATASETS)

    if not args.skip_tuning:
        log.info("── Phase 1: Tuning ──")
        _tune(DATASETS, args.trials_lssvm, args.trials_ft,
              args.folds, args.seed_tune, args.drive_path)

    log.info("── Phase 2: Experiments ──")
    _run_experiments(list(range(args.seeds)), args.drive_path)


if __name__ == "__main__":
    main()

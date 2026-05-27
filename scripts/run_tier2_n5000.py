#!/usr/bin/env python3
"""Tier 2 com N=5000 amostras (subamostradas estratificadamente).

Roda TODOS os 19 modelos em 6 datasets:
  ADULT (45k), CREDIT (30k), BANK (45k), TELCO (7k), SHOPPERS (12k), HIGGS50K (50k)

Cada (modelo × dataset × seed) usa um subsample estratificado de N=5000
determinístico pela seed, garantindo variância amostral entre runs.

Tuning policy
-------------
Tuning Optuna leva muito tempo (~12h só ele). Estratégia adotada:
  - Tunar via Optuna apenas: XGBoost, DualFISTA, Nyström-SVM, FT-CUR, SAINT
  - LSSVMs baselines usam defaults razoáveis (StandardScaler + sigma=1.0)
  - FT-Transformers baselines usam defaults consagrados (embedding_dim=64,
    num_blocks=3, num_heads=4, dropout=0.1, lr=1e-3, batch_size=256)

Saída: results/tier2_n5000.json
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
        logging.FileHandler("results/tier2_n5000.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Constants ─────────────────────────────────────────────────────────────────

# Protocolo Tier 2: N_TRAIN = 5000 (treino). Como o split é 70/30, subamostramos
# N_TOTAL_CAP = round(N_TRAIN / 0.70) ≈ 7143 do dataset original, e o split
# 70/30 resulta em ~5000 treino + ~2143 teste.
N_TRAIN = 5000
N_TOTAL_CAP = round(N_TRAIN / 0.70)   # = 7143
DATASETS = ["ADULT", "CREDIT", "BANK", "TELCO", "SHOPPERS", "HIGGS50K"]
OUTPUT_FILE = Path("results/tier2_n5000.json")
PARAMS_FILE = Path("results/tuning/best_params_tier2_n5000.json")

# Model groups: (runner_name, variant_name, extra_params)
LSSVM_BASELINES = [
    ("StandardLSSVM",     "StandardLSSVM",     {}),
    ("PCPLSSVm",          "PCPLSSVm",          {}),
    ("FSALSSVm",          "FSALSSVm",          {}),
    ("IPLSSVm",           "IPLSSVm",           {}),
    ("PruningLSSVM",      "PruningLSSVM",      {}),
    ("OppositeMapsLSSVM", "OppositeMapsLSSVM", {}),
    ("FISTANesterovLSSVM","FISTANesterov",     {}),
]
LSSVM_PROPOSED = [
    ("ADMMNesterovLSSVM",         "ADMMNesterovLSSVM",  {}),  # paper-base
    ("OriginalADMMNesterovLSSVM", "ADMMElasticNet",     {}),  # variante
    ("DualFISTALSSVM",            "DualFISTA",          {}),  # tuna
    ("NystromLSSVMColnorm",       "NystromLSSVMColnorm",{}),  # tuna
]
FT_VARIANTS = [
    ("FTTransformer", "FTTransformer_softmax",   {"attention_type": "softmax"}),
    ("FTTransformer", "FTTransformer_topk",      {"attention_type": "topk", "topk_ratio": 0.10}),
    ("FTTransformer", "FTTransformer_entmax",    {"attention_type": "entmax", "alpha": 1.5}),
    ("FTTransformer", "FTTransformer_sparsemax", {"attention_type": "sparsemax"}),
]
INTER_INSTANCE = [
    ("SAINTColnorm",            "SAINTColnorm",            {}),  # tuna
    ("FTTransformerCURColnorm", "FTTransformerCURColnorm", {}),  # tuna
]
GENERAL_BASELINE = [
    ("XGBoost", "XGBoost", {}),  # tuna
]

ALL_MODELS = (GENERAL_BASELINE + LSSVM_BASELINES + LSSVM_PROPOSED
              + FT_VARIANTS + INTER_INSTANCE)

# Modelos que serão tunados (os outros usam defaults razoáveis)
TUNABLE = {"XGBoost", "DualFISTA", "NystromLSSVMColnorm",
           "FTTransformerCURColnorm", "SAINTColnorm"}

# Defaults com nomes de parâmetros corretos (verificados em inspect)
DEFAULT_PARAMS = {
    "StandardLSSVM":             {"sigma": 1.0, "tau": 1.0},
    "PCPLSSVm":                  {"sigma": 1.0, "tau": 1.0, "rank": 100},
    "FSALSSVm":                  {"sigma": 1.0, "tau": 1.0, "n_components": 200},
    "IPLSSVm":                   {"sigma": 1.0, "tau": 1.0, "selection_ratio": 0.20},
    "PruningLSSVM":              {"sigma": 1.0, "tau": 1.0, "pruning_rate": 0.30},
    "OppositeMapsLSSVM":         {"sigma": 1.0, "tau": 1.0, "n_prototypes": 100},
    "FISTANesterovLSSVM":        {"sigma": 1.0, "tau": 1.0, "lambda_": 0.01},
    "ADMMNesterovLSSVM":         {"sigma": 1.0, "tau": 1.0, "rho": 1.0, "lambda_": 0.01},
    "OriginalADMMNesterovLSSVM": {"sigma": 1.0, "tau": 1.0, "rho": 1.0, "lambda_": 0.01},
    "DualFISTALSSVM":            {"sigma": 1.0, "tau": 1.0, "lambda_": 0.01},
    "NystromLSSVMColnorm":       {"sigma": 1.0, "gamma": 1.0, "m_ratio": 0.10},
    "FTTransformerCURColnorm":   {"d_model": 32, "n_heads": 4, "n_layers": 2,
                                  "m_ratio": 0.10, "lr": 5e-4, "epochs": 60,
                                  "patience": 8},
    "SAINTColnorm":              {"d_model": 32, "n_heads": 4, "n_layers": 2,
                                  "lr": 5e-4, "epochs": 60, "patience": 8},
    "XGBoost":                   {"n_estimators": 200, "max_depth": 6,
                                  "learning_rate": 0.1},
}
for v, extra in [("softmax", {}), ("topk", {"topk_ratio": 0.10}),
                  ("entmax", {"alpha": 1.5}), ("sparsemax", {})]:
    DEFAULT_PARAMS[f"FTTransformer_{v}"] = {
        "embedding_dim": 64, "num_blocks": 3, "num_heads": 4,
        "dropout": 0.1, "lr": 1e-3, "batch_size": 256,
        "max_epochs": 60, "patience": 8, "attention_type": v, **extra,
    }


GROUPS = {
    "all": ALL_MODELS,
    "lssvm": LSSVM_BASELINES + LSSVM_PROPOSED,
    "transformer": FT_VARIANTS + INTER_INSTANCE,
    "fast": GENERAL_BASELINE + [
        ("NystromLSSVMColnorm", "NystromLSSVMColnorm", {}),
        ("DualFISTALSSVM",      "DualFISTA",          {}),
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _load_params(path):
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {k: v["best_params"] for k, v in raw.items() if "best_params" in v}


def _resolve_params(variant_name, dataset, tuned, extra):
    key = f"{variant_name}__{dataset}"
    params = dict(tuned[key]) if key in tuned else dict(DEFAULT_PARAMS.get(variant_name, {}))
    if extra:
        params.update(extra)
    return params


def _existing_keys(results):
    keys = set()
    for r in results:
        mv = r.get("model_variant") or r.get("model")
        keys.add(f"{mv}__{r.get('dataset')}__{r.get('seed')}")
    return keys


# ── CV helper ─────────────────────────────────────────────────────────────────

def _cv_eval(model_factory, X, y, folds, seed, signed=False):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    y_target = (y * 2 - 1).astype(int) if signed else y
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for tr, val in cv.split(X, y_target):
        sc = StandardScaler()
        Xt = sc.fit_transform(X[tr]); Xv = sc.transform(X[val])
        try:
            m = model_factory()
            m.fit(Xt, y_target[tr])
            pred = m.predict(Xv)
            if signed:
                pb = ((pred + 1) // 2).astype(int)
                yb = ((y_target[val] + 1) // 2).astype(int)
                scores.append(f1_score(yb, pb, average="macro", zero_division=0))
            else:
                scores.append(f1_score(y_target[val], pred, average="macro", zero_division=0))
        except Exception as e:
            log.debug("CV fold failed: %s", e)
            scores.append(0.0)
    return float(sum(scores) / len(scores))


# ── Objetivos Optuna (só dos modelos tunáveis) ────────────────────────────────

def _obj_dual_fista(trial, X, y, folds, seed):
    from src.models.lssvm.dual.fista_dual_lssvm import DualFISTALSSVM
    p = {"sigma":   trial.suggest_float("sigma", 0.1, 50.0, log=True),
         "tau":     trial.suggest_float("tau",   0.01, 1000.0, log=True),
         "lambda_": trial.suggest_float("lambda_", 1e-4, 1.0, log=True)}
    return _cv_eval(lambda: DualFISTALSSVM(**p, max_iter=200), X, y, folds, seed, signed=True)


def _obj_nystrom(trial, X, y, folds, seed):
    from src.models.nystrom_lssvm_wrapper import NystromLSSVMColnorm
    p = {"sigma":   trial.suggest_float("sigma", 0.1, 50.0, log=True),
         "gamma":   trial.suggest_float("gamma", 0.01, 1000.0, log=True),
         "m_ratio": trial.suggest_float("m_ratio", 0.02, 0.30)}
    return _cv_eval(lambda: NystromLSSVMColnorm(**p, random_state=seed), X, y, folds, seed, signed=True)


def _obj_ftcur(trial, X, y, folds, seed):
    from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm
    d_model = trial.suggest_categorical("d_model", [16, 32, 64])
    n_heads = trial.suggest_categorical("n_heads", [2, 4])
    if d_model % n_heads != 0:
        return 0.0
    p = dict(d_model=d_model, n_heads=n_heads,
             n_layers=trial.suggest_int("n_layers", 1, 3),
             m_ratio=trial.suggest_float("m_ratio", 0.02, 0.15),
             lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
             epochs=30, patience=5, random_state=seed)
    return _cv_eval(lambda: FTTransformerCURColnorm(**p), X, y, folds, seed)


def _obj_saint(trial, X, y, folds, seed):
    from src.models.ft_transformer_saint_wrapper import SAINTColnorm
    d_model = trial.suggest_categorical("d_model", [16, 32, 64])
    n_heads = trial.suggest_categorical("n_heads", [2, 4])
    if d_model % n_heads != 0:
        return 0.0
    p = dict(d_model=d_model, n_heads=n_heads,
             n_layers=trial.suggest_int("n_layers", 1, 3),
             lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
             epochs=30, patience=5, random_state=seed)
    return _cv_eval(lambda: SAINTColnorm(**p), X, y, folds, seed)


def _obj_xgb(trial, X, y, folds, seed):
    from src.models.xgboost_wrapper import XGBoostBaseline
    p = dict(n_estimators=trial.suggest_int("n_estimators", 50, 500),
             max_depth=trial.suggest_int("max_depth", 3, 10),
             learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
             subsample=trial.suggest_float("subsample", 0.5, 1.0),
             colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
             reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
             reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
             random_state=seed)
    return _cv_eval(lambda: XGBoostBaseline(**p), X, y, folds, seed)


OBJ_FN = {
    "XGBoost":                _obj_xgb,
    "DualFISTA":              _obj_dual_fista,
    "NystromLSSVMColnorm":    _obj_nystrom,
    "FTTransformerCURColnorm":_obj_ftcur,
    "SAINTColnorm":           _obj_saint,
}


# ── Tuning ────────────────────────────────────────────────────────────────────

def _tune(models, datasets, n_trials, folds, seed_tune):
    import optuna
    from sklearn.model_selection import StratifiedShuffleSplit
    from src.data.loaders import DatasetLoader

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    existing = json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else {}

    normalized = [(t[0], t[1], t[2] if len(t) == 3 else {}) for t in models]

    for runner_name, variant_name, extra in normalized:
        if variant_name not in TUNABLE:
            continue
        for dataset in datasets:
            key = f"{variant_name}__{dataset}"
            if key in existing:
                log.info("[SKIP-TUNE] %s", key); continue

            log.info("[TUNE] %s (%d trials)...", key, n_trials)
            X, y, _ = DatasetLoader.load(dataset)
            if len(X) > N_TOTAL_CAP:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=N_TOTAL_CAP,
                                              random_state=seed_tune)
                idx, _ = next(sss.split(X, y))
                X, y = X[idx], y[idx]

            obj_fn = OBJ_FN[variant_name]
            try:
                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=seed_tune))
                study.optimize(lambda tr: obj_fn(tr, X, y, folds, seed_tune),
                               n_trials=n_trials, show_progress_bar=False)
                best = dict(study.best_params)
                # Restore full training epochs for FT models
                if variant_name in {"FTTransformerCURColnorm", "SAINTColnorm"}:
                    best["epochs"] = 60; best["patience"] = 8
                existing[key] = {"best_params": best,
                                 "best_value": study.best_value,
                                 "metric": "f1_macro_cv"}
                _save_json(PARAMS_FILE, existing)
                log.info("[OK]   %s  f1=%.4f", key, study.best_value)
            except Exception as e:
                log.warning("[FAIL] %s — %s", key, e)


# ── Experiments ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",        type=int, default=30)
    parser.add_argument("--trials",       type=int, default=20)
    parser.add_argument("--folds",        type=int, default=3)
    parser.add_argument("--seed-tune",    type=int, default=0)
    parser.add_argument("--models-group", choices=list(GROUPS.keys()), default="all")
    parser.add_argument("--datasets",     nargs="*", default=None)
    parser.add_argument("--skip-tuning",  action="store_true")
    args = parser.parse_args()

    from src.experiments.runner import run_single_experiment

    datasets = args.datasets or DATASETS
    models = GROUPS[args.models_group]
    seeds = list(range(args.seeds))

    log.info("=== Tier 2 N=%d ===", N_TOTAL_CAP)
    log.info("Group: %s  |  %d models  |  %d datasets  |  %d seeds",
             args.models_group, len(models), len(datasets), len(seeds))
    log.info("Datasets: %s", datasets)

    if not args.skip_tuning:
        log.info("── Tuning (Optuna, %d trials) ──", args.trials)
        _tune(models, datasets, args.trials, args.folds, args.seed_tune)

    existing = json.loads(OUTPUT_FILE.read_text()) if OUTPUT_FILE.exists() else []
    done_keys = _existing_keys(existing)
    all_results = list(existing)
    tuned = _load_params(PARAMS_FILE)

    total = len(models) * len(datasets) * len(seeds)
    completed = errors = 0
    t0 = time.perf_counter()

    log.info("── Running %d experiments (N=%d cap) ──", total, N_TOTAL_CAP)

    normalized = [(t[0], t[1], t[2] if len(t) == 3 else {}) for t in models]

    for runner_name, variant_name, extra in normalized:
        for dataset in datasets:
            params = _resolve_params(variant_name, dataset, tuned, extra)

            for seed in seeds:
                run_key = f"{variant_name}__{dataset}__{seed}"
                if run_key in done_keys:
                    continue

                result = run_single_experiment(
                    model_name=runner_name,
                    dataset_name=dataset,
                    seed=seed,
                    model_params=params,
                    n_samples_cap=N_TOTAL_CAP,
                )
                result["model_variant"] = variant_name
                result["n_samples_cap"] = N_TOTAL_CAP

                all_results.append(result)
                done_keys.add(run_key)
                completed += 1
                if result["status"] != "ok":
                    errors += 1

                if completed % 5 == 0:
                    _save_json(OUTPUT_FILE, all_results)

                elapsed = time.perf_counter() - t0
                eta = (elapsed / completed * (total - completed)) if completed else 0
                f1 = result.get("f1_macro", float("nan"))
                log.info("[%d/%d] %s / %s / seed=%d — %s f1=%.4f | ETA %.0fm",
                         completed, total, variant_name, dataset, seed,
                         result["status"], f1, eta / 60)

    _save_json(OUTPUT_FILE, all_results)
    log.info("=== Done: %d/%d ok, %d errors ===", completed - errors, total, errors)


if __name__ == "__main__":
    main()

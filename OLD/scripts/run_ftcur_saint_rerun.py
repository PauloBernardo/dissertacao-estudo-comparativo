#!/usr/bin/env python3
"""Rerun focado de FT-CUR e SAINT com early_stop_metric e/ou balance_train.

Após a ablação H2 confirmar que val_loss > val_acc para esses modelos em
dados desbalanceados (p < 10⁻⁵), os resultados do Tier 2 atual com val_acc
estão metodologicamente comprometidos para FT-CUR e SAINT.

Este script tuna independentemente cada (modelo × dataset × early_stop_metric)
via Optuna e roda experimentos com 30 seeds.

Casos de uso típicos:

  # 1. Re-rodar Tier 2 N=5000 FT-CUR/SAINT com val_loss em todos 6 datasets
  python scripts/run_ftcur_saint_rerun.py \\
      --early-stop-metric val_loss \\
      --datasets ADULT CREDIT BANK TELCO SHOPPERS HIGGS50K \\
      --output-file results/tier2_n5000_ftcur_saint_valloss.json \\
      --params-file results/tuning/best_params_ftcur_saint_valloss.json

  # 2. H1 balanced para FT-CUR/SAINT com val_loss (protocolo corrigido)
  python scripts/run_ftcur_saint_rerun.py \\
      --early-stop-metric val_loss \\
      --datasets CREDIT BANK SHOPPERS \\
      --balance-train \\
      --output-file results/tier2_balanced_ftcur_saint.json \\
      --params-file results/tuning/best_params_tier2_balanced_ftcur_saint.json

Compatível com checkpoint resume (--seeds N) — re-executar para retomar.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

N_TOTAL_CAP = 7143
MODELS = [
    ("FTTransformerCURColnorm", "FTTransformerCURColnorm"),
    ("SAINTColnorm",            "SAINTColnorm"),
]


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _cv_eval_f1(model_factory, X, y, folds, seed):
    """CV evaluation com F1-macro como métrica de comparação no Optuna."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for tr, val in cv.split(X, y):
        sc = StandardScaler()
        Xt = sc.fit_transform(X[tr]); Xv = sc.transform(X[val])
        try:
            m = model_factory()
            m.fit(Xt, y[tr])
            pred = m.predict(Xv)
            scores.append(f1_score(y[val], pred, average="macro", zero_division=0))
        except Exception as e:
            log.debug("CV fold failed: %s", e)
            scores.append(0.0)
    return float(sum(scores) / len(scores))


def _obj_ftcur(trial, X, y, folds, seed, metric):
    from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm
    d_model = trial.suggest_categorical("d_model", [16, 32, 64])
    n_heads = trial.suggest_categorical("n_heads", [2, 4])
    if d_model % n_heads != 0:
        return 0.0
    p = dict(d_model=d_model, n_heads=n_heads,
             n_layers=trial.suggest_int("n_layers", 1, 3),
             m_ratio=trial.suggest_float("m_ratio", 0.02, 0.15),
             lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
             epochs=30, patience=5, random_state=seed,
             early_stop_metric=metric)
    return _cv_eval_f1(lambda: FTTransformerCURColnorm(**p), X, y, folds, seed)


def _obj_saint(trial, X, y, folds, seed, metric):
    from src.models.ft_transformer_saint_wrapper import SAINTColnorm
    d_model = trial.suggest_categorical("d_model", [16, 32, 64])
    n_heads = trial.suggest_categorical("n_heads", [2, 4])
    if d_model % n_heads != 0:
        return 0.0
    p = dict(d_model=d_model, n_heads=n_heads,
             n_layers=trial.suggest_int("n_layers", 1, 3),
             lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
             epochs=30, patience=5, random_state=seed,
             early_stop_metric=metric)
    return _cv_eval_f1(lambda: SAINTColnorm(**p), X, y, folds, seed)


OBJ_FN = {
    "FTTransformerCURColnorm": _obj_ftcur,
    "SAINTColnorm":            _obj_saint,
}


def _tune(datasets, params_file, metric, n_trials, folds, seed_tune,
          balance_train):
    """Tuna FT-CUR e SAINT em datasets dados, todos com metric fixo.

    Se balance_train, aplica undersampling no subsample durante tuning também.
    """
    import optuna
    from sklearn.model_selection import StratifiedShuffleSplit
    from src.data.loaders import DatasetLoader
    import numpy as np

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    existing = json.loads(params_file.read_text()) if params_file.exists() else {}

    for runner_name, variant_name in MODELS:
        for dataset in datasets:
            key = f"{variant_name}__{dataset}__{metric}"
            if key in existing:
                log.info("[SKIP-TUNE] %s", key); continue

            log.info("[TUNE] %s (%d trials)...", key, n_trials)
            X, y, _ = DatasetLoader.load(dataset)
            if len(X) > N_TOTAL_CAP:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=N_TOTAL_CAP,
                                              random_state=seed_tune)
                idx, _ = next(sss.split(X, y))
                X, y = X[idx], y[idx]

            # Se balance_train, balancear o subsample inteiro para o tuning
            # (consistente com o protocolo de experimento, onde train é balanceado)
            if balance_train:
                idx_pos = np.where(y == 1)[0]
                idx_neg = np.where(y == 0)[0]
                n_minor = min(len(idx_pos), len(idx_neg))
                rng = np.random.RandomState(seed_tune)
                if len(idx_pos) > n_minor:
                    idx_pos = rng.choice(idx_pos, size=n_minor, replace=False)
                if len(idx_neg) > n_minor:
                    idx_neg = rng.choice(idx_neg, size=n_minor, replace=False)
                idx = np.sort(np.concatenate([idx_pos, idx_neg]))
                X, y = X[idx], y[idx]

            obj_fn = OBJ_FN[variant_name]
            try:
                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=seed_tune))
                study.optimize(
                    lambda tr: obj_fn(tr, X, y, folds, seed_tune, metric),
                    n_trials=n_trials, show_progress_bar=False)
                best = dict(study.best_params)
                best["epochs"] = 60; best["patience"] = 8
                best["early_stop_metric"] = metric
                existing[key] = {
                    "best_params": best,
                    "best_value": study.best_value,
                    "metric": "f1_macro_cv",
                    "balance_train": balance_train,
                }
                _save_json(params_file, existing)
                log.info("[OK]   %s  f1_cv=%.4f", key, study.best_value)
            except Exception as e:
                log.warning("[FAIL] %s — %s", key, e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--early-stop-metric", required=True,
                        choices=["val_acc", "val_loss", "val_f1_macro"],
                        help="Critério de early stopping para FT-CUR/SAINT")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Datasets para rodar")
    parser.add_argument("--balance-train", action="store_true",
                        help="Balanceia treino via undersampling")
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--params-file", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=30)
    # Defaults idênticos ao Tier 2 N=5000 para garantir comparabilidade direta
    # dos resultados FT-CUR/SAINT corrigidos (val_loss) com o restante do Tier 2.
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed-tune", type=int, default=0)
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args()

    from src.experiments.runner import run_single_experiment

    seeds = list(range(args.seeds))
    log.info("=" * 70)
    log.info("Rerun FT-CUR / SAINT — metric=%s, balance_train=%s",
             args.early_stop_metric, args.balance_train)
    log.info("Datasets: %s", args.datasets)
    log.info("=" * 70)

    # Tuning
    if not args.skip_tuning:
        _tune(args.datasets, args.params_file, args.early_stop_metric,
              args.trials, args.folds, args.seed_tune, args.balance_train)

    # Carrega params
    tuned = {}
    if args.params_file.exists():
        raw = json.loads(args.params_file.read_text())
        for k, v in raw.items():
            if "best_params" in v:
                tuned[k] = v["best_params"]
    log.info("Params carregados: %d combos", len(tuned))

    # Experimentos
    existing = json.loads(args.output_file.read_text()) if args.output_file.exists() else []
    done_keys = set()
    for r in existing:
        key = (r.get("model_variant"), r.get("dataset"), r.get("seed"))
        done_keys.add(key)
    log.info("Resumindo: %d runs já feitos", len(existing))

    all_results = list(existing)
    total = len(MODELS) * len(args.datasets) * len(seeds)
    completed = errors = 0
    t_start = time.perf_counter()

    for runner_name, variant_name in MODELS:
        for dataset in args.datasets:
            key = f"{variant_name}__{dataset}__{args.early_stop_metric}"
            if key not in tuned:
                log.warning("Sem params para %s — pulando", key)
                continue
            params = dict(tuned[key])

            for seed in seeds:
                run_key = (variant_name, dataset, seed)
                if run_key in done_keys:
                    continue

                result = run_single_experiment(
                    model_name=runner_name,
                    dataset_name=dataset,
                    seed=seed,
                    model_params=params,
                    n_samples_cap=N_TOTAL_CAP,
                    balance_train=args.balance_train,
                )
                result["model_variant"] = variant_name
                result["early_stop_metric"] = args.early_stop_metric
                result["balance_train"] = args.balance_train
                result["n_samples_cap"] = N_TOTAL_CAP

                all_results.append(result)
                done_keys.add(run_key)
                completed += 1
                if result["status"] != "ok":
                    errors += 1

                if completed % 5 == 0:
                    _save_json(args.output_file, all_results)

                elapsed = time.perf_counter() - t_start
                eta = (elapsed / completed * (total - completed)) if completed else 0
                f1 = result.get("f1_macro", float("nan"))
                log.info("[%d/%d] %s/%s/seed=%d — f1=%.4f | ETA %.0fm",
                         completed, total, variant_name, dataset, seed,
                         f1, eta / 60)

    _save_json(args.output_file, all_results)
    log.info("=" * 70)
    log.info("Concluído: %d/%d ok, %d errors", completed - errors, total, errors)
    log.info("=" * 70)


if __name__ == "__main__":
    main()

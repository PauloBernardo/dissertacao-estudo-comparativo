#!/usr/bin/env python3
"""Ablação: critério de early stopping para FT-CUR e SAINT em imbalanceados.

Testa a hipótese H2 do relatório:
  O colapso do FT-CUR (e potencialmente SAINT) em CREDIT/BANK pode ser causado
  por uso de val_acc como critério de early stopping. Em datasets imbalanceados,
  val_acc favorece checkpoints que acertam majoritariamente a classe dominante,
  mas que ainda têm F1-macro baixo.

Comparação rigorosa (retuna para cada métrica):
  - val_acc       (default original)
  - val_loss      (BCE loss em validação)
  - val_f1_macro  (F1-macro, alinhado à métrica do estudo)

Para cada (modelo × dataset × early_stop_metric), tuna INDEPENDENTEMENTE
via Optuna com aquela métrica como critério de early stopping. Isso evita
contaminação: usar hiperparâmetros otimizados para val_acc em treino com
val_loss daria uma comparação injusta.

Modelos: FT-CUR (Nyströmformer), SAINT
Datasets: CREDIT, BANK (os 2 onde houve colapso significativo)

Outputs:
  results/tier2_early_stop_ablation.json
  results/tuning/best_params_early_stop_ablation.json

Usage:
    # Tuning + experimentos (recomendado, ~3-5h em T4)
    python scripts/run_early_stop_ablation.py --seeds 30 --trials 15

    # Skip tuning (usa val_acc-tuned params para todas as métricas — menos rigoroso)
    python scripts/run_early_stop_ablation.py --skip-tuning
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
        logging.FileHandler("results/early_stop_ablation.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

N_TOTAL_CAP = 7143
DATASETS = ["CREDIT", "BANK"]
MODELS = [
    ("FTTransformerCURColnorm", "FTTransformerCURColnorm"),
    ("SAINTColnorm",            "SAINTColnorm"),
]
EARLY_STOP_METRICS = ["val_acc", "val_loss", "val_f1_macro"]


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


# ── Tuning helpers ────────────────────────────────────────────────────────────

def _cv_eval_with_metric(model_factory, X, y, folds, seed):
    """CV evaluation com F1-macro (independente do early_stop_metric usado)."""
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
    return _cv_eval_with_metric(lambda: FTTransformerCURColnorm(**p),
                                 X, y, folds, seed)


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
    return _cv_eval_with_metric(lambda: SAINTColnorm(**p),
                                 X, y, folds, seed)


OBJ_FN = {
    "FTTransformerCURColnorm": _obj_ftcur,
    "SAINTColnorm":            _obj_saint,
}


def _tune(params_file, n_trials, folds, seed_tune):
    """Tuna FT-CUR e SAINT em CREDIT/BANK × 3 métricas (12 combos)."""
    import optuna
    from sklearn.model_selection import StratifiedShuffleSplit
    from src.data.loaders import DatasetLoader

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    existing = json.loads(params_file.read_text()) if params_file.exists() else {}

    for runner_name, variant_name in MODELS:
        for dataset in DATASETS:
            for metric in EARLY_STOP_METRICS:
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
                    }
                    _save_json(params_file, existing)
                    log.info("[OK]   %s  f1_cv=%.4f", key, study.best_value)
                except Exception as e:
                    log.warning("[FAIL] %s — %s", key, e)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed-tune", type=int, default=0)
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Não tunar; reusar params do tier2_n5000_snapshot "
                             "(menos rigoroso mas mais rápido)")
    parser.add_argument("--output-file", type=Path,
                        default=Path("results/tier2_early_stop_ablation.json"))
    parser.add_argument("--params-file", type=Path,
                        default=Path("results/tuning/best_params_early_stop_ablation.json"))
    parser.add_argument("--fallback-params", type=Path,
                        default=Path("results/tuning/best_params_tier2_n5000_snapshot.json"),
                        help="Params de fallback quando --skip-tuning")
    args = parser.parse_args()

    from src.experiments.runner import run_single_experiment

    seeds = list(range(args.seeds))
    log.info("=" * 70)
    log.info("Ablação H2 — early_stop_metric × FT-CUR/SAINT × CREDIT/BANK")
    log.info("=" * 70)

    # ── Fase de tuning ────────────────────────────────────────────────────────
    if not args.skip_tuning:
        log.info("Tuning: %d combos × %d trials × %d folds",
                 len(MODELS) * len(DATASETS) * len(EARLY_STOP_METRICS),
                 args.trials, args.folds)
        _tune(args.params_file, args.trials, args.folds, args.seed_tune)

    # Carrega params (tunados ou fallback)
    if args.params_file.exists():
        raw = json.loads(args.params_file.read_text())
        tuned = {k: v["best_params"] for k, v in raw.items() if "best_params" in v}
        log.info("Params tunados específicos: %d combos", len(tuned))
    else:
        tuned = {}

    fallback = {}
    if args.fallback_params.exists():
        raw = json.loads(args.fallback_params.read_text())
        fallback = {k: v["best_params"] for k, v in raw.items() if "best_params" in v}
        log.info("Fallback params (val_acc tuned): %d combos", len(fallback))

    # ── Experimentos ─────────────────────────────────────────────────────────
    existing = json.loads(args.output_file.read_text()) if args.output_file.exists() else []
    done_keys = set()
    for r in existing:
        key = (r.get("model_variant"), r.get("dataset"),
               r.get("seed"), r.get("early_stop_metric", "val_acc"))
        done_keys.add(key)
    log.info("Resumindo: %d runs já feitos", len(existing))

    total = len(MODELS) * len(DATASETS) * len(seeds) * len(EARLY_STOP_METRICS)
    all_results = list(existing)
    completed = errors = 0
    t_start = time.perf_counter()

    for runner_name, variant_name in MODELS:
        for dataset in DATASETS:
            for metric in EARLY_STOP_METRICS:
                # Resolve params: tunado específico > fallback (val_acc)
                key_tuned = f"{variant_name}__{dataset}__{metric}"
                key_fallback = f"{variant_name}__{dataset}"
                if key_tuned in tuned:
                    params = dict(tuned[key_tuned])
                elif key_fallback in fallback:
                    params = dict(fallback[key_fallback])
                    params["early_stop_metric"] = metric  # override
                else:
                    log.warning("Sem params para %s — pulando", key_tuned)
                    continue

                for seed in seeds:
                    key = (variant_name, dataset, seed, metric)
                    if key in done_keys:
                        continue

                    result = run_single_experiment(
                        model_name=runner_name,
                        dataset_name=dataset,
                        seed=seed,
                        model_params=params,
                        n_samples_cap=N_TOTAL_CAP,
                    )
                    result["model_variant"] = variant_name
                    result["early_stop_metric"] = metric
                    result["n_samples_cap"] = N_TOTAL_CAP

                    all_results.append(result)
                    done_keys.add(key)
                    completed += 1
                    if result["status"] != "ok":
                        errors += 1

                    if completed % 5 == 0:
                        _save_json(args.output_file, all_results)

                    elapsed = time.perf_counter() - t_start
                    eta = (elapsed / completed * (total - completed)) if completed else 0
                    f1 = result.get("f1_macro", float("nan"))
                    log.info("[%d/%d] %s/%s/%s/seed=%d — f1=%.4f | ETA %.0fm",
                             completed, total, variant_name, dataset, metric,
                             seed, f1, eta / 60)

    _save_json(args.output_file, all_results)
    log.info("=" * 70)
    log.info("=== Concluído: %d/%d ok, %d errors ===", completed - errors, total, errors)
    log.info("=" * 70)

    # Análise rápida
    import numpy as np
    scores = defaultdict(list)
    for r in all_results:
        if r.get("status") != "ok": continue
        key = (r["model_variant"], r["dataset"], r.get("early_stop_metric", "val_acc"))
        scores[key].append(r.get("f1_macro", float("nan")))

    print("\n" + "=" * 90)
    print("Resumo F1-macro por (modelo × dataset × early_stop_metric)")
    print("=" * 90)
    print(f'\n{"Modelo":<28} {"Dataset":<10} {"val_acc":>10} {"val_loss":>10} {"val_f1":>10} {"Δ best":>10}')
    print("─" * 90)
    for m, _ in MODELS:
        for ds in DATASETS:
            acc = np.mean(scores.get((m, ds, "val_acc"), [float("nan")]))
            loss = np.mean(scores.get((m, ds, "val_loss"), [float("nan")]))
            f1m = np.mean(scores.get((m, ds, "val_f1_macro"), [float("nan")]))
            best = max(loss, f1m) if not (np.isnan(loss) and np.isnan(f1m)) else float("nan")
            delta = best - acc if not (np.isnan(best) or np.isnan(acc)) else float("nan")
            print(f"{m:<28} {ds:<10} {acc:>10.4f} {loss:>10.4f} {f1m:>10.4f} {delta:>+10.4f}")


if __name__ == "__main__":
    main()

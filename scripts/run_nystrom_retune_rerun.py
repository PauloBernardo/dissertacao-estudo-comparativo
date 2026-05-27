#!/usr/bin/env python3
"""Retune e reexperimento do NystromLSSVMColnorm com implementação corrigida.

O tuning anterior foi feito com a implementação bugada (predição usava kernel
completo em vez dos landmarks). Os params encontrados são subótimos para o
modelo honesto.

Este script:
  1. Retuna NystromLSSVMColnorm em todos os datasets com a implementação corrigida
     → salva em results/tuning/best_params_nystrom_corrected.json
  2. Remove resultados antigos do Nyström dos 4 arquivos de resultado
  3. Reexecuta experimentos com os novos params (Tier 1, Scaling, 5f, MK5)

Flags
-----
  --retune-only : faz só o tuning (não toca nos JSONs de resultado).
                  Usar enquanto run_saint_ftcur_rerun.py ainda está rodando.
  --skip-retune : pula o tuning (usa best_params_nystrom_corrected.json existente)
                  e vai direto para os experimentos.

Usage
-----
  # Fase 1: tuning em paralelo com outro script
  python scripts/run_nystrom_retune_rerun.py --retune-only

  # Fase 2: depois que run_saint_ftcur_rerun.py terminar
  python scripts/run_nystrom_retune_rerun.py --skip-retune
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
        logging.FileHandler("results/nystrom_retune.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Constants ─────────────────────────────────────────────────────────────────

TIER1_DS   = ["BCW", "PID", "HAB", "VCP", "GCR", "AUS", "TWS", "TWM", "TWC"]
SCALING_DS = ["TWS_2k", "TWM_2k", "TWC_2k"]
F5_DS      = ["TWS_5f", "TWM_5f", "TWC_5f"]
MK5_DS     = ["MKE", "MKM", "MKH"]

PARAMS_CORRECTED = Path("results/tuning/best_params_nystrom_corrected.json")
PARAMS_5F        = Path("results/tuning/best_params_nystrom_corrected_5f.json")
PARAMS_MK5       = Path("results/tuning/best_params_nystrom_corrected_mk5.json")

DEFAULT_PARAMS = {"sigma": 1.0, "gamma": 1.0, "m_ratio": 0.20}

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


def _existing_keys(results: list) -> set[str]:
    keys = set()
    for r in results:
        mv = r.get("model_variant") or r.get("model")
        keys.add(f"{mv}__{r.get('dataset')}__{r.get('seed')}")
    return keys


def _resolve_params(dataset: str, tuned: dict) -> dict:
    key = f"NystromLSSVMColnorm__{dataset}"
    stripped = dataset.split("_")[0]
    for k in [key, f"NystromLSSVMColnorm__{stripped}"]:
        if k in tuned:
            return dict(tuned[k])
    log.warning("No tuned params for NystromLSSVMColnorm / %s — using defaults", dataset)
    return dict(DEFAULT_PARAMS)


# ── Tuning ────────────────────────────────────────────────────────────────────

def _tune_nystrom(datasets: list[str], params_file: Path,
                  n_trials: int, folds: int, seed: int) -> None:
    import optuna
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    from src.data.loaders import DatasetLoader
    from src.models.nystrom_lssvm_wrapper import NystromLSSVMColnorm

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    existing = json.loads(params_file.read_text()) if params_file.exists() else {}

    for dataset in datasets:
        key = f"NystromLSSVMColnorm__{dataset}"
        if key in existing:
            bp = existing[key].get("best_params", {})
            bv = existing[key].get("best_value", 0)
            log.info("[SKIP ] %s  m_ratio=%.3f  f1_cv=%.4f", key,
                     bp.get("m_ratio", 0), bv)
            continue

        log.info("[TUNE ] NystromLSSVMColnorm / %s (%d trials)...", dataset, n_trials)
        X, y, _ = DatasetLoader.load(dataset)
        y_signed = (y * 2 - 1).astype(int)

        def objective(trial):
            sigma   = trial.suggest_float("sigma",   0.01, 100.0, log=True)
            gamma   = trial.suggest_float("gamma",   0.01, 1000.0, log=True)
            m_ratio = trial.suggest_float("m_ratio", 0.05, 0.50)
            model = NystromLSSVMColnorm(sigma=sigma, gamma=gamma,
                                        m_ratio=m_ratio, random_state=seed)
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            scores = []
            for tr, val in cv.split(X, y_signed):
                from sklearn.preprocessing import StandardScaler as SS
                sc = SS()
                X_tr = sc.fit_transform(X[tr])
                X_val = sc.transform(X[val])
                try:
                    model.fit(X_tr, y_signed[tr])
                    pred = model.predict(X_val)
                    # Convert signed → binary for f1
                    pb = ((pred + 1) // 2).astype(int)
                    yb = ((y_signed[val] + 1) // 2).astype(int)
                    scores.append(f1_score(yb, pb, average="macro", zero_division=0))
                except Exception:
                    scores.append(0.0)
            return float(sum(scores) / len(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        bp = study.best_params
        existing[key] = {
            "best_params": bp,
            "best_value": study.best_value,
            "metric": "f1_macro_cv_corrected",
            "n_trials": len(study.trials),
        }
        _save_json(params_file, existing)
        log.info("[OK   ] %s  m_ratio=%.3f  sparsity=%.1f%%  f1_cv=%.4f",
                 key, bp["m_ratio"], (1 - bp["m_ratio"]) * 100, study.best_value)


# ── Experiment phase ──────────────────────────────────────────────────────────

def _strip_nystrom(files: list[Path]) -> None:
    log.info("=== Stripping old NystromLSSVMColnorm results ===")
    for path in files:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        before = len(data)
        data = [r for r in data
                if (r.get("model_variant") or r.get("model")) != "NystromLSSVMColnorm"]
        removed = before - len(data)
        _save_json(path, data)
        log.info("  %s — removed %d (%d remain)", path.name, removed, len(data))


def _run_phase(phase_name: str, datasets: list[str], params_file: Path,
               output_file: Path, seeds: list[int], run_single_experiment,
               extra_field: dict | None = None) -> None:
    tuned = _load_params(params_file)

    existing = _load_json(output_file)
    existing_keys = _existing_keys(existing)
    log.info("Resuming %s: %d in %s", phase_name, len(existing), output_file)

    all_results = list(existing)
    total = len(datasets) * len(seeds)
    completed = errors = 0
    t_start = time.perf_counter()

    for dataset in datasets:
        params = _resolve_params(dataset, tuned)
        for seed in seeds:
            run_key = f"NystromLSSVMColnorm__{dataset}__{seed}"
            if run_key in existing_keys:
                continue

            result = run_single_experiment(
                model_name="NystromLSSVMColnorm",
                dataset_name=dataset,
                seed=seed,
                model_params=params,
            )
            result["model_variant"] = "NystromLSSVMColnorm"
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
            spar = result.get("sparsity_ratio", float("nan"))
            log.info("[%s %d/%d] %s / seed=%d — %s  f1=%.4f  spar=%.1f%%  ETA %.0fm",
                     phase_name, completed, total, dataset, seed,
                     result["status"], f1, spar * 100, eta / 60)

    _save_json(output_file, all_results)
    log.info("=== %s done: %d errors / %d runs ===", phase_name, errors, total)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",        type=int,   default=30)
    parser.add_argument("--trials",       type=int,   default=50)
    parser.add_argument("--folds",        type=int,   default=5)
    parser.add_argument("--seed-tune",    type=int,   default=0)
    parser.add_argument("--retune-only",  action="store_true",
                        help="Só faz tuning, não toca nos JSONs de resultado")
    parser.add_argument("--skip-retune",  action="store_true",
                        help="Pula tuning, vai direto para experimentos")
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    log.info("=== NystromLSSVMColnorm Retune + Rerun (implementação corrigida) ===")
    log.info("Seeds: %d | Trials: %d | retune_only=%s | skip_retune=%s",
             args.seeds, args.trials, args.retune_only, args.skip_retune)

    # ── Tuning ────────────────────────────────────────────────────────────────
    if not args.skip_retune:
        log.info("=== Tuning: Tier 1 datasets ===")
        _tune_nystrom(TIER1_DS, PARAMS_CORRECTED, args.trials, args.folds, args.seed_tune)

        log.info("=== Tuning: 5f datasets ===")
        _tune_nystrom(F5_DS, PARAMS_5F, args.trials, args.folds, args.seed_tune)

        log.info("=== Tuning: MK5 datasets ===")
        _tune_nystrom(MK5_DS, PARAMS_MK5, args.trials, args.folds, args.seed_tune)

        # Print summary of new params vs old
        log.info("=== Comparação: params antigos vs corrigidos ===")
        old = {}
        for f in [
            Path("results/tuning/best_params_custom.json"),
            Path("results/tuning/best_params_custom_mk5.json"),
        ]:
            if f.exists():
                raw = json.loads(f.read_text())
                for k, v in raw.items():
                    if "NystromLSSVM" in k and "best_params" in v:
                        old[k] = v
        new = {}
        for f in [PARAMS_CORRECTED, PARAMS_5F, PARAMS_MK5]:
            if f.exists():
                raw = json.loads(f.read_text())
                for k, v in raw.items():
                    new[k] = v

        print("\n{:<45} {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8}".format(
            "Dataset", "σ_old", "γ_old", "m_old", "σ_new", "γ_new", "m_new"))
        print("-" * 90)
        for k in sorted(new.keys()):
            nbp = new[k].get("best_params", {})
            obp = old.get(k, {}).get("best_params", {})
            print("{:<45} {:>8.3f} {:>8.1f} {:>8.3f} | {:>8.3f} {:>8.1f} {:>8.3f}".format(
                k,
                obp.get("sigma", float("nan")), obp.get("gamma", float("nan")),
                obp.get("m_ratio", float("nan")),
                nbp.get("sigma", float("nan")), nbp.get("gamma", float("nan")),
                nbp.get("m_ratio", float("nan")),
            ))

    if args.retune_only:
        log.info("=== --retune-only: tuning concluído. Rode sem a flag para os experimentos. ===")
        return

    # ── Experiment phase ──────────────────────────────────────────────────────
    from src.experiments.runner import run_single_experiment

    _strip_nystrom(RESULT_FILES)

    log.info("=== Tier 1 (9 datasets × %d seeds) ===", args.seeds)
    _run_phase("TIER1", TIER1_DS, PARAMS_CORRECTED,
               Path("results/tier1_custom_models.json"), seeds, run_single_experiment)

    log.info("=== Scaling N=2000 ===")
    _run_phase("SCALING", SCALING_DS, PARAMS_CORRECTED,
               Path("results/synthetic_scaling_n2000.json"), seeds, run_single_experiment)

    log.info("=== 5-features (Tier 1 params) ===")
    _run_phase("5F-FIXED", F5_DS, PARAMS_CORRECTED,
               Path("results/synthetic_5features.json"), seeds, run_single_experiment)

    log.info("=== 5-features (params retuned no 5f) ===")
    _run_phase("5F-RETUNED", F5_DS, PARAMS_5F,
               Path("results/synthetic_5features_tuned.json"), seeds, run_single_experiment)

    log.info("=== MK5 ===")
    _run_phase("MK5", MK5_DS, PARAMS_MK5,
               Path("results/synthetic_mk5.json"), seeds, run_single_experiment,
               extra_field={"n_features_informative": 5})

    # ── Final summary ─────────────────────────────────────────────────────────
    import numpy as np
    from collections import defaultdict

    log.info("=== Tudo concluído ===")
    print("\n" + "=" * 65)
    print("NystromLSSVMColnorm — Tier 1 F1-macro (9 datasets × 30 seeds)")
    print("=" * 65)
    data = _load_json(Path("results/tier1_custom_models.json"))
    scores: dict = defaultdict(list)
    spars: dict = defaultdict(list)
    for r in data:
        mv = r.get("model_variant") or r.get("model")
        if mv == "NystromLSSVMColnorm":
            scores[r.get("dataset", "")].append(r.get("f1_macro", float("nan")))
            spars[r.get("dataset", "")].append(r.get("sparsity_ratio", float("nan")))

    all_f1 = [v for vals in scores.values() for v in vals]
    all_sp  = [v for vals in spars.values()  for v in vals]
    for ds in TIER1_DS:
        f1s = scores.get(ds, [])
        sps = spars.get(ds, [])
        if f1s:
            print(f"  {ds:<8} f1={np.mean(f1s):.4f}  spar={np.mean(sps):.1%}")
    if all_f1:
        print(f"  {'Média':<8} f1={np.mean(all_f1):.4f}  spar={np.mean(all_sp):.1%}")


if __name__ == "__main__":
    main()

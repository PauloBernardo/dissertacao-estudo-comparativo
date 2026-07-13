#!/usr/bin/env python3
"""Ablação — seleção de landmarks do Nyström-SVM: random vs colnorm.

Objetivo
--------
Avaliar se a seleção de landmarks por norma de coluna (``colnorm``, usada
na dissertação via ``NystromLSSVMColnorm``) é de fato melhor que a
amostragem aleatória uniforme (``random``) em acurácia/F1. A rodada
anterior (3 seeds) sugeriu que o método pouco importa; aqui repetimos com
30 seeds e teste estatístico pareado.

Protocolo (idêntico ao Tier 1 de ``run_tier1_gridcv.py``)
--------------------------------------------------------
    1. Hold-out 70/30 estratificado com ``random_state=seed``.
    2. GridSearchCV(Pipeline(StandardScaler → NystromLSSVMRandom),
       cv=StratifiedKFold(5, shuffle, seed), scoring='f1_macro', refit).
    3. refit no treino completo com os melhores params.
    4. Avaliação única no teste 30%.

Grid casado por dataset (reproduz EXATAMENTE o grid que gerou os
resultados de colnorm em ``results/tier1_gridcv.json``):
    - AI4I (adicionado depois): 18 combos, m_ratio fixo 0.30.
    - Demais 9 datasets:        96 combos, m_ratio tunado.

Resumível: rerun pula entradas já gravadas.

Uso
---
    python scripts/run_nystrom_random_ablation.py \
        [--datasets D1 ...] [--seeds 0 1 ...] [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels, make_splits
from src.experiments.reproducibility import set_global_seed
from src.experiments.runner import _build_model
from scripts.run_tier1_gridcv import (
    _collect_sparsity_metrics,
    _compute_test_metrics,
)

# ── Configuração ────────────────────────────────────────────────────────────

# Variante/modelo default (sobrescrevível por --variant). Todos os seletores
# de Nyström (Random/Kmeans/Opposite) compartilham a MESMA grade do colnorm.
VARIANT   = "NystromLSSVMRandom"
MODEL     = "NystromLSSVMRandom"
TIER1_DATASETS = ["BCW", "PID", "HAB", "VCP", "GCR", "AUS", "AI4I", "TWS", "TWM", "TWC"]
DEFAULT_SEEDS  = list(range(30))
OUTPUT_FILE    = Path("results/tier1_nystrom_random.json")

# Grid casado por dataset — reproduz o grid que gerou os resultados de
# NystromLSSVMColnorm em results/tier1_gridcv.json (recuperado do histórico
# do git: commit 61908e7 para os 9 datasets, commit 05cf344 para AI4I).
GRID_96 = {  # AUS BCW GCR HAB PID TWC TWM TWS VCP
    "grid":  {
        "sigma":   [0.1, 0.5, 2.0, 8.0],
        "gamma":   [0.1, 1.0, 10.0, 30.0, 50.0, 100.0],
        "m_ratio": [0.05, 0.10, 0.20, 0.30],
    },
    "fixed": {},
}
GRID_18 = {  # AI4I
    "grid":  {
        "sigma": [0.5, 1.5, 5.0],
        "gamma": [0.1, 1.0, 10.0, 30.0, 50.0, 100.0],
    },
    "fixed": {"m_ratio": 0.30},
}
DATASET_GRID = {ds: (GRID_18 if ds == "AI4I" else GRID_96) for ds in TIER1_DATASETS}

logger = logging.getLogger("nystrom_random")


def _grid_size(cfg: dict) -> int:
    n = 1
    for v in cfg["grid"].values():
        n *= len(v)
    return n


def _run_key(dataset: str, seed: int) -> str:
    return f"{VARIANT}__{dataset}__seed{seed}"


def _existing_keys(records: list[dict]) -> set[str]:
    return {
        _run_key(r["dataset"], r["seed"])
        for r in records
        if r.get("status") == "ok"
    }


def _save(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(records, indent=2, default=str))
    tmp.replace(path)


def _build_pipeline(cfg: dict, seed: int) -> tuple[Pipeline, dict]:
    fixed = dict(cfg["fixed"])
    estimator, _ = _build_model(MODEL, fixed, label_format="signed")
    if hasattr(estimator, "set_params") and "random_state" in estimator.get_params():
        estimator.set_params(random_state=seed)
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    param_grid = {f"clf__{k}": v for k, v in cfg["grid"].items()}
    return pipeline, param_grid


def run_one(dataset: str, seed: int) -> dict[str, Any]:
    cfg = DATASET_GRID[dataset]
    set_global_seed(seed)

    record: dict[str, Any] = {
        "variant":  VARIANT,
        "model":    MODEL,
        "dataset":  dataset,
        "seed":     seed,
        "label_format": "signed",
        "grid_size":    _grid_size(cfg),
    }

    try:
        X, y, meta = DatasetLoader.load(dataset)
        X_train, X_test, y_train_raw, y_test_raw = make_splits(
            X, y, test_size=0.30, seed=seed)
        y_train = _convert_labels(y_train_raw, "signed")
        y_test  = _convert_labels(y_test_raw,  "signed")

        record.update({
            "n_train": int(len(X_train)),
            "n_test":  int(len(X_test)),
            "n_features": int(X.shape[1]),
            "dataset_tier": meta.get("tier"),
        })

        pipeline, param_grid = _build_pipeline(cfg, seed)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        search = GridSearchCV(
            pipeline, param_grid=param_grid, cv=cv, scoring="f1_macro",
            refit=True, n_jobs=-1, error_score=0.0, return_train_score=False,
        )

        t0 = time.perf_counter()
        search.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        test_metrics = _compute_test_metrics(
            search.best_estimator_, X_test, y_test, "signed")
        predict_time = time.perf_counter() - t0

        clf = search.best_estimator_.named_steps["clf"]
        sparsity = _collect_sparsity_metrics(clf, X_test)
        best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

        record.update({
            "status":        "ok",
            "best_params":   best_params,
            "cv_score_f1_macro": float(search.best_score_),
            "fit_time_s":    fit_time,
            "predict_time_s": predict_time,
            **test_metrics,
            **sparsity,
        })
    except Exception as exc:
        record.update({
            "status":    "error",
            "error":     f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })

    return record


def main() -> int:
    global VARIANT, MODEL
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variant", default=VARIANT,
                   choices=["NystromLSSVMRandom", "NystromLSSVMKmeans",
                            "NystromLSSVMOpposite", "NystromLSSVMColnorm"],
                   help="Seletor de landmarks (grade casada por dataset).")
    p.add_argument("--datasets", nargs="+", default=TIER1_DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--output", type=Path, default=OUTPUT_FILE)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    VARIANT = MODEL = args.variant

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logger.setLevel(args.log_level)

    records = json.loads(args.output.read_text()) if args.output.exists() else []
    done = _existing_keys(records)
    plan = [(d, s) for d in args.datasets for s in args.seeds]
    total = len(plan)
    logger.info("Resuming: %d/%d entries already complete.", len(done), total)

    interrupted = {"flag": False}

    def _on_signal(signum, _frame):
        logger.warning("Signal %d — exiting after current entry.", signum)
        interrupted["flag"] = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    t_start = time.perf_counter()
    for i, (dataset, seed) in enumerate(plan, start=1):
        if interrupted["flag"]:
            break
        key = _run_key(dataset, seed)
        if key in done:
            continue

        logger.info("[%d/%d] %s  seed=%d  (grid=%d)",
                    i, total, dataset, seed, _grid_size(DATASET_GRID[dataset]))
        rec = run_one(dataset, seed)
        records.append(rec)
        _save(args.output, records)

        if rec["status"] == "ok":
            logger.info("    F1=%.4f  acc=%.4f  sparsity=%.3f  %.1fs",
                        rec["test_f1_macro"], rec["test_accuracy"],
                        rec.get("sparsity_ratio", float("nan")), rec["fit_time_s"])
        else:
            logger.warning("    ERROR: %s", rec["error"])

    elapsed = time.perf_counter() - t_start
    ok = sum(1 for r in records if r.get("status") == "ok")
    logger.info("Done. %d ok / %d total. %.1fs elapsed.", ok, len(records), elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

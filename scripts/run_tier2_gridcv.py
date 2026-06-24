#!/usr/bin/env python3
"""Tier 2 — Hold-out + GridSearchCV, datasets grandes com N fixo.

Protocolo
---------
Mesmo pipeline do Tier 1 (run_tier1_gridcv.py), com subsampling
estratificado e uniforme para TODOS os modelos:
  - N_TRAIN = 2 000  →  cap total = 2 857  (split 70/30)

O subsampling é determinístico por seed, garantindo variância amostral
entre runs sem vazamento de teste.

18 modelos × 6 datasets × 20 seeds = 2 160 runs.

Uso
---
    python scripts/run_tier2_gridcv.py [--n-train 2000]
                                       [--models M1 M2 ...]
                                       [--datasets D1 D2 ...]
                                       [--seeds 0..19]
                                       [--output results/tier2_gridcv.json]
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

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, StratifiedShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels, make_splits
from src.experiments.reproducibility import set_global_seed
from src.experiments.runner import _build_model
from src.metrics.sparsity import transformer_sparsity
from src.tuning.grids import GRIDS, grid_size

# ── Configuração ──────────────────────────────────────────────────────────────

TIER2_DATASETS = ["ADULT", "BANK", "CREDIT", "HIGGS50K", "SHOPPERS", "TELCO"]

DEFAULT_VARIANTS = [
    # Baselines LSSVM clássicos (viáveis a N=2000, < 3 min/run)
    "StandardLSSVM", "DualFISTA",
    "PCPLSSVm", "FSALSSVm", "IPLSSVm",
    # Contribuições escaláveis (Nyström — foco do Tier 2)
    "NystromLSSVMColnorm", "ADMMNystromLSSVM", "FISTANystrom",
    # Baseline ML
    "XGBoost",
    # Transformers (rodados separadamente no Kaggle)
    "FTTransformer_softmax", "FTTransformer_topk",
    "FTTransformer_entmax", "FTTransformer_sparsemax",
    "FTTransformerCURColnorm", "SAINTColnorm",
]

DEFAULT_SEEDS  = list(range(20))
DEFAULT_N_TRAIN = 2000

OUTPUT_FILE = Path("results/tier2_gridcv.json")

# Usado apenas para determinar o formato de label (signed vs binary)
LSSVM_VARIANTS = {
    "StandardLSSVM", "DualFISTA",
    "PCPLSSVm", "FSALSSVm", "IPLSSVm",
    "NystromLSSVMColnorm", "ADMMNystromLSSVM", "FISTANystrom",
}

logger = logging.getLogger("tier2")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_key(variant: str, dataset: str, seed: int) -> str:
    return f"{variant}__{dataset}__seed{seed}"


def _existing_keys(records: list[dict]) -> set[str]:
    return {
        _run_key(r["variant"], r["dataset"], r["seed"])
        for r in records
        if r.get("status") == "ok"
    }


def _save(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(records, indent=2, default=str))
    tmp.replace(path)


def _label_format(variant: str) -> str:
    return "signed" if variant in LSSVM_VARIANTS else "binary"


def _subsample(X: np.ndarray, y: np.ndarray, n_total_cap: int,
               seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Subsampling estratificado determinístico por seed."""
    if len(X) <= n_total_cap:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_total_cap,
                                  random_state=seed)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]


def _build_pipeline(variant: str, seed: int) -> tuple[Pipeline, dict]:
    cfg   = GRIDS[variant]
    fixed = dict(cfg["fixed"])
    estimator, _ = _build_model(cfg["model_name"], fixed, label_format="signed")
    if hasattr(estimator, "set_params") and "random_state" in estimator.get_params():
        estimator.set_params(random_state=seed)
    pipeline   = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    param_grid = {f"clf__{k}": v for k, v in cfg["grid"].items()}
    return pipeline, param_grid


def _compute_test_metrics(estimator, X_test, y_test, label_format: str) -> dict:
    y_pred = estimator.predict(X_test)
    if label_format == "signed":
        y_test_e = ((y_test + 1) // 2).astype(int)
        y_pred_e = ((y_pred + 1) // 2).astype(int)
    else:
        y_test_e, y_pred_e = y_test, y_pred

    metrics = {
        "test_f1_macro":  float(f1_score(y_test_e, y_pred_e, average="macro", zero_division=0)),
        "test_f1_binary": float(f1_score(y_test_e, y_pred_e, average="binary", zero_division=0)),
        "test_accuracy":  float(accuracy_score(y_test_e, y_pred_e)),
        "test_mcc":       float(matthews_corrcoef(y_test_e, y_pred_e)),
    }
    if hasattr(estimator, "predict_proba"):
        try:
            proba = estimator.predict_proba(X_test)
            score = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
            metrics["test_auc_roc"] = float(roc_auc_score(y_test_e, score))
        except Exception:
            metrics["test_auc_roc"] = None
    elif hasattr(estimator, "decision_function"):
        try:
            metrics["test_auc_roc"] = float(
                roc_auc_score(y_test_e, estimator.decision_function(X_test)))
        except Exception:
            metrics["test_auc_roc"] = None
    else:
        metrics["test_auc_roc"] = None
    return metrics


def _collect_sparsity(estimator, X_test) -> dict:
    metrics: dict = {}
    if hasattr(estimator, "n_support_"):
        try:
            metrics["n_support_vectors"] = int(estimator.n_support_)
        except Exception:
            pass
    if hasattr(estimator, "sparsity_ratio_"):
        try:
            metrics["sparsity_ratio"] = float(estimator.sparsity_ratio_)
        except Exception:
            pass
    if hasattr(estimator, "attention_sparsity"):
        try:
            estimator.predict_proba(X_test[: min(32, len(X_test))])
            metrics.update(transformer_sparsity(estimator))
        except Exception:
            pass
    return metrics


# ── Experimento único ─────────────────────────────────────────────────────────

def run_one(variant: str, dataset: str, seed: int, n_train: int) -> dict[str, Any]:
    cfg          = GRIDS[variant]
    label_format = _label_format(variant)
    n_total_cap  = round(n_train / 0.70)
    n_jobs_cv    = 1 if cfg["needs_gpu"] else -1

    set_global_seed(seed)

    record: dict[str, Any] = {
        "variant":        variant,
        "model":          cfg["model_name"],
        "dataset":        dataset,
        "seed":           seed,
        "label_format":   label_format,
        "grid_size":      grid_size(variant),
        "n_train_target": n_train,
    }

    try:
        X_full, y_full, meta = DatasetLoader.load(dataset)
        X_sub, y_sub = _subsample(X_full, y_full, n_total_cap, seed)

        X_train, X_test, y_train_raw, y_test_raw = make_splits(
            X_sub, y_sub, test_size=0.30, seed=seed)

        y_train = _convert_labels(y_train_raw, label_format)
        y_test  = _convert_labels(y_test_raw,  label_format)

        record.update({
            "n_total_sub":  int(len(X_sub)),
            "n_train":      int(len(X_train)),
            "n_test":       int(len(X_test)),
            "n_features":   int(X_full.shape[1]),
            "dataset_tier": meta.get("tier"),
        })

        pipeline, param_grid = _build_pipeline(variant, seed)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        search = GridSearchCV(
            pipeline, param_grid=param_grid, cv=cv,
            scoring="f1_macro", refit=True,
            n_jobs=n_jobs_cv, error_score=0.0,
            return_train_score=False,
        )

        t0 = time.perf_counter()
        search.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        test_metrics = _compute_test_metrics(
            search.best_estimator_, X_test, y_test, label_format)
        predict_time = time.perf_counter() - t0

        clf      = search.best_estimator_.named_steps["clf"]
        sparsity = _collect_sparsity(clf, X_test)

        best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

        record.update({
            "status":            "ok",
            "best_params":       best_params,
            "cv_score_f1_macro": float(search.best_score_),
            "fit_time_s":        fit_time,
            "predict_time_s":    predict_time,
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


# ── Orquestrador ──────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-train",  type=int, default=DEFAULT_N_TRAIN,
                   help="N de treino alvo para TODOS os modelos (default: 2000)")
    p.add_argument("--models",   nargs="+", default=DEFAULT_VARIANTS)
    p.add_argument("--datasets", nargs="+", default=TIER2_DATASETS)
    p.add_argument("--seeds",    nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--output",   type=Path, default=OUTPUT_FILE)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("tier2").setLevel(args.log_level)

    records      = json.loads(args.output.read_text()) if args.output.exists() else []
    done         = _existing_keys(records)
    total_planned = len(args.models) * len(args.datasets) * len(args.seeds)
    logger.info("Resumindo: %d/%d entradas já completas.", len(done), total_planned)
    logger.info("Protocolo: N_train=%d, cap_total=%d", args.n_train, round(args.n_train / 0.70))

    plan = [(m, d, s) for m in args.models for d in args.datasets for s in args.seeds]

    interrupted = {"flag": False}
    def _on_signal(signum, _frame):
        logger.warning("Sinal %d — finalizando após entrada atual.", signum)
        interrupted["flag"] = True
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT,  _on_signal)

    t_start = time.perf_counter()
    for i, (variant, dataset, seed) in enumerate(plan, start=1):
        if interrupted["flag"]:
            logger.warning("Interrompido; %d restantes (resumível).", len(plan) - i + 1)
            break

        key = _run_key(variant, dataset, seed)
        if key in done:
            logger.debug("[%d/%d] SKIP %s", i, len(plan), key)
            continue

        logger.info("[%d/%d] %s  grid=%d", i, len(plan), key, grid_size(variant))

        t0  = time.perf_counter()
        rec = run_one(variant, dataset, seed, args.n_train)
        wall = time.perf_counter() - t0
        rec["wall_time_s"] = wall

        records.append(rec)
        _save(args.output, records)

        if rec.get("status") == "ok":
            logger.info(
                "    OK  cv=%.4f  test=%.4f  fit=%.1fs  n_train=%d  wall=%.1fs",
                rec["cv_score_f1_macro"], rec["test_f1_macro"],
                rec["fit_time_s"], rec["n_train"], wall,
            )
        else:
            logger.warning("    FAIL  %s", rec.get("error", "?")[:120])

    elapsed = time.perf_counter() - t_start
    n_ok = sum(1 for r in records if r.get("status") == "ok")
    logger.info("=== Concluído. %.1f min. OK=%d/%d. %s ===",
                elapsed / 60, n_ok, total_planned, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

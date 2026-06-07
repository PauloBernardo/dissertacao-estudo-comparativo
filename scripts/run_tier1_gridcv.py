#!/usr/bin/env python3
"""Tier 1 — Hold-out + Grid Search + 5-fold CV protocol.

Protocol (per Marinho et al. supervisor's spec):
    1. Hold-out 70/30 stratified split with ``random_state=seed`` on the
       full dataset. The test set is locked.
    2. ``GridSearchCV(estimator, grid, cv=StratifiedKFold(5, shuffle, seed),
       scoring='f1_macro', refit=True)`` on the 70% train.
       ``StandardScaler`` is fit inside each CV fold via a sklearn
       Pipeline, so no leakage from val into scaler.
    3. ``refit=True`` re-trains the estimator on the FULL train with the
       best params (scaler fit on full train as well).
    4. Single evaluation on the 30% test → final reported metric.

For each (model, dataset, seed) one record is written to the output JSON.
The script is resumable: rerun skips already-recorded entries.

Usage
-----
    python scripts/run_tier1_gridcv.py [--models M1 M2 ...]
                                       [--datasets D1 D2 ...]
                                       [--seeds 0 1 2 ...]
                                       [--output FILE]
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
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels, make_splits
from src.experiments.reproducibility import set_global_seed
from src.experiments.runner import _build_model
from src.metrics.sparsity import transformer_sparsity
from src.tuning.grids import GRIDS, grid_size


# ── Configuration ───────────────────────────────────────────────────────────

TIER1_DATASETS = ["BCW", "PID", "HAB", "VCP", "GCR", "AUS", "TWS", "TWM", "TWC"]

DEFAULT_VARIANTS = list(GRIDS.keys())

DEFAULT_SEEDS = list(range(30))

OUTPUT_FILE = Path("results/tier1_gridcv.json")

LSSVM_VARIANTS = {
    "StandardLSSVM", "PCPLSSVm", "FSALSSVm", "IPLSSVm",
    "PruningLSSVM", "OppositeMapsLSSVM",
    "ADMMNesterovLSSVM", "ADMMElasticNet",
    "FISTANesterov", "DualFISTA", "NystromLSSVMColnorm",
}

logger = logging.getLogger("tier1")


# ── Helpers ─────────────────────────────────────────────────────────────────

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


def _build_pipeline(variant: str, seed: int) -> tuple[Pipeline, dict]:
    """Build (Pipeline, param_grid) ready for GridSearchCV.

    Returns
    -------
    pipeline : Pipeline of StandardScaler → estimator with fixed params set.
    param_grid : dict with keys prefixed by ``clf__`` for sklearn.
    """
    cfg = GRIDS[variant]
    fixed = dict(cfg["fixed"])

    # Sklearn-style random_state injection where supported.
    estimator, _ = _build_model(cfg["model_name"], fixed, label_format="signed")
    if hasattr(estimator, "set_params") and "random_state" in estimator.get_params():
        estimator.set_params(random_state=seed)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    estimator),
    ])
    param_grid = {f"clf__{k}": v for k, v in cfg["grid"].items()}
    return pipeline, param_grid


def _compute_test_metrics(
    estimator, X_test, y_test, label_format: str,
) -> dict[str, float]:
    """Compute test-set F1-macro, accuracy, MCC, AUC.

    Maps LSSVM signed predictions to {0,1} for sklearn metrics that
    expect non-negative labels (e.g. matthews_corrcoef).
    """
    y_pred = estimator.predict(X_test)

    if label_format == "signed":
        y_test_eval = ((y_test + 1) // 2).astype(int)
        y_pred_eval = ((y_pred + 1) // 2).astype(int)
    else:
        y_test_eval = y_test
        y_pred_eval = y_pred

    metrics = {
        "test_f1_macro":  float(f1_score(y_test_eval, y_pred_eval, average="macro", zero_division=0)),
        "test_f1_binary": float(f1_score(y_test_eval, y_pred_eval, average="binary", zero_division=0)),
        "test_accuracy":  float(accuracy_score(y_test_eval, y_pred_eval)),
        "test_mcc":       float(matthews_corrcoef(y_test_eval, y_pred_eval)),
    }

    # AUC requires probability or decision function.
    proba = None
    if hasattr(estimator, "predict_proba"):
        try:
            proba = estimator.predict_proba(X_test)
            score = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
            metrics["test_auc_roc"] = float(roc_auc_score(y_test_eval, score))
        except Exception:
            metrics["test_auc_roc"] = None
    elif hasattr(estimator, "decision_function"):
        try:
            score = estimator.decision_function(X_test)
            metrics["test_auc_roc"] = float(roc_auc_score(y_test_eval, score))
        except Exception:
            metrics["test_auc_roc"] = None
    else:
        metrics["test_auc_roc"] = None

    return metrics


def _collect_sparsity_metrics(estimator, X_test) -> dict[str, float]:
    """Collect model-specific sparsity metrics from the fitted pipeline clf."""
    metrics: dict[str, float] = {}

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

    # FTTransformer attention sparsity is distinct from support-vector sparsity:
    # it measures zeros / entropy in attention weights, not sample selection.
    if hasattr(estimator, "attention_sparsity"):
        try:
            estimator.predict_proba(X_test[: min(32, len(X_test))])
            metrics.update(transformer_sparsity(estimator))
        except Exception:
            pass

    return metrics


# ── Single (variant, dataset, seed) run ─────────────────────────────────────

def run_one(variant: str, dataset: str, seed: int) -> dict[str, Any]:
    cfg = GRIDS[variant]
    label_format = _label_format(variant)
    n_jobs = 1 if cfg["needs_gpu"] else -1

    set_global_seed(seed)

    record: dict[str, Any] = {
        "variant":  variant,
        "model":    cfg["model_name"],
        "dataset":  dataset,
        "seed":     seed,
        "label_format": label_format,
        "grid_size":    grid_size(variant),
    }

    try:
        # 1. Load and hold-out 70/30
        X, y, meta = DatasetLoader.load(dataset)
        X_train, X_test, y_train_raw, y_test_raw = make_splits(
            X, y, test_size=0.30, seed=seed)

        # 2. Convert labels (scaler is handled inside the Pipeline).
        y_train = _convert_labels(y_train_raw, label_format)
        y_test  = _convert_labels(y_test_raw,  label_format)

        record.update({
            "n_train": int(len(X_train)),
            "n_test":  int(len(X_test)),
            "n_features": int(X.shape[1]),
            "dataset_tier": meta.get("tier"),
        })

        # 3. GridSearchCV: 5-fold StratifiedKFold inside the Pipeline.
        pipeline, param_grid = _build_pipeline(variant, seed)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="f1_macro",
            refit=True,
            n_jobs=n_jobs,
            error_score=0.0,
            return_train_score=False,
        )

        t0 = time.perf_counter()
        search.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        # 4. Single evaluation on test.
        t0 = time.perf_counter()
        test_metrics = _compute_test_metrics(
            search.best_estimator_, X_test, y_test, label_format)
        predict_time = time.perf_counter() - t0

        # 5. Collect support-vector or attention sparsity, depending on model.
        clf = search.best_estimator_.named_steps["clf"]
        sparsity = _collect_sparsity_metrics(clf, X_test)

        # 6. Strip the clf__ prefix from best params for readability.
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
            "status":     "error",
            "error":      f"{type(exc).__name__}: {exc}",
            "traceback":  traceback.format_exc(),
        })

    return record


# ── Orchestrator ────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", nargs="+", default=DEFAULT_VARIANTS,
                   help="Variant names (keys of GRIDS).")
    p.add_argument("--datasets", nargs="+", default=TIER1_DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--output", type=Path, default=OUTPUT_FILE)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("tier1").setLevel(args.log_level)

    # Resume support.
    records = json.loads(args.output.read_text()) if args.output.exists() else []
    done = _existing_keys(records)
    logger.info("Resuming: %d/%d entries already complete.",
                len(done), len(args.models) * len(args.datasets) * len(args.seeds))

    plan = [(m, d, s) for m in args.models for d in args.datasets for s in args.seeds]
    total = len(plan)

    # Graceful shutdown: on SIGTERM/SIGINT, finish saving the current
    # record (already protected by atomic _save) and exit cleanly. This is
    # the relevant path for Colab session timeouts and Ctrl-C.
    interrupted = {"flag": False}

    def _on_signal(signum, _frame):
        logger.warning("Received signal %d — will exit after current entry.",
                       signum)
        interrupted["flag"] = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT,  _on_signal)

    t_start = time.perf_counter()
    for i, (variant, dataset, seed) in enumerate(plan, start=1):
        if interrupted["flag"]:
            logger.warning("Interrupted; %d entries left (resumable).",
                           total - i + 1)
            break

        key = _run_key(variant, dataset, seed)
        if key in done:
            logger.debug("[%d/%d] SKIP %s", i, total, key)
            continue

        gsz = grid_size(variant)
        logger.info("[%d/%d] %s (grid=%d)", i, total, key, gsz)

        t0 = time.perf_counter()
        rec = run_one(variant, dataset, seed)
        wall = time.perf_counter() - t0
        rec["wall_time_s"] = wall

        records.append(rec)
        _save(args.output, records)

        if rec.get("status") == "ok":
            logger.info(
                "    OK  cv=%.4f  test=%.4f  fit=%.1fs  wall=%.1fs",
                rec["cv_score_f1_macro"], rec["test_f1_macro"],
                rec["fit_time_s"], wall,
            )
        else:
            logger.warning("    FAIL  %s", rec.get("error", "?")[:120])

    elapsed = time.perf_counter() - t_start
    logger.info("=== Done. Elapsed %.1f min. Output: %s ===",
                elapsed / 60, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

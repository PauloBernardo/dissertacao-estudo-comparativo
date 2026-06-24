#!/usr/bin/env python3
"""ADMM-Nyström Mode B — identity check + speedup benchmark.

Part 1 — Identity check
    Run ADMMNystromDistributed (n_blocks=4) on 5 datasets × 10 seeds.
    Compare F1-macro vs Mode A results already in tier1_admm_nystrom.json.
    Pass criterion: |ΔF1| < 1e-6 for all runs.

Part 2 — Speedup benchmark
    Generate a synthetic N=5000 binary dataset.
    Fit Mode A (n_blocks=1) and Mode B (n_blocks=4) with the same params.
    Report CᵀC wall time and total fit-time speedup.

Output
    results/admm_nystrom_distributed.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels, make_splits
from src.experiments.reproducibility import set_global_seed
from src.models.lssvm.primal.admm_nystrom import ADMMNystromLSSVM
from src.tuning.grids import GRIDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("admm_distrib")

OUTPUT = ROOT / "results" / "admm_nystrom_distributed.json"

CHECK_DATASETS = ["AUS", "BCW", "PID", "TWS", "TWC"]
CHECK_SEEDS    = list(range(10))


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_search(n_blocks: int, n_jobs_model: int, seed: int) -> tuple[GridSearchCV, dict]:
    cfg_a = GRIDS["ADMMNystromLSSVM"]
    fixed = dict(cfg_a["fixed"]) | {"n_blocks": n_blocks, "n_jobs": n_jobs_model}
    estimator = ADMMNystromLSSVM(**fixed)
    pipeline  = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    param_grid = {f"clf__{k}": v for k, v in cfg_a["grid"].items()}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1_macro",
                        refit=True, n_jobs=-1, error_score=0.0), param_grid


def run_one_mode_b(dataset: str, seed: int) -> dict:
    set_global_seed(seed)
    X, y, _ = DatasetLoader.load(dataset)
    X_train, X_test, y_train_raw, y_test_raw = make_splits(X, y, test_size=0.30, seed=seed)
    y_train = _convert_labels(y_train_raw, "signed")
    y_test  = _convert_labels(y_test_raw, "signed")

    search, _ = _build_search(n_blocks=4, n_jobs_model=4, seed=seed)
    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    from sklearn.metrics import f1_score
    y_pred = search.best_estimator_.predict(X_test)
    y_test_e = ((y_test + 1) // 2).astype(int)
    y_pred_e = ((y_pred + 1) // 2).astype(int)
    f1 = float(f1_score(y_test_e, y_pred_e, average="macro", zero_division=0))

    return {
        "dataset": dataset, "seed": seed, "mode": "B",
        "test_f1_macro": f1, "fit_time_s": fit_time,
        "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
    }


# ── Part 1: identity check ────────────────────────────────────────────────────

def identity_check(mode_a_records: list[dict]) -> list[dict]:
    log.info("=== Part 1: Identity check (%d datasets × %d seeds) ===",
             len(CHECK_DATASETS), len(CHECK_SEEDS))

    # Index Mode A results
    a_idx = {
        (r["dataset"], r["seed"]): r["test_f1_macro"]
        for r in mode_a_records
        if r.get("status") == "ok"
    }

    results = []
    total = len(CHECK_DATASETS) * len(CHECK_SEEDS)
    for i, dataset in enumerate(CHECK_DATASETS):
        for seed in CHECK_SEEDS:
            idx = i * len(CHECK_SEEDS) + seed + 1
            log.info("[%d/%d] %s seed=%d", idx, total, dataset, seed)
            rec = run_one_mode_b(dataset, seed)
            f1_a = a_idx.get((dataset, seed), float("nan"))
            rec["f1_mode_a"] = f1_a
            rec["delta_f1"]  = abs(rec["test_f1_macro"] - f1_a)
            results.append(rec)
            log.info("  F1_B=%.6f  F1_A=%.6f  |Δ|=%.2e",
                     rec["test_f1_macro"], f1_a, rec["delta_f1"])

    # Summary
    deltas = [r["delta_f1"] for r in results]
    log.info("Identity check: max|ΔF1|=%.2e  mean|ΔF1|=%.2e",
             max(deltas), sum(deltas) / len(deltas))
    passed = max(deltas) < 1e-4
    log.info("PASS=%s (threshold 1e-4)", passed)
    return results


# ── Part 2: speedup benchmark ─────────────────────────────────────────────────

def speedup_benchmark() -> dict:
    log.info("=== Part 2: Speedup benchmark (N=5000) ===")
    rng = np.random.default_rng(42)
    X, y = make_classification(
        n_samples=5000, n_features=20, n_informative=10,
        n_redundant=5, random_state=42,
    )
    y_signed = (y * 2 - 1).astype(float)

    # Use median best_params from Mode A tier1 results as fixed params
    # (sigma=0.5, tau=0.05, lambda_=0.1 — reasonable defaults from grids)
    fixed_params = {"sigma": 0.5, "tau": 0.05, "lambda_": 0.1,
                    "m_ratio": 0.10, "rho": None, "max_iter": 500}

    n_repeats = 5
    times_a, times_b = [], []
    ctc_a, ctc_b = [], []

    for rep in range(n_repeats):
        # Mode A
        model_a = ADMMNystromLSSVM(**fixed_params, n_blocks=1, n_jobs=1)
        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X)
        t0 = time.perf_counter()
        model_a.fit(X_sc, y_signed)
        times_a.append(time.perf_counter() - t0)
        ctc_a.append(model_a.ctc_wall_time_)
        log.info("  Mode A rep %d: fit=%.3fs  CᵀC=%.3fs", rep + 1, times_a[-1], ctc_a[-1])

        # Mode B
        model_b = ADMMNystromLSSVM(**fixed_params, n_blocks=4, n_jobs=4)
        t0 = time.perf_counter()
        model_b.fit(X_sc, y_signed)
        times_b.append(time.perf_counter() - t0)
        ctc_b.append(model_b.ctc_wall_time_)
        log.info("  Mode B rep %d: fit=%.3fs  CᵀC=%.3fs", rep + 1, times_b[-1], ctc_b[-1])

    result = {
        "n_samples": 5000,
        "n_features": 20,
        "n_repeats": n_repeats,
        "mode_a_fit_mean_s":  float(np.mean(times_a)),
        "mode_a_fit_std_s":   float(np.std(times_a)),
        "mode_b_fit_mean_s":  float(np.mean(times_b)),
        "mode_b_fit_std_s":   float(np.std(times_b)),
        "mode_a_ctc_mean_s":  float(np.mean(ctc_a)),
        "mode_b_ctc_mean_s":  float(np.mean(ctc_b)),
        "speedup_fit":        float(np.mean(times_a) / np.mean(times_b)),
        "speedup_ctc":        float(np.mean(ctc_a)   / np.mean(ctc_b)),
    }
    log.info("Fit speedup B/A:  %.2fx", result["speedup_fit"])
    log.info("CᵀC speedup B/A: %.2fx", result["speedup_ctc"])
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    mode_a = json.loads((ROOT / "results" / "tier1_admm_nystrom.json").read_text())

    identity_results = identity_check(mode_a)
    speedup_result   = speedup_benchmark()

    output = {
        "identity_check": identity_results,
        "speedup_benchmark": speedup_result,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, default=str))
    log.info("Saved to %s", OUTPUT)

    # Print final summary
    deltas = [r["delta_f1"] for r in identity_results]
    print("\n=== SUMMARY ===")
    print(f"Identity check: {len(identity_results)} runs | max|ΔF1|={max(deltas):.2e} | "
          f"PASS={max(deltas) < 1e-4}")
    print(f"Speedup fit:   {speedup_result['speedup_fit']:.2f}x")
    print(f"Speedup CᵀC:  {speedup_result['speedup_ctc']:.2f}x")


if __name__ == "__main__":
    main()

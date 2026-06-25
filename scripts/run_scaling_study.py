#!/usr/bin/env python3
"""Scaling study: best_params do Tier 2 × N crescente.

Extrai automaticamente os melhores hiperparâmetros do Tier 2
(tier2_gridcv.json) e mede tempo, RAM e F1 para N crescente.
Inclui todos os modelos — O(N²/N³) e O(N·m²) — para mostrar o colapso.

Uso:
    python scripts/run_scaling_study.py
    python scripts/run_scaling_study.py --dataset ADULT
    python scripts/run_scaling_study.py --models StandardLSSVM NystromLSSVMColnorm
    python scripts/run_scaling_study.py --n-values 1000 5000 10000 20000
    python scripts/run_scaling_study.py --timeout 1800
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels
from src.experiments.reproducibility import set_global_seed
from src.experiments.runner import _build_model

logger = logging.getLogger("scaling_study")

# ── Configuração ──────────────────────────────────────────────────────────────

DEFAULT_N_VALUES  = [1_000, 2_000, 5_000, 10_000, 20_000]
DEFAULT_SEEDS     = 5
DEFAULT_TIMEOUT_S = 3_600   # 1 h; acima disso registra "timeout"
DEFAULT_DATASET   = "ADULT"
DEFAULT_OUTPUT    = Path("results/scaling_study.json")
TIER2_JSON        = Path("results/tier2_gridcv.json")

LSSVM_VARIANTS = {
    "StandardLSSVM", "DualFISTA",
    "PCPLSSVm", "FSALSSVm", "IPLSSVm",
    "NystromLSSVMColnorm", "ADMMNystromLSSVM", "FISTANystrom",
}

DEFAULT_VARIANTS = [
    "StandardLSSVM", "DualFISTA",
    "PCPLSSVm", "FSALSSVm", "IPLSSVm",
    "NystromLSSVMColnorm", "ADMMNystromLSSVM", "FISTANystrom",
    "XGBoost",
]

# Fallback params se modelo ainda não estiver no Tier 2
FALLBACK_PARAMS: dict[str, dict] = {
    "StandardLSSVM":       {"sigma": 5.0,  "tau": 2.5},
    "DualFISTA":           {"sigma": 0.5,  "tau": 0.1, "lambda_": 0.01},
    "PCPLSSVm":            {"sigma": 5.0,  "tau": 0.1, "rank": 100},
    "FSALSSVm":            {"sigma": 0.5,  "tau": 0.1, "n_components": 100},
    "IPLSSVm":             {"sigma": 0.5,  "tau": 0.1, "selection_ratio": 0.2},
    "NystromLSSVMColnorm": {"sigma": 0.5,  "gamma": 10.0, "m_ratio": 0.30},
    "ADMMNystromLSSVM":    {"sigma": 0.5,  "tau": 0.05, "lambda_": 0.01,
                            "m_ratio": 0.30, "rho": None, "max_iter": 500,
                            "landmark_method": "colnorm", "n_blocks": 1, "n_jobs": 1},
    "FISTANystrom":        {"sigma": 0.5,  "tau": 0.05, "lambda_": 0.01,
                            "m_ratio": 0.30, "landmark_method": "colnorm",
                            "max_iter": 5000},
    "XGBoost":             {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
}

# ── Extração de best_params do Tier 2 ────────────────────────────────────────

def _extract_best_params(tier2_path: Path) -> dict[str, dict]:
    """Para cada variante no Tier 2, retorna o modo de cada hiperparâmetro."""
    if not tier2_path.exists():
        logger.warning("Tier 2 JSON não encontrado — usando fallback params.")
        return {}

    records = json.loads(tier2_path.read_text())
    ok = [r for r in records if r.get("status") == "ok"]
    if not ok:
        return {}

    by_variant: dict[str, list[dict]] = {}
    for r in ok:
        by_variant.setdefault(r["variant"], []).append(r["best_params"])

    result: dict[str, dict] = {}
    for variant, param_list in by_variant.items():
        all_keys = {k for p in param_list for k in p}
        mode_params: dict[str, Any] = {}
        for k in all_keys:
            vals = [p[k] for p in param_list if k in p]
            # mode: para floats/ints, usa Counter
            try:
                mode_params[k] = Counter(vals).most_common(1)[0][0]
            except Exception:
                mode_params[k] = vals[0]
        result[variant] = mode_params
        logger.info("  %s: %s (n=%d runs)", variant, mode_params, len(param_list))

    return result


# ── Subsampling ───────────────────────────────────────────────────────────────

def _subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int):
    if len(X) <= n:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]


# ── Monitoramento de RAM ──────────────────────────────────────────────────────

_PROC = psutil.Process(os.getpid())

def _monitor_rss(stop: threading.Event, peak_mb: list) -> None:
    while not stop.is_set():
        try:
            rss = _PROC.memory_info().rss / 1024 / 1024
            if rss > peak_mb[0]:
                peak_mb[0] = rss
        except psutil.NoSuchProcess:
            break
        time.sleep(0.01)


# ── Worker (rodado em processo separado para isolamento de memória/timeout) ───

def _worker(
    result_queue: mp.Queue,
    variant: str,
    params: dict,
    X_train: np.ndarray,
    y_train_raw: np.ndarray,
    X_test: np.ndarray,
    y_test_raw: np.ndarray,
    seed: int,
) -> None:
    try:
        label_fmt = "signed" if variant in LSSVM_VARIANTS else "binary"
        y_train = _convert_labels(y_train_raw, label_fmt)
        y_test  = _convert_labels(y_test_raw,  label_fmt)

        from src.tuning.grids import GRIDS
        cfg = GRIDS[variant]
        # Merge: fixed do grid + best_params do Tier 2
        fixed = dict(cfg.get("fixed", {}))
        merged = {**fixed, **params}

        estimator, _ = _build_model(cfg["model_name"], merged, label_format=label_fmt)
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])

        baseline_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        peak_mb = [baseline_mb]
        stop = threading.Event()
        mon = threading.Thread(target=_monitor_rss, args=(stop, peak_mb), daemon=True)
        mon.start()

        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        stop.set(); mon.join()
        delta_mb = max(peak_mb[0] - baseline_mb, 0.0)

        t0 = time.perf_counter()
        y_pred = pipeline.predict(X_test)
        predict_time = time.perf_counter() - t0

        if label_fmt == "signed":
            y_test_e = ((y_test + 1) // 2).astype(int)
            y_pred_e = ((y_pred + 1) // 2).astype(int)
        else:
            y_test_e, y_pred_e = y_test, y_pred

        f1 = float(f1_score(y_test_e, y_pred_e, average="macro", zero_division=0))

        # Sparsity
        clf = pipeline.named_steps["clf"]
        sparsity = float(clf.sparsity_ratio_) if hasattr(clf, "sparsity_ratio_") else None
        n_support = int(clf.n_support_) if hasattr(clf, "n_support_") else None

        result_queue.put({
            "status":        "ok",
            "fit_time_s":    fit_time,
            "predict_time_s": predict_time,
            "ram_delta_mb":  delta_mb,
            "test_f1_macro": f1,
            "sparsity_ratio": sparsity,
            "n_support":     n_support,
            "params_used":   params,
        })
    except Exception as exc:
        result_queue.put({
            "status":    "error",
            "error":     f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


def _run_with_timeout(
    variant: str,
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    timeout_s: int,
) -> dict:
    q: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_worker,
        args=(q, variant, params, X_train, y_train, X_test, y_test, seed),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
        return {"status": "timeout", "timeout_s": timeout_s}

    if not q.empty():
        return q.get()
    return {"status": "error", "error": "worker died without result"}


# ── Orquestrador ──────────────────────────────────────────────────────────────

def _run_key(variant: str, n: int, seed: int) -> str:
    return f"{variant}__N{n}__seed{seed}"


def _save(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(records, indent=2, default=str))
    tmp.replace(path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset",   default=DEFAULT_DATASET)
    p.add_argument("--n-values",  nargs="+", type=int, default=DEFAULT_N_VALUES)
    p.add_argument("--seeds",     type=int, default=DEFAULT_SEEDS,
                   help="Número de seeds (0..seeds-1)")
    p.add_argument("--timeout",   type=int, default=DEFAULT_TIMEOUT_S,
                   help="Timeout em segundos por run (default: 3600)")
    p.add_argument("--models",    nargs="+", default=DEFAULT_VARIANTS)
    p.add_argument("--tier2-json", type=Path, default=TIER2_JSON)
    p.add_argument("--output",    type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("scaling_study").setLevel(args.log_level)

    # Carrega best_params do Tier 2
    logger.info("Extraindo best_params de %s …", args.tier2_json)
    tier2_params = _extract_best_params(args.tier2_json)

    # Carrega dataset completo
    logger.info("Carregando dataset %s …", args.dataset)
    X_full, y_full, _ = DatasetLoader.load(args.dataset)
    logger.info("Dataset: N=%d  features=%d", len(X_full), X_full.shape[1])

    # Resume
    records: list[dict] = (
        json.loads(args.output.read_text()) if args.output.exists() else []
    )
    done = {
        _run_key(r["variant"], r["n"], r["seed"])
        for r in records
        if r.get("status") in ("ok", "timeout")
    }

    seeds = list(range(args.seeds))
    plan = [(m, n, s) for m in args.models for n in args.n_values for s in seeds]
    total = len(plan)
    logger.info("Plano: %d runs  (%d modelos × %d N-values × %d seeds)",
                total, len(args.models), len(args.n_values), len(seeds))
    logger.info("Resumindo: %d/%d já completos.", len(done), total)

    interrupted = {"flag": False}
    def _on_signal(signum, _frame):
        logger.warning("Sinal %d — finalizando após run atual.", signum)
        interrupted["flag"] = True
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT,  _on_signal)

    for i, (variant, n, seed) in enumerate(plan, 1):
        if interrupted["flag"]:
            break

        key = _run_key(variant, n, seed)
        if key in done:
            logger.debug("[%d/%d] SKIP %s", i, total, key)
            continue

        # Params: Tier 2 > fallback
        params = tier2_params.get(variant, FALLBACK_PARAMS.get(variant, {}))

        # Subsample com split 70/30
        n_total = round(n / 0.70)
        set_global_seed(seed)
        X_sub, y_sub = _subsample(X_full, y_full, n_total, seed)
        split_idx = int(len(X_sub) * 0.70)
        X_train, X_test = X_sub[:split_idx], X_sub[split_idx:]
        y_train, y_test = y_sub[:split_idx], y_sub[split_idx:]

        logger.info("[%d/%d] %s  N=%d  seed=%d  timeout=%ds",
                    i, total, variant, n, seed, args.timeout)

        t0   = time.perf_counter()
        rec  = _run_with_timeout(
            variant, params, X_train, y_train, X_test, y_test, seed, args.timeout
        )
        wall = time.perf_counter() - t0

        rec.update({
            "variant": variant,
            "dataset": args.dataset,
            "n":       n,
            "n_train": len(X_train),
            "n_test":  len(X_test),
            "seed":    seed,
            "wall_time_s": wall,
        })
        records.append(rec)
        _save(args.output, records)

        if rec["status"] == "ok":
            logger.info(
                "    OK  fit=%.1fs  RAM=+%.0fMB  F1=%.4f",
                rec["fit_time_s"], rec["ram_delta_mb"], rec["test_f1_macro"],
            )
        elif rec["status"] == "timeout":
            logger.warning("    TIMEOUT (>%ds)", args.timeout)
        else:
            logger.warning("    FAIL  %s", str(rec.get("error", "?"))[:100])

    n_ok      = sum(1 for r in records if r.get("status") == "ok")
    n_timeout = sum(1 for r in records if r.get("status") == "timeout")
    logger.info("=== Concluído. OK=%d  TIMEOUT=%d  %s ===",
                n_ok, n_timeout, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

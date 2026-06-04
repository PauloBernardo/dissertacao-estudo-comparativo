#!/usr/bin/env python3
"""LSSVM efficiency benchmark across N_train in {1k, 2k, 5k, 7k, 10k}.

Goal
----
Characterize fit_time, predict_time, peak_ram, sparsity and (sanity) F1 as
a function of N_train for the 11 LSSVM variants, reusing the best
hyperparameters already tuned at N_train=5000 (Tier 2, post-bugfix
b83243e). NO tuning is performed.

Protocol
--------
* For each (model, dataset, N_train), one stratified subsample of size
  N_total = N_train / 0.7 is drawn deterministically per seed, then split
  70/30. Same convention as the Tier 2 N=5000 protocol.
* Single seed by default (seed=0): we are measuring computational cost,
  not F1 statistics.
* Each run is executed in a fresh subprocess so that peak RSS is
  attributable to that single fit/predict cycle.

Outputs
-------
results/benchmark_lssvm_scaling.json (list of dicts, resumable).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Heavy imports (numpy, sklearn, src.*) happen in worker mode only.

# (model_name, variant_name_in_best_params)
LSSVM_MODELS: list[tuple[str, str]] = [
    ("StandardLSSVM",            "StandardLSSVM"),
    ("PCPLSSVm",                 "PCPLSSVm"),
    ("FSALSSVm",                 "FSALSSVm"),
    ("IPLSSVm",                  "IPLSSVm"),
    ("PruningLSSVM",             "PruningLSSVM"),
    ("OppositeMapsLSSVM",        "OppositeMapsLSSVM"),
    ("FISTANesterovLSSVM",       "FISTANesterov"),
    ("ADMMNesterovLSSVM",        "ADMMNesterovLSSVM"),     # paper-base
    ("ADMMNesterovLSSVM",        "ADMMElasticNet"),        # variant: same class, lambda2_>0
    ("DualFISTALSSVM",           "DualFISTA"),
    ("NystromLSSVMColnorm",      "NystromLSSVMColnorm"),
]

DEFAULT_DATASETS = ["ADULT", "CREDIT", "HIGGS50K"]
DEFAULT_N_TRAIN  = [1000, 2000, 5000, 7000, 10000]
DEFAULT_SEED     = 0

PARAMS_FILE = Path("results/tuning/best_params_tier2_n5000_cpu.json")
OUTPUT_FILE = Path("results/benchmark_lssvm_scaling.json")


# ── helpers ─────────────────────────────────────────────────────────────────

def _peak_ram_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _load_params(variant: str, dataset: str) -> dict:
    if not PARAMS_FILE.exists():
        return {}
    with PARAMS_FILE.open() as f:
        data = json.load(f)
    key = f"{variant}__{dataset}"
    return dict(data.get(key, {}).get("best_params", {}))


def _n_total_for(n_train: int, test_size: float = 0.30) -> int:
    return int(round(n_train / (1.0 - test_size)))


def _run_key(variant: str, dataset: str, n_train: int, seed: int) -> str:
    return f"{variant}__{dataset}__N{n_train}__seed{seed}"


# ── worker (single fit/predict in a fresh subprocess) ───────────────────────

def _worker(model_name: str, variant: str, dataset: str, n_train: int, seed: int) -> dict:
    """Run one experiment, return a result dict (printed as JSON to stdout)."""
    from src.experiments.runner import run_single_experiment

    gc.collect()
    params = _load_params(variant, dataset)
    n_total = _n_total_for(n_train)

    t0 = time.perf_counter()
    try:
        res = run_single_experiment(
            model_name=model_name,
            dataset_name=dataset,
            seed=seed,
            model_params=params,
            test_size=0.30,
            n_samples_cap=n_total,
        )
        total = time.perf_counter() - t0
        peak_ram = _peak_ram_mb()
        out = {
            "model": model_name,
            "variant": variant,
            "dataset": dataset,
            "n_train_target": n_train,
            "n_total_cap": n_total,
            "seed": seed,
            "params": params,
            "total_time_s": total,
            "peak_ram_mb": peak_ram,
            "status": res.get("status", "ok"),
            "fit_time_s": res.get("train_time_s"),
            "predict_time_s": res.get("predict_time_s"),
            "f1_macro": res.get("f1_macro"),
            "accuracy": res.get("accuracy"),
            "n_train": res.get("n_train"),
            "n_test": res.get("n_test"),
            "n_features": res.get("n_features"),
            "sparsity_ratio": res.get("sparsity_ratio"),
            "n_support_vectors": res.get("n_support_vectors"),
        }
        if res.get("status") == "error":
            out["error"] = res.get("error_message")
            out["traceback"] = res.get("traceback")
    except Exception as e:
        total = time.perf_counter() - t0
        out = {
            "model": model_name,
            "variant": variant,
            "dataset": dataset,
            "n_train_target": n_train,
            "n_total_cap": n_total,
            "seed": seed,
            "params": params,
            "total_time_s": total,
            "peak_ram_mb": _peak_ram_mb(),
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
    return out


def _spawn_worker(model_name: str, variant: str, dataset: str, n_train: int,
                  seed: int, timeout_s: int) -> dict:
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker",
        "--model", model_name,
        "--variant", variant,
        "--dataset", dataset,
        "--n-train", str(n_train),
        "--seed", str(seed),
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            check=False, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {
            "model": model_name, "variant": variant, "dataset": dataset,
            "n_train_target": n_train, "seed": seed,
            "status": "timeout",
            "total_time_s": time.perf_counter() - t0,
            "error": f"worker exceeded {timeout_s}s",
        }

    if proc.returncode != 0:
        return {
            "model": model_name, "variant": variant, "dataset": dataset,
            "n_train_target": n_train, "seed": seed,
            "status": "worker_error",
            "total_time_s": time.perf_counter() - t0,
            "returncode": proc.returncode,
            "stderr": proc.stderr.strip()[-2000:],
            "stdout_tail": proc.stdout.strip()[-2000:],
        }

    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        return {
            "model": model_name, "variant": variant, "dataset": dataset,
            "n_train_target": n_train, "seed": seed,
            "status": "parse_error",
            "error": str(e),
            "stdout_tail": proc.stdout.strip()[-2000:],
            "stderr_tail": proc.stderr.strip()[-2000:],
        }


# ── orchestrator ────────────────────────────────────────────────────────────

def _existing_keys(results: list[dict]) -> set[str]:
    keys = set()
    for r in results:
        if r.get("status") in ("ok",):  # only skip successes; retry errors/timeouts
            v = r.get("variant") or r.get("model")
            keys.add(_run_key(v, r["dataset"], r["n_train_target"], r["seed"]))
    return keys


def _save_jsonl(path: Path, data: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _main_orchestrator(args) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("benchmark")

    datasets = args.datasets
    n_trains = args.n_trains
    seed = args.seed

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(OUTPUT_FILE.read_text()) if OUTPUT_FILE.exists() else []
    done = _existing_keys(results)

    plan = [(m, v, d, n) for (m, v) in LSSVM_MODELS
                          for d in datasets
                          for n in n_trains]
    total = len(plan)
    log.info("Planned %d runs (%d models x %d datasets x %d sizes); %d already done.",
             total, len(LSSVM_MODELS), len(datasets), len(n_trains), len(done))

    t_start = time.perf_counter()
    for i, (model_name, variant, dataset, n_train) in enumerate(plan, start=1):
        key = _run_key(variant, dataset, n_train, seed)
        if key in done:
            log.info("[%d/%d] SKIP %s (already done)", i, total, key)
            continue

        # Per-N timeout heuristic: 10 min at 5k, scale with n^3
        per_n = max(600, int(600 * (n_train / 5000) ** 3))
        timeout_s = min(per_n, args.max_timeout)

        log.info("[%d/%d] %s (timeout=%ds)", i, total, key, timeout_s)
        t0 = time.perf_counter()
        out = _spawn_worker(model_name, variant, dataset, n_train, seed, timeout_s)
        wall = time.perf_counter() - t0
        out.setdefault("wall_time_s", wall)
        results.append(out)
        _save_jsonl(OUTPUT_FILE, results)

        status = out.get("status", "?")
        f1 = out.get("f1_macro")
        fit_s = out.get("fit_time_s")
        peak = out.get("peak_ram_mb")
        log.info("    -> status=%s  wall=%.1fs  fit=%s  f1=%s  peak_ram=%s MB",
                 status,
                 wall,
                 f"{fit_s:.1f}s" if isinstance(fit_s, (int, float)) else "-",
                 f"{f1:.3f}" if isinstance(f1, (int, float)) else "-",
                 f"{peak:.0f}" if isinstance(peak, (int, float)) else "-")

    elapsed = time.perf_counter() - t_start
    log.info("=== Done. Elapsed %.1f min. Output: %s ===",
             elapsed / 60, OUTPUT_FILE)
    return 0


# ── entry point ─────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true",
                   help="Internal: run as worker (single experiment).")
    p.add_argument("--model", type=str)
    p.add_argument("--variant", type=str)
    p.add_argument("--dataset", type=str)
    p.add_argument("--n-train", type=int)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--n-trains", nargs="+", type=int, default=DEFAULT_N_TRAIN)
    p.add_argument("--max-timeout", type=int, default=3600,
                   help="Hard cap on per-worker timeout in seconds.")
    args = p.parse_args()

    if args.worker:
        out = _worker(args.model, args.variant, args.dataset, args.n_train, args.seed)
        # Last stdout line is parsed by the orchestrator.
        print(json.dumps(out))
        return 0

    return _main_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())

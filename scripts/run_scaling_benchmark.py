#!/usr/bin/env python3
"""Benchmark de escalabilidade: tempo de fit × N e pico de RAM × N.

Compara 5 modelos com hiperparâmetros fixos em dados sintéticos:
    - StandardLSSVM       O(N³) — full kernel
    - NystromLSSVMColnorm O(N·m²) — Nyström sem L1
    - ADMMNesterovLSSVM   O(N³) — ADMM full kernel
    - ADMMNystromLSSVM    O(N·m²) — ADMM no espaço Nyström
    - FISTANystromLSSVM   O(N·m²) — FISTA no espaço Nyström

Varia N = [500, 1000, 2000, 5000, 10000, 20000].
Para StandardLSSVM e ADMMNesterovLSSVM, para antes de explodir memória.

Uso:
    python scripts/run_scaling_benchmark.py [--output FILE] [--repeats N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("benchmark")

# ── Configuração ─────────────────────────────────────────────────────────────

N_VALUES = [500, 1000, 2000, 5000, 10000, 20000]

# N máximo para modelos O(N²/N³) — acima disso memória/tempo explodem
N_MAX_FULL = 5000

_PROC = psutil.Process(os.getpid())


def _make_clf(variant: str, seed: int):
    """Instancia o classificador com hiperparâmetros fixos."""
    if variant == "StandardLSSVM":
        from src.models.lssvm.standard import StandardLSSVM
        return StandardLSSVM(sigma=0.5, tau=2.5)
    elif variant == "ADMMNesterovLSSVM":
        from src.models.lssvm.primal.admm_nesterov import ADMMNesterovLSSVM
        return ADMMNesterovLSSVM(sigma=0.5, tau=0.5, lambda_=0.01, rho=None, max_iter=500)
    elif variant == "NystromLSSVMColnorm":
        from src.models.nystrom_lssvm_wrapper import NystromLSSVMColnorm
        return NystromLSSVMColnorm(sigma=0.5, gamma=10.0, m_ratio=0.30)
    elif variant == "ADMMNystromLSSVM":
        from src.models.lssvm.primal.admm_nystrom import ADMMNystromLSSVM
        return ADMMNystromLSSVM(sigma=0.5, tau=0.5, lambda_=0.01,
                                m_ratio=0.30, landmark_method="colnorm",
                                rho=None, max_iter=500, random_state=seed)
    elif variant == "FISTANystromLSSVM":
        from src.models.lssvm.primal.fista_nystrom import FISTANystromLSSVM
        return FISTANystromLSSVM(sigma=0.5, tau=0.5, lambda_=0.01,
                                 m_ratio=0.30, landmark_method="colnorm",
                                 max_iter=5000, random_state=seed)
    raise ValueError(variant)


MODELS = {
    "StandardLSSVM":       {"max_n": N_MAX_FULL, "label": "LSSVM padrão  O(N³)"},
    "ADMMNesterovLSSVM":   {"max_n": N_MAX_FULL, "label": "ADMM-Nesterov  O(N³)"},
    "NystromLSSVMColnorm": {"max_n": None,        "label": "Nyström-Colnorm  O(N·m²)"},
    "ADMMNystromLSSVM":    {"max_n": None,        "label": "ADMM-Nyström  O(N·m²)"},
    "FISTANystromLSSVM":   {"max_n": None,        "label": "FISTA-Nyström  O(N·m²)"},
}


def make_data(n: int, seed: int) -> tuple:
    X, y = make_classification(
        n_samples=n + 200,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=seed,
    )
    return X[:n], X[n:], y[:n], y[n:]


def _monitor_rss(stop_event: threading.Event, peak_mb: list) -> None:
    """Thread auxiliar: amostra RSS a cada 10 ms e registra o pico."""
    while not stop_event.is_set():
        try:
            rss = _PROC.memory_info().rss / 1024 / 1024  # MB
            if rss > peak_mb[0]:
                peak_mb[0] = rss
        except psutil.NoSuchProcess:
            break
        time.sleep(0.01)


def fit_measure(variant: str, X_tr, y_tr, seed: int) -> tuple[float, float]:
    """Retorna (tempo_fit_s, pico_ram_mb) durante o fit."""
    clf = _make_clf(variant, seed)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    # Baseline RSS antes do fit (dados já em memória)
    baseline_mb = _PROC.memory_info().rss / 1024 / 1024
    peak_mb = [baseline_mb]
    stop = threading.Event()
    mon = threading.Thread(target=_monitor_rss, args=(stop, peak_mb), daemon=True)
    mon.start()

    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0

    stop.set()
    mon.join()

    delta_mb = peak_mb[0] - baseline_mb
    return elapsed, max(delta_mb, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/scaling_benchmark.json")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repetições por (modelo, N) para mediana estável")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for variant, cfg in MODELS.items():
        log.info("=== %s ===", variant)
        for n in N_VALUES:
            if cfg["max_n"] and n > cfg["max_n"]:
                log.info("  N=%6d — pulando (N_max=%d)", n, cfg["max_n"])
                results.append({
                    "variant": variant, "label": cfg["label"],
                    "n": n, "skipped": True,
                    "fit_time_median": None, "ram_peak_mb_median": None,
                })
                continue

            times, rams = [], []
            for rep in range(args.repeats):
                X_tr, X_te, y_tr, y_te = make_data(n, seed=rep)
                try:
                    t, ram = fit_measure(variant, X_tr, y_tr, seed=rep)
                    times.append(t)
                    rams.append(ram)
                except Exception as e:
                    log.warning("  N=%d rep=%d ERRO: %s", n, rep, e)
                    break

            if not times:
                results.append({
                    "variant": variant, "label": cfg["label"],
                    "n": n, "skipped": True,
                    "fit_time_median": None, "ram_peak_mb_median": None,
                })
                continue

            t_med = float(np.median(times))
            r_med = float(np.median(rams))
            log.info("  N=%6d  tempo=%.2fs  RAM=%.1f MB  (%d reps)",
                     n, t_med, r_med, len(times))
            results.append({
                "variant": variant, "label": cfg["label"],
                "n": n, "skipped": False,
                "fit_time_median": t_med,
                "fit_time_all": [round(t, 4) for t in times],
                "ram_peak_mb_median": r_med,
                "ram_peak_mb_all": [round(r, 2) for r in rams],
            })

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Salvo em %s", out_path)

    # ── Tabela resumo ──────────────────────────────────────────────────────
    print()
    print(f"{'Modelo':<22}  {'N':>6}  {'Tempo':>8}  {'RAM':>10}")
    print("-" * 54)
    for r in results:
        t = f"{r['fit_time_median']:.2f}s" if r['fit_time_median'] else "—"
        m = f"{r['ram_peak_mb_median']:.1f} MB" if r['ram_peak_mb_median'] else "—"
        sk = "  (skip)" if r['skipped'] else ""
        print(f"{r['variant']:<22}  {r['n']:>6}  {t:>8}  {m:>10}{sk}")


if __name__ == "__main__":
    main()

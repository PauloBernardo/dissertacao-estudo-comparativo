#!/usr/bin/env python3
"""Benchmark de escalabilidade para Transformers: tempo de treino, predição e memória × N.

Modelos (hiperparâmetros fixos — moda do GridCV Tier 2):
    - FTTransformer_softmax   (num_blocks=2, num_heads=2)
    - SAINTColnorm            (n_heads=2, n_layers=1)
    - FTTransformerCURColnorm (m_ratio=0.2, n_heads=4, n_layers=2)

Métricas por N:
    - fit_time_s      — tempo total de treino (40 épocas ou patience=6)
    - pred_time_ms    — tempo de predição sobre N_TEST=500 amostras fixas
    - ram_delta_mb    — pico de RAM adicional (psutil RSS, thread de monitoramento)
    - vram_mb         — pico de VRAM (torch.cuda.max_memory_allocated)

Uso:
    python scripts/run_transformer_scaling.py [--output FILE] [--device cuda|cpu]
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
import torch
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("transformer_bench")

# ── Configuração ──────────────────────────────────────────────────────────────

N_VALUES  = [500, 1000, 2000, 5000, 10000, 20000]
N_TEST    = 500   # subconjunto fixo para medir predição isolada
REPEATS   = 3     # medianas sobre 3 repetições
EPOCHS    = 40
PATIENCE  = 6

# N máximo por modelo — atenção densa O(N²) explode para N grande
N_MAX = {
    "FTTransformer_softmax":   10000,
    "SAINTColnorm":            5000,   # atenção inter-instâncias densa
    "FTTransformerCURColnorm": 20000,  # Nyströmformer O(N·m)
}

_PROC = psutil.Process(os.getpid())


# ── Instanciação dos modelos ──────────────────────────────────────────────────

def _make_model(variant: str):
    if variant == "FTTransformer_softmax":
        from src.models.transformers.ft_transformer import FTTransformer
        return FTTransformer(
            num_blocks=2, num_heads=2,
            max_epochs=EPOCHS, patience=PATIENCE,
            attention_type="softmax",
        )
    elif variant == "SAINTColnorm":
        from src.models.ft_transformer_saint_wrapper import SAINTColnorm
        return SAINTColnorm(
            n_heads=2, n_layers=1,
            epochs=EPOCHS, patience=PATIENCE, early_stop_metric="val_loss",
        )
    elif variant == "FTTransformerCURColnorm":
        from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm
        return FTTransformerCURColnorm(
            n_heads=4, n_layers=2, m_ratio=0.2,
            epochs=EPOCHS, patience=PATIENCE, early_stop_metric="val_loss",
        )
    raise ValueError(variant)


# ── Monitoramento de memória ──────────────────────────────────────────────────

def _monitor_rss(stop: threading.Event, peak_mb: list) -> None:
    while not stop.is_set():
        try:
            rss = _PROC.memory_info().rss / 1024 / 1024
            if rss > peak_mb[0]:
                peak_mb[0] = rss
        except psutil.NoSuchProcess:
            break
        time.sleep(0.01)


def _measure(variant: str, X_tr, y_tr, X_te) -> dict:
    """Retorna métricas de fit e predição para um (variant, N)."""
    model = _make_model(variant)

    # Reset VRAM counter
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    baseline_mb = _PROC.memory_info().rss / 1024 / 1024
    peak_mb = [baseline_mb]
    stop = threading.Event()
    mon = threading.Thread(target=_monitor_rss, args=(stop, peak_mb), daemon=True)
    mon.start()

    # ── Treino ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0

    stop.set()
    mon.join()

    ram_delta = max(peak_mb[0] - baseline_mb, 0.0)
    vram_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024
               if torch.cuda.is_available() else 0.0)

    # ── Predição ──────────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    model.predict(X_te)
    pred_ms = (time.perf_counter() - t1) * 1000

    return {
        "fit_s": round(fit_s, 3),
        "pred_ms": round(pred_ms, 3),
        "ram_delta_mb": round(ram_delta, 1),
        "vram_mb": round(vram_mb, 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/transformer_scaling.json")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    results: list[dict] = []

    variants = ["FTTransformer_softmax", "SAINTColnorm", "FTTransformerCURColnorm"]

    for variant in variants:
        log.info("=== %s ===", variant)
        max_n = N_MAX[variant]

        for n in N_VALUES:
            if n > max_n:
                log.info("  N=%6d — pulando (N_max=%d)", n, max_n)
                results.append({"variant": variant, "n": n, "skipped": True})
                continue

            fits, preds, rams, vrams = [], [], [], []

            for rep in range(args.repeats):
                try:
                    rng = np.random.default_rng(rep)
                    X_all, y_all = make_classification(
                        n_samples=n + N_TEST,
                        n_features=20, n_informative=10, n_redundant=5,
                        random_state=rep,
                    )
                    X_tr_raw, X_te_raw = X_all[:n], X_all[n:]
                    y_tr = y_all[:n]

                    X_tr = scaler.fit_transform(X_tr_raw)
                    X_te = scaler.transform(X_te_raw)

                    m = _measure(variant, X_tr, y_tr, X_te)
                    fits.append(m["fit_s"])
                    preds.append(m["pred_ms"])
                    rams.append(m["ram_delta_mb"])
                    vrams.append(m["vram_mb"])

                    log.info("  N=%6d rep=%d  fit=%.1fs  pred=%.1fms  "
                             "RAM=%.1fMB  VRAM=%.1fMB",
                             n, rep, m["fit_s"], m["pred_ms"],
                             m["ram_delta_mb"], m["vram_mb"])

                except Exception as e:
                    log.warning("  N=%6d rep=%d ERRO: %s", n, rep, e)
                    break

            if not fits:
                results.append({"variant": variant, "n": n, "skipped": True,
                                 "error": str(e)})
                continue

            results.append({
                "variant": variant, "n": n, "skipped": False,
                "fit_s_median":      round(float(np.median(fits)), 3),
                "pred_ms_median":    round(float(np.median(preds)), 3),
                "ram_delta_mb_median": round(float(np.median(rams)), 1),
                "vram_mb_median":    round(float(np.median(vrams)), 1),
                "fit_s_all":    [round(x, 3) for x in fits],
                "pred_ms_all":  [round(x, 3) for x in preds],
                "ram_mb_all":   [round(x, 1) for x in rams],
                "vram_mb_all":  [round(x, 1) for x in vrams],
            })

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Salvo em %s", out_path)

    # ── Tabela resumo ──────────────────────────────────────────────────────────
    print()
    print(f"{'Variant':<26}  {'N':>6}  {'fit':>7}  {'pred':>8}  {'RAM':>8}  {'VRAM':>8}")
    print("-" * 72)
    for r in results:
        if r.get("skipped"):
            print(f"{r['variant']:<26}  {r['n']:>6}  {'—':>7}  {'—':>8}  {'—':>8}  {'—':>8}")
        else:
            print(f"{r['variant']:<26}  {r['n']:>6}  "
                  f"{r['fit_s_median']:>6.1f}s  "
                  f"{r['pred_ms_median']:>7.1f}ms  "
                  f"{r['ram_delta_mb_median']:>6.0f}MB  "
                  f"{r['vram_mb_median']:>6.0f}MB")


if __name__ == "__main__":
    main()

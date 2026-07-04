#!/usr/bin/env python3
"""Benchmark de escalabilidade para Transformers: tempo de treino, predição e memória × N.

Modelos (hiperparâmetros fixos — moda do GridCV Tier 2):
    - FTTransformer_softmax   (num_blocks=2, num_heads=2)
    - SAINTColnorm            (n_heads=2, n_layers=1)
    - FTTransformerCURColnorm (m_ratio=0.2, n_heads=4, n_layers=2)

Métricas por N (treino e predição medidos SEPARADAMENTE):
    - fit_s_median        — tempo total de treino (40 épocas ou patience=6)
    - ram_delta_mb_median — pico de RAM adicional no TREINO (psutil RSS)
    - vram_mb_median      — pico de VRAM no TREINO (torch.cuda.max_memory_allocated)
    - pred_ms_median      — tempo de predição sobre N_TEST=500 amostras fixas
    - pred_ram_mb_median  — pico de RAM adicional na PREDIÇÃO
    - pred_vram_mb_median — pico de VRAM na PREDIÇÃO
    - pred_oom            — True se a inferência estourou a memória (SAINT O(N²))

Modelos transdutivos (SAINT/FT-CUR) carregam o contexto de treino na inferência,
então a predição tem custo/memória próprios: SAINT é O(N²) (pode dar OOM mesmo com
treino mini-batch), FT-CUR é O(N·m) com m fixo. Por isso a predição é medida à parte.

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

N_VALUES  = [500, 1000, 2000, 5000, 10000, 20000, 50000]
N_TEST    = 500   # subconjunto fixo para medir predição isolada
REPEATS   = 3     # medianas sobre 3 repetições
EPOCHS    = 40
PATIENCE  = 6

# Todos os modelos tentam até N=50K — OOM é capturado e registrado como erro.
# Duas variantes inter-instâncias por modelo, para comparação JUSTA:
#   *_minibatch  : batch_size=256 → atenção limitada ao batch (memória constante)
#   *_full/mfixed: batch completo → SAINT O(N²) (OOM); FT-CUR O(N·m) com m FIXO (escala)
BATCH = 256
M_FIXED = 256          # landmarks fixos p/ FT-CUR full-batch → O(N·m) sub-quadrático
M_MINIBATCH = 64       # landmarks p/ FT-CUR mini-batch

N_MAX = {
    "FTTransformer_softmax":   50000,
    "FTTransformer_topk":      50000,
    "FTTransformer_entmax":    50000,
    "FTTransformer_sparsemax": 50000,
    "SAINT_minibatch":         50000,
    "SAINT_fullbatch":         50000,
    "FTCUR_minibatch":         50000,
    "FTCUR_mfixed_full":       50000,
}

_PROC = psutil.Process(os.getpid())


# ── Instanciação dos modelos ──────────────────────────────────────────────────

def _make_model(variant: str):
    if variant in ("FTTransformer_softmax", "FTTransformer_topk",
                   "FTTransformer_entmax", "FTTransformer_sparsemax"):
        from src.models.transformers.ft_transformer import FTTransformer
        attn_map = {
            "FTTransformer_softmax":   "softmax",
            "FTTransformer_topk":      "topk",
            "FTTransformer_entmax":    "entmax",
            "FTTransformer_sparsemax": "sparsemax",
        }
        return FTTransformer(
            num_blocks=2, num_heads=2,
            max_epochs=EPOCHS, patience=PATIENCE,
            attention_type=attn_map[variant],
        )
    elif variant in ("SAINT_minibatch", "SAINT_fullbatch"):
        from src.models.ft_transformer_saint_wrapper import SAINTColnorm
        # mini-batch: atenção B×B constante | full-batch: atenção N×N → O(N²)
        bs = BATCH if variant == "SAINT_minibatch" else None
        return SAINTColnorm(
            n_heads=2, n_layers=1,
            epochs=EPOCHS, patience=PATIENCE, early_stop_metric="val_loss",
            batch_size=bs,
        )
    elif variant in ("FTCUR_minibatch", "FTCUR_mfixed_full"):
        from src.models.ft_transformer_cur_wrapper import FTTransformerCURColnorm
        if variant == "FTCUR_minibatch":
            # mini-batch (como o SAINT); m fixo pequeno → memória de treino limitada
            return FTTransformerCURColnorm(
                n_heads=4, n_layers=2, m_landmarks=M_MINIBATCH, batch_size=BATCH,
                epochs=EPOCHS, patience=PATIENCE, early_stop_metric="val_loss",
            )
        else:
            # batch completo, m FIXO → atenção O(N·m) genuinamente sub-quadrática
            return FTTransformerCURColnorm(
                n_heads=4, n_layers=2, m_landmarks=M_FIXED, batch_size=None,
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


def _run_phase(fn):
    """Executa fn() medindo tempo (s), pico de RAM adicional (MB) e pico de VRAM (MB).

    Reseta o contador de VRAM antes da fase, então cada fase (treino/predição)
    mede seu próprio pico isoladamente.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    baseline_mb = _PROC.memory_info().rss / 1024 / 1024
    peak_mb = [baseline_mb]
    stop = threading.Event()
    mon = threading.Thread(target=_monitor_rss, args=(stop, peak_mb), daemon=True)
    mon.start()

    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0

    stop.set()
    mon.join()

    ram_delta = max(peak_mb[0] - baseline_mb, 0.0)
    vram_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024
               if torch.cuda.is_available() else 0.0)
    return elapsed, ram_delta, vram_mb


def _measure(variant: str, X_tr, y_tr, X_te) -> dict:
    """Retorna métricas de fit e predição para um (variant, N).

    Treino e predição são medidos SEPARADAMENTE (tempo + RAM + VRAM). Isso é
    essencial para modelos transdutivos (SAINT/FT-CUR): a inferência carrega o
    contexto de treino e tem custo próprio — no SAINT é O(N²) e pode dar OOM
    mesmo quando o treino (mini-batch) coube na memória. O OOM de predição é
    capturado aqui, preservando as métricas de treino já coletadas.
    """
    model = _make_model(variant)

    # ── Treino ────────────────────────────────────────────────────────────────
    fit_s, fit_ram, fit_vram = _run_phase(lambda: model.fit(X_tr, y_tr))

    # ── Predição (memória medida à parte; OOM não descarta o treino) ───────────
    pred_oom = False
    try:
        pred_s, pred_ram, pred_vram = _run_phase(lambda: model.predict(X_te))
        pred_ms = pred_s * 1000
    except torch.cuda.OutOfMemoryError:
        pred_oom = True
        pred_ms = pred_ram = pred_vram = float("nan")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "fit_s": round(fit_s, 3),
        "ram_delta_mb": round(fit_ram, 1),   # RAM de TREINO (compat. retroativa)
        "vram_mb": round(fit_vram, 1),        # VRAM de TREINO (compat. retroativa)
        "pred_oom": pred_oom,
        "pred_ms": None if pred_oom else round(pred_ms, 3),
        "pred_ram_mb": None if pred_oom else round(pred_ram, 1),
        "pred_vram_mb": None if pred_oom else round(pred_vram, 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/transformer_scaling.json")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--variants", default=None,
                        help="Lista separada por vírgula p/ rodar só um subconjunto "
                             "(ex.: SAINT_minibatch,SAINT_fullbatch,FTCUR_minibatch,"
                             "FTCUR_mfixed_full). Default: todas.")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    results: list[dict] = []

    variants = [
        "FTTransformer_softmax", "FTTransformer_topk",
        "FTTransformer_entmax", "FTTransformer_sparsemax",
        "SAINT_minibatch", "SAINT_fullbatch",
        "FTCUR_minibatch", "FTCUR_mfixed_full",
    ]
    if args.variants:
        wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in wanted if v not in variants]
        if unknown:
            raise SystemExit(f"Variantes desconhecidas: {unknown}\nDisponíveis: {variants}")
        variants = wanted
        log.info("Rodando apenas: %s", variants)

    for variant in variants:
        log.info("=== %s ===", variant)
        max_n = N_MAX[variant]

        for n in N_VALUES:
            if n > max_n:
                log.info("  N=%6d — pulando (N_max=%d)", n, max_n)
                results.append({"variant": variant, "n": n, "skipped": True})
                continue

            fits, preds, rams, vrams = [], [], [], []
            pred_rams, pred_vrams = [], []
            pred_oom = False

            oom = False
            last_error = None
            for rep in range(args.repeats):
                try:
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
                    rams.append(m["ram_delta_mb"])
                    vrams.append(m["vram_mb"])
                    if m["pred_oom"]:
                        pred_oom = True   # inferência O(N²) estourou (SAINT)
                    else:
                        preds.append(m["pred_ms"])
                        pred_rams.append(m["pred_ram_mb"])
                        pred_vrams.append(m["pred_vram_mb"])

                    log.info("  N=%6d rep=%d  fit=%.1fs  pred=%s  "
                             "RAM=%.1fMB  VRAM=%.1fMB  predVRAM=%s",
                             n, rep, m["fit_s"],
                             "OOM" if m["pred_oom"] else f"{m['pred_ms']:.1f}ms",
                             m["ram_delta_mb"], m["vram_mb"],
                             "OOM" if m["pred_oom"] else f"{m['pred_vram_mb']:.1f}MB")

                except torch.cuda.OutOfMemoryError as e:
                    oom = True
                    last_error = "OOM"
                    log.warning("  N=%6d rep=%d OOM — registrando e pulando N maiores", n, rep)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                except Exception as e:
                    last_error = str(e)
                    log.warning("  N=%6d rep=%d ERRO: %s", n, rep, e)
                    break

            if not fits:
                results.append({
                    "variant": variant, "n": n, "skipped": True,
                    "oom": oom, "error": last_error,
                })
                if oom:
                    # OOM vai piorar com N maior — para este modelo
                    log.warning("  OOM em N=%d — pulando N maiores para %s", n, variant)
                    for n_skip in N_VALUES[N_VALUES.index(n) + 1:]:
                        results.append({"variant": variant, "n": n_skip,
                                        "skipped": True, "oom": True, "error": "OOM (skip após primeiro OOM)"})
                    break
                continue

            have_pred = len(preds) > 0
            results.append({
                "variant": variant, "n": n, "skipped": False,
                "fit_s_median":      round(float(np.median(fits)), 3),
                "ram_delta_mb_median": round(float(np.median(rams)), 1),
                "vram_mb_median":    round(float(np.median(vrams)), 1),
                # Predição medida à parte; pred_oom=True → inferência O(N²) estourou
                "pred_oom":          pred_oom,
                "pred_ms_median":    round(float(np.median(preds)), 3) if have_pred else None,
                "pred_ram_mb_median":  round(float(np.median(pred_rams)), 1) if have_pred else None,
                "pred_vram_mb_median": round(float(np.median(pred_vrams)), 1) if have_pred else None,
                "fit_s_all":    [round(x, 3) for x in fits],
                "ram_mb_all":   [round(x, 1) for x in rams],
                "vram_mb_all":  [round(x, 1) for x in vrams],
                "pred_ms_all":  [round(x, 3) for x in preds],
                "pred_vram_mb_all": [round(x, 1) for x in pred_vrams],
            })

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Salvo em %s", out_path)

    # ── Tabela resumo ──────────────────────────────────────────────────────────
    print()
    print(f"{'Variant':<26}  {'N':>6}  {'fit':>7}  {'fitVRAM':>8}  "
          f"{'pred':>8}  {'predVRAM':>9}")
    print("-" * 78)
    for r in results:
        if r.get("skipped"):
            marker = "OOM" if r.get("oom") else "—"
            print(f"{r['variant']:<26}  {r['n']:>6}  {marker:>7}  {marker:>8}  "
                  f"{marker:>8}  {marker:>9}")
        else:
            pred_ms = r.get("pred_ms_median")
            pred_vram = r.get("pred_vram_mb_median")
            pred_str  = "OOM" if pred_ms is None else f"{pred_ms:>6.1f}ms"
            pvram_str = "OOM" if pred_vram is None else f"{pred_vram:>6.0f}MB"
            print(f"{r['variant']:<26}  {r['n']:>6}  "
                  f"{r['fit_s_median']:>6.1f}s  "
                  f"{r['vram_mb_median']:>6.0f}MB  "
                  f"{pred_str:>8}  {pvram_str:>9}")


if __name__ == "__main__":
    main()

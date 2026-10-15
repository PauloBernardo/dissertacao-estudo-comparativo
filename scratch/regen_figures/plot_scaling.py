#!/usr/bin/env python3
"""Regenera as 4 figuras de benchmark de escalabilidade (LSSVM + Transformers)
a partir dos JSONs atualizados -- mesmos nomes de arquivo, mesmo layout de
2 painéis (Tempo | Memória) das figuras originais, mas agora cada linha
distingue FIT (sólido) de PREDIÇÃO (tracejado) -- a distinção que motivou
toda a atualização das Tabelas 18/19.

NÃO toca no LaTeX -- só regenera os .pdf que Capitulo4.tex já referencia
(Figuras/benchmark_lssvm_time.pdf, _memory.pdf, benchmark_transformers_time.pdf,
_memory.pdf). Script fica em scratch/ (utilitário, não parte do pipeline
rastreado) -- promover para scripts/ do repo principal se for útil de novo.

Uso:
    python plot_scaling.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY_ROOT = Path(__file__).resolve().parents[2]
DISS_FIGURAS = Path("/home/paulo/Documentos/dissertacao-estudo-comparativo/dissertacao-latex/Figuras")

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _plot_family(records: list[dict], model_labels: dict[str, str],
                  fit_key: str, fit_mem_key: str, pred_key: str, pred_mem_key: str,
                  pred_unit_divisor: float, time_out: Path, mem_out: Path,
                  xlabel: str = "Número de Amostras de Treino (N)") -> None:
    by_variant: dict[str, list[dict]] = {}
    for r in records:
        if r.get("skipped"):
            continue
        by_variant.setdefault(r["variant"], []).append(r)

    fig_t, ax_t = plt.subplots(figsize=(6.0, 4.6))
    fig_m, ax_m = plt.subplots(figsize=(6.0, 4.6))

    for i, (variant, label) in enumerate(model_labels.items()):
        rows = sorted(by_variant.get(variant, []), key=lambda r: r["n"])
        if not rows:
            continue
        color = COLORS[i % len(COLORS)]
        ns = [r["n"] for r in rows]

        fit_t = [r.get(fit_key) for r in rows]
        pred_t = [r[pred_key] / pred_unit_divisor if r.get(pred_key) is not None else None for r in rows]
        fit_m = [r.get(fit_mem_key) for r in rows]
        pred_m = [r.get(pred_mem_key) for r in rows]

        # Uma entrada de legenda por modelo (a linha fit, sólida); a curva de
        # predição usa a MESMA cor, sem entrada própria -- o título do eixo já
        # explica sólido=fit/tracejado=predição, then repetir por modelo só
        # dobraria a legenda sem acrescentar informação.
        ax_t.plot(ns, fit_t, marker="o", linestyle="-", color=color, label=label)
        if any(v is not None for v in pred_t):
            ax_t.plot([n for n, v in zip(ns, pred_t) if v is not None],
                       [v for v in pred_t if v is not None],
                       marker="s", linestyle="--", color=color, alpha=0.75)

        ax_m.plot(ns, fit_m, marker="o", linestyle="-", color=color, label=label)
        if any(v is not None for v in pred_m):
            ax_m.plot([n for n, v in zip(ns, pred_m) if v is not None],
                       [v for v in pred_m if v is not None],
                       marker="s", linestyle="--", color=color, alpha=0.75)

    for ax, ylabel, title in (
        (ax_t, "Tempo (s)", "Tempo vs N --- sólido = fit, tracejado = predição"),
        (ax_m, "Memória (MB)", "Memória vs N --- sólido = fit, tracejado = predição"),
    ):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7.5, ncol=1, loc="best")

    fig_t.tight_layout()
    fig_m.tight_layout()
    fig_t.savefig(time_out)
    fig_m.savefig(mem_out)
    plt.close(fig_t)
    plt.close(fig_m)
    print(f"Salvos: {time_out}, {mem_out}")


def main() -> None:
    # ── LSSVM ────────────────────────────────────────────────────────────────
    lssvm_records = json.loads(
        (STUDY_ROOT / "scratch/lssvm_table18_package/table18_results.json").read_text()
    )
    lssvm_labels = {
        "StandardLSSVM":             "LSSVM padrão O(N³)",
        "ADMMNesterovLSSVM":         "ADMM-Nesterov O(N³)",
        "NystromLSSVMColnorm":       "Nyström-SVM O(N·m²)",
        "ADMMNystromLSSVM":          "ADMM-Nyström O(N·m²)",
        "FISTANystromLSSVM":         "FISTA-Nyström O(N·m²)",
        "PCPLSSVm":                  "LSSVM-PCP O(N·r²)",
        "IPLSSVmOriginal":           "LSSVM-IP O(N³)",
        "PruningLSSVM":              "LSSVM-Pruning O(N³)",
        "OppositeMapsOriginalLSSVM": "LSSVM-OppMaps O(N²)",
    }
    _plot_family(
        lssvm_records, lssvm_labels,
        fit_key="fit_s_median", fit_mem_key="fit_ram_mb_median",
        pred_key="pred_s_median", pred_mem_key="pred_ram_mb_median",
        pred_unit_divisor=1.0,
        time_out=DISS_FIGURAS / "benchmark_lssvm_time.pdf",
        mem_out=DISS_FIGURAS / "benchmark_lssvm_memory.pdf",
    )

    # ── Transformers ─────────────────────────────────────────────────────────
    tf_records = json.loads((STUDY_ROOT / "results/table19_results.json").read_text())
    tf_labels = {
        "FTTransformer_softmax":   "FT-Softmax",
        "FTTransformer_topk":      "FT-TopK",
        "FTTransformer_entmax":    "FT-Entmax",
        "FTTransformer_sparsemax": "FT-Sparsemax",
        "SAINT_minibatch":         "SAINT (mini-batch)",
        "SAINT_fullbatch":         "SAINT (full-batch)",
        "FTCUR_minibatch":         "FT-CUR (mini-batch)",
        "FTCUR_mfixed_full":       "FT-CUR (full-batch, m fixo)",
    }
    _plot_family(
        tf_records, tf_labels,
        fit_key="fit_s_median", fit_mem_key="vram_mb_median",
        pred_key="pred_ms_median", pred_mem_key="pred_vram_mb_median",
        pred_unit_divisor=1000.0,  # ms -> s, pra dividir a mesma escala do fit
        time_out=DISS_FIGURAS / "benchmark_transformers_time.pdf",
        mem_out=DISS_FIGURAS / "benchmark_transformers_memory.pdf",
    )


if __name__ == "__main__":
    main()

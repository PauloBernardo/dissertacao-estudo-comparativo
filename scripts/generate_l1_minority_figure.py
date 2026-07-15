#!/usr/bin/env python3
"""Dose-resposta: poda ℓ1 da classe minoritária vs desbalanceamento.

Eixo x: razão de desbalanceamento do dataset (majoritária:minoritária).
Eixo y: r = (% minoria entre coeficientes sobreviventes) / (% minoria no treino),
        medida como o MENOR valor no trajeto de esparsificação até s=0,90 (r_min).
        r = 1  -> a poda é neutra quanto à classe (linha de referência).
        r -> 0 -> a minoria é eliminada do modelo.

Cada dataset: 5 sementes (pontos claros) + média (marcador cheio) com barra de ±dp.
(σ,τ) fixos por dataset -> a única fonte de variação entre sementes é o sorteio dos
dados; os desvios pequenos mostram que a seed NÃO dirige o efeito.

Legível em P&B (tese impressa): um único traço/cor, distinção por marcador e rótulo
direto. Saída: results/report_figs/fig_l1_minority_pruning.{pdf,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "report_figs"

# paleta validada (mesma da figura de convergência)
C_ACC, C_NEUTRAL = "#2a78d6", "#1baf7a"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

ORDER = ["HIGGS50K", "COVER", "ADULT", "CREDIT", "SHOPPERS", "BANK"]


def main():
    rows = json.loads((ROOT / "results" / "l1_minority_doseresponse.json").read_text())

    by = {}
    for r in rows:
        by.setdefault(r["dataset"], []).append(r)

    xs_pts, ys_pts = [], []          # todos os pontos (p/ Spearman)
    xm, ym, yerr, labels = [], [], [], []
    for ds in ORDER:
        rr = by[ds]
        imb = rr[0]["imbalance"]
        vals = [r["r_min_pre"] for r in rr if r["r_min_pre"] is not None]
        if not vals:
            continue
        for v in vals:
            xs_pts.append(imb); ys_pts.append(v)
        xm.append(imb); ym.append(float(np.mean(vals)))
        yerr.append(float(np.std(vals))); labels.append(ds)

    rho, p = spearmanr(xs_pts, ys_pts)

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6,
                         "font.family": "serif", "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    # faixa "minoria sub-representada" (r < 1) — sombra sutil
    ax.axhspan(0, 1, color=C_ACC, alpha=0.05, zorder=0)
    # referência de neutralidade r = 1
    ax.axhline(1.0, color=C_NEUTRAL, lw=1.2, ls="--", zorder=2)
    ax.text(7.7, 1.02, "representação neutra ($r=1$)", fontsize=7.5,
            color=C_NEUTRAL, ha="right", va="bottom")

    # pontos por semente (claros)
    ax.scatter(xs_pts, ys_pts, s=14, color=INK_MUTED, alpha=0.30,
               zorder=3, linewidths=0)
    # média por dataset + barra de erro + traço conectando
    ax.plot(xm, ym, color=C_ACC, lw=1.6, zorder=4)
    ax.errorbar(xm, ym, yerr=yerr, fmt="o", color=C_ACC, ms=6,
                capsize=3, elinewidth=1, mec="white", mew=0.8, zorder=5)

    # rótulos diretos dos datasets (offsets ajustados p/ evitar colisão)
    off = {"HIGGS50K": (-6, 9, "right", "bottom"),
           "COVER":    (0, -9, "center", "top"),
           "ADULT":    (-10, 9, "right", "bottom"),
           "CREDIT":   (12, 9, "left", "bottom"),
           "SHOPPERS": (10, -6, "left", "top"),
           "BANK":     (0, -9, "center", "top")}
    for x, y, lab in zip(xm, ym, labels):
        ox, oy, ha, va = off[lab]
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(ox, oy), ha=ha, va=va, fontsize=7.5, color=INK)

    ax.set_xlabel("Razão de desbalanceamento (majoritária : minoritária)")
    ax.set_ylabel(r"$r$: representação da minoria entre sobreviventes")
    ax.set_xlim(0.5, 8.3)
    ax.set_ylim(0.0, 1.18)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])

    ax.text(0.03, 0.06, rf"Spearman $\rho = {rho:+.2f}$  ($p = {p:.1e}$)",
            transform=ax.transAxes, fontsize=8, color=INK,
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, lw=0.6))

    ax.grid(True, which="major", color=GRID, lw=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_l1_minority_pruning.{ext}", dpi=200,
                    bbox_inches="tight")
    print(f"Spearman rho={rho:+.3f} p={p:.2e} (n={len(xs_pts)})")
    for x, y, e, lab in zip(xm, ym, yerr, labels):
        print(f"  {lab:10s} {x:.1f}:1  r_min={y:.3f}±{e:.3f}")
    print(f"salvo em {OUT}/fig_l1_minority_pruning.pdf")


if __name__ == "__main__":
    main()

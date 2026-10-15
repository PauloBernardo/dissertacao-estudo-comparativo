#!/usr/bin/env python3
"""Curvas de convergência: ADMM-Nyström vs FISTA-Nyström.

2026-10-15: formulação nova — os dois minimizam o MESMO objetivo (eq. admm_vs_fista).
Painel (a): sub-otimalidade relativa (f_k − f*)/f* × iteração, escala log-log, contra o
            melhor valor obtido por qualquer solver: ADMM com ρ fixo (regra da referência,
            1/λ_max(H)), ADMM com ρ adaptativo (Boyd 2011 §3.4.1) e FISTA.
Painel (b): resíduo primal do ADMM com ρ fixo e adaptativo, com a tolerância. O FISTA
            não possui análogo (não há splitting), logo não aparece aqui.

A mensagem: com ρ fixo o ADMM estaciona longe do ótimo; com ρ adaptativo chega lá
em centenas de iterações, antes do FISTA.

Saída: results/report_figs/fig_convergence_admm_fista.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels, make_splits
from src.models.lssvm.primal.admm_nystrom import ADMMNystromLSSVM
from src.models.lssvm.primal.fista_nystrom import FISTANystromLSSVM

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "report_figs"

# Paleta validada (dataviz/references/palette.md, checagem light: pass).
# Encoding SECUNDÁRIO por estilo de linha -> figura legível em P&B (tese impressa).
C_ADMM, C_FISTA, C_DUAL = "#2a78d6", "#eda100", "#1baf7a"
C_FIX = "#c0392b"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

DATASET, SEED, SIGMA, TAU, LREL = "ADULT", 0, 5.0, 0.01, 0.001  # moda do Tier 2


def load_data():
    X, y, _ = DatasetLoader.load(DATASET)
    if len(X) > 2857:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=2857, random_state=SEED)
        idx, _ = next(sss.split(X, y))
        X, y = X[idx], y[idx]
    Xtr, _, ytr, _ = make_splits(X, y, test_size=0.30, seed=SEED)
    return StandardScaler().fit_transform(Xtr), _convert_labels(ytr, "signed").astype(float)


def main():
    Xtr, ytr = load_data()
    kw = dict(sigma=SIGMA, tau=TAU, lambda_rel=LREL, kernel_reg=True, m_ratio=0.30,
              landmark_method="random", random_state=SEED, track_history=True, tol=1e-10)

    fix = ADMMNystromLSSVM(**kw, rho=None, max_iter=20000); fix.fit(Xtr, ytr)
    adp = ADMMNystromLSSVM(**kw, rho="adaptive", max_iter=3000); adp.fit(Xtr, ytr)
    fista = FISTANystromLSSVM(**kw, max_iter=100000); fista.fit(Xtr, ytr)

    hx, ha, hf = fix.history_, adp.history_, fista.history_
    fx, fa, ff = (np.array([h["objective"] for h in H]) for H in (hx, ha, hf))
    fs = min(fx.min(), fa.min(), ff.min())
    sub = lambda f: np.maximum((f - fs) / abs(fs), 1e-16)
    sub_x, sub_a, sub_f = sub(fx), sub(fa), sub(ff)

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6,
                         "font.family": "serif", "mathtext.fontset": "cm"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9))

    # ── (a) sub-otimalidade ────────────────────────────────────────────────
    ax1.loglog(np.arange(1, len(sub_x) + 1), sub_x, color=C_FIX, lw=2,
               ls=":", label=r"ADMM, $\rho$ fixo")
    ax1.loglog(np.arange(1, len(sub_a) + 1), sub_a, color=C_ADMM, lw=2,
               ls="-", label=r"ADMM, $\rho$ adaptativo")
    ax1.loglog(np.arange(1, len(sub_f) + 1), sub_f, color=C_FISTA, lw=2,
               ls="--", label="FISTA")
    ax1.axvline(500, color=INK_MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax1.axvline(5000, color=INK_MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1)
    ax1.set_xlabel("Iteração"); ax1.set_ylabel(r"$(f_k - f^\star)\,/\,f^\star$")
    ax1.set_title("(a) Sub-otimalidade no objetivo comum",
                  fontsize=9, color=INK, pad=8)
    ax1.legend(frameon=False, fontsize=7.5, loc="lower left")

    # ── (b) resíduo primal do ADMM ─────────────────────────────────────────
    for H, c, ls, lab in ((hx, C_FIX, ":", r"$\rho$ fixo"), (ha, C_ADMM, "-", r"$\rho$ adaptativo")):
        ax2.loglog([h["iter"] for h in H], np.maximum([h["primal_res"] for h in H], 1e-16),
                   color=c, lw=2, ls=ls, label=lab)
    ax2.axhline(1e-6, color=INK_MUTED, lw=1, ls=":", zorder=1)
    ax2.text(1.15, 2.2e-6, r"tol $=10^{-6}$", fontsize=7.5, color=INK_MUTED,
             ha="left", va="bottom")
    ax2.set_xlabel("Iteração"); ax2.set_ylabel("Resíduo primal")
    ax2.set_title("(b) Resíduo primal do ADMM",
                  fontsize=9, color=INK, pad=8)
    ax2.legend(frameon=False, fontsize=8, loc="lower left")

    for ax in (ax1, ax2):
        ax.grid(True, which="major", color=GRID, lw=0.5, alpha=0.8)
        ax.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.4)
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
        fig.savefig(OUT / f"fig_convergence_admm_fista.{ext}", dpi=200,
                    bbox_inches="tight")
    for n, H, f in (("ADMM fixo", hx, sub_x), ("ADMM adapt", ha, sub_a), ("FISTA", hf, sub_f)):
        first = next((i + 1 for i, v in enumerate(f) if v < 1e-6), None)
        print(f"{n:10s}: {len(H):6d} iters | subopt final = {f[-1]:.2e} | 1e-6 em {first}")
    print(f"salvo em {OUT}/fig_convergence_admm_fista.pdf")


if __name__ == "__main__":
    main()

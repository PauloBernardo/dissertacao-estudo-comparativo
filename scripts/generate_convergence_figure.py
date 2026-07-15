#!/usr/bin/env python3
"""Curvas de convergência: ADMM-Nyström vs FISTA-Nyström.

Painel (a): sub-otimalidade relativa (f_k − f*)/f* × iteração, escala log-log.
            Cada solver é medido contra o ótimo do SEU PRÓPRIO objetivo (o ADMM
            no objetivo cru, o FISTA no centrado) — pois minimizam problemas
            distintos (ver Seção 3, eq. admm_vs_fista).
Painel (b): resíduos primal e dual do ADMM × iteração, com a tolerância. O FISTA
            não possui análogo (não há splitting), logo não aparece aqui.

A mensagem: ambos atingem um platô — nenhum parou prematuramente.

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
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

DATASET, SEED, SIGMA, TAU, LAM = "ADULT", 0, 0.5, 5.0, 0.1


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
    kw = dict(sigma=SIGMA, tau=TAU, lambda_=LAM, m_ratio=0.30,
              landmark_method="colnorm", random_state=SEED, track_history=True)

    # tol apertada + teto alto: garante que cada um chegue ao SEU ótimo,
    # para que f* seja uma referência confiável de sub-otimalidade.
    admm = ADMMNystromLSSVM(**kw, tol=1e-12, max_iter=3000); admm.fit(Xtr, ytr)
    fista = FISTANystromLSSVM(**kw, tol=1e-12, max_iter=20000); fista.fit(Xtr, ytr)

    ha, hf = admm.history_, fista.history_
    fa = np.array([h["objective"] for h in ha]); fs_a = fa.min()
    ff = np.array([h["objective"] for h in hf]); fs_f = ff.min()
    sub_a = np.maximum((fa - fs_a) / abs(fs_a), 1e-16)
    sub_f = np.maximum((ff - fs_f) / abs(fs_f), 1e-16)

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6,
                         "font.family": "serif", "mathtext.fontset": "cm"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9))

    # ── (a) sub-otimalidade ────────────────────────────────────────────────
    ax1.loglog(np.arange(1, len(sub_a) + 1), sub_a, color=C_ADMM, lw=2,
               ls="-", label="ADMM-Nyström")
    ax1.loglog(np.arange(1, len(sub_f) + 1), sub_f, color=C_FISTA, lw=2,
               ls="--", label="FISTA-Nyström")
    ax1.set_xlabel("Iteração"); ax1.set_ylabel(r"$(f_k - f^\star)\,/\,f^\star$")
    ax1.set_title("(a) Sub-otimalidade — cada solver no seu objetivo",
                  fontsize=9, color=INK, pad=8)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")

    # ── (b) resíduos do ADMM ───────────────────────────────────────────────
    it = np.array([h["iter"] for h in ha])
    pr = np.maximum([h["primal_res"] for h in ha], 1e-16)
    dr = np.maximum([h["dual_res"] for h in ha], 1e-16)
    ax2.loglog(it, pr, color=C_ADMM, lw=2, ls="-", label="resíduo primal")
    ax2.loglog(it, dr, color=C_DUAL, lw=2, ls="-.", label="resíduo dual")
    ax2.axhline(1e-6, color=INK_MUTED, lw=1, ls=":", zorder=1)
    ax2.text(1.15, 2.2e-6, r"tol $=10^{-6}$", fontsize=7.5, color=INK_MUTED,
             ha="left", va="bottom")
    ax2.set_xlabel("Iteração"); ax2.set_ylabel("Norma do resíduo")
    ax2.set_title("(b) Convergência do splitting (ADMM)",
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
    print(f"ADMM : {len(ha):5d} iters | f* = {fs_a:.4f} | subopt final = {sub_a[-1]:.2e}")
    print(f"FISTA: {len(hf):5d} iters | f* = {fs_f:.4f} | subopt final = {sub_f[-1]:.2e}")
    print(f"ADMM  resíduos finais: primal={pr[-1]:.2e}  dual={dr[-1]:.2e}")
    print(f"salvo em {OUT}/fig_convergence_admm_fista.pdf")


if __name__ == "__main__":
    main()

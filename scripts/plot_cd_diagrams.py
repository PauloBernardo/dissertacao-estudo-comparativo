#!/usr/bin/env python3
"""Diagramas de diferença crítica (Demšar, 2006) legíveis na largura do texto da tese.

Lê os ranks médios publicados (tables/tier{1,2}_ranks.tex), calcula a CD de Nemenyi e grava
SÓ as figuras Figuras/fig_cd_tier{1,2}.pdf — não toca nas tabelas de Nemenyi (que têm edições
manuais; ver generate_nemenyi_analysis.py, que reescreveria as duas coisas).

    .venv/bin/python scripts/plot_cd_diagrams.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from generate_nemenyi_analysis import read_ranks  # noqa: E402
from src.metrics.statistical import nemenyi_cd  # noqa: E402

THESIS = ROOT.parent / "dissertacao-latex"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 9})


def cliques(r: list[float], cd: float) -> list[tuple[int, int]]:
    """Grupos maximais de modelos consecutivos cuja amplitude de rank não excede a CD."""
    out, last = [], -1
    for i in range(len(r)):
        j = max(k for k in range(i, len(r)) if r[k] - r[i] <= cd)
        if j > i and j > last:
            out.append((i, j)); last = j
    return out


def plot(rows: list[tuple[str, float]], cd: float, out: Path) -> None:
    names = [n for n, _ in rows]; r = [v for _, v in rows]
    n = len(r); nl = (n + 1) // 2
    lo, hi = math.floor(min(r)), math.ceil(max(r))
    W, M = 6.3, 1.55                       # largura total e margem dos rótulos (pol.)
    x = lambda v: M + (v - lo) / (hi - lo) * (W - 2 * M)
    cl = cliques(r, cd)
    y_cd, y_ax = 0.0, -0.45
    y_cl = [y_ax - 0.16 - 0.11 * c for c in range(len(cl))]
    y_lab0 = (y_cl[-1] if cl else y_ax) - 0.22
    row = 0.19
    H = -(y_lab0 - row * (nl - 1)) + 0.25
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(y_lab0 - row * (nl - 1) - 0.12, 0.18); ax.axis("off")

    # eixo de ranks
    ax.hlines(y_ax, x(lo), x(hi), color="black", lw=0.9)
    for t in range(lo, hi + 1):
        ax.vlines(x(t), y_ax, y_ax + 0.07, color="black", lw=0.9)
        ax.text(x(t), y_ax + 0.10, str(t), ha="center", va="bottom", fontsize=8.5)
    # barra da CD
    ax.hlines(y_cd, x(lo), x(lo + cd), color="#b2182b", lw=1.6)
    for e in (lo, lo + cd):
        ax.vlines(x(e), y_cd - 0.04, y_cd + 0.04, color="#b2182b", lw=1.6)
    ax.text(x(lo + cd) + 0.06, y_cd, f"CD = {cd:.2f}".replace(".", ","), color="#b2182b",
            va="center", ha="left", fontsize=9)
    # grupos sem diferença significativa
    for (i, j), yc in zip(cl, y_cl):
        ax.hlines(yc, x(r[i]) - 0.03, x(r[j]) + 0.03, color="black", lw=2.4)
    # rótulos: melhores à esquerda, piores à direita
    for k in range(n):
        left = k < nl
        idx = k if left else n - 1 - k + nl      # direita de cima para baixo: do pior ao melhor
        yy = y_lab0 - row * (k if left else k - nl)
        xr = x(r[idx])
        xe = M - 0.08 if left else W - M + 0.08
        ax.plot([xr, xr, xe], [y_ax, yy, yy], color="0.35", lw=0.6)
        lab = f"{names[idx]} ({r[idx]:.2f})".replace(".", ",")
        ax.text(xe - 0.04 if left else xe + 0.04, yy, lab, ha="right" if left else "left",
                va="center", fontsize=8.5)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("->", out, f"(CD={cd:.3f}, {len(cl)} grupos)")


def main() -> int:
    for tier, nds in [(1, 10), (2, 6)]:
        rows = read_ranks(THESIS / "tables" / f"tier{tier}_ranks.tex")
        plot(rows, nemenyi_cd(len(rows), nds, alpha=0.05), THESIS / "Figuras" / f"fig_cd_tier{tier}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Figura combinada: cada modelo esparso posicionado por (taxa de vitoria
em F1, taxa de vitoria em estabilidade) contra o baseline ingenuo, no
proprio ponto de operacao oficial. Quadrantes a 50%/50% separam quem
passa nos dois eixos, so num, so no outro, ou em nenhum.
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS = HERE / "validation_results_2026_10_15.json"
PARSIMONY_RESULTS = HERE / "validation_parsimony_results_2026_10_14.json"

LABELS = {
    "NystromLSSVMRandom": "Nyström-SVM", "ADMMNystromRandom": "ADMM-Nyström",
    "FISTANystromRandom": "FISTA-Nyström", "PCPLSSVm": "LSSVM-PCP",
    "IPLSSVmOriginal": "LSSVM-IP", "FSALSSVmOriginal": "LSSVM-FSA",
    "PruningLSSVM": "LSSVM-Pruning", "OppositeMapsOriginalLSSVM": "LSSVM-OppMaps",
    "DualFISTA": "DualFISTA", "ADMMNesterovLSSVM": "ADMM-Nesterov",
    "ADMMElasticNet": "ADMM-ElasticNet", "FISTANesterov": "FISTA-Nesterov",
}
STRUCTURAL = {"NystromLSSVMRandom", "ADMMNystromRandom", "FISTANystromRandom", "PCPLSSVm",
              "IPLSSVmOriginal", "FSALSSVmOriginal", "PruningLSSVM", "OppositeMapsOriginalLSSVM"}


def compute():
    data = json.loads(RESULTS.read_text())
    by_vd = defaultdict(lambda: {"off": [], "nai": []})
    for r in data:
        by_vd[(r["variant"], r["dataset"])]["off"].append(r["f1_official"])
        by_vd[(r["variant"], r["dataset"])]["nai"].append(r["f1_naive"])

    by_variant = defaultdict(lambda: {"f1_wins": 0, "f1_n": 0, "stab_wins": 0, "stab_n": 0})
    for (v, d), vals in by_vd.items():
        off, nai = vals["off"], vals["nai"]
        for o, n in zip(off, nai):
            by_variant[v]["f1_n"] += 1
            if o > n:
                by_variant[v]["f1_wins"] += 1
        if len(off) >= 5:
            so, sn = st.stdev(off), st.stdev(nai)
            by_variant[v]["stab_n"] += 1
            if so < sn:
                by_variant[v]["stab_wins"] += 1
    return by_variant


def compute_parsimony():
    data = [r for r in json.loads(PARSIMONY_RESULTS.read_text()) if r["status"] == "ok"]
    by_vd = defaultdict(lambda: {"par": [], "nai": []})
    for r in data:
        by_vd[(r["variant"], r["dataset"])]["par"].append(r["f1_parsimony"])
        by_vd[(r["variant"], r["dataset"])]["nai"].append(r["f1_naive"])

    by_variant = defaultdict(lambda: {"f1_wins": 0, "f1_n": 0, "stab_wins": 0, "stab_n": 0})
    for (v, d), vals in by_vd.items():
        par, nai = vals["par"], vals["nai"]
        for p, n in zip(par, nai):
            by_variant[v]["f1_n"] += 1
            if p > n:
                by_variant[v]["f1_wins"] += 1
        if len(par) >= 5:
            sp, sn = st.stdev(par), st.stdev(nai)
            by_variant[v]["stab_n"] += 1
            if sp < sn:
                by_variant[v]["stab_wins"] += 1
    return by_variant


# offsets manuais (dx, dy em pontos, ha) -- calibrados pras coordenadas
# reais (varios pontos quase coincidem, sobretudo os 3 primais l1 em
# ~(13,30) e o aglomerado estrutural em y~80-82).
OFFSETS = {
    "PCPLSSVm": (6, 5, "left"),
    "NystromLSSVMRandom": (-7, 6, "right"),
    "IPLSSVmOriginal": (-8, 0, "right"),
    "OppositeMapsOriginalLSSVM": (-2, 9, "right"),
    "FSALSSVmOriginal": (3, -11, "left"),
    "FISTANystromRandom": (-7, -2, "right"),
    "ADMMNystromRandom": (2, -15, "right"),
    "PruningLSSVM": (-7, -3, "right"),
    "DualFISTA": (7, 2, "left"),
    "ADMMNesterovLSSVM": (7, 10, "left"),
    "ADMMElasticNet": (7, 0, "left"),
    "FISTANesterov": (7, -10, "left"),
}


L1_EMERGENT = {"DualFISTA", "ADMMNesterovLSSVM", "ADMMElasticNet", "FISTANesterov"}


def main() -> None:
    by_variant = compute()
    by_variant_par = compute_parsimony()

    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    ax.axhline(50, color="gray", linestyle="--", linewidth=1)
    ax.axvline(50, color="gray", linestyle="--", linewidth=1)
    ax.axhspan(50, 100, xmin=0.5, xmax=1.0, color="#2a9d5c", alpha=0.08)
    ax.axhspan(0, 50, xmin=0, xmax=0.5, color="#d94f4f", alpha=0.08)

    # jitter minusculo so pra separar visualmente os 3 primais l1, que
    # caem quase no mesmo ponto (13,30) -- nao afeta a leitura numerica,
    # os valores reais estao na tabela.
    JITTER = {"ADMMNesterovLSSVM": (-1.2, 1.0), "ADMMElasticNet": (0.0, 0.0), "FISTANesterov": (1.2, -1.0)}

    for v, d in by_variant.items():
        f1pct = 100 * d["f1_wins"] / d["f1_n"] + JITTER.get(v, (0, 0))[0]
        stabpct = 100 * d["stab_wins"] / d["stab_n"] + JITTER.get(v, (0, 0))[1]
        color = "#2a9d5c" if v in STRUCTURAL else "#d94f4f"
        marker = "o" if v in STRUCTURAL else "^"
        ax.scatter([f1pct], [stabpct], s=100, color=color, marker=marker,
                   edgecolors="black", linewidths=0.8, zorder=5)
        dx, dy, ha = OFFSETS[v]
        ax.annotate(LABELS[v], (f1pct, stabpct), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, ha=ha, zorder=6)

        if v in L1_EMERGENT and v in by_variant_par:
            dp = by_variant_par[v]
            f1p = 100 * dp["f1_wins"] / dp["f1_n"]
            stabp = 100 * dp["stab_wins"] / dp["stab_n"]
            ax.annotate("", xy=(f1p, stabp), xytext=(f1pct, stabpct),
                        arrowprops=dict(arrowstyle="->", color="#d94f4f", alpha=0.7, lw=1.3),
                        zorder=4)
            ax.scatter([f1p], [stabp], s=100, facecolors="white", edgecolors="#d94f4f",
                       marker="^", linewidths=1.6, zorder=5)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("% dos casos em que o modelo bate o ingênuo em $F_1$")
    ax.set_ylabel("% das bases em que o modelo é mais estável\n(menor desvio-padrão entre sementes) que o ingênuo")
    ax.set_title("Validação de cada modelo esparso: acurácia × estabilidade\nvs. baseline ingênuo --- ponto padrão e (família $\\ell_1$) ponto de parcimônia", fontsize=10.5)

    ax.text(75, 23, "ganha em F1,\nperde em estabilidade", fontsize=8, ha="center", style="italic", color="gray")
    ax.text(25, 96, "perde em F1,\nganha em estabilidade", fontsize=8, ha="center", style="italic", color="gray")
    ax.text(75, 96, "aprovado nos\ndois eixos", fontsize=8.5, ha="center", style="italic", color="#1d6b3e", weight="bold")
    ax.text(37, 4, "reprovado nos dois eixos", fontsize=8.5, ha="center", style="italic", color="#a13030", weight="bold")

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([], [], marker="o", color="#2a9d5c", linestyle="", markersize=9,
               markeredgecolor="black", label="esparsidade imposta (estrutural)"),
        Line2D([], [], marker="^", color="#d94f4f", linestyle="", markersize=9,
               markeredgecolor="black", label="família $\\ell_1$, ponto padrão"),
        Line2D([], [], marker="^", color="white", linestyle="", markersize=9,
               markeredgecolor="#d94f4f", markeredgewidth=1.6, label="família $\\ell_1$, ponto de parcimônia"),
    ]
    ax.legend(handles=legend_elems, fontsize=8.5, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out = HERE / "validation_f1_vs_stability.pdf"
    fig.savefig(out)
    print(f"Salvo: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Figura-resumo: taxa de vitoria de cada modelo esparso contra o baseline
'esparso ingenuo' (denso, treinado direto em k=n_support pontos), no
proprio ponto de operacao oficial (GridSearchCV) de cada um -- Tier1 e
Tier2 lado a lado. Verde = modelo bate o ingenuo na maioria das vezes;
vermelho = perde na maioria das vezes.
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "validation_results.json"

LABELS = {
    "NystromLSSVMColnorm": "Nyström-SVM", "ADMMNystromLSSVM": "ADMM-Nyström",
    "FISTANystrom": "FISTA-Nyström", "PCPLSSVm": "LSSVM-PCP",
    "IPLSSVmOriginal": "LSSVM-IP", "FSALSSVmOriginal": "LSSVM-FSA",
    "PruningLSSVM": "LSSVM-Pruning", "OppositeMapsOriginalLSSVM": "LSSVM-OppMaps",
    "DualFISTA": "DualFISTA", "ADMMNesterovLSSVM": "ADMM-Nesterov",
    "ADMMElasticNet": "ADMM-ElasticNet", "FISTANesterov": "FISTA-Nesterov",
}
STRUCTURAL = ["NystromLSSVMColnorm", "ADMMNystromLSSVM", "FISTANystrom", "PCPLSSVm",
              "IPLSSVmOriginal", "FSALSSVmOriginal", "PruningLSSVM", "OppositeMapsOriginalLSSVM"]
L1_EMERGENT = ["DualFISTA", "ADMMNesterovLSSVM", "ADMMElasticNet", "FISTANesterov"]
ORDER = STRUCTURAL + L1_EMERGENT


def main() -> None:
    data = json.loads(RESULTS.read_text())
    by_model_tier = defaultdict(lambda: {"off": [], "nai": []})
    for r in data:
        by_model_tier[(r["variant"], r["tier"])]["off"].append(r["f1_official"])
        by_model_tier[(r["variant"], r["tier"])]["nai"].append(r["f1_naive"])

    variants = [v for v in ORDER if (v, "tier1") in by_model_tier]
    x = np.arange(len(variants))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 6.4))
    for tier, offset, hatch in [("tier1", -width / 2, None), ("tier2", width / 2, "//")]:
        win_pcts, colors = [], []
        for v in variants:
            key = (v, tier)
            if key not in by_model_tier:
                win_pcts.append(0)
                colors.append("none")
                continue
            off, nai = by_model_tier[key]["off"], by_model_tier[key]["nai"]
            wins = sum(1 for o, n in zip(off, nai) if o > n)
            pct = 100 * wins / len(off)
            win_pcts.append(pct)
            colors.append("#2a9d5c" if pct >= 50 else "#d94f4f")
        bars = ax.bar(x + offset, win_pcts, width, color=colors, edgecolor="black",
                       linewidth=0.6, hatch=hatch, label=f"{'Tier 1' if tier=='tier1' else 'Tier 2'}")

    ax.axhline(50, color="gray", linestyle="--", linewidth=1)
    ax.axvline(len(STRUCTURAL) - 0.5, color="black", linestyle=":", linewidth=1.2)
    ax.text(len(STRUCTURAL) / 2 - 0.5, 104, "esparsidade imposta\n(seleção/estrutural)",
            ha="center", fontsize=8.5, style="italic")
    ax.text(len(STRUCTURAL) + len(L1_EMERGENT) / 2 - 0.5, 104, "esparsidade emergente\n(família $\\ell_1$)",
            ha="center", fontsize=8.5, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[v] for v in variants], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("% de vezes que o modelo bate o baseline ingênuo\n(mesmo k = n_support oficial)")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("Cada modelo esparso vs. baseline ingênuo, no próprio ponto de operação oficial", fontsize=11, pad=32)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#2a9d5c", edgecolor="black", label="bate o ingênuo (≥50%)"),
        Patch(facecolor="#d94f4f", edgecolor="black", label="perde pro ingênuo (<50%)"),
        Patch(facecolor="white", edgecolor="black", label="Tier 1 (sem hachura)"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Tier 2 (hachurado)"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.42))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0.14, 1, 1))

    out = HERE / "validation_win_rate.pdf"
    fig.savefig(out)
    print(f"Salvo: {out}")


if __name__ == "__main__":
    main()

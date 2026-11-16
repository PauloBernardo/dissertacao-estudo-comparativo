#!/usr/bin/env python3
"""Gera fig4 (ablação B — ruído/5f) e fig5 (ablação C — mk5/complexidade).

Lê os arquivos de ablação consolidados (CPU fiel + transformers do Kaggle):
    results/ablation_b_noise.json   — TWS_5f/TWM_5f/TWC_5f
    results/ablation_c_mk5.json     — MKE/MKM/MKH
Substitui a geração antiga (OLD/scripts/generate_report_figs.py, que lia
results antigos com os IP/FSA/OppMaps adaptados). Usa os nomes fiéis.
"""
import json
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))
from variant_aliases import canon  # noqa: E402
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "report_figs"

MODEL_LABEL = {
    "StandardLSSVM": "LSSVM-Std", "PCPLSSVm": "LSSVM-PCP",
    "FSALSSVmOriginal": "LSSVM-FSA", "IPLSSVmOriginal": "LSSVM-IP",
    "PruningLSSVM": "LSSVM-Pruning", "OppositeMapsOriginalLSSVM": "LSSVM-OppMaps",
    "ADMMNesterovLSSVM": "ADMM-Nesterov", "ADMMElasticNet": "ADMM-ElasticNet",
    "FISTANesterov": "FISTA-Nesterov", "DualFISTA": "DualFISTA★",
    "NystromLSSVMColnorm": "Nyström-SVM★", "FTTransformerCURColnorm": "FT-CUR★",
    "FTTransformer_softmax": "FT-Softmax", "FTTransformer_topk": "FT-Top-$k$",
    "FTTransformer_entmax": "FT-Entmax", "FTTransformer_sparsemax": "FT-Sparsemax",
    "SAINTColnorm": "SAINT", "XGBoost": "XGBoost",
}
PROPOSED = {"DualFISTA", "NystromLSSVMColnorm", "FTTransformerCURColnorm"}
INVESTIGATED = {"ADMMElasticNet"}
PAPER_BASE = {"ADMMNesterovLSSVM"}
FT_BASELINE = {"FTTransformer_softmax", "FTTransformer_topk",
               "FTTransformer_entmax", "FTTransformer_sparsemax", "SAINTColnorm"}
GENERAL_BASE = {"XGBoost"}


def model_color(m):
    if m in PROPOSED:
        return {"DualFISTA": "#d62728", "NystromLSSVMColnorm": "#2ca02c",
                "FTTransformerCURColnorm": "#ff7f0e"}[m]
    if m in GENERAL_BASE: return "#e377c2"
    if m in INVESTIGATED: return "#9467bd"
    if m in PAPER_BASE:   return "#8c564b"
    if m in FT_BASELINE:  return "#1f77b4"
    return "#888888"      # LSSVM baseline


def collect(path, datasets):
    out = defaultdict(lambda: defaultdict(list))
    for r in canon(json.loads(Path(path).read_text())):
        if r.get("status") != "ok":
            continue
        m = r.get("variant") or r.get("model")
        d = r["dataset"]
        if d in datasets and r.get("test_f1_macro") is not None:
            out[m][d].append(r["test_f1_macro"])
    return out


def barpanels(sc, models, ds_list, ds_labels, suptitle, fname, ref=None, ref_ds=None, ref_label=None,
              main_label=None):
    """Painéis de pontos horizontais (2026-11-16, revisão: rótulos legíveis na largura do texto).

    Cada modelo é uma linha; o ponto cheio é a média ± desvio-padrão entre sementes. Com
    ``ref``, um círculo vazio marca o mesmo modelo na condição de referência (ex.: 2D do
    Cenário 1), ligado ao ponto cheio. ``suptitle`` não é desenhado: a legenda da tese explica.
    """
    plt.rcParams.update({"font.size": 8.5})
    fig, axes = plt.subplots(1, len(ds_list), figsize=(6.3, 0.30 * len(models) + 1.25),
                             sharey=True, sharex=True)
    ypos = np.arange(len(models))[::-1]
    for col, ds in enumerate(ds_list):
        ax = axes[col]
        for y, m in zip(ypos, models):
            c = model_color(m)
            v = sc[m].get(ds)
            if ref is not None:
                r = ref[m].get(ref_ds[col])
                if r and v:
                    ax.plot([np.mean(r), np.mean(v)], [y, y], color=c, lw=1.2, alpha=0.6, zorder=2)
                if r:
                    ax.scatter(np.mean(r), y, s=24, facecolor="white", edgecolor=c, lw=1.2, zorder=3)
            if v:
                ax.errorbar(np.mean(v), y, xerr=np.std(v), fmt="o", ms=4.5, color=c,
                            elinewidth=1, capsize=2, zorder=4)
        ax.set_title(ds_labels[col], fontsize=9)
        ax.set_yticks(ypos)
        ax.set_yticklabels([MODEL_LABEL.get(m, m) for m in models], fontsize=8.5)
        ax.set_xlim(0.4, 1.02)
        ax.set_xticks([0.4, 0.6, 0.8, 1.0])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("F1-macro", fontsize=8.5)
        ax.grid(True, axis="x", alpha=0.3)
    if ref is not None:
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", label=ref_label),
             Line2D([], [], marker="o", ls="", color="0.3", label=main_label)]
        fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, fontsize=8,
                   bbox_to_anchor=(0.6, 1.0))
        plt.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.92))
    else:
        plt.tight_layout(w_pad=0.6)
    plt.savefig(OUT / f"{fname}.pdf", dpi=150)
    plt.savefig(OUT / f"{fname}.png", dpi=150)
    plt.close()
    print(f"{fname} done")


# fig4 — Ablação B (5 features com ruído)
NOISE_MODELS = ["StandardLSSVM", "ADMMNesterovLSSVM", "DualFISTA",
                "FTTransformer_sparsemax", "FTTransformer_softmax",
                "NystromLSSVMColnorm", "FTTransformerCURColnorm", "XGBoost"]
sc_b = collect(ROOT / "results" / "ablation_b_noise.json", ["TWS_5f", "TWM_5f", "TWC_5f"])
sc_2d = collect(ROOT / "results" / "tier1_gridcv.json", ["TWS", "TWM", "TWC"])
barpanels(sc_b, NOISE_MODELS, ["TWS_5f", "TWM_5f", "TWC_5f"],
          ["Espiral (TWS)", "Luas (TWM)", "Tabuleiro (TWC)"],
          "Análise de Sensibilidade B — 5 features informativas + ruído", "fig4_5features",
          ref=sc_2d, ref_ds=["TWS", "TWM", "TWC"], ref_label="2D (Cenário 1)",
          main_label="5D (3 atributos de ruído): média ± desvio-padrão")
for m in NOISE_MODELS:
    print(f"  {MODEL_LABEL.get(m, m):16s}", "  ".join(
        f"{d}: 2D={np.mean(sc_2d[m][d]) if sc_2d[m].get(d) else float('nan'):.3f} "
        f"5D={np.mean(sc_b[m][d + '_5f']) if sc_b[m].get(d + '_5f') else float('nan'):.3f}"
        for d in ["TWS", "TWM", "TWC"]))

# fig5 — Ablação C (MK5, complexidade)
MK_MODELS = ["StandardLSSVM", "ADMMNesterovLSSVM", "DualFISTA",
             "FSALSSVmOriginal", "IPLSSVmOriginal", "OppositeMapsOriginalLSSVM",
             "FTTransformer_softmax", "NystromLSSVMColnorm", "FTTransformerCURColnorm"]
sc_c = collect(ROOT / "results" / "ablation_c_mk5.json", ["MKE", "MKM", "MKH"])
barpanels(sc_c, MK_MODELS, ["MKE", "MKM", "MKH"],
          ["MKE (fácil)", "MKM (médio)", "MKH (difícil)"],
          "Análise de Sensibilidade C — MK5: complexidade de atributos", "fig5_mk5")

print("figs salvas em", OUT)

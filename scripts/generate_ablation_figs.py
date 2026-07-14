#!/usr/bin/env python3
"""Gera fig4 (ablação B — ruído/5f) e fig5 (ablação C — mk5/complexidade).

Lê os arquivos de ablação consolidados (CPU fiel + transformers do Kaggle):
    results/ablation_b_noise.json   — TWS_5f/TWM_5f/TWC_5f
    results/ablation_c_mk5.json     — MKE/MKM/MKH
Substitui a geração antiga (OLD/scripts/generate_report_figs.py, que lia
results antigos com os IP/FSA/OppMaps adaptados). Usa os nomes fiéis.
"""
import json
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
    "PruningLSSVM": "LSSVM-Prun", "OppositeMapsOriginalLSSVM": "LSSVM-OppM",
    "ADMMNesterovLSSVM": "LSSVM-ADMM", "ADMMElasticNet": "ADMM-ElasticNet",
    "FISTANesterov": "FISTA-Nesterov", "DualFISTA": "DualFISTA★",
    "NystromLSSVMColnorm": "Nyström-SVM★", "FTTransformerCURColnorm": "FT-CUR★",
    "FTTransformer_softmax": "FT-Softmax", "FTTransformer_topk": "FT-TopK",
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
    for r in json.loads(Path(path).read_text()):
        if r.get("status") != "ok":
            continue
        m = r.get("variant") or r.get("model")
        d = r["dataset"]
        if d in datasets and r.get("test_f1_macro") is not None:
            out[m][d].append(r["test_f1_macro"])
    return out


def barpanels(sc, models, ds_list, ds_labels, suptitle, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for col, ds in enumerate(ds_list):
        ax = axes[col]
        vals = [np.mean(sc[m][ds]) if sc[m].get(ds) else np.nan for m in models]
        stds = [np.std(sc[m][ds]) if sc[m].get(ds) else 0 for m in models]
        cols = [model_color(m) for m in models]
        ax.bar(range(len(models)), vals, color=cols, alpha=0.85, yerr=stds, capsize=3)
        ax.set_title(ds_labels[col], fontsize=10)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models],
                           rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel("F1-macro" if col == 0 else "")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / f"{fname}.pdf", dpi=150)
    plt.savefig(OUT / f"{fname}.png", dpi=150)
    plt.close()
    print(f"{fname} done")


# fig4 — Ablação B (5 features com ruído)
NOISE_MODELS = ["StandardLSSVM", "ADMMNesterovLSSVM", "DualFISTA",
                "FTTransformer_sparsemax", "FTTransformer_softmax",
                "NystromLSSVMColnorm", "FTTransformerCURColnorm"]
sc_b = collect(ROOT / "results" / "ablation_b_noise.json", ["TWS_5f", "TWM_5f", "TWC_5f"])
barpanels(sc_b, NOISE_MODELS, ["TWS_5f", "TWM_5f", "TWC_5f"],
          ["Espiral (5f)", "Luas (5f)", "Tabuleiro (5f)"],
          "Ablação B — 5 features informativas + ruído", "fig4_5features")

# fig5 — Ablação C (MK5, complexidade)
MK_MODELS = ["StandardLSSVM", "ADMMNesterovLSSVM", "DualFISTA",
             "FSALSSVmOriginal", "IPLSSVmOriginal", "OppositeMapsOriginalLSSVM",
             "FTTransformer_softmax", "NystromLSSVMColnorm", "FTTransformerCURColnorm"]
sc_c = collect(ROOT / "results" / "ablation_c_mk5.json", ["MKE", "MKM", "MKH"])
barpanels(sc_c, MK_MODELS, ["MKE", "MKM", "MKH"],
          ["MKE (fácil)", "MKM (médio)", "MKH (difícil)"],
          "Ablação C — MK5: complexidade de atributos", "fig5_mk5")

print("figs salvas em", OUT)

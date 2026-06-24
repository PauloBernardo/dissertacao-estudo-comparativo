"""Generate figures for the dissertation report.

Reads:
  results/tier1_gridcv.json           — 18 models × 9 real datasets (N≈400)
  results/ablation_a_scaling.json     — LSSVM × TWS/TWM/TWC N=2000
  results/ablation_a_transformers.json — Transformers × TWS/TWM/TWC N=2000

Outputs (results/report_figs/):
  fig1_f1_all_models.pdf/png
  fig2_sparsity_tradeoff.pdf/png
  fig3_scaling.pdf/png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "results" / "report_figs"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS_T1 = ["AUS", "BCW", "GCR", "HAB", "PID", "TWC", "TWM", "TWS", "VCP"]
SYNTH_DS    = ["TWS", "TWM", "TWC"]

MODEL_LABEL = {
    "StandardLSSVM":          "LSSVM-Std",
    "PCPLSSVm":               "LSSVM-PCP",
    "FSALSSVm":               "LSSVM-FSA",
    "IPLSSVm":                "LSSVM-IP",
    "PruningLSSVM":           "LSSVM-Prun",
    "OppositeMapsLSSVM":      "LSSVM-OppM",
    "ADMMNesterovLSSVM":      "LSSVM-ADMM",
    "ADMMElasticNet":         "LSSVM-ADMM-EN",
    "FISTANesterov":          "LSSVM-FISTA",
    "DualFISTA":              "DualFISTA★",
    "NystromLSSVMColnorm":    "Nystřöm-SVM★",
    "FTTransformerCURColnorm":"FT-CUR★",
    "FTTransformer_softmax":  "FT-Softmax",
    "FTTransformer_topk":     "FT-TopK",
    "FTTransformer_entmax":   "FT-Entmax",
    "FTTransformer_sparsemax":"FT-Sparsemax",
    "SAINTColnorm":           "SAINT",
    "XGBoost":                "XGBoost",
}

PROPOSED   = {"DualFISTA", "NystromLSSVMColnorm", "FTTransformerCURColnorm"}
ADMM_FAM   = {"ADMMNesterovLSSVM", "ADMMElasticNet", "FISTANesterov"}
LSSVM_BL   = {"StandardLSSVM", "PCPLSSVm", "FSALSSVm", "IPLSSVm", "PruningLSSVM", "OppositeMapsLSSVM"}
FT_BL      = {"FTTransformer_softmax", "FTTransformer_topk", "FTTransformer_entmax",
               "FTTransformer_sparsemax", "SAINTColnorm"}
GEN_BL     = {"XGBoost"}

COLOR_MAP = {
    "DualFISTA":              "#d62728",
    "NystromLSSVMColnorm":    "#2ca02c",
    "FTTransformerCURColnorm":"#ff7f0e",
    "XGBoost":                "#e377c2",
}


def model_color(m: str) -> str:
    if m in COLOR_MAP:
        return COLOR_MAP[m]
    if m in ADMM_FAM:
        return "#9467bd"
    if m in LSSVM_BL:
        return "#1f77b4"
    return "#888888"


def collect(records: list, datasets: list[str]) -> dict:
    """Return {model: {dataset: [f1, ...]}}."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("status") != "ok":
            continue
        m = r.get("variant") or r.get("model")
        d = r.get("dataset", "")
        if d in datasets:
            out[m][d].append(r.get("test_f1_macro", float("nan")))
    return out


def load(path: Path) -> list:
    return json.loads(path.read_text())


def main() -> None:
    t1      = load(ROOT / "results" / "tier1_gridcv.json")
    sc_lssvm = load(ROOT / "results" / "ablation_a_scaling.json")
    sc_tf    = load(ROOT / "results" / "ablation_a_transformers.json")
    sc_2k_all = sc_lssvm + sc_tf

    # ── Figure 1: F1-macro all models ────────────────────────────────────────
    sc = collect(t1, DATASETS_T1)

    model_stats: dict = {}
    for m, ds_map in sc.items():
        vals = [v for lst in ds_map.values() for v in lst if not np.isnan(v)]
        if vals:
            model_stats[m] = (np.mean(vals), np.std(vals))

    order  = sorted(model_stats, key=lambda m: -model_stats[m][0])
    labels = [MODEL_LABEL.get(m, m) for m in order]
    means  = [model_stats[m][0] for m in order]
    stds   = [model_stats[m][1] for m in order]
    colors = [model_color(m) for m in order]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(order)), means, xerr=stds, color=colors,
            align="center", alpha=0.85, capsize=3, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("F1-macro médio (30 sementes × 9 datasets)", fontsize=10)
    ax.set_title("Tier 1 — Desempenho Geral dos Modelos\n(★ = modelo proposto/investigado)", fontsize=11)
    ax.axvline(means[0], color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlim(0.55, 0.95)

    patches = [
        mpatches.Patch(color="#1f77b4", label="LSSVM baselines"),
        mpatches.Patch(color="#888888", label="FT-Transformer / SAINT"),
        mpatches.Patch(color="#9467bd", label="ADMM/FISTA family"),
        mpatches.Patch(color="#d62728", label="DualFISTA (proposto)"),
        mpatches.Patch(color="#2ca02c", label="Nyström-SVM (proposto)"),
        mpatches.Patch(color="#ff7f0e", label="FT-CUR (proposto)"),
        mpatches.Patch(color="#e377c2", label="XGBoost"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(OUT / "fig1_f1_all_models.pdf", dpi=150)
    fig.savefig(OUT / "fig1_f1_all_models.png", dpi=150)
    plt.close(fig)
    print("fig1 done")

    # ── Figure 2: Sparsity trade-off ─────────────────────────────────────────
    SPARSE_MODELS = ["ADMMNesterovLSSVM", "ADMMElasticNet", "FISTANesterov",
                     "DualFISTA", "FSALSSVm", "IPLSSVm", "PruningLSSVM",
                     "OppositeMapsLSSVM", "NystromLSSVMColnorm"]

    spar_stats: dict = {}
    for r in t1:
        if r.get("status") != "ok":
            continue
        m = r.get("variant") or r.get("model")
        if m not in SPARSE_MODELS:
            continue
        if r.get("dataset") not in DATASETS_T1:
            continue
        spar_stats.setdefault(m, {"f1": [], "spar": []})
        spar_stats[m]["f1"].append(r.get("test_f1_macro", float("nan")))
        spar_stats[m]["spar"].append((r.get("sparsity_ratio") or 0) * 100)

    fig, ax = plt.subplots(figsize=(8, 6))
    for m in SPARSE_MODELS:
        if m not in spar_stats:
            continue
        f1_mean = np.nanmean(spar_stats[m]["f1"])
        f1_std  = np.nanstd(spar_stats[m]["f1"])
        sp_mean = np.mean(spar_stats[m]["spar"])
        sp_std  = np.std(spar_stats[m]["spar"])
        col = model_color(m)
        lbl = MODEL_LABEL.get(m, m)
        mrk = "*" if m in PROPOSED else "o"
        ax.errorbar(sp_mean, f1_mean, xerr=sp_std, yerr=f1_std,
                    fmt=mrk, color=col, markersize=10 if m in PROPOSED else 7,
                    capsize=4, linewidth=1.2, zorder=5 if m in PROPOSED else 3)
        offset_y = 0.004 if m not in {"FISTANesterov", "ADMMElasticNet"} else -0.009
        ax.annotate(lbl, (sp_mean, f1_mean + offset_y), fontsize=7.5, ha="center")

    ax.set_xlabel("Esparsidade média (%)", fontsize=10)
    ax.set_ylabel("F1-macro médio", fontsize=10)
    ax.set_title("Trade-off Esparsidade × Desempenho\n(★ = modelos propostos)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 100)
    plt.tight_layout()
    fig.savefig(OUT / "fig2_sparsity_tradeoff.pdf", dpi=150)
    fig.savefig(OUT / "fig2_sparsity_tradeoff.png", dpi=150)
    plt.close(fig)
    print("fig2 done")

    # ── Figure 3: Scaling N=400 vs N=2000 ────────────────────────────────────
    sc400  = collect(t1, SYNTH_DS)
    sc2000 = collect(sc_2k_all, ["TWS_2k", "TWM_2k", "TWC_2k"])

    SCALE_MODELS = ["StandardLSSVM", "ADMMNesterovLSSVM", "DualFISTA",
                    "NystromLSSVMColnorm", "FTTransformer_sparsemax",
                    "FTTransformerCURColnorm", "SAINTColnorm"]
    ds_pairs = [("TWS", "TWS_2k"), ("TWM", "TWM_2k"), ("TWC", "TWC_2k")]
    ds_names  = ["TWS", "TWM", "TWC"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for col_idx, (ds400, ds2k) in enumerate(ds_pairs):
        ax = axes[col_idx]
        for i, m in enumerate(SCALE_MODELS):
            v400_list = sc400[m].get(ds400, [])
            v2k_list  = sc2000[m].get(ds2k, [])
            v400 = float(np.nanmean(v400_list)) if v400_list else float("nan")
            v2k  = float(np.nanmean(v2k_list))  if v2k_list  else float("nan")
            col = model_color(m)
            lbl = MODEL_LABEL.get(m, m)
            ax.bar(i - 0.2, v400, 0.35, color=col, alpha=0.45,
                   label=f"{lbl} N=400" if col_idx == 0 else "")
            ax.bar(i + 0.2, v2k,  0.35, color=col, alpha=0.95,
                   label=f"{lbl} N=2000" if col_idx == 0 else "")
        ax.set_title(f"Dataset: {ds_names[col_idx]}", fontsize=10)
        ax.set_xticks(range(len(SCALE_MODELS)))
        ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in SCALE_MODELS],
                           rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel("F1-macro" if col_idx == 0 else "")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Ablação A — Escalabilidade Amostral: N=400 (translúcido) vs N=2000 (sólido)", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "fig3_scaling.pdf", dpi=150)
    fig.savefig(OUT / "fig3_scaling.png", dpi=150)
    plt.close(fig)
    print("fig3 done")

    print(f"All figures saved to {OUT}")


if __name__ == "__main__":
    main()

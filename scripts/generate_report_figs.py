"""Generate figures for the dissertation report.

Reads:
  results/tier1_gridcv.json           — 18 models × 10 real datasets (N≈400)
  results/ablation_a_scaling.json     — LSSVM × TWS/TWM/TWC N=2000
  results/ablation_a_transformers.json — Transformers × TWS/TWM/TWC N=2000

Outputs (results/report_figs/):
  fig1_f1_all_models.pdf/png
  fig2_sparsity_tradeoff.pdf/png
  fig3_scaling.pdf/png
"""
import json
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))
from variant_aliases import canon  # noqa: E402
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

DATASETS_T1 = ["AI4I", "AUS", "BCW", "GCR", "HAB", "PID", "TWC", "TWM", "TWS", "VCP"]
SYNTH_DS    = ["TWS", "TWM", "TWC"]

MODEL_LABEL = {
    "StandardLSSVM":          "LSSVM-Std",
    "PCPLSSVm":               "LSSVM-PCP",
    "FSALSSVmOriginal":       "LSSVM-FSA",
    "IPLSSVmOriginal":        "LSSVM-IP",
    "PruningLSSVM":           "LSSVM-Pruning",
    "OppositeMapsOriginalLSSVM":"LSSVM-OppMaps",
    "ADMMNesterovLSSVM":      "ADMM-Nesterov",
    "ADMMElasticNet":         "ADMM-ElasticNet",
    "FISTANesterov":          "FISTA-Nesterov",
    "DualFISTA":              "DualFISTA★",
    "NystromLSSVMColnorm":    "Nyström-SVM★",
    "ADMMNystromLSSVM":       "ADMM-Nyström★",
    "FISTANystrom":           "FISTA-Nyström★",
    "FTTransformerCURColnorm":"FT-CUR★",
    "FTTransformer_softmax":  "FT-Softmax",
    "FTTransformer_topk":     "FT-Top-$k$",
    "FTTransformer_entmax":   "FT-Entmax",
    "FTTransformer_sparsemax":"FT-Sparsemax",
    "SAINTColnorm":           "SAINT",
    "XGBoost":                "XGBoost",
}

PROPOSED   = {"DualFISTA", "NystromLSSVMColnorm", "FTTransformerCURColnorm",
              "ADMMNystromLSSVM", "FISTANystrom"}
ADMM_FAM   = {"ADMMNesterovLSSVM", "ADMMElasticNet", "FISTANesterov"}
LSSVM_BL   = {"StandardLSSVM", "PCPLSSVm", "FSALSSVmOriginal", "IPLSSVmOriginal",
              "PruningLSSVM", "OppositeMapsOriginalLSSVM"}
FT_BL      = {"FTTransformer_softmax", "FTTransformer_topk", "FTTransformer_entmax",
               "FTTransformer_sparsemax", "SAINTColnorm"}
GEN_BL     = {"XGBoost"}

COLOR_MAP = {
    "DualFISTA":              "#d62728",
    "NystromLSSVMColnorm":    "#2ca02c",
    "FTTransformerCURColnorm":"#ff7f0e",
    "XGBoost":                "#e377c2",
    "ADMMNystromLSSVM":       "#8c564b",
    "FISTANystrom":           "#17becf",
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
    return canon(json.loads(path.read_text()))


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
    ax.set_xlabel("F1-macro médio (30 sementes × 10 datasets)", fontsize=10)
    ax.set_title("Cenário 1 — Desempenho Geral dos Modelos\n(★ = modelo proposto/investigado)", fontsize=11)
    ax.axvline(means[0], color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlim(0.55, 0.95)

    patches = [
        mpatches.Patch(color="#1f77b4", label="LSSVMs de referência"),
        mpatches.Patch(color="#888888", label="FT-Transformer / SAINT"),
        mpatches.Patch(color="#9467bd", label="Família ADMM/FISTA"),
        mpatches.Patch(color="#d62728", label="DualFISTA (proposto)"),
        mpatches.Patch(color="#2ca02c", label="Nyström-SVM (proposto)"),
        mpatches.Patch(color="#8c564b", label="ADMM-Nyström (proposto)"),
        mpatches.Patch(color="#17becf", label="FISTA-Nyström (proposto)"),
        mpatches.Patch(color="#ff7f0e", label="FT-CUR (proposto)"),
        mpatches.Patch(color="#e377c2", label="XGBoost"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(OUT / "fig1_f1_all_models.pdf", dpi=150)
    fig.savefig(OUT / "fig1_f1_all_models.png", dpi=150)
    plt.close(fig)
    print("fig1 done")

    # ── Figure 2: Sparsity trade-off ─────────────────────────────────────────
    SPARSE_MODELS = ["ADMMNesterovLSSVM", "ADMMElasticNet", "FISTANesterov",
                     "DualFISTA", "FSALSSVmOriginal", "IPLSSVmOriginal", "PruningLSSVM",
                     "OppositeMapsOriginalLSSVM", "NystromLSSVMColnorm",
                     "ADMMNystromLSSVM", "FISTANystrom", "PCPLSSVm"]

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

    # offsets manuais (dx, dy em pontos, ha) -- o aglomerado em 58-83% de
    # esparsidade tem 8 modelos muito proximos entre si (F1 0,82-0,843);
    # offset fixo nao separa os rotulos.
    LABEL_OFFSETS = {
        "ADMMNesterovLSSVM":          (-6, 6, "right"),
        "ADMMElasticNet":             (-6, -14, "right"),
        "FISTANesterov":              (0, -14, "center"),
        "DualFISTA":                  (0, 10, "center"),
        "PruningLSSVM":               (-6, -16, "right"),
        "IPLSSVmOriginal":            (0, 10, "center"),
        "FSALSSVmOriginal":           (10, 18, "left"),
        "PCPLSSVm":                   (-8, 10, "right"),
        "NystromLSSVMColnorm":        (8, 18, "left"),
        "FISTANystrom":               (-8, -16, "right"),
        "ADMMNystromLSSVM":           (8, -22, "left"),
        "OppositeMapsOriginalLSSVM":  (10, -30, "left"),
    }

    fig, ax = plt.subplots(figsize=(9, 6.5))
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
        dx, dy, ha = LABEL_OFFSETS.get(m, (0, 10, "center"))
        ax.annotate(lbl, (sp_mean, f1_mean), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, ha=ha)

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

    # Halteres horizontais: rótulos dos modelos na horizontal (legíveis na largura do
    # texto da tese, ~16 cm) e a variação N=400 -> N=2000 lida como o comprimento do traço.
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.0), sharey=True, sharex=True)
    ypos = np.arange(len(SCALE_MODELS))[::-1]
    for col_idx, (ds400, ds2k) in enumerate(ds_pairs):
        ax = axes[col_idx]
        for y, m in zip(ypos, SCALE_MODELS):
            v400_list = sc400[m].get(ds400, [])
            v2k_list  = sc2000[m].get(ds2k, [])
            v400 = float(np.nanmean(v400_list)) if v400_list else float("nan")
            v2k  = float(np.nanmean(v2k_list))  if v2k_list  else float("nan")
            col = model_color(m)
            ax.plot([v400, v2k], [y, y], color=col, lw=1.6, zorder=2)
            ax.scatter(v400, y, s=26, facecolor="white", edgecolor=col, lw=1.3, zorder=3,
                       label="$N=400$" if (col_idx == 0 and y == ypos[0]) else None)
            ax.scatter(v2k, y, s=26, color=col, zorder=4,
                       label="$N=2000$" if (col_idx == 0 and y == ypos[0]) else None)
        ax.set_title(ds_names[col_idx], fontsize=9)
        ax.set_yticks(ypos)
        ax.set_yticklabels([MODEL_LABEL.get(m, m) for m in SCALE_MODELS], fontsize=8.5)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("F1-macro", fontsize=8.5)
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_xlim(0.85, 1.008)   # mesma escala nos três painéis: o tamanho do ganho é comparável
    h_, l_ = axes[0].get_legend_handles_labels()
    leg = fig.legend(h_, l_, fontsize=8, loc="upper center", ncol=2, frameon=False,
                     bbox_to_anchor=(0.6, 1.0), handletextpad=0.3)
    for h in leg.legend_handles:
        h.set_edgecolor("0.3")
        if h.get_label() == "$N=2000$":
            h.set_facecolor("0.3")
    plt.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig3_scaling.pdf", dpi=150)
    fig.savefig(OUT / "fig3_scaling.png", dpi=150)
    plt.close(fig)
    print("fig3 done")

    print(f"All figures saved to {OUT}")


if __name__ == "__main__":
    main()

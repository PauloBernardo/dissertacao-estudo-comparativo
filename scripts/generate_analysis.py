#!/usr/bin/env python3
"""Generate all tables and figures for the dissertation.

Reads results/tier1_results.json and outputs:
  results/tables/  — LaTeX .tex files
  results/plots/   — PDF figures

Usage
-----
    python scripts/generate_analysis.py [--results results/tier1_results.json]
                                        [--metric f1_macro]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.tables import results_table, sparsity_table, wilcoxon_table, ranks_table
from src.analysis.plots import (
    cd_diagram, boxplots, sparsity_accuracy_scatter, training_time_barplot,
)
from src.metrics.statistical import (
    wilcoxon_pairwise, friedman_test, average_ranks, nemenyi_cd,
)

TABLES_DIR = ROOT / "results" / "tables"
PLOTS_DIR  = ROOT / "results" / "plots"

# Friendly display names for the paper
MODEL_LABELS: dict[str, str] = {
    "StandardLSSVM":          "LSSVM (Standard)",
    "PCPLSSVm":               "LSSVM-PCP",
    "FSALSSVm":               "LSSVM-FSA",
    "PruningLSSVM":           "LSSVM-Pruning",
    "IPLSSVm":                "LSSVM-IP",
    "OppositeMapsLSSVM":      "LSSVM-OppMaps",
    "ADMMNesterovLSSVM":      "LSSVM-ADMM",
    "FTTransformer_softmax":  "FT-Softmax",
    "FTTransformer_topk":     "FT-TopK",
    "FTTransformer_entmax":   "FT-Entmax",
    "FTTransformer_sparsemax":"FT-Sparsemax",
}


def load_results(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text())
    df = pd.DataFrame(raw)
    df = df[df["status"] == "ok"].copy()
    # Use model_variant to distinguish transformer attention types
    df["model"] = df["model_variant"].fillna(df["model"])
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    return df


def build_score_matrix(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Build (n_models, n_datasets) matrix of mean scores across seeds."""
    models = sorted(df["model"].unique())
    datasets = sorted(df["dataset"].unique())
    mat = np.full((len(models), len(datasets)), np.nan)
    for i, m in enumerate(models):
        for j, d in enumerate(datasets):
            sub = df[(df["model"] == m) & (df["dataset"] == d)][metric]
            if len(sub) > 0:
                mat[i, j] = sub.mean()
    return mat, models, datasets


def run(results_path: Path, metric: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {results_path} ...")
    df = load_results(results_path)
    print(f"  {len(df)} ok runs | {df['model'].nunique()} models | {df['dataset'].nunique()} datasets")

    # Use labelled names for display
    df_display = df.copy()
    df_display["model"] = df_display["model_label"]

    models_raw    = sorted(df["model"].unique())
    models_label  = [MODEL_LABELS.get(m, m) for m in models_raw]
    datasets      = sorted(df["dataset"].unique())

    # ── 1. Main results table (F1-macro) ──────────────────────────────────────
    print("Generating results table ...")
    tex = results_table(
        df_display.to_dict("records"),
        metric=metric,
        caption=f"Média $\\pm$ desvio padrão de F1-macro (30 sementes, 9 conjuntos de dados Tier 1).",
        label="tab:tier1_results",
    )
    (TABLES_DIR / "tier1_results.tex").write_text(tex)
    print(f"  Saved tables/tier1_results.tex")

    # ── 2. Accuracy table ─────────────────────────────────────────────────────
    tex_acc = results_table(
        df_display.to_dict("records"),
        metric="accuracy",
        caption="Média $\\pm$ desvio padrão de Acurácia (30 sementes, 9 conjuntos de dados Tier 1).",
        label="tab:tier1_accuracy",
    )
    (TABLES_DIR / "tier1_accuracy.tex").write_text(tex_acc)
    print(f"  Saved tables/tier1_accuracy.tex")

    # ── 3. Sparsity table ─────────────────────────────────────────────────────
    print("Generating sparsity table ...")
    tex_sp = sparsity_table(
        df_display.to_dict("records"),
        caption="Taxa de esparsidade média por modelo (todos os datasets e sementes).",
        label="tab:sparsity",
    )
    (TABLES_DIR / "sparsity.tex").write_text(tex_sp)
    print(f"  Saved tables/sparsity.tex")

    # ── 4. Statistical tests ──────────────────────────────────────────────────
    print("Running statistical tests ...")
    score_mat, m_order, d_order = build_score_matrix(df, metric)

    # Remove models with any NaN dataset (incomplete)
    valid_mask = ~np.isnan(score_mat).any(axis=1)
    score_mat_v = score_mat[valid_mask]
    models_v    = [models_raw[i] for i, v in enumerate(valid_mask) if v]
    labels_v    = [MODEL_LABELS.get(m, m) for m in models_v]

    # score_mat_v is (n_models, n_datasets); statistical functions expect (n_datasets, n_models)
    score_mat_T = score_mat_v.T

    # Friedman test
    friedman = friedman_test(score_mat_T)
    print(f"  Friedman: stat={friedman['statistic']:.3f}, p={friedman['pvalue']:.4f}")
    (TABLES_DIR / "friedman.txt").write_text(
        f"Friedman test\nStatistic: {friedman['statistic']:.4f}\np-value:   {friedman['pvalue']:.6f}\n"
    )

    # Average ranks
    ranks = average_ranks(score_mat_T)
    tex_ranks = ranks_table(
        ranks, labels_v,
        caption="Ranks médios de Friedman (menor = melhor).",
        label="tab:ranks",
    )
    (TABLES_DIR / "ranks.tex").write_text(tex_ranks)
    print(f"  Saved tables/ranks.tex")

    # Nemenyi CD
    cd = nemenyi_cd(len(models_v), len(d_order))
    print(f"  Nemenyi CD (α=0.05): {cd:.3f}")

    # Wilcoxon pairwise with Holm-Bonferroni correction
    # Use per-dataset mean scores as paired observations (n_datasets,)
    n_m = len(models_v)
    n_comparisons = n_m * (n_m - 1) // 2
    raw_pvalues: list[tuple[str, str, float]] = []
    for i in range(n_m):
        for j in range(i + 1, n_m):
            a = score_mat_T[:, i]
            b = score_mat_T[:, j]
            valid = ~(np.isnan(a) | np.isnan(b))
            if valid.sum() < 2:
                continue
            res = wilcoxon_pairwise(a[valid], b[valid])
            raw_pvalues.append((labels_v[i], labels_v[j], res["pvalue"]))

    # Holm-Bonferroni: sort ascending, multiply each by (n_comparisons - rank),
    # then enforce monotonicity with cumulative max so p_adj[i] >= p_adj[i-1].
    raw_pvalues.sort(key=lambda x: x[2])
    adjusted: list[float] = []
    running_max = 0.0
    for rank, (_, _, p_raw) in enumerate(raw_pvalues):
        p_adj = min(1.0, p_raw * (n_comparisons - rank))
        running_max = max(running_max, p_adj)
        adjusted.append(running_max)

    pmat = pd.DataFrame(np.nan, index=labels_v, columns=labels_v)
    pmat_raw = pd.DataFrame(np.nan, index=labels_v, columns=labels_v)
    for (li, lj, p_raw), p_adj in zip(raw_pvalues, adjusted):
        pmat.loc[li, lj] = p_adj
        pmat.loc[lj, li] = p_adj
        pmat_raw.loc[li, lj] = p_raw
        pmat_raw.loc[lj, li] = p_raw

    tex_wilcoxon = wilcoxon_table(
        pmat,
        caption=(
            "P-valores do teste de Wilcoxon signed-rank com correção de Holm-Bonferroni "
            r"($n_{\text{comp}}=" + str(n_comparisons) + r"$; "
            r"negrito = significativo em $\alpha=0{,}05$)."
        ),
        label="tab:wilcoxon",
    )
    tex_wilcoxon_raw = wilcoxon_table(
        pmat_raw,
        caption="P-valores brutos do teste de Wilcoxon signed-rank (sem correção; negrito = $p<0{,}05$).",
        label="tab:wilcoxon_raw",
    )
    (TABLES_DIR / "wilcoxon_raw.tex").write_text(tex_wilcoxon_raw)
    print(f"  Saved tables/wilcoxon_raw.tex")
    (TABLES_DIR / "wilcoxon.tex").write_text(tex_wilcoxon)
    print(f"  Saved tables/wilcoxon.tex")

    # ── 5. CD diagram ─────────────────────────────────────────────────────────
    print("Generating CD diagram ...")
    fig = cd_diagram(ranks, labels_v, cd,
                     title=f"Diagrama de Diferença Crítica — {metric}")
    fig.savefig(PLOTS_DIR / "cd_diagram.pdf", bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "cd_diagram.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved plots/cd_diagram.pdf")

    # ── 6. Box plots ──────────────────────────────────────────────────────────
    print("Generating box plots ...")
    fig = boxplots(df_display.to_dict("records"), metric=metric,
                   title=f"Distribuição de {metric} por dataset")
    fig.savefig(PLOTS_DIR / "boxplots.pdf", bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "boxplots.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved plots/boxplots.pdf")

    # ── 7. Sparsity vs accuracy scatter ──────────────────────────────────────
    print("Generating sparsity scatter ...")
    fig = sparsity_accuracy_scatter(df_display.to_dict("records"), metric=metric)
    fig.savefig(PLOTS_DIR / "sparsity_scatter.pdf", bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "sparsity_scatter.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved plots/sparsity_scatter.pdf")

    # ── 8. Training time bar plot ─────────────────────────────────────────────
    print("Generating training time plot ...")
    fig = training_time_barplot(df_display.to_dict("records"))
    fig.savefig(PLOTS_DIR / "training_time.pdf", bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "training_time.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved plots/training_time.pdf")

    # ── 9. Summary CSV ────────────────────────────────────────────────────────
    summary = (
        df.groupby("model")[metric]
        .agg(["mean", "std", "median", "min", "max"])
        .round(4)
        .sort_values("mean", ascending=False)
    )
    summary.index = [MODEL_LABELS.get(m, m) for m in summary.index]
    summary.to_csv(TABLES_DIR / "summary.csv")
    print(f"  Saved tables/summary.csv")

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n=== {metric} Summary ===")
    print(summary.to_string())
    print(f"\nFriedman p-value: {friedman['pvalue']:.6f}")
    print(f"Nemenyi CD (α=0.05): {cd:.3f}")
    print(f"\nAverage ranks:")
    for name, rank in sorted(zip(labels_v, ranks), key=lambda x: x[1]):
        print(f"  {name:30s} {rank:.3f}")

    print("\nDone. All files saved to results/tables/ and results/plots/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path,
                        default=ROOT / "results" / "tier1_results.json")
    parser.add_argument("--metric", default="f1_macro")
    args = parser.parse_args()
    run(args.results, args.metric)


if __name__ == "__main__":
    main()

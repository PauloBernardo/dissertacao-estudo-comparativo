"""Figures for the dissertation.

Generates:
    - CD (Critical Difference) diagram
    - Per-dataset box plots of F1 scores
    - Sparsity vs. accuracy scatter
    - Training time comparison
    - Attention weight heatmap (for FT-Transformer)

All functions return matplotlib Figure objects (not saved to disk).
Use fig.savefig(path, bbox_inches="tight") to persist.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# ── CD diagram ────────────────────────────────────────────────────────────────

def cd_diagram(
    ranks: np.ndarray,
    model_names: list[str],
    cd: float,
    title: str = "Critical Difference Diagram",
) -> plt.Figure:
    """Draw a Demsar-style CD diagram.

    Parameters
    ----------
    ranks : (n_models,) average Friedman ranks
    model_names : model labels in the same order as ranks
    cd : critical difference threshold
    title : plot title
    """
    order = np.argsort(ranks)
    sorted_ranks = ranks[order]
    sorted_names = [model_names[i] for i in order]
    n = len(sorted_ranks)

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * n), 3))
    ax.set_xlim(sorted_ranks[0] - 0.5, sorted_ranks[-1] + 0.5)
    ax.set_ylim(-1, 1.5)
    ax.axis("off")

    y_line = 0.8
    ax.hlines(y_line, sorted_ranks[0], sorted_ranks[-1], colors="black", linewidths=1.5)

    for rank, name in zip(sorted_ranks, sorted_names):
        ax.plot(rank, y_line, "o", color="black", markersize=5, zorder=3)
        # Alternate labels above/below
        idx = list(sorted_ranks).index(rank)
        y_text = y_line + 0.35 if idx % 2 == 0 else y_line - 0.55
        ax.text(rank, y_text, name, ha="center", va="bottom" if idx % 2 == 0 else "top",
                fontsize=8, rotation=45)

    # Draw CD bar
    mid = (sorted_ranks[0] + sorted_ranks[-1]) / 2
    ax.annotate(
        "", xy=(mid + cd / 2, y_line + 0.2),
        xytext=(mid - cd / 2, y_line + 0.2),
        arrowprops=dict(arrowstyle="|-|", color="red", lw=1.5),
    )
    ax.text(mid, y_line + 0.30, f"CD={cd:.2f}", ha="center", color="red", fontsize=9)

    ax.set_title(title, pad=10)
    fig.tight_layout()
    return fig


# ── Box plots ─────────────────────────────────────────────────────────────────

def boxplots(
    results: list[dict],
    metric: str = "f1_macro",
    model_col: str = "model",
    dataset_col: str = "dataset",
    title: str = "",
) -> plt.Figure:
    """Box plots of metric per model, grouped by dataset.

    One subplot per dataset.
    """
    df = pd.DataFrame(results)
    if df.empty or metric not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    datasets = sorted(df[dataset_col].unique())
    models = sorted(df[model_col].unique())
    n_datasets = len(datasets)
    ncols = min(3, n_datasets)
    nrows = (n_datasets + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                              sharey=True, squeeze=False)

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    color_map = dict(zip(models, colors))

    for i, dataset in enumerate(datasets):
        ax = axes[i // ncols][i % ncols]
        sub = df[df[dataset_col] == dataset]
        data = [sub[sub[model_col] == m][metric].dropna().values for m in models]
        bp = ax.boxplot(data, patch_artist=True, notch=False, widths=0.6)
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(color_map[m])
            patch.set_alpha(0.7)
        ax.set_title(dataset, fontsize=10)
        ax.set_xticks([])
        ax.set_ylabel(metric if i % ncols == 0 else "")

    # Turn off empty subplots
    for j in range(n_datasets, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    # Legend
    handles = [mpatches.Patch(facecolor=color_map[m], label=m, alpha=0.7) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(models)),
               bbox_to_anchor=(0.5, 0.0), fontsize=8)

    fig.suptitle(title or f"{metric} per dataset", y=1.01)
    fig.tight_layout()
    return fig


# ── Sparsity vs. accuracy scatter ────────────────────────────────────────────

def sparsity_accuracy_scatter(
    results: list[dict],
    metric: str = "f1_macro",
    sparsity_col: str | None = None,
    model_col: str = "model",
) -> plt.Figure:
    """Scatter plot: sparsity ratio (x) vs. metric (y) per run, coloured by model."""
    df = pd.DataFrame(results)

    if sparsity_col is None:
        for col in ("sparsity_ratio", "mean_zero_fraction"):
            if col in df.columns:
                sparsity_col = col
                break
    if sparsity_col is None or sparsity_col not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No sparsity column found", ha="center")
        return fig

    models = sorted(df[model_col].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    color_map = dict(zip(models, colors))

    fig, ax = plt.subplots(figsize=(7, 5))
    for model in models:
        sub = df[df[model_col] == model]
        ax.scatter(sub[sparsity_col], sub[metric],
                   color=color_map[model], label=model, alpha=0.5, s=20)

    ax.set_xlabel("Sparsity ratio")
    ax.set_ylabel(metric)
    ax.set_title("Sparsity vs. Performance")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


# ── Training time comparison ──────────────────────────────────────────────────

def training_time_barplot(
    results: list[dict],
    model_col: str = "model",
    dataset_col: str = "dataset",
    time_col: str = "train_time_s",
) -> plt.Figure:
    """Horizontal bar chart: median train time per model (log scale)."""
    df = pd.DataFrame(results)
    if df.empty or time_col not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No timing data", ha="center")
        return fig

    med = df.groupby(model_col)[time_col].median().sort_values()
    q1 = df.groupby(model_col)[time_col].quantile(0.25).reindex(med.index)
    q3 = df.groupby(model_col)[time_col].quantile(0.75).reindex(med.index)

    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(med))))
    y = np.arange(len(med))
    ax.barh(y, med.values, color="steelblue", alpha=0.8)
    ax.errorbar(med.values, y, xerr=[med.values - q1.values, q3.values - med.values],
                fmt="none", color="black", capsize=3, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(med.index.tolist())
    ax.set_xscale("log")
    ax.set_xlabel("Train time (s) — log scale")
    ax.set_title("Median training time per model (IQR bars)")
    fig.tight_layout()
    return fig


# ── Attention heatmap ─────────────────────────────────────────────────────────

def attention_heatmap(
    attn_weights: np.ndarray,
    feature_names: list[str] | None = None,
    layer: int = 0,
    head: int = 0,
    title: str = "Attention Weights",
) -> plt.Figure:
    """Heatmap of FT-Transformer attention weights.

    Parameters
    ----------
    attn_weights : (n_layers, B, H, L, L) or (B, H, L, L) array
    feature_names : token labels (excluding CLS); if None, uses Feature-i
    layer, head : which layer/head to plot
    """
    if attn_weights.ndim == 5:
        w = attn_weights[layer, 0, head]
    elif attn_weights.ndim == 4:
        w = attn_weights[0, head]
    else:
        w = attn_weights

    L = w.shape[-1]
    n_features = L - 1
    labels = ["[CLS]"] + (feature_names or [f"F{i}" for i in range(n_features)])

    fig, ax = plt.subplots(figsize=(max(4, 0.4 * L), max(4, 0.4 * L)))
    im = ax.imshow(w, cmap="Blues", vmin=0, vmax=w.max())
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

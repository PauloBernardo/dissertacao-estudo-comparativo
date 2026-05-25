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
    """Draw a Demsar (2006) style CD diagram.

    Models are split left/right of the axis; lines connect labels to rank
    positions. Horizontal bars below the axis mark groups of models that are
    NOT significantly different (rank difference < CD).

    Parameters
    ----------
    ranks : (n_models,) average Friedman ranks (lower = better)
    model_names : labels in the same order as ranks
    cd : critical difference threshold
    title : plot title
    """
    order = np.argsort(ranks)
    sorted_ranks = ranks[order]
    sorted_names = [model_names[i] for i in order]
    n = len(sorted_ranks)

    # Split: left side = better half, right side = worse half
    n_left  = (n + 1) // 2
    n_right = n - n_left

    left_names  = sorted_names[:n_left]
    left_ranks  = sorted_ranks[:n_left]
    right_names = sorted_names[n_left:]
    right_ranks = sorted_ranks[n_left:]

    # Layout constants — scale axis so each rank unit = 1 inch
    rank_min    = sorted_ranks[0]
    rank_max    = sorted_ranks[-1]
    axis_span   = rank_max - rank_min or 1.0
    label_col_w = 2.8             # inches for label columns (each side)
    axis_w      = max(axis_span * 0.95, 5.0)   # axis width in inches (≥5)
    scale       = axis_w / axis_span            # inches per rank unit

    fig_w = label_col_w * 2 + axis_w
    fig_h = max(4.0, 0.55 * n_left + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    def rank_to_x(r: float) -> float:
        return label_col_w + (r - rank_min) * scale

    # y positions (normalised 0-1 in data coords → use figure inches via transform)
    y_axis      = 0.72
    y_label_top = 0.92
    row_h       = max(0.07, (y_label_top - 0.18) / max(n_left, 1))

    ax.set_xlim(0, fig_w)
    ax.set_ylim(-0.40, 1.15)

    # ── Axis line ─────────────────────────────────────────────────────────────
    ax.hlines(y_axis, rank_to_x(rank_min), rank_to_x(rank_max),
              colors="black", linewidths=1.8, zorder=2)

    # Tick marks
    for r in sorted_ranks:
        ax.vlines(rank_to_x(r), y_axis - 0.03, y_axis + 0.03,
                  colors="black", linewidths=1.2, zorder=3)

    # Rank labels on axis
    for r in sorted_ranks:
        ax.text(rank_to_x(r), y_axis + 0.05, f"{r:.1f}",
                ha="center", va="bottom", fontsize=7.5, color="#444444")

    # ── Left side labels (better models) ──────────────────────────────────────
    for i, (name, r) in enumerate(zip(left_names, left_ranks)):
        y_text = y_label_top - i * row_h
        x_tick = rank_to_x(r)
        # Horizontal line from label to axis tick
        ax.plot([0.15, x_tick], [y_text, y_axis],
                color="#555555", linewidth=0.8, zorder=1)
        ax.text(0.10, y_text, name, ha="right", va="center", fontsize=9)

    # ── Right side labels (worse models) ─────────────────────────────────────
    x_right_edge = fig_w - 0.10
    for i, (name, r) in enumerate(zip(right_names, right_ranks)):
        y_text = y_label_top - i * row_h
        x_tick = rank_to_x(r)
        ax.plot([x_tick, x_right_edge - 0.05], [y_axis, y_text],
                color="#555555", linewidth=0.8, zorder=1)
        ax.text(x_right_edge, y_text, name, ha="left", va="center", fontsize=9)

    # ── CD bar (top-left corner) ───────────────────────────────────────────────
    cd_y   = y_axis + 0.28
    cd_x0  = rank_to_x(rank_min)
    cd_x1  = cd_x0 + cd * scale
    ax.hlines(cd_y, cd_x0, cd_x1, colors="#cc0000", linewidths=2.5)
    ax.vlines([cd_x0, cd_x1], cd_y - 0.025, cd_y + 0.025,
              colors="#cc0000", linewidths=2.0)
    ax.text((cd_x0 + cd_x1) / 2, cd_y + 0.04, f"CD = {cd:.2f}",
            ha="center", va="bottom", fontsize=9, color="#cc0000", fontweight="bold")

    # ── Clique bars (groups not significantly different) ──────────────────────
    # For each model i, find the rightmost model j where rank_j - rank_i < cd
    # Draw a bar from rank_i to rank_j below the axis
    bar_y_start = y_axis - 0.08
    bar_step    = 0.055
    drawn: list[tuple[float, float]] = []

    for i in range(n):
        j = i
        while j + 1 < n and sorted_ranks[j + 1] - sorted_ranks[i] < cd:
            j += 1
        if j > i:
            seg = (sorted_ranks[i], sorted_ranks[j])
            # Avoid drawing duplicate or fully contained segments
            if not any(s[0] <= seg[0] and s[1] >= seg[1] for s in drawn):
                # Stack bars so they don't overlap
                level = sum(
                    1 for s in drawn
                    if not (s[1] < seg[0] or s[0] > seg[1])
                )
                y_bar = bar_y_start - level * bar_step
                ax.hlines(y_bar, rank_to_x(seg[0]), rank_to_x(seg[1]),
                          colors="black", linewidths=3.5, zorder=4)
                drawn.append(seg)

    ax.set_title(title, fontsize=11, pad=6)
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

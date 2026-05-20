"""LaTeX table generation for dissertation results.

Generates:
    - Main results table (models × datasets, mean ± std of F1-macro)
    - Sparsity summary table
    - Statistical comparison table (Wilcoxon p-values)
    - CD diagram data (average ranks)
"""

from __future__ import annotations

import textwrap
from typing import Any

import numpy as np
import pandas as pd


def _fmt(mean: float, std: float, bold: bool = False, decimals: int = 3) -> str:
    """Format mean ± std for LaTeX cells."""
    fmt = f"{{:.{decimals}f}}"
    s = fmt.format(mean) + r" $\pm$ " + fmt.format(std)
    return r"\textbf{" + s + r"}" if bold else s


def results_table(
    results: list[dict],
    metric: str = "f1_macro",
    model_col: str = "model",
    dataset_col: str = "dataset",
    caption: str = "",
    label: str = "tab:results",
    highlight_best: bool = True,
) -> str:
    """Generate the main results LaTeX table.

    Parameters
    ----------
    results : list of per-run result dicts (each has model, dataset, metric)
    metric : column to summarise
    model_col, dataset_col : column names
    caption : LaTeX caption string
    label : LaTeX label
    highlight_best : bold the best model per dataset

    Returns
    -------
    str : complete LaTeX table environment
    """
    df = pd.DataFrame(results)
    if df.empty or metric not in df.columns:
        return f"% No data for metric={metric!r}\n"

    # Aggregate: mean and std across seeds per (model, dataset)
    agg = df.groupby([model_col, dataset_col])[metric].agg(["mean", "std"]).reset_index()
    agg["std"] = agg["std"].fillna(0.0)

    models = sorted(agg[model_col].unique())
    datasets = sorted(agg[dataset_col].unique())

    # Build a (model, dataset) → (mean, std) lookup
    lookup: dict[tuple, tuple] = {}
    for _, row in agg.iterrows():
        lookup[(row[model_col], row[dataset_col])] = (row["mean"], row["std"])

    # Best model per dataset
    best: dict[str, str] = {}
    if highlight_best:
        for d in datasets:
            best_mean = -np.inf
            best_model = ""
            for m in models:
                mean, _ = lookup.get((m, d), (np.nan, np.nan))
                if not np.isnan(mean) and mean > best_mean:
                    best_mean = mean
                    best_model = m
            best[d] = best_model

    n_datasets = len(datasets)
    col_spec = "l" + "r" * n_datasets
    header = " & ".join(["Model"] + [d.replace("_", r"\_") for d in datasets])

    rows = []
    for m in models:
        cells = [m.replace("_", r"\_")]
        for d in datasets:
            mean, std = lookup.get((m, d), (np.nan, 0.0))
            if np.isnan(mean):
                cells.append("--")
            else:
                is_best = highlight_best and best.get(d) == m
                cells.append(_fmt(mean, std, bold=is_best))
        rows.append(" & ".join(cells) + r" \\")

    body = "\n        ".join(rows)
    cap = caption or f"Mean $\\pm$ std of {metric} across 30 seeds."

    table = textwrap.dedent(rf"""
        \begin{{table}}[ht]
          \centering
          \caption{{{cap}}}
          \label{{{label}}}
          \resizebox{{\textwidth}}{{!}}{{%
          \begin{{tabular}}{{{col_spec}}}
            \toprule
            {header} \\
            \midrule
            {body}
            \bottomrule
          \end{{tabular}}}}
        \end{{table}}
    """).strip()
    return table


def sparsity_table(
    results: list[dict],
    model_col: str = "model",
    caption: str = "",
    label: str = "tab:sparsity",
) -> str:
    """Generate a sparsity summary table (mean sparsity ratio per model)."""
    df = pd.DataFrame(results)
    if df.empty:
        return "% No sparsity data\n"

    # Collect sparsity columns
    sparsity_col = None
    for col in ("sparsity_ratio", "mean_zero_fraction"):
        if col in df.columns:
            sparsity_col = col
            break
    if sparsity_col is None:
        return f"% No sparsity column found\n"

    agg = df.groupby(model_col)[sparsity_col].agg(["mean", "std"]).reset_index()
    agg["std"] = agg["std"].fillna(0.0)

    rows = []
    for _, row in agg.iterrows():
        name = str(row[model_col]).replace("_", r"\_")
        rows.append(f"        {name} & {_fmt(row['mean'], row['std'])} \\\\")
    body = "\n".join(rows)

    cap = caption or "Mean sparsity ratio across all datasets and seeds."
    table = textwrap.dedent(rf"""
        \begin{{table}}[ht]
          \centering
          \caption{{{cap}}}
          \label{{{label}}}
          \begin{{tabular}}{{lr}}
            \toprule
            Model & Sparsity ratio \\
            \midrule
{body}
            \bottomrule
          \end{{tabular}}
        \end{{table}}
    """).strip()
    return table


def wilcoxon_table(
    pvalue_matrix: pd.DataFrame,
    caption: str = "",
    label: str = "tab:wilcoxon",
) -> str:
    """Format a symmetric Wilcoxon p-value matrix as a LaTeX table.

    Parameters
    ----------
    pvalue_matrix : DataFrame with model names as index and columns
    """
    models = list(pvalue_matrix.index)
    col_spec = "l" + "r" * len(models)
    header = " & ".join([""] + [m.replace("_", r"\_") for m in models])

    rows = []
    for m_row in models:
        cells = [m_row.replace("_", r"\_")]
        for m_col in models:
            if m_row == m_col:
                cells.append("--")
            else:
                p = pvalue_matrix.loc[m_row, m_col]
                if np.isnan(p):
                    cells.append("--")
                elif p < 0.001:
                    cells.append(r"$<$0.001")
                elif p < 0.05:
                    cells.append(r"\textbf{" + f"{p:.3f}" + r"}")
                else:
                    cells.append(f"{p:.3f}")
        rows.append(" & ".join(cells) + r" \\")

    body = "\n        ".join(rows)
    cap = caption or "Wilcoxon signed-rank test p-values (bold = significant at $\\alpha=0.05$)."

    table = textwrap.dedent(rf"""
        \begin{{table}}[ht]
          \centering
          \caption{{{cap}}}
          \label{{{label}}}
          \resizebox{{\textwidth}}{{!}}{{%
          \begin{{tabular}}{{{col_spec}}}
            \toprule
            {header} \\
            \midrule
            {body}
            \bottomrule
          \end{{tabular}}}}
        \end{{table}}
    """).strip()
    return table


def ranks_table(
    ranks: np.ndarray,
    model_names: list[str],
    caption: str = "",
    label: str = "tab:ranks",
) -> str:
    """Format average Friedman ranks as a LaTeX table (sorted best→worst)."""
    order = np.argsort(ranks)
    rows = []
    for rank_pos, idx in enumerate(order, start=1):
        name = model_names[idx].replace("_", r"\_")
        rows.append(f"        {rank_pos} & {name} & {ranks[idx]:.3f} \\\\")
    body = "\n".join(rows)

    cap = caption or "Average Friedman ranks (lower = better)."
    table = textwrap.dedent(rf"""
        \begin{{table}}[ht]
          \centering
          \caption{{{cap}}}
          \label{{{label}}}
          \begin{{tabular}}{{clr}}
            \toprule
            Rank & Model & Avg.\ rank \\
            \midrule
{body}
            \bottomrule
          \end{{tabular}}
        \end{{table}}
    """).strip()
    return table

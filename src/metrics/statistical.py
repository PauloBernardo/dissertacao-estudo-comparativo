"""Statistical comparison utilities for the experiment results.

Implements:
    - Wilcoxon signed-rank test (pairwise)
    - Friedman test (multi-model)
    - Nemenyi post-hoc test via critical difference (CD diagram data)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from numpy.typing import NDArray


def wilcoxon_pairwise(
    scores_a: NDArray,
    scores_b: NDArray,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Wilcoxon signed-rank test between two vectors of per-dataset scores.

    Parameters
    ----------
    scores_a, scores_b : arrays of length n_datasets
    alternative : "two-sided", "greater", or "less"

    Returns
    -------
    dict with keys: statistic, pvalue
    """
    stat, pvalue = stats.wilcoxon(scores_a, scores_b, alternative=alternative)
    return {"statistic": float(stat), "pvalue": float(pvalue)}


def friedman_test(score_matrix: NDArray) -> dict[str, float]:
    """Friedman test across models.

    Parameters
    ----------
    score_matrix : (n_datasets, n_models) array of per-dataset scores

    Returns
    -------
    dict with keys: statistic, pvalue
    """
    stat, pvalue = stats.friedmanchisquare(*[score_matrix[:, j]
                                              for j in range(score_matrix.shape[1])])
    return {"statistic": float(stat), "pvalue": float(pvalue)}


def average_ranks(score_matrix: NDArray) -> NDArray:
    """Compute average ranks (lower rank = better score).

    Parameters
    ----------
    score_matrix : (n_datasets, n_models) — higher score is better

    Returns
    -------
    ranks : (n_models,) average ranks across datasets
    """
    n_datasets, n_models = score_matrix.shape
    # Rank within each dataset (descending: best → rank 1)
    ranks = np.zeros_like(score_matrix)
    for i in range(n_datasets):
        # scipy.stats.rankdata ranks ascending; negate for descending
        ranks[i] = stats.rankdata(-score_matrix[i])
    return ranks.mean(axis=0)


def nemenyi_cd(n_models: int, n_datasets: int, alpha: float = 0.05) -> float:
    """Critical difference for the Nemenyi post-hoc test.

    Uses the Studentised range distribution approximation.

    Parameters
    ----------
    n_models : number of classifiers compared
    n_datasets : number of datasets
    alpha : significance level (0.05 or 0.10)

    Returns
    -------
    cd : critical difference value
    """
    # q_alpha values from standard table (alpha=0.05)
    # Demsar (2006) Table 5, k=2..10
    q_alpha_table = {
        0.05: [np.nan, np.nan, 1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164],
        0.10: [np.nan, np.nan, 1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920],
    }
    if alpha not in q_alpha_table:
        raise ValueError(f"alpha must be 0.05 or 0.10, got {alpha}")
    table = q_alpha_table[alpha]
    if n_models > len(table) - 1:
        raise ValueError(f"n_models={n_models} exceeds table size (max {len(table)-1})")
    q = table[n_models]
    cd = q * np.sqrt(n_models * (n_models + 1) / (6.0 * n_datasets))
    return float(cd)


def summary_table(
    results: list[dict],
    metric: str = "f1_macro",
    model_col: str = "model",
    dataset_col: str = "dataset",
) -> pd.DataFrame:
    """Pivot results list into a (dataset × model) summary table.

    Parameters
    ----------
    results : list of dicts, each with at least model_col, dataset_col, metric
    metric : column to pivot as values
    model_col, dataset_col : column names for model and dataset identifiers

    Returns
    -------
    DataFrame of shape (n_datasets, n_models)
    """
    df = pd.DataFrame(results)
    return df.pivot_table(index=dataset_col, columns=model_col, values=metric,
                          aggfunc="mean")

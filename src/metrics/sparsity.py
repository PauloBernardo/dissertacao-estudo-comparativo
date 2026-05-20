"""Sparsity metrics for LSSVM and Transformer models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def lssvm_sparsity(model) -> dict[str, float]:
    """Extract sparsity metrics from a fitted LSSVM model.

    Parameters
    ----------
    model : fitted BaseLSSVM subclass

    Returns
    -------
    dict with keys: n_support_vectors, sparsity_ratio, n_total_samples
    """
    return {
        "n_support_vectors": int(model.n_support_),
        "sparsity_ratio": float(model.sparsity_ratio_),
        "n_total_samples": int(model.n_samples_fit_),
    }


def transformer_sparsity(model) -> dict[str, float]:
    """Extract sparsity metrics from a fitted FTTransformer model.

    Calls model.attention_sparsity(), which reads the last recorded
    attention weights. Requires a forward pass to have been run first.

    Parameters
    ----------
    model : fitted FTTransformer

    Returns
    -------
    dict with keys: mean_zero_fraction, mean_entropy, effective_n_tokens
    """
    return model.attention_sparsity()


def alpha_vector_sparsity(alpha: NDArray, threshold: float = 1e-6) -> dict[str, float]:
    """Sparsity of an arbitrary alpha/weight vector.

    Parameters
    ----------
    alpha : coefficient vector
    threshold : values below this are treated as zero

    Returns
    -------
    dict with keys: n_nonzero, n_total, sparsity_ratio, l0_norm
    """
    n_total = len(alpha)
    n_nonzero = int(np.sum(np.abs(alpha) > threshold))
    return {
        "n_nonzero": n_nonzero,
        "n_total": n_total,
        "sparsity_ratio": 1.0 - n_nonzero / max(n_total, 1),
        "l0_norm": float(n_nonzero),
    }

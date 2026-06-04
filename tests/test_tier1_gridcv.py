"""Structural tests for the Tier 1 GridSearchCV protocol."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone

from scripts.run_tier1_gridcv import _collect_sparsity_metrics
from src.experiments.runner import _build_model
from src.tuning.grids import GRIDS


def test_all_grid_variants_are_sklearn_compatible():
    """Every Tier 1 variant must be cloneable and accept its grid params.

    `scripts/run_tier1_gridcv.py` relies on sklearn Pipeline + GridSearchCV,
    so each estimator must expose `get_params`/`set_params` through the
    BaseEstimator contract.
    """
    for variant, cfg in GRIDS.items():
        estimator, _ = _build_model(cfg["model_name"], dict(cfg["fixed"]), label_format="signed")
        clone(estimator)
        params = estimator.get_params()

        for name in cfg["fixed"]:
            assert name in params, f"{variant}: missing fixed param {name!r}"

        for name in cfg["grid"]:
            assert name in params, f"{variant}: missing grid param {name!r}"


def test_collect_sparsity_metrics_includes_transformer_attention_metrics():
    estimator, _ = _build_model(
        "FTTransformer",
        {
            "embedding_dim": 16,
            "num_blocks": 1,
            "num_heads": 2,
            "max_epochs": 2,
            "batch_size": 32,
            "val_fraction": 0.0,
            "attention_type": "topk",
            "topk_ratio": 0.5,
            "random_state": 0,
        },
        label_format="binary",
    )

    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 6))
    y = rng.integers(0, 2, size=64)
    estimator.fit(X, y)

    metrics = _collect_sparsity_metrics(estimator, X[:16])
    assert "mean_zero_fraction" in metrics
    assert "mean_entropy" in metrics
    assert "effective_n_tokens" in metrics

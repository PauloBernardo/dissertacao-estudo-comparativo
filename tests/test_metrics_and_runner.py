"""Tests for metrics modules and the experiment runner."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.performance import compute_performance
from src.metrics.sparsity import alpha_vector_sparsity
from src.metrics.efficiency import efficiency_metrics, measure_fit_time
from src.metrics.statistical import (
    wilcoxon_pairwise, friedman_test, average_ranks, nemenyi_cd, summary_table
)
from src.experiments.runner import run_single_experiment


# ── Performance metrics ────────────────────────────────────────────────────────

class TestComputePerformance:
    def _perfect(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        return y, y.copy()

    def test_perfect_accuracy(self):
        y, yhat = self._perfect()
        m = compute_performance(y, yhat)
        assert m["accuracy"] == 1.0
        assert m["f1_macro"] == 1.0
        assert m["mcc"] == 1.0

    def test_returns_required_keys(self):
        y = np.array([0, 1, 0, 1])
        m = compute_performance(y, y)
        assert {"accuracy", "f1_macro", "f1_binary", "mcc"}.issubset(m.keys())

    def test_proba_adds_auc_and_ap(self):
        y = np.array([0, 0, 1, 1])
        yhat = y.copy()
        proba = np.column_stack([1 - y, y]).astype(float)
        m = compute_performance(y, yhat, y_proba=proba)
        assert "auc_roc" in m
        assert "avg_precision" in m
        assert m["auc_roc"] == 1.0

    def test_random_prediction_below_perfect(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 100)
        yhat = rng.integers(0, 2, 100)
        m = compute_performance(y, yhat)
        assert m["accuracy"] < 1.0


# ── Sparsity metrics ───────────────────────────────────────────────────────────

class TestAlphaVectorSparsity:
    def test_all_zeros(self):
        m = alpha_vector_sparsity(np.zeros(10))
        assert m["n_nonzero"] == 0
        assert m["sparsity_ratio"] == 1.0

    def test_all_nonzero(self):
        m = alpha_vector_sparsity(np.ones(10))
        assert m["n_nonzero"] == 10
        assert m["sparsity_ratio"] == 0.0

    def test_half_sparse(self):
        a = np.array([1.0, 0.0, 1.0, 0.0])
        m = alpha_vector_sparsity(a)
        assert m["n_nonzero"] == 2
        assert m["sparsity_ratio"] == 0.5

    def test_custom_threshold(self):
        a = np.array([0.1, 0.5, 0.0, 0.05])
        m = alpha_vector_sparsity(a, threshold=0.2)
        assert m["n_nonzero"] == 1  # only 0.5 > 0.2


# ── Efficiency metrics ─────────────────────────────────────────────────────────

class TestEfficiencyMetrics:
    def test_keys(self):
        m = efficiency_metrics(1.0, 0.1, 1000, 200)
        assert {"train_time_s", "predict_time_s",
                "train_time_per_sample_ms", "predict_time_per_sample_ms"}.issubset(m.keys())

    def test_per_sample_ms(self):
        m = efficiency_metrics(1.0, 0.2, 1000, 200)
        assert abs(m["train_time_per_sample_ms"] - 1.0) < 1e-9
        assert abs(m["predict_time_per_sample_ms"] - 1.0) < 1e-9

    def test_measure_fit_time_returns_model_and_float(self):
        from sklearn.linear_model import LogisticRegression
        X = np.random.randn(50, 4)
        y = np.random.randint(0, 2, 50)
        model, t = measure_fit_time(LogisticRegression(max_iter=200), X, y)
        assert hasattr(model, "coef_")
        assert isinstance(t, float) and t >= 0


# ── Statistical tests ──────────────────────────────────────────────────────────

class TestStatisticalMetrics:
    def _sample_scores(self, n=12, seed=0):
        rng = np.random.default_rng(seed)
        a = rng.uniform(0.7, 0.9, n)
        b = rng.uniform(0.6, 0.8, n)
        return a, b

    def test_wilcoxon_keys(self):
        a, b = self._sample_scores()
        result = wilcoxon_pairwise(a, b)
        assert {"statistic", "pvalue"}.issubset(result.keys())

    def test_wilcoxon_identical_gives_pvalue_one(self):
        # Identical arrays → all differences are zero → p-value is 1.0 (or NaN)
        a = np.array([0.8, 0.7, 0.9, 0.85, 0.75, 0.82] * 2, dtype=float)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = wilcoxon_pairwise(a, a)
        # p-value should be 1.0 or NaN (no differences)
        assert result["pvalue"] >= 1.0 or np.isnan(result["pvalue"])

    def test_friedman_keys(self):
        mat = np.random.default_rng(0).uniform(0, 1, (12, 4))
        result = friedman_test(mat)
        assert {"statistic", "pvalue"}.issubset(result.keys())

    def test_average_ranks_shape(self):
        mat = np.array([[0.9, 0.8, 0.7], [0.6, 0.95, 0.85]])
        ranks = average_ranks(mat)
        assert ranks.shape == (3,)

    def test_average_ranks_best_is_lowest(self):
        # Column 0 always best → should have rank ~1
        mat = np.array([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7], [0.9, 0.8, 0.7]])
        ranks = average_ranks(mat)
        assert ranks[0] < ranks[1] < ranks[2]

    def test_nemenyi_cd_positive(self):
        cd = nemenyi_cd(n_models=5, n_datasets=12, alpha=0.05)
        assert cd > 0

    def test_summary_table_shape(self):
        records = [
            {"model": "A", "dataset": "D1", "f1_macro": 0.8},
            {"model": "B", "dataset": "D1", "f1_macro": 0.75},
            {"model": "A", "dataset": "D2", "f1_macro": 0.85},
            {"model": "B", "dataset": "D2", "f1_macro": 0.9},
        ]
        tbl = summary_table(records)
        assert tbl.shape == (2, 2)
        assert "A" in tbl.columns
        assert "D1" in tbl.index


# ── Runner integration tests ───────────────────────────────────────────────────

class TestRunnerSingleExperiment:
    def test_standard_lssvm_bcw(self):
        result = run_single_experiment(
            "StandardLSSVM",
            "BCW",
            seed=0,
            model_params={"sigma": 3.0, "tau": 1.0},
        )
        assert result["status"] == "ok"
        assert result["accuracy"] > 0.80
        assert "sparsity_ratio" in result
        assert "train_time_s" in result

    def test_ft_transformer_bcw(self):
        result = run_single_experiment(
            "FTTransformer",
            "BCW",
            seed=0,
            model_params={
                "embedding_dim": 16,
                "num_blocks": 1,
                "num_heads": 2,
                "max_epochs": 5,
                "val_fraction": 0.0,
                "batch_size": 64,
            },
        )
        assert result["status"] == "ok"
        assert "accuracy" in result
        assert "mean_zero_fraction" in result

    def test_unknown_model_returns_error(self):
        result = run_single_experiment("NONEXISTENT_MODEL", "BCW", seed=0)
        assert result["status"] == "error"
        assert "error_message" in result

    def test_pruning_lssvm_synthetic(self):
        result = run_single_experiment(
            "PruningLSSVM",
            "TWM",
            seed=0,
            model_params={"sigma": 1.0, "tau": 1.0},
        )
        assert result["status"] == "ok"
        assert result["accuracy"] > 0.70

    def test_result_keys_complete(self):
        result = run_single_experiment(
            "StandardLSSVM", "TWM", seed=0,
            model_params={"sigma": 1.0, "tau": 1.0},
        )
        for key in ["model", "dataset", "seed", "accuracy", "f1_macro",
                    "train_time_s", "n_train", "n_test", "n_features"]:
            assert key in result, f"Missing key: {key}"

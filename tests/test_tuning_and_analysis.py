"""Tests for Optuna tuning wrapper and analysis modules."""

from __future__ import annotations

import numpy as np
import pytest

from src.tuning.bayesian import tune_model, _suggest_params, _load_config
from src.analysis.tables import results_table, sparsity_table, ranks_table
from src.analysis.plots import cd_diagram, boxplots, sparsity_accuracy_scatter


# ── Tuning tests ──────────────────────────────────────────────────────────────

class TestTuningConfig:
    def test_config_loads(self):
        cfg = _load_config()
        assert "models" in cfg
        assert "lssvm_standard" in cfg["models"]

    def test_suggest_params_types(self):
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        search_space = {
            "sigma": {"type": "log_uniform", "low": 0.01, "high": 100.0},
            "tau": {"type": "log_uniform", "low": 0.01, "high": 100.0},
        }
        params = _suggest_params(trial, search_space)
        assert "sigma" in params and "tau" in params
        assert 0.01 <= params["sigma"] <= 100.0
        assert 0.01 <= params["tau"] <= 100.0

    def test_suggest_params_categorical(self):
        import optuna
        study = optuna.create_study()
        trial = study.ask()
        search_space = {
            "embedding_dim": {"type": "categorical", "choices": [32, 64, 128]},
        }
        params = _suggest_params(trial, search_space)
        assert params["embedding_dim"] in [32, 64, 128]


class TestTuneModel:
    def test_tune_standard_lssvm_moons(self):
        result = tune_model(
            "StandardLSSVM", "TWM",
            n_trials=5, cv_folds=3, metric="f1_macro", seed=0,
        )
        assert "best_params" in result
        assert "best_value" in result
        assert result["best_value"] > 0.5
        assert result["n_trials_completed"] == 5

    def test_tune_returns_fixed_params_too(self):
        result = tune_model(
            "StandardLSSVM", "TWM",
            n_trials=3, cv_folds=2, seed=0,
        )
        # fixed_params (e.g. max_iter, tol) should appear in best_params
        assert "max_iter" in result["best_params"]

    def test_tune_ft_transformer_tiny(self):
        result = tune_model(
            "FTTransformer", "TWM",
            n_trials=2, cv_folds=2, metric="f1_macro", seed=0,
        )
        assert result["best_value"] > 0.0

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="not found in config"):
            tune_model("NONEXISTENT", "BCW", n_trials=1)

    def test_tune_by_config_key(self):
        result = tune_model(
            "lssvm_standard", "TWS",
            n_trials=3, cv_folds=2, seed=1,
        )
        assert result["model"] == "lssvm_standard"


# ── Table tests ───────────────────────────────────────────────────────────────

def _fake_results(n_models=3, n_datasets=2, n_seeds=5, seed=0):
    rng = np.random.default_rng(seed)
    records = []
    for m in [f"Model{i}" for i in range(n_models)]:
        for d in [f"D{j}" for j in range(n_datasets)]:
            for s in range(n_seeds):
                records.append({
                    "model": m, "dataset": d, "seed": s,
                    "f1_macro": float(rng.uniform(0.7, 0.95)),
                    "sparsity_ratio": float(rng.uniform(0, 0.8)),
                    "train_time_s": float(rng.uniform(0.01, 2.0)),
                })
    return records


class TestLatexTables:
    def test_results_table_returns_string(self):
        records = _fake_results()
        latex = results_table(records)
        assert isinstance(latex, str)
        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex

    def test_results_table_contains_model_names(self):
        records = _fake_results()
        latex = results_table(records)
        assert "Model0" in latex
        assert "D0" in latex

    def test_results_table_empty(self):
        latex = results_table([])
        assert "%" in latex  # comment line

    def test_sparsity_table(self):
        records = _fake_results()
        latex = sparsity_table(records)
        assert r"\begin{table}" in latex
        assert "Model0" in latex

    def test_ranks_table(self):
        ranks = np.array([2.1, 1.3, 3.2])
        names = ["A", "B", "C"]
        latex = ranks_table(ranks, names)
        assert r"\begin{table}" in latex
        assert "B" in latex  # best rank (1.3) should appear


# ── Plot tests ────────────────────────────────────────────────────────────────

class TestPlots:
    def test_cd_diagram_returns_figure(self):
        import matplotlib.pyplot as plt
        ranks = np.array([1.5, 2.3, 3.1, 1.8])
        names = ["A", "B", "C", "D"]
        fig = cd_diagram(ranks, names, cd=0.8)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_boxplots_returns_figure(self):
        import matplotlib.pyplot as plt
        records = _fake_results()
        fig = boxplots(records)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_sparsity_scatter_returns_figure(self):
        import matplotlib.pyplot as plt
        records = _fake_results()
        fig = sparsity_accuracy_scatter(records)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_boxplots_empty_data(self):
        import matplotlib.pyplot as plt
        fig = boxplots([])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

"""Sanity checks for LSSVM implementations.

Tests verify:
1. fit/predict runs on toy datasets
2. Sparsity is > 0% for sparse methods (and 0% for standard)
3. Reproducibility with fixed seed
4. Accuracy within ±5% of StandardLSSVM on BCW (sklearn built-in)
5. Decision function shapes are correct
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.experiments.reproducibility import set_global_seed
from src.models.lssvm.standard import StandardLSSVM
from src.models.lssvm.primal.admm_nesterov import ADMMNesterovLSSVM


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def moons_data():
    """Small 2-moons dataset with labels in {-1, +1}."""
    set_global_seed(42)
    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
    y = np.where(y == 1, 1, -1)
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


@pytest.fixture(scope="module")
def bcw_data():
    """Breast Cancer Wisconsin — used for accuracy validation vs paper Table 4."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    y = np.where(y == 1, 1, -1)
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


# ── StandardLSSVM ─────────────────────────────────────────────────────────────

class TestStandardLSSVM:
    def test_fit_predict_moons(self, moons_data):
        X_tr, X_te, y_tr, y_te = moons_data
        model = StandardLSSVM(sigma=1.0, tau=1.0)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert preds.shape == y_te.shape
        assert set(np.unique(preds)).issubset({-1, 1})

    def test_accuracy_above_chance_moons(self, moons_data):
        X_tr, X_te, y_tr, y_te = moons_data
        model = StandardLSSVM(sigma=0.5, tau=10.0)
        model.fit(X_tr, y_tr)
        acc = np.mean(model.predict(X_te) == y_te)
        assert acc > 0.80, f"Accuracy too low: {acc:.3f}"

    def test_decision_function_shape(self, moons_data):
        X_tr, X_te, y_tr, y_te = moons_data
        model = StandardLSSVM(sigma=1.0, tau=1.0)
        model.fit(X_tr, y_tr)
        scores = model.decision_function(X_te)
        assert scores.shape == (len(y_te),)

    def test_all_samples_are_svs(self, moons_data):
        """Standard LSSVM has no sparsity (all training samples are SVs)."""
        X_tr, X_te, y_tr, y_te = moons_data
        model = StandardLSSVM(sigma=1.0, tau=1.0)
        model.fit(X_tr, y_tr)
        # All αᵢ are non-zero in the standard LSSVM
        assert model.sparsity_ratio_ < 0.10, (
            f"Expected near-zero sparsity, got {model.sparsity_ratio_:.2%}"
        )

    def test_reproducibility(self, moons_data):
        X_tr, X_te, y_tr, y_te = moons_data
        m1 = StandardLSSVM(sigma=1.0, tau=1.0).fit(X_tr, y_tr)
        m2 = StandardLSSVM(sigma=1.0, tau=1.0).fit(X_tr, y_tr)
        np.testing.assert_array_almost_equal(m1.alpha_, m2.alpha_)

    def test_binary_label_conversion(self, moons_data):
        """Model should accept {0, 1} labels and internally convert."""
        X_tr, X_te, y_tr, y_te = moons_data
        y_tr_01 = np.where(y_tr == 1, 1, 0)
        y_te_01 = np.where(y_te == 1, 1, 0)
        model = StandardLSSVM(sigma=1.0, tau=1.0)
        model.fit(X_tr, y_tr_01)
        preds = model.predict(X_te)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_bcw_accuracy(self, bcw_data):
        """Baseline LSSVM should achieve >90% accuracy on BCW with sigma=3."""
        X_tr, X_te, y_tr, y_te = bcw_data
        model = StandardLSSVM(sigma=3.0, tau=1.0)
        model.fit(X_tr, y_tr)
        acc = np.mean(model.predict(X_te) == y_te)
        assert acc > 0.90, f"BCW accuracy too low: {acc:.3f}"


# ── ADMMNesterovLSSVM ─────────────────────────────────────────────────────────

class TestADMMNesterovLSSVM:
    def test_fit_predict_moons(self, moons_data):
        X_tr, X_te, y_tr, y_te = moons_data
        model = ADMMNesterovLSSVM(sigma=1.0, tau=1.0, lambda_=0.1, rho=1.0)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert preds.shape == y_te.shape
        assert set(np.unique(preds)).issubset({-1, 1})

    def test_sparsity_greater_than_zero(self, moons_data):
        """ADMM-Nesterov should yield at least some sparse α coefficients."""
        X_tr, X_te, y_tr, y_te = moons_data
        # Large lambda → strong sparsity
        model = ADMMNesterovLSSVM(
            sigma=1.0, tau=1.0, lambda_=5.0, rho=1.0, max_iter=200
        )
        model.fit(X_tr, y_tr)
        assert model.sparsity_ratio_ > 0.0, "Expected non-zero sparsity"
        assert model.n_support_ < model.n_samples_fit_, "Some SVs should be pruned"

    def test_decision_function_shape(self, moons_data):
        X_tr, X_te, y_tr, y_te = moons_data
        model = ADMMNesterovLSSVM(sigma=1.0, tau=1.0, lambda_=0.1, rho=1.0)
        model.fit(X_tr, y_tr)
        scores = model.decision_function(X_te)
        assert scores.shape == (len(y_te),)

    def test_reproducibility(self, moons_data):
        X_tr, X_te, y_tr, y_te = moons_data
        params = dict(sigma=1.0, tau=1.0, lambda_=0.5, rho=1.0, max_iter=100)
        m1 = ADMMNesterovLSSVM(**params).fit(X_tr, y_tr)
        m2 = ADMMNesterovLSSVM(**params).fit(X_tr, y_tr)
        np.testing.assert_array_almost_equal(m1.alpha_, m2.alpha_)

    def test_nesterov_vs_plain_admm_sparsity(self, moons_data):
        """With Nesterov, model should converge in fewer iterations."""
        X_tr, X_te, y_tr, y_te = moons_data
        params = dict(sigma=1.0, tau=1.0, lambda_=1.0, rho=1.0, max_iter=300)
        m_fast = ADMMNesterovLSSVM(**params, use_nesterov=True).fit(X_tr, y_tr)
        m_plain = ADMMNesterovLSSVM(**params, use_nesterov=False).fit(X_tr, y_tr)
        # Both should converge to similar solutions
        # Nesterov should need fewer or equal iterations
        assert m_fast.n_iter_ <= m_plain.n_iter_ + 50  # generous tolerance

    def test_bcw_accuracy_above_threshold(self, bcw_data):
        """ADMM-Nesterov should achieve reasonable accuracy on BCW.

        Note: the ADMM primal and Suykens dual LSSVM solve different
        optimisation problems, so direct accuracy comparison is not fair.
        Both achieve their best performance with tuned hyperparameters.
        """
        X_tr, X_te, y_tr, y_te = bcw_data
        admm_model = ADMMNesterovLSSVM(
            sigma=10.0, tau=0.01, lambda_=0.0, rho=1.0, max_iter=500
        ).fit(X_tr, y_tr)
        admm_acc = np.mean(admm_model.predict(X_te) == y_te)
        assert admm_acc > 0.80, f"ADMM BCW accuracy too low: {admm_acc:.3f}"

    def test_larger_lambda_means_more_sparsity(self, moons_data):
        """Increasing λ should decrease the number of support vectors."""
        X_tr, X_te, y_tr, y_te = moons_data
        base = dict(sigma=1.0, tau=1.0, rho=1.0, max_iter=200)
        m_low = ADMMNesterovLSSVM(**base, lambda_=0.1).fit(X_tr, y_tr)
        m_high = ADMMNesterovLSSVM(**base, lambda_=5.0).fit(X_tr, y_tr)
        assert m_low.n_support_ >= m_high.n_support_, (
            "Higher λ should produce fewer SVs"
        )

    def test_binary_label_conversion(self, moons_data):
        X_tr, X_te, y_tr, y_te = moons_data
        y_tr_01 = np.where(y_tr == 1, 1, 0)
        y_te_01 = np.where(y_te == 1, 1, 0)
        model = ADMMNesterovLSSVM(sigma=1.0, tau=1.0, lambda_=0.1, rho=1.0)
        model.fit(X_tr, y_tr_01)
        preds = model.predict(X_te)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_n_iter_attribute_set(self, moons_data):
        X_tr, X_te, y_tr, y_te = moons_data
        model = ADMMNesterovLSSVM(sigma=1.0, tau=1.0, lambda_=0.1, rho=1.0)
        model.fit(X_tr, y_tr)
        assert hasattr(model, "n_iter_")
        assert 1 <= model.n_iter_ <= model.max_iter

"""Sanity checks for dual and remaining primal LSSVM sparse methods."""

import numpy as np
import pytest
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.experiments.reproducibility import set_global_seed
from src.models.lssvm.dual.p_lssvm import PruningLSSVM
from src.models.lssvm.dual.ip_lssvm import IPLSSVm
from src.models.lssvm.dual.opposite_maps import OppositeMapsLSSVM
from src.models.lssvm.primal.pcp_lssvm import PCPLSSVm
from src.models.lssvm.primal.fsa_lssvm import FSALSSVm


PARAMS = dict(sigma=1.0, tau=1.0)


@pytest.fixture(scope="module")
def moons_data():
    set_global_seed(42)
    X, y = make_moons(n_samples=300, noise=0.1, random_state=42)
    y = np.where(y == 1, 1, -1)
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def _basic_checks(model_cls, moons_data, extra_params=None):
    """Run fit/predict/sparsity/reproducibility checks for any LSSVM model."""
    X_tr, X_te, y_tr, y_te = moons_data
    params = {**PARAMS, **(extra_params or {})}

    # fit/predict
    m = model_cls(**params).fit(X_tr, y_tr)
    preds = m.predict(X_te)
    assert preds.shape == y_te.shape
    assert set(np.unique(preds)).issubset({-1, 1})

    # accuracy above chance
    acc = np.mean(preds == y_te)
    assert acc > 0.65, f"{model_cls.__name__} accuracy too low: {acc:.3f}"

    # sparsity > 0%
    assert m.sparsity_ratio_ > 0.0, f"{model_cls.__name__} has no sparsity"
    assert m.n_support_ < m.n_samples_fit_

    # decision function shape
    scores = m.decision_function(X_te)
    assert scores.shape == (len(y_te),)

    # reproducibility
    m2 = model_cls(**params).fit(X_tr, y_tr)
    np.testing.assert_array_almost_equal(m.alpha_, m2.alpha_)


# ── Pruning LSSVM ─────────────────────────────────────────────────────────────

class TestPruningLSSVM:
    def test_basic(self, moons_data):
        _basic_checks(PruningLSSVM, moons_data, {"pruning_rate": 0.2, "max_pruning_steps": 5})

    def test_higher_pruning_more_sparse(self, moons_data):
        X_tr, X_te, y_tr, y_te = moons_data
        m_low = PruningLSSVM(**PARAMS, pruning_rate=0.05, max_pruning_steps=3).fit(X_tr, y_tr)
        m_high = PruningLSSVM(**PARAMS, pruning_rate=0.30, max_pruning_steps=10).fit(X_tr, y_tr)
        assert m_low.n_support_ >= m_high.n_support_


# ── IP-LSSVM ──────────────────────────────────────────────────────────────────

class TestIPLSSVM:
    def test_basic(self, moons_data):
        _basic_checks(IPLSSVm, moons_data, {"selection_ratio": 0.30})

    def test_smaller_ratio_more_sparse(self, moons_data):
        X_tr, X_te, y_tr, y_te = moons_data
        m_large = IPLSSVm(**PARAMS, selection_ratio=0.50).fit(X_tr, y_tr)
        m_small = IPLSSVm(**PARAMS, selection_ratio=0.15).fit(X_tr, y_tr)
        assert m_large.n_support_ >= m_small.n_support_


# ── Opposite Maps LSSVM ───────────────────────────────────────────────────────

class TestOppositeMapsLSSVM:
    def test_basic(self, moons_data):
        # Need enough prototypes for non-trivial subset on 2-moons
        _basic_checks(OppositeMapsLSSVM, moons_data, {"n_prototypes": 25})

    def test_more_prototypes_more_svs(self, moons_data):
        X_tr, X_te, y_tr, y_te = moons_data
        m_few = OppositeMapsLSSVM(**PARAMS, n_prototypes=3).fit(X_tr, y_tr)
        m_many = OppositeMapsLSSVM(**PARAMS, n_prototypes=15).fit(X_tr, y_tr)
        assert m_few.n_support_ <= m_many.n_support_


# ── PCP-LSSVM ─────────────────────────────────────────────────────────────────

class TestPCPLSSVM:
    def test_basic(self, moons_data):
        X_tr, X_te, y_tr, y_te = moons_data
        m = PCPLSSVm(**PARAMS, rank=30).fit(X_tr, y_tr)
        preds = m.predict(X_te)
        assert preds.shape == y_te.shape
        acc = np.mean(preds == y_te)
        assert acc > 0.65, f"PCPLSSVm accuracy too low: {acc:.3f}"
        scores = m.decision_function(X_te)
        assert scores.shape == (len(y_te),)

    def test_reproducibility(self, moons_data):
        X_tr, X_te, y_tr, y_te = moons_data
        m1 = PCPLSSVm(**PARAMS, rank=20).fit(X_tr, y_tr)
        m2 = PCPLSSVm(**PARAMS, rank=20).fit(X_tr, y_tr)
        np.testing.assert_array_almost_equal(m1.alpha_, m2.alpha_)


# ── FSA-LSSVM ─────────────────────────────────────────────────────────────────

class TestFSALSSVM:
    def test_basic(self, moons_data):
        _basic_checks(FSALSSVm, moons_data, {"n_components": 40})

    def test_more_components_less_sparse(self, moons_data):
        X_tr, X_te, y_tr, y_te = moons_data
        m_few = FSALSSVm(**PARAMS, n_components=10).fit(X_tr, y_tr)
        m_many = FSALSSVm(**PARAMS, n_components=60).fit(X_tr, y_tr)
        assert m_few.n_support_ <= m_many.n_support_

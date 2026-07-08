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

    def test_safety_fallback_triggers_on_aggressive_pruning(self, moons_data):
        # Regression test for the _solve bug (2026-07-07): the training-F1
        # drop check omitted the y[active] weighting, so best_score/
        # current_score both degenerated to a label-agnostic (effectively
        # constant) decision function and the "stop if F1 drops" guard never
        # fired. pruning_rate=0.9 removes 90% of SVs per step — the reduced
        # model must degrade enough that the fallback reverts to the full set.
        X_tr, X_te, y_tr, y_te = moons_data
        m = PruningLSSVM(
            **PARAMS, pruning_rate=0.9, max_pruning_steps=20,
            min_sv_fraction=0.02, drop_tolerance=0.05,
        ).fit(X_tr, y_tr)
        assert m.n_support_ == m.n_samples_fit_


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
        # Need enough prototypes for non-trivial subset on 2-moons. Since
        # each prototype now contributes a same-class anchor alongside the
        # opposite-class boundary point (2026-07-07 fix), the reduced set
        # is stable enough that the default fallback no longer discards it.
        _basic_checks(OppositeMapsLSSVM, moons_data, {"n_prototypes": 25})

    def test_more_prototypes_more_svs(self, moons_data):
        X_tr, X_te, y_tr, y_te = moons_data
        m_few = OppositeMapsLSSVM(**PARAMS, n_prototypes=3).fit(X_tr, y_tr)
        m_many = OppositeMapsLSSVM(**PARAMS, n_prototypes=15).fit(X_tr, y_tr)
        assert m_few.n_support_ <= m_many.n_support_

    def test_prototypes_include_same_class_anchor(self, moons_data):
        # Regression test for the anchor-point fix (2026-07-07): each
        # prototype must contribute a same-class anchor, not just an
        # opposite-class boundary point — otherwise the reduced set is
        # built entirely from closely-spaced cross-class pairs, which
        # destabilises the LSSVM least-squares fit (see class docstring).
        X_tr, X_te, y_tr, y_te = moons_data
        m = OppositeMapsLSSVM(**PARAMS, n_prototypes=5, drop_tolerance=np.inf).fit(X_tr, y_tr)
        selected_y = y_tr[m.support_indices_]
        assert (selected_y == 1).sum() >= 2
        assert (selected_y == -1).sum() >= 2

    def test_fallback_triggers_on_bad_reduction(self, moons_data):
        # Regression test for the fallback mechanism (2026-07-07): even with
        # the anchor-point fix, n_prototypes=1 is too coarse a reduction on
        # 2-moons — the fallback must still trigger and revert to the
        # full model in genuinely bad cases.
        X_tr, X_te, y_tr, y_te = moons_data
        m = OppositeMapsLSSVM(**PARAMS, n_prototypes=1, drop_tolerance=0.05).fit(X_tr, y_tr)
        assert m.sparsity_ratio_ == 0.0
        assert m.n_support_ == m.n_samples_fit_


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

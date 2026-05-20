"""Tests for data loaders, preprocessing and synthetic generators."""

import os
import numpy as np
import pytest

from src.data.loaders import DatasetLoader
from src.data.preprocessing import make_splits, preprocess
from src.data.synthetic import make_tws, make_twm, make_twc

# Skip network-requiring tests unless RUN_NETWORK_TESTS=1 is set
_network = pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS", "0") != "1",
    reason="requires network download (set RUN_NETWORK_TESTS=1 to enable)",
)


# ── Synthetic datasets ────────────────────────────────────────────────────────

class TestSyntheticDatasets:
    @pytest.mark.parametrize("fn,kwargs", [
        (make_tws, {"n_samples": 200, "random_state": 0}),
        (make_twm, {"n_samples": 200, "random_state": 0}),
        (make_twc, {"n_samples": 200, "random_state": 0}),
    ])
    def test_shape_and_labels(self, fn, kwargs):
        X, y = fn(**kwargs)
        assert X.ndim == 2
        assert y.ndim == 1
        assert len(X) == len(y) == kwargs["n_samples"]
        assert set(np.unique(y)) == {-1, 1}

    def test_reproducibility(self):
        X1, y1 = make_twm(n_samples=100, random_state=7)
        X2, y2 = make_twm(n_samples=100, random_state=7)
        np.testing.assert_array_equal(X1, X2)

    def test_different_seeds_differ(self):
        X1, _ = make_tws(n_samples=100, random_state=0)
        X2, _ = make_tws(n_samples=100, random_state=99)
        assert not np.allclose(X1, X2)


# ── DatasetLoader — sklearn/cached datasets ───────────────────────────────────

class TestDatasetLoaderBCW:
    """BCW uses sklearn — always available, fast."""

    def test_load_returns_correct_types(self):
        X, y, meta = DatasetLoader.load("BCW")
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(meta, dict)

    def test_shape(self):
        X, y, meta = DatasetLoader.load("BCW")
        assert X.ndim == 2 and y.ndim == 1
        assert len(X) == len(y)
        assert X.shape == (569, 30)

    def test_binary_labels(self):
        _, y, _ = DatasetLoader.load("BCW")
        assert set(np.unique(y)).issubset({0, 1})
        assert len(np.unique(y)) == 2

    def test_no_nans(self):
        X, _, _ = DatasetLoader.load("BCW")
        assert not np.isnan(X).any()

    def test_meta_fields(self):
        _, _, meta = DatasetLoader.load("BCW")
        assert "tier" in meta
        assert "n_samples" in meta
        assert "class_ratio" in meta


class TestDatasetLoaderSynthetic:
    @pytest.mark.parametrize("name", ["TWS", "TWM", "TWC"])
    def test_synthetic_load(self, name):
        X, y, meta = DatasetLoader.load(name)
        assert X.shape == (400, 2)
        assert set(np.unique(y)).issubset({0, 1})

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            DatasetLoader.load("NONEXISTENT_DATASET")

    def test_available_lists_known_datasets(self):
        avail = DatasetLoader.available()
        for name in ["BCW", "TWS", "BANK", "TELCO", "SHOPPERS", "HIGGS50K", "HIGGS500K"]:
            assert name in avail, f"{name} missing from available()"


# ── Preprocessing pipeline ────────────────────────────────────────────────────

class TestPreprocessing:
    def setup_method(self):
        self.X, self.y, _ = DatasetLoader.load("BCW")

    def test_make_splits_shapes(self):
        X_tr, X_te, y_tr, y_te = make_splits(self.X, self.y, test_size=0.30, seed=0)
        assert len(X_tr) + len(X_te) == len(self.X)
        assert abs(len(X_te) / len(self.X) - 0.30) < 0.02

    def test_split_is_stratified(self):
        X_tr, X_te, y_tr, y_te = make_splits(self.X, self.y, test_size=0.30, seed=0)
        ratio_train = np.mean(y_tr)
        ratio_test = np.mean(y_te)
        ratio_full = np.mean(self.y)
        assert abs(ratio_train - ratio_full) < 0.05
        assert abs(ratio_test - ratio_full) < 0.05

    def test_split_reproducibility(self):
        X_tr1, X_te1, y_tr1, y_te1 = make_splits(self.X, self.y, seed=42)
        X_tr2, X_te2, y_tr2, y_te2 = make_splits(self.X, self.y, seed=42)
        np.testing.assert_array_equal(X_tr1, X_tr2)

    def test_preprocess_signed_labels(self):
        X_tr, X_te, y_tr, y_te = make_splits(self.X, self.y)
        X_tr_s, X_te_s, y_tr_s, y_te_s = preprocess(X_tr, X_te, y_tr, y_te, "signed")
        assert set(np.unique(y_tr_s)).issubset({-1, 1})
        assert set(np.unique(y_te_s)).issubset({-1, 1})

    def test_preprocess_binary_labels(self):
        X_tr, X_te, y_tr, y_te = make_splits(self.X, self.y)
        _, _, y_tr_b, y_te_b = preprocess(X_tr, X_te, y_tr, y_te, "binary")
        assert set(np.unique(y_tr_b)).issubset({0, 1})

    def test_scaler_no_data_leakage(self):
        """Test data mean comes exclusively from training split."""
        X_tr, X_te, y_tr, y_te = make_splits(self.X, self.y)
        X_tr_s, X_te_s, _, _ = preprocess(X_tr, X_te, y_tr, y_te)
        # Training set should be approximately zero-mean after standardisation
        assert np.abs(np.mean(X_tr_s, axis=0)).max() < 1e-10
        # Test set mean need not be zero (no leakage)
        assert not np.allclose(np.mean(X_te_s, axis=0), 0)


# ── Tier 2/3 dataset loaders (network-gated) ──────────────────────────────────

def _check_dataset(name: str, min_samples: int = 100, n_features_min: int = 2):
    X, y, meta = DatasetLoader.load(name)
    assert X.ndim == 2
    assert y.ndim == 1
    assert len(X) == len(y)
    assert len(X) >= min_samples
    assert X.shape[1] >= n_features_min
    assert set(np.unique(y)).issubset({0, 1})
    assert len(np.unique(y)) == 2
    assert not np.isnan(X).any()
    assert "tier" in meta
    return X, y, meta


class TestTier2Datasets:
    @_network
    def test_bank_loads(self):
        X, y, meta = _check_dataset("BANK", min_samples=40000, n_features_min=10)
        assert meta["tier"] == 2

    @_network
    def test_telco_loads(self):
        X, y, meta = _check_dataset("TELCO", min_samples=6000, n_features_min=15)
        assert meta["tier"] == 2

    @_network
    def test_shoppers_loads(self):
        X, y, meta = _check_dataset("SHOPPERS", min_samples=10000, n_features_min=10)
        assert meta["tier"] == 2

    @_network
    def test_higgs50k_loads(self):
        X, y, meta = _check_dataset("HIGGS50K", min_samples=50000, n_features_min=28)
        assert X.shape == (50000, 28)
        assert meta["tier"] == 2

    @_network
    def test_higgs500k_loads(self):
        X, y, meta = _check_dataset("HIGGS500K", min_samples=500000, n_features_min=28)
        assert X.shape == (500000, 28)
        assert meta["tier"] == 3

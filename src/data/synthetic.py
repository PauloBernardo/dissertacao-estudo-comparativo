"""Synthetic dataset generators used in the original paper.

TWS — Two-class Spiral
TWM — Two-class Moons (sklearn variant)
TWC — Two-class Checkerboard

Variants with suffix _5f embed the 2D problem in 5D by appending 3 Gaussian
noise features (mean=0, std=1). The label depends only on the first 2 features.

MK5 series — 5-feature make_classification datasets (all features informative):
  MKE — Easy   (class_sep=2.0, 1 cluster/class,  flip_y=0.01)
  MKM — Medium (class_sep=1.0, 2 clusters/class, flip_y=0.05)
  MKH — Hard   (class_sep=0.5, 3 clusters/class, flip_y=0.10)

All generators return (X, y) with y ∈ {-1, +1}.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_tws(
    n_samples: int = 400,
    noise: float = 0.05,
    n_turns: float = 1.5,
    random_state: int = 42,
) -> tuple[NDArray, NDArray]:
    """Two-class Spiral dataset (TWS).

    Parameters
    ----------
    n_samples : total samples (equally split between classes)
    noise : Gaussian noise standard deviation
    n_turns : number of spiral turns
    random_state : seed
    """
    rng = np.random.default_rng(random_state)
    n = n_samples // 2

    theta = np.linspace(0, n_turns * 2 * np.pi, n)
    r = np.linspace(0.1, 1.0, n)

    # Class +1: spiral 1
    x1_pos = r * np.cos(theta)
    x2_pos = r * np.sin(theta)

    # Class -1: spiral 2 (offset by π)
    x1_neg = r * np.cos(theta + np.pi)
    x2_neg = r * np.sin(theta + np.pi)

    X_pos = np.column_stack([x1_pos, x2_pos])
    X_neg = np.column_stack([x1_neg, x2_neg])

    X = np.vstack([X_pos, X_neg])
    X += rng.normal(0, noise, X.shape)

    y = np.hstack([np.ones(n), -np.ones(n)]).astype(int)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def make_twm(
    n_samples: int = 400,
    noise: float = 0.10,
    random_state: int = 42,
) -> tuple[NDArray, NDArray]:
    """Two-class Moons dataset (TWM) with labels {-1, +1}."""
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    y = np.where(y == 1, 1, -1).astype(int)
    return X, y


def make_twc(
    n_samples: int = 400,
    n_tiles: int = 4,
    noise: float = 0.02,
    random_state: int = 42,
) -> tuple[NDArray, NDArray]:
    """Two-class Checkerboard dataset (TWC).

    Parameters
    ----------
    n_samples : total samples
    n_tiles : number of tiles per row/column (checkerboard grid = n_tiles × n_tiles)
    noise : Gaussian noise standard deviation
    """
    rng = np.random.default_rng(random_state)
    X = rng.uniform(0, n_tiles, (n_samples, 2))
    X += rng.normal(0, noise, X.shape)

    # Assign label based on checkerboard pattern
    col = np.floor(X[:, 0]).astype(int)
    row = np.floor(X[:, 1]).astype(int)
    y = np.where((col + row) % 2 == 0, 1, -1).astype(int)
    return X, y


# ── 5-feature variants (2D structure embedded in 5D with 3 noise features) ────

def _add_noise_features(
    X: NDArray,
    y: NDArray,
    n_noise: int = 3,
    noise_std: float = 1.0,
    random_state: int = 42,
) -> tuple[NDArray, NDArray]:
    rng = np.random.default_rng(random_state + 1000)
    noise = rng.normal(0, noise_std, (len(X), n_noise))
    return np.hstack([X, noise]), y


def make_tws_5f(n_samples: int = 400, random_state: int = 42) -> tuple[NDArray, NDArray]:
    """TWS embedded in 5D: 2 spiral features + 3 Gaussian noise features."""
    X, y = make_tws(n_samples=n_samples, random_state=random_state)
    return _add_noise_features(X, y, random_state=random_state)


def make_twm_5f(n_samples: int = 400, random_state: int = 42) -> tuple[NDArray, NDArray]:
    """TWM embedded in 5D: 2 moon features + 3 Gaussian noise features."""
    X, y = make_twm(n_samples=n_samples, random_state=random_state)
    return _add_noise_features(X, y, random_state=random_state)


def make_twc_5f(n_samples: int = 400, random_state: int = 42) -> tuple[NDArray, NDArray]:
    """TWC embedded in 5D: 2 checkerboard features + 3 Gaussian noise features."""
    X, y = make_twc(n_samples=n_samples, random_state=random_state)
    return _add_noise_features(X, y, random_state=random_state)


# ── MK5 series — 5-feature make_classification (all features informative) ─────

def _make_mk5(
    n_samples: int,
    class_sep: float,
    n_clusters_per_class: int,
    flip_y: float,
    random_state: int,
) -> tuple[NDArray, NDArray]:
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=random_state,
    )
    return X, np.where(y == 1, 1, -1).astype(int)


def make_mke(n_samples: int = 400, random_state: int = 42) -> tuple[NDArray, NDArray]:
    """MKE — Easy: class_sep=2.0, 1 cluster/class, flip_y=0.01."""
    return _make_mk5(n_samples, class_sep=2.0, n_clusters_per_class=1,
                     flip_y=0.01, random_state=random_state)


def make_mkm(n_samples: int = 400, random_state: int = 42) -> tuple[NDArray, NDArray]:
    """MKM — Medium: class_sep=1.0, 2 clusters/class, flip_y=0.05."""
    return _make_mk5(n_samples, class_sep=1.0, n_clusters_per_class=2,
                     flip_y=0.05, random_state=random_state)


def make_mkh(n_samples: int = 400, random_state: int = 42) -> tuple[NDArray, NDArray]:
    """MKH — Hard: class_sep=0.5, 3 clusters/class, flip_y=0.10."""
    return _make_mk5(n_samples, class_sep=0.5, n_clusters_per_class=3,
                     flip_y=0.10, random_state=random_state)

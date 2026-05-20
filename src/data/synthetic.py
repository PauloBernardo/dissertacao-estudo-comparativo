"""Synthetic dataset generators used in the original paper.

TWS — Two-class Spiral
TWM — Two-class Moons (sklearn variant)
TWC — Two-class Checkerboard

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

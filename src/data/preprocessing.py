"""Standardised preprocessing pipeline for all experiments.

All preprocessing is fit ONLY on the training split and applied to the
test split to avoid data leakage.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def make_splits(
    X: NDArray,
    y: NDArray,
    test_size: float = 0.30,
    seed: int = 0,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Stratified 70/30 split (or any test_size) with a fixed seed.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)  — binary labels
    test_size : float
    seed : int

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def preprocess(
    X_train: NDArray,
    X_test: NDArray,
    y_train: NDArray,
    y_test: NDArray,
    label_format: Literal["signed", "binary"] = "signed",
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """StandardScaler (fit on train only) + label conversion.

    Parameters
    ----------
    X_train, X_test : raw feature arrays
    y_train, y_test : raw label arrays (any encoding)
    label_format : "signed" → {-1, +1};  "binary" → {0, 1}

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train_conv, y_test_conv
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    y_tr = _convert_labels(y_train, label_format)
    y_te = _convert_labels(y_test, label_format)

    return X_tr, X_te, y_tr, y_te


def _convert_labels(y: NDArray, fmt: str) -> NDArray:
    """Map any binary labels to {-1,+1} or {0,1}."""
    unique = np.unique(y)
    if len(unique) != 2:
        raise ValueError(f"Expected exactly 2 classes, got {unique}.")

    # Identify negative/positive class (lower value = negative)
    neg, pos = sorted(unique)
    y_binary = np.where(y == pos, 1, 0).astype(int)

    if fmt == "binary":
        return y_binary
    elif fmt == "signed":
        return np.where(y_binary == 1, 1, -1).astype(int)
    else:
        raise ValueError(f"Unknown label_format: {fmt!r}")

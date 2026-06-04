"""Abstract base class for all LSSVM variants."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

logger = logging.getLogger(__name__)


class BaseLSSVM(ABC, BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible base for all LSSVM implementations.

    Subclasses must implement ``_solve``, which populates ``alpha_`` and
    ``bias_`` given the training data in primal or dual form.

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth (γ = 1 / (2σ²)).
    tau : float
        Regularisation parameter (C in primal, γ in dual formulation).
    kernel : str
        Only 'rbf' is currently supported.
    tol : float
        Convergence tolerance used by solvers.
    max_iter : int
        Maximum number of solver iterations.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        kernel: str = "rbf",
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> None:
        self.sigma = sigma
        self.tau = tau
        self.kernel = kernel
        self.tol = tol
        self.max_iter = max_iter

    # ── Kernel ────────────────────────────────────────────────────────────────

    def _rbf_kernel(self, X: NDArray, Y: NDArray) -> NDArray:
        """Compute the RBF kernel matrix K(X, Y).

        K_ij = exp(-||x_i - y_j||² / (2σ²))
        """
        gamma = 1.0 / (2.0 * self.sigma**2)
        # pairwise squared distances via ||x-y||² = ||x||² + ||y||² - 2 x·y
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        Y_sq = np.sum(Y**2, axis=1, keepdims=True)
        sq_dists = X_sq + Y_sq.T - 2.0 * X @ Y.T
        sq_dists = np.maximum(sq_dists, 0.0)  # numerical floor
        return np.exp(-gamma * sq_dists)

    def kernel_matrix(self, X: NDArray, Y: Optional[NDArray] = None) -> NDArray:
        """Return the kernel matrix for X (and optionally Y)."""
        if self.kernel != "rbf":
            raise ValueError(f"Unsupported kernel: {self.kernel!r}")
        Y = X if Y is None else Y
        return self._rbf_kernel(X, Y)

    # ── Fit / Predict ─────────────────────────────────────────────────────────

    def fit(self, X: NDArray, y: NDArray) -> "BaseLSSVM":
        """Train the LSSVM.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,) with labels in {-1, +1}

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if set(self.classes_) == {0, 1}:
            # internally always use {-1, +1}
            y = np.where(y == 1, 1.0, -1.0)
        elif not set(self.classes_).issubset({-1, 1}):
            raise ValueError("Labels must be in {-1, +1} or {0, 1}.")

        self.X_train_ = X
        self.y_train_ = y
        self.n_samples_fit_, self.n_features_in_ = X.shape

        logger.info(
            "%s.fit — n=%d, p=%d, sigma=%.4f, tau=%.4f",
            self.__class__.__name__,
            self.n_samples_fit_,
            self.n_features_in_,
            self.sigma,
            self.tau,
        )

        self._solve(X, y)
        return self

    @abstractmethod
    def _solve(self, X: NDArray, y: NDArray) -> None:
        """Solve the LSSVM optimisation problem.

        Must set ``self.alpha_`` (dual coefficients, shape (n_samples,)),
        ``self.bias_`` (scalar intercept), and optionally
        ``self.support_vectors_`` with ``self.support_indices_``.
        """

    # Threshold for treating an α coefficient as zero (drops it from predict).
    # Must match the threshold used by ``n_support_`` so that reported
    # sparsity and inference cost stay consistent.
    _ALPHA_ZERO_TOL: float = 1e-6

    def _kernel_predict(self, X: NDArray, coef: NDArray) -> NDArray:
        """Compute K(X, X_sv) @ coef_sv + bias, skipping α=0 columns.

        Subclasses call this from their ``decision_function`` with the
        appropriate coefficient vector (``α·y`` for dual form, ``α`` for
        primal form where the label is already absorbed).
        """
        nz = np.abs(self.alpha_) > self._ALPHA_ZERO_TOL
        if nz.all():
            K = self.kernel_matrix(X, self.X_train_)
            return K @ coef + self.bias_
        K = self.kernel_matrix(X, self.X_train_[nz])
        return K @ coef[nz] + self.bias_

    def decision_function(self, X: NDArray) -> NDArray:
        """Compute raw decision scores f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b."""
        check_is_fitted(self, ["alpha_", "bias_", "X_train_", "y_train_"])
        X = check_array(X)
        return self._kernel_predict(X, self.alpha_ * self.y_train_)

    def predict(self, X: NDArray) -> NDArray:
        """Return predicted class labels in the original label space."""
        scores = self.decision_function(X)
        raw = np.sign(scores)
        raw[raw == 0] = 1.0
        if set(self.classes_) == {0, 1}:
            return np.where(raw == 1.0, 1, 0)
        return raw.astype(int)

    # ── Sparsity helpers ──────────────────────────────────────────────────────

    @property
    def n_support_(self) -> int:
        """Number of non-zero dual coefficients (support vectors)."""
        check_is_fitted(self, ["alpha_"])
        return int(np.sum(np.abs(self.alpha_) > 1e-6))

    @property
    def sparsity_ratio_(self) -> float:
        """Fraction of training samples that are NOT support vectors."""
        return 1.0 - self.n_support_ / self.n_samples_fit_

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sigma={self.sigma}, tau={self.tau}, kernel={self.kernel!r})"
        )

"""Pivoted Cholesky Primal LSSVM (PCP-LSSVM).

Direct Method: computes a low-rank approximation of the kernel matrix
using Pivoted (incomplete) Cholesky, then trains a primal LSSVM on the
reduced feature space.

Reference:
    Zhou S. et al., "PCP-LSSVM: A Pivoted Cholesky Decomposition based
    Primal Formulation of the LSSVM Classifier", 2015.

Sparsity analogy with Nyström
------------------------------
PCP selects r pivot points greedily (max residual diagonal); Nyström
selects m landmarks by column-norm sampling. Both compress N → r/m and
predict in O(n_test × r). Reported sparsity = 1 − r/N, exactly as
NystromLSSVMColnorm reports 1 − m/N.

Paper-fonte (BASE TEORICA, fora do repo — ver docs/model_references.md):
    LSSVM/CLASSICOS/zhou2016.pdf
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import BaseLSSVM

logger = logging.getLogger(__name__)


class PCPLSSVm(BaseLSSVM):
    """PCP-LSSVM via incomplete Pivoted Cholesky of the kernel matrix.

    The rank-r Cholesky approximation K ≈ L_r L_r^T is used to map samples
    to a r-dimensional feature space. Then a primal LSSVM is solved
    directly in this reduced space via normal equations.

    Prediction uses only the r pivot points: f(x) = φ(x)·w + b,
    where φ(x) ∈ ℝʳ is computed from K(x, X_pivots) — O(n_test × r).

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth.
    tau : float
        Regularisation parameter.
    rank : int
        Target rank for the Pivoted Cholesky approximation.
    tol : float
        Residual tolerance for early stopping of Cholesky pivoting.
    max_iter : int
        Maximum Cholesky pivot steps (bounded by n).
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        rank: int = 50,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> None:
        super().__init__(sigma=sigma, tau=tau, tol=tol, max_iter=max_iter)
        self.rank = rank

    def _pivoted_cholesky(self, K: NDArray) -> tuple[NDArray, NDArray]:
        """Incomplete Pivoted Cholesky: K ≈ L Lᵀ, L ∈ ℝ^{n×r}.

        Returns (L, piv) where piv[j] is the training-set index of the
        j-th pivot (analogous to Nyström landmark indices).
        """
        n = K.shape[0]
        r = min(self.rank, n)

        L = np.zeros((n, r))
        diag = np.diag(K).copy()
        piv = np.arange(n)

        actual_r = r
        for j in range(r):
            best = np.argmax(diag[j:]) + j
            if diag[best] < self.tol:
                actual_r = j
                break

            piv[[j, best]] = piv[[best, j]]
            diag[[j, best]] = diag[[best, j]]
            L[[j, best], :j] = L[[best, j], :j]

            L[j, j] = np.sqrt(diag[j])

            if j + 1 < n:
                v = (K[piv[j + 1:], piv[j]] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]
                L[j + 1:, j] = v
                diag[j + 1:] -= v ** 2

        return L[:, :actual_r], piv

    def _cholesky_features(self, K_cross: NDArray) -> NDArray:
        """Map test-kernel columns K(X_test, X_pivots) → Cholesky features φ.

        φ(x)[j] = (K(x, X_piv[j]) - φ(x)[:j] · L_diag[:j,j]) / L_diag[j,j]

        This is the triangular solve that gives the same low-rank features
        as the training L, so f(x) = φ(x) · w + b.
        """
        n_test, r = K_cross.shape[0], self.L_.shape[1]
        phi = np.zeros((n_test, r))
        L = self.L_          # (n_train, r) — rows reordered by pivot during fit
        for j in range(r):
            phi[:, j] = (K_cross[:, j] - phi[:, :j] @ L[j, :j]) / L[j, j]
        return phi

    def _solve(self, X: NDArray, y: NDArray) -> None:
        n = len(y)
        K = self.kernel_matrix(X)                    # N×N — needed for Cholesky

        L, piv = self._pivoted_cholesky(K)
        r = L.shape[1]

        # L has rows in PIVOTED order: L[j,:] = features for training point piv[j].
        # Normal equations in reduced space: (τIᵣ + LᵀL) w = Lᵀỹ  where ỹ = y[piv]
        y_piv = y[piv]                               # labels reordered to match L rows
        w = np.linalg.solve(self.tau * np.eye(r) + L.T @ L, L.T @ y_piv)

        # Bias: mean residual (in pivoted space)
        self.bias_ = float(np.mean(y_piv - L @ w))

        # Store pivot indices (analogous to Nyström landmark indices)
        self.pivot_indices_: NDArray = piv[:r]       # r training-set indices
        self.piv_: NDArray = piv                     # full permutation (kept for reference)
        self.w_: NDArray = w                          # primal weights in ℝʳ
        self.L_: NDArray = L                          # Cholesky factor (n×r, pivoted order)
        self.rank_: int = r
        self.n_samples_fit_ = n

        # alpha_ in pivot space for BaseLSSVM compatibility (not used in predict)
        self.alpha_: NDArray = (y_piv - L @ w) / self.tau

        logger.info(
            "PCPLSSVm solved — rank=%d/%d  compression=%.1f%%  bias=%.4f",
            r, self.rank, 100.0 * (1.0 - r / n), self.bias_,
        )

    def decision_function(self, X: NDArray) -> NDArray:
        """f(x) = φ(x)·w + b  using only the r pivot points — O(n_test × r)."""
        check_is_fitted(self, ["w_", "pivot_indices_", "L_", "bias_", "X_train_"])
        X = check_array(X)
        # K(X_test, X_pivots): r kernel evaluations per test point
        K_cross = self.kernel_matrix(X, self.X_train_[self.pivot_indices_])
        phi = self._cholesky_features(K_cross)
        return phi @ self.w_ + self.bias_

    # ── Sparsity interface (analogous to NystromLSSVMColnorm) ─────────────────

    @property
    def n_support_(self) -> int:
        """Number of pivot points used (analogous to Nyström landmarks)."""
        check_is_fitted(self, ["rank_"])
        return self.rank_

    @property
    def sparsity_ratio_(self) -> float:
        """Compression ratio: fraction of training points NOT used as pivots."""
        return 1.0 - self.rank_ / self.n_samples_fit_

"""Pivoted Cholesky Primal LSSVM (PCP-LSSVM).

Direct Method: computes a low-rank approximation of the kernel matrix
using Pivoted (incomplete) Cholesky, then trains a primal LSSVM on the
reduced feature space.

Reference:
    Zhou S. et al., "PCP-LSSVM: A Pivoted Cholesky Decomposition based
    Primal Formulation of the LSSVM Classifier", 2015.
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
    directly in this reduced space.

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth.
    tau : float
        Regularisation parameter.
    rank : int
        Target rank for the Pivoted Cholesky approximation.
    tol : float
        Residual tolerance for stopping the Cholesky approximation.
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

    def _pivoted_cholesky(self, K: NDArray) -> NDArray:
        """Compute incomplete Pivoted Cholesky: K ≈ L L^T.

        Returns L of shape (n, r) where r ≤ self.rank.

        Algorithm: greedily selects the pivot that maximises the residual
        diagonal entry (equivalent to maximising the Frobenius error reduction).
        """
        n = K.shape[0]
        r = min(self.rank, n)

        L = np.zeros((n, r))
        diag = np.diag(K).copy()
        piv = np.arange(n)

        for j in range(r):
            # Find pivot (largest residual diagonal entry)
            best = np.argmax(diag[j:]) + j
            if diag[best] < self.tol:
                r = j
                break

            # Swap pivot into position j
            piv[[j, best]] = piv[[best, j]]
            diag[[j, best]] = diag[[best, j]]
            L[[j, best], :j] = L[[best, j], :j]

            L[j, j] = np.sqrt(diag[j])

            if j + 1 < n:
                v = (K[piv[j + 1 :], piv[j]] - L[j + 1 :, :j] @ L[j, :j]) / L[j, j]
                L[j + 1 :, j] = v
                diag[j + 1 :] -= v**2

        return L[:, :r], piv

    def _solve(self, X: NDArray, y: NDArray) -> None:
        """Pivoted Cholesky decomposition + primal LSSVM solve."""
        n = len(y)
        K = self.kernel_matrix(X)

        # Low-rank Cholesky: K ≈ L L^T  (L is n×r, r ≤ rank)
        L, piv = self._pivoted_cholesky(K)
        r = L.shape[1]
        logger.debug("PCP-LSSVM: used rank r=%d (requested %d).", r, self.rank)

        # Primal LSSVM in the reduced feature space:
        # min_w ½||w||² + 1/(2τ) Σεᵢ²  s.t.  L[i] · w = yᵢ - εᵢ
        # Normal equations: (τI_r + L^T L) w = L^T y
        A = tau_reg = self.tau * np.eye(r) + L.T @ L   # (r, r)
        rhs = L.T @ y                                    # (r,)
        w = np.linalg.solve(A, rhs)                      # primal weights in ℝ^r

        # α in feature space: εᵢ = yᵢ - L[i] w, αᵢ = εᵢ / τ
        epsilon = y - L @ w
        alpha_primal = epsilon / self.tau                 # (n,)

        self.alpha_: NDArray = alpha_primal
        self.w_: NDArray = w
        self.L_: NDArray = L
        self.bias_: float = float(np.mean(y - K @ alpha_primal))

        logger.info(
            "PCPLSSVm solved — rank=%d, n_sv=%d/%d (%.1f%% sparse)",
            r,
            self.n_support_,
            n,
            100.0 * self.sparsity_ratio_,
        )

    def decision_function(self, X: NDArray) -> NDArray:
        """f(x) = K(x, X_train) @ α + b  (primal formulation)."""
        check_is_fitted(self, ["alpha_", "bias_", "X_train_"])
        X = check_array(X)
        K = self.kernel_matrix(X, self.X_train_)
        return K @ self.alpha_ + self.bias_

"""Fast Sparse Approximation LSSVM (FSA-LSSVM).

Direct Method: greedy forward selection of the most informative kernel
basis functions. At each step, adds the training sample that most reduces
the approximation error (functional margin) using a Matching Pursuit-style
criterion.

Reference:
    Jiao L. et al., "Fast Sparse Approximation for Least Squares Support
    Vector Machine", IEEE Transactions on Neural Networks, 2007.

Paper-fonte (BASE TEORICA, fora do repo — ver docs/model_references.md):
    LSSVM/CLASSICOS/tnn07a.pdf
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import BaseLSSVM

logger = logging.getLogger(__name__)


class FSALSSVm(BaseLSSVM):
    """Fast Sparse Approximation LSSVM (Jiao et al., 2007).

    Greedy forward selection: starts with an empty support vector set and
    iteratively adds the sample that maximises the correlation between the
    current residual and the kernel column.

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth.
    tau : float
        Regularisation parameter.
    n_components : int
        Maximum number of basis functions (support vectors) to select.
    tol : float
        Stop early if the residual norm falls below this threshold.
    max_iter : int
        Alias for n_components (interface consistency with BaseLSSVM).
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        n_components: int = 50,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> None:
        super().__init__(sigma=sigma, tau=tau, tol=tol, max_iter=max_iter)
        self.n_components = n_components

    def _solve(self, X: NDArray, y: NDArray) -> None:
        """Greedy kernel basis selection + LSSVM solve on selected set."""
        n = len(y)
        K = self.kernel_matrix(X)

        # Greedy forward selection ─────────────────────────────────────────────
        selected = []
        residual = y.copy().astype(float)
        remaining = set(range(n))

        max_k = min(self.n_components, n)

        for _ in range(max_k):
            if not remaining:
                break

            # Select index with highest |correlation| between residual and K col
            rem_list = np.array(sorted(remaining))
            corr = np.abs(K[:, rem_list].T @ residual)
            best_local = int(np.argmax(corr))
            best = int(rem_list[best_local])

            selected.append(best)
            remaining.discard(best)

            # Update residual: project out the selected kernel column
            k_col = K[:, best]
            k_norm = float(k_col @ k_col)
            if k_norm < 1e-12:
                continue
            residual -= (float(k_col @ residual) / k_norm) * k_col

            if np.linalg.norm(residual) < self.tol:
                logger.debug("FSA-LSSVM: early stop at %d basis functions.", len(selected))
                break

        selected = np.array(sorted(selected))

        # Retrain LSSVM on selected subset ────────────────────────────────────
        X_sub = X[selected]
        y_sub = y[selected]
        n_sub = len(y_sub)

        K_sub = self.kernel_matrix(X_sub)
        Omega = (y_sub[:, None] * K_sub) * y_sub[None, :]
        H = Omega + np.eye(n_sub) / self.tau
        ones = np.ones(n_sub)

        from scipy.sparse.linalg import cg

        eta, _ = cg(H, y_sub, rtol=self.tol, maxiter=self.max_iter)
        mu, _ = cg(H, ones, rtol=self.tol, maxiter=self.max_iter)

        s = float(y_sub @ eta)
        self.bias_: float = float(ones @ eta) / s if abs(s) > 1e-12 else 0.0
        alpha_sub = mu - self.bias_ * eta

        self.alpha_: NDArray = np.zeros(n)
        self.alpha_[selected] = alpha_sub
        self.support_indices_: NDArray = selected

        logger.info(
            "FSALSSVm solved — %d SVs / %d (%.1f%% sparse)",
            len(selected),
            n,
            100.0 * self.sparsity_ratio_,
        )

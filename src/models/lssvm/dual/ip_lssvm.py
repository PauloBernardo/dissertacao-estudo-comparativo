"""Two-step Sparse LSSVM (IP-LSSVM).

Reduction Method: identifies candidate support vectors via QR/SVD
decomposition of the kernel matrix, then retrains on the reduced subset.

Reference:
    Carvalho B.P.R. & Braga A.P., "IP-LSSVM: A two-step sparse
    classifier based on LS-SVM", Pattern Recognition Letters, 2009.

Paper-fonte (BASE TEORICA, fora do repo — ver docs/model_references.md):
    LSSVM/CLASSICOS/carvalho2009.pdf
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import cg

from ..base import BaseLSSVM

logger = logging.getLogger(__name__)


class IPLSSVm(BaseLSSVM):
    """Two-step Sparse LSSVM (Carvalho & Braga, 2009).

    Algorithm:
        Step 1 — Candidate identification:
            Compute pivoted QR decomposition of the kernel matrix.
            The first `k` pivot columns correspond to the most "linearly
            independent" kernel evaluations, i.e., the best candidate SVs.
        Step 2 — Retrain on the selected subset.

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth.
    tau : float
        Regularisation parameter.
    selection_ratio : float
        Fraction of training samples to keep as SVs (0 < r < 1).
    tol : float
        CG convergence tolerance.
    max_iter : int
        Maximum CG iterations.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        selection_ratio: float = 0.20,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> None:
        super().__init__(sigma=sigma, tau=tau, tol=tol, max_iter=max_iter)
        self.selection_ratio = selection_ratio

    def _solve(self, X: NDArray, y: NDArray) -> None:
        """Two-step: identify SVs via pivoted QR, retrain on subset."""
        n = len(y)
        K = self.kernel_matrix(X)

        # ── Step 1: identify candidates via column-pivoted QR ─────────────────
        n_select = max(2, int(self.selection_ratio * n))

        # scipy's QR with column pivoting (mode='economic', pivoting=True)
        import scipy.linalg
        _, _, pivot = scipy.linalg.qr(K, pivoting=True)
        selected = np.sort(pivot[:n_select])

        # ── Step 2: retrain LSSVM on selected subset ──────────────────────────
        X_sub = X[selected]
        y_sub = y[selected]
        n_sub = len(y_sub)

        K_sub = self.kernel_matrix(X_sub)
        Omega = (y_sub[:, None] * K_sub) * y_sub[None, :]
        H = Omega + np.eye(n_sub) / self.tau
        ones = np.ones(n_sub)

        eta, _ = cg(H, y_sub, rtol=self.tol, maxiter=self.max_iter)
        mu, _ = cg(H, ones, rtol=self.tol, maxiter=self.max_iter)

        s = float(y_sub @ eta)
        self.bias_: float = float(ones @ eta) / s if abs(s) > 1e-12 else 0.0
        alpha_sub = mu - self.bias_ * eta

        # Store in full-N format
        self.alpha_: NDArray = np.zeros(n)
        self.alpha_[selected] = alpha_sub
        self.support_indices_: NDArray = selected

        logger.info(
            "IPLSSVm solved — %d SVs / %d (%.1f%% sparse)",
            len(selected),
            n,
            100.0 * self.sparsity_ratio_,
        )

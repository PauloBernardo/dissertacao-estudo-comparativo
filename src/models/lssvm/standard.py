"""Standard LSSVM (Suykens & Vandewalle, 1999).

Solves the dual KKT system:
    [0,  1ᵀ ] [b ]   [0]
    [1, Ω+I/τ] [α ] = [y]

where Ωᵢⱼ = yᵢ yⱼ K(xᵢ, xⱼ) and τ = gamma (regularisation).

The system is solved via Conjugate Gradient (Hestenes-Stiefel variant)
applied to the positive-definite sub-system after elimination of b.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import cg

from .base import BaseLSSVM

logger = logging.getLogger(__name__)


class StandardLSSVM(BaseLSSVM):
    """Classic LSSVM solved via Conjugate Gradient on the dual KKT system.

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth.
    tau : float
        Regularisation (inverse of sparsity — larger = denser).
    tol : float
        CG convergence tolerance.
    max_iter : int
        Maximum CG iterations.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> None:
        super().__init__(sigma=sigma, tau=tau, tol=tol, max_iter=max_iter)

    def _solve(self, X: NDArray, y: NDArray) -> None:
        """Solve the LSSVM dual KKT system via Conjugate Gradient.

        Solves the block KKT system (Eq. 3, Suykens & Vandewalle 1999):
            [0,   y^T ] [b]   [0]
            [y, Ω+I/τ] [α] = [1]
        where Ωᵢⱼ = yᵢ yⱼ K(xᵢ, xⱼ).

        Following Algorithm 1 of the paper (Hestenes-Stiefel CG):
            Hη = y,  Hμ = 1,  s = y^T η
            b  = (1^T η) / s,  α = μ - b·η
        """
        n = len(y)
        K = self.kernel_matrix(X)

        # H = Ω + I/τ  (positive definite — safe for CG)
        Omega = (y[:, None] * K) * y[None, :]   # Ωᵢⱼ = yᵢ yⱼ Kᵢⱼ
        H = Omega + np.eye(n) / self.tau

        ones = np.ones(n)
        eta, info1 = cg(H, y, rtol=self.tol, maxiter=self.max_iter)
        mu, info2 = cg(H, ones, rtol=self.tol, maxiter=self.max_iter)

        if info1 != 0:
            logger.warning("CG (Hη=y) did not converge (info=%d).", info1)
        if info2 != 0:
            logger.warning("CG (Hμ=1) did not converge (info=%d).", info2)

        s = float(y @ eta)                           # s = y^T η > 0
        self.bias_: float = float(ones @ eta) / s    # b = (1^T η) / s
        self.alpha_: NDArray = mu - self.bias_ * eta  # α = μ - b·η

        logger.info(
            "StandardLSSVM solved — n_support=%d / %d (%.1f%% sparse), bias=%.4f",
            self.n_support_,
            n,
            100.0 * self.sparsity_ratio_,
            self.bias_,
        )

"""Pruning LSSVM (P-LSSVM).

Reduction Method: trains a full standard LSSVM, then iteratively prunes
training points with the smallest |αᵢ| and retrains on the reduced set.
Stops when the desired sparsity level is reached or accuracy drops.

Reference:
    Suykens J.A.K. et al., "Least Squares Support Vector Machines",
    World Scientific, 2002. (Chapter 5 — Fixed-Size LSSVM / Pruning)
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import cg

from ..base import BaseLSSVM

logger = logging.getLogger(__name__)


class PruningLSSVM(BaseLSSVM):
    """Pruning LSSVM — dual sparsification via iterative pruning and retraining.

    Algorithm:
        1. Train full LSSVM on all N samples (Suykens CG).
        2. Remove the `floor(pruning_rate * current_n)` samples with
           smallest |αᵢ|.
        3. Retrain LSSVM on the remaining subset.
        4. Repeat until `max_pruning_steps` or a minimum set size is reached.

    The model stores the FULL training set for kernel computation at test
    time, but only the pruned subset's αᵢ are non-zero.

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth.
    tau : float
        Regularisation parameter.
    pruning_rate : float
        Fraction of current support vectors to prune per step (0 < r < 1).
    max_pruning_steps : int
        Maximum number of prune-retrain cycles.
    min_sv_fraction : float
        Stop pruning when the SV set reaches this fraction of the original N.
    tol : float
        CG convergence tolerance.
    max_iter : int
        Maximum CG iterations per solve.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        pruning_rate: float = 0.10,
        max_pruning_steps: int = 20,
        min_sv_fraction: float = 0.02,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> None:
        super().__init__(sigma=sigma, tau=tau, tol=tol, max_iter=max_iter)
        self.pruning_rate = pruning_rate
        self.max_pruning_steps = max_pruning_steps
        self.min_sv_fraction = min_sv_fraction

    def _solve_on_subset(
        self, X_sub: NDArray, y_sub: NDArray
    ) -> tuple[NDArray, float]:
        """Solve the LSSVM KKT system on a subset of the training data.

        Returns
        -------
        alpha : NDArray of shape (n_sub,)
        bias : float
        """
        n_sub = len(y_sub)
        K_sub = self.kernel_matrix(X_sub)
        Omega = (y_sub[:, None] * K_sub) * y_sub[None, :]
        H = Omega + np.eye(n_sub) / self.tau
        ones = np.ones(n_sub)

        eta, _ = cg(H, y_sub, rtol=self.tol, maxiter=self.max_iter)
        mu, _ = cg(H, ones, rtol=self.tol, maxiter=self.max_iter)

        s = float(y_sub @ eta)
        b = float(ones @ eta) / s if abs(s) > 1e-12 else 0.0
        alpha = mu - b * eta
        return alpha, b

    def _solve(self, X: NDArray, y: NDArray) -> None:
        """Iterative pruning: train → prune smallest |αᵢ| → retrain."""
        n = len(y)
        min_sv = max(2, int(self.min_sv_fraction * n))

        # Full training set indices (active set)
        active = np.arange(n)

        # Initial full solve
        alpha_sub, bias = self._solve_on_subset(X[active], y[active])

        for step in range(self.max_pruning_steps):
            n_active = len(active)
            if n_active <= min_sv:
                logger.info("Pruning stopped: reached min SV set (%d).", n_active)
                break

            # How many to prune this step
            n_prune = max(1, int(self.pruning_rate * n_active))
            n_keep = n_active - n_prune
            if n_keep < min_sv:
                n_keep = min_sv
                n_prune = n_active - n_keep

            # Keep indices with largest |αᵢ|
            keep_mask = np.argsort(np.abs(alpha_sub))[-n_keep:]
            active = active[keep_mask]

            # Retrain on pruned subset
            alpha_sub, bias = self._solve_on_subset(X[active], y[active])

            logger.debug(
                "Pruning step %d: %d → %d support vectors.",
                step + 1,
                n_active,
                len(active),
            )

        # Store in full-N format (zeros for pruned samples)
        self.alpha_: NDArray = np.zeros(n)
        self.alpha_[active] = alpha_sub
        self.bias_: float = bias
        self.support_indices_: NDArray = active

        logger.info(
            "PruningLSSVM solved — %d SVs / %d (%.1f%% sparse)",
            len(active),
            n,
            100.0 * self.sparsity_ratio_,
        )

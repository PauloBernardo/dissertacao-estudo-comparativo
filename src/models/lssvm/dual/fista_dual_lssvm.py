"""Sparse LSSVM in Dual via FISTA (Fast Iterative Shrinkage-Thresholding).

Solves the dual LSSVM problem with L1 regularisation directly on the dual
variables (Lagrange multipliers), as opposed to the primal formulation of
Marinho et al. which uses L1 on Cholesky-space coefficients.

Dual LSSVM LASSO (Beck & Teboulle, 2009 applied to LSSVM dual):

    min_α  ½αᵀΩα - 1ᵀα + λ‖α‖₁
where:
    Ω  = YKY + (1/τ)I      Y = diag(y), K = RBF kernel matrix
    α  ∈ ℝⁿ                dual variables (Lagrange multipliers)
    λ  > 0                 L1 regularisation — controls sparsity

FISTA iteration:
    η      = 1/L                          step size (L = max_eig(Ω))
    v_k    = y_k - η(Ω y_k - 1)          gradient step on extrapolated y
    α_k    = S_{η·λ}(v_k)                soft-threshold (prox of λ‖·‖₁)
    t_{k+1} = (1 + √(1+4t_k²)) / 2       FISTA momentum
    y_{k+1} = α_k + (t_k-1)/t_{k+1} (α_k - α_{k-1})

Prediction:
    f(x) = Σ_i α_i y_i K(x_i, x) + b    (dual decision function)

Key difference from primal FISTA/ADMM:
    - Ω is positive definite by construction → better conditioned
    - α_i = 0 means point i is exactly NOT a support vector
    - λ_max = 1 always (gradient at α=0 is -1, scale-invariant)

Paper-fonte (BASE TEORICA, fora do repo — ver docs/model_references.md):
    LSSVM/CLASSICOS/beck2009.pdf
"""

from __future__ import annotations

import logging
from math import log, sqrt

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import BaseLSSVM

logger = logging.getLogger(__name__)


class DualFISTALSSVM(BaseLSSVM):
    """Sparse LSSVM in Dual via FISTA with Nesterov momentum.

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth (γ = 1/(2σ²)).
    tau : float
        LSSVM regularisation — controls margin/slack trade-off.
        Enters Ω as (1/τ)I diagonal term.
    lambda_ : float or None
        L1 regularisation on dual variables. If None, auto-set to
        0.1 (10% of λ_max=1, which is scale-invariant in the dual).
    tol : float
        Convergence tolerance on ‖α_k - α_{k-1}‖.
    max_iter : int
        Maximum FISTA iterations.
    adaptive_restart : bool
        O'Donoghue & Candès gradient-based restart.
    estimate_bias : bool
        Whether to estimate intercept as mean(y - K α_eff).
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        lambda_: float | None = None,
        tol: float = 1e-6,
        max_iter: int = 5000,
        adaptive_restart: bool = True,
        estimate_bias: bool = True,
    ) -> None:
        super().__init__(sigma=sigma, tau=tau, tol=tol, max_iter=max_iter)
        self.lambda_ = lambda_
        self.adaptive_restart = adaptive_restart
        self.estimate_bias = estimate_bias

    @staticmethod
    def _soft_threshold(v: NDArray, threshold: float) -> NDArray:
        return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)

    def _solve(self, X: NDArray, y: NDArray) -> None:
        n = len(y)

        # ── Kernel matrix ─────────────────────────────────────────────────────
        K = self.kernel_matrix(X)

        # ── Omega = YKY + (1/τ)I ─────────────────────────────────────────────
        # YKY: Omega_ij = y_i * y_j * K_ij
        Omega = np.outer(y, y) * K + (1.0 / self.tau) * np.eye(n)

        # ── Lipschitz constant L = max_eigenvalue(Ω) ─────────────────────────
        L = float(np.linalg.eigvalsh(Omega).max())
        eta = 1.0 / L      # step size

        # ── Auto-select λ ─────────────────────────────────────────────────────
        # In dual, grad f(0) = -1 → λ_max = 1 (scale-invariant).
        # Default: 10% of λ_max — mild sparsity, let Optuna refine.
        lam = self.lambda_ if self.lambda_ is not None else 0.1
        threshold = eta * lam

        ones = np.ones(n)

        # ── FISTA loop ────────────────────────────────────────────────────────
        alpha      = np.zeros(n)
        alpha_prev = np.zeros(n)
        y_ext      = np.zeros(n)   # extrapolated point
        t          = 1.0

        converged = False
        for k in range(self.max_iter):
            # proximal-gradient step on extrapolated point
            grad    = Omega @ y_ext - ones    # ∇f(y_ext) = Ω y - 1
            v       = y_ext - eta * grad
            alpha_new = self._soft_threshold(v, threshold)

            # convergence check
            if float(np.linalg.norm(alpha_new - alpha)) < self.tol:
                alpha = alpha_new
                converged = True
                break

            # FISTA momentum
            t_new    = (1.0 + sqrt(1.0 + 4.0 * t * t)) / 2.0
            momentum = (t - 1.0) / t_new

            if self.adaptive_restart:
                # restart when step opposes gradient: (α_new - α_prev)ᵀ(α_new - v) > 0
                if float(np.dot(alpha_new - alpha_prev, alpha_new - v)) > 0:
                    t_new    = 1.0
                    momentum = 0.0

            alpha_prev = alpha.copy()
            alpha      = alpha_new
            t          = t_new
            y_ext      = alpha + momentum * (alpha - alpha_prev)

        self.n_iter_: int = k + 1
        self.converged_ = converged
        self.lambda_used_ = lam

        # ── Store α_eff = α ⊙ y so BaseLSSVM decision function works ─────────
        # f(x) = K(X_train, x) @ alpha_eff + b = Σ α_i y_i K(x_i,x) + b
        alpha_eff = alpha * y
        self.alpha_ = alpha_eff

        self.bias_: float = (
            float(np.mean(y - K @ self.alpha_)) if self.estimate_bias else 0.0
        )

        logger.info(
            "DualFISTALSSVM — n_sv=%d/%d (%.1f%% sparse), %d iters, "
            "λ=%.4f, L=%.4f, bias=%.4f, converged=%s",
            self.n_support_, n, 100.0 * self.sparsity_ratio_,
            self.n_iter_, lam, L, self.bias_, converged,
        )

    def decision_function(self, X: NDArray) -> NDArray:
        check_is_fitted(self, ["alpha_", "bias_", "X_train_"])
        X = check_array(X)
        return self._kernel_predict(X, self.alpha_)

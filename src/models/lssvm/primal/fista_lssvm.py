"""Sparse LSSVM in Primal via FISTA (Fast Iterative Shrinkage-Thresholding).

Solves the same LASSO problem as ADMMNesterovLSSVM but using a pure
proximal-gradient approach instead of variable splitting:

    min_α  ½‖Aα - b‖₂² + λ‖α‖₁
where:
    K̃  = K + σ_tik·I = LLᵀ   (Cholesky; L lower-triangular)
    A  = Lᵀ                    (upper-triangular, n×n)
    b  = (τI + K̃)⁻¹ Lᵀ y     (n-vector)

FISTA iteration (Beck & Teboulle, 2009):
    v_k    = y_k + (1/L)(r - K̃ y_k)      proximal-gradient step on extrapolated y
    x_k    = S_{λ/L}(v_k)                 soft-threshold (prox of λ‖·‖₁)
    t_{k+1} = (1 + √(1 + 4t_k²)) / 2     FISTA momentum
    y_{k+1} = x_k + ((t_k-1)/t_{k+1})(x_k - x_{k-1})   Nesterov extrapolation

where L = max_eigenvalue(K̃) is the Lipschitz constant of ∇f.

Adaptive restart (O'Donoghue & Candès, 2015):
    Restart when: (x_k - x_{k-1})ᵀ (x_k - v_k) > 0
    (gradient condition: step and function gradient point in the same direction)

Comparison with ADMM:
- FISTA is purely first-order (gradient + prox); ADMM uses a linear system solve
  which implicitly provides second-order information via (K̃+ρI)⁻¹.
- ADMM typically converges in fewer iterations; FISTA is simpler to implement.
- Both achieve O(1/k²) convergence with Nesterov momentum.
"""

from __future__ import annotations

import logging
from math import log, sqrt

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import BaseLSSVM

logger = logging.getLogger(__name__)


class FISTANesterovLSSVM(BaseLSSVM):
    """Sparse LSSVM in Primal via FISTA with Nesterov momentum.

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth (γ = 1/(2σ²)).
    tau : float
        LSSVM regularisation parameter.
    lambda_ : float or None
        L1 regularisation. If None, auto-set to sqrt(2 log n) * ‖b‖∞
        (scale-adaptive, same fix as ADMMNesterovLSSVM).
    tol : float
        Convergence tolerance on ‖x_k - x_{k-1}‖.
    max_iter : int
        Maximum FISTA iterations.
    adaptive_restart : bool
        O'Donoghue & Candès gradient-based restart: reset momentum when
        the step direction conflicts with the gradient direction.
    tikhonov_reg : float
        Tikhonov regularisation for Cholesky stability: K̃ = K + reg·I.
    estimate_bias : bool
        Whether to estimate an intercept as b = mean(y - Kα).
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        lambda_: float | None = None,
        tol: float = 1e-6,
        max_iter: int = 5000,
        adaptive_restart: bool = True,
        tikhonov_reg: float = 0.01,
        estimate_bias: bool = True,
    ) -> None:
        super().__init__(sigma=sigma, tau=tau, tol=tol, max_iter=max_iter)
        self.lambda_ = lambda_
        self.adaptive_restart = adaptive_restart
        self.tikhonov_reg = tikhonov_reg
        self.estimate_bias = estimate_bias

    @staticmethod
    def _soft_threshold(v: NDArray, threshold: float) -> NDArray:
        return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)

    def _solve(self, X: NDArray, y: NDArray) -> None:
        n = len(y)

        # ── Kernel + Cholesky ─────────────────────────────────────────────────
        K = self.kernel_matrix(X)
        K_tik = K + self.tikhonov_reg * np.eye(n)

        try:
            L_chol = np.linalg.cholesky(K_tik)
        except np.linalg.LinAlgError:
            logger.warning("Cholesky failed — adding extra regularisation 1e-3.")
            L_chol = np.linalg.cholesky(K_tik + 1e-3 * np.eye(n))

        A = L_chol.T                                     # upper triangular n×n
        Pt_y = A @ y
        b_vec = np.linalg.solve(self.tau * np.eye(n) + K_tik, Pt_y)
        r = L_chol @ b_vec                               # = Aᵀ b_vec

        # ── Lipschitz constant L = max_eigenvalue(K̃) ─────────────────────────
        eig_max = float(np.linalg.eigvalsh(K_tik).max())
        step = 1.0 / eig_max if eig_max > 0 else 1.0    # 1/L

        # ── Auto-select λ (scale-adaptive) ────────────────────────────────────
        if self.lambda_ is not None:
            lam = self.lambda_
        else:
            b_scale = float(np.abs(b_vec).max())
            b_scale = b_scale if b_scale > 0 else 1.0
            lam = sqrt(2.0 * log(n)) * b_scale

        threshold = lam * step                           # λ/L for soft-threshold

        # ── FISTA loop ────────────────────────────────────────────────────────
        x     = np.zeros(n)   # current iterate
        x_prev= np.zeros(n)   # previous iterate
        y_ext = np.zeros(n)   # extrapolated point
        t     = 1.0

        converged = False
        for k in range(self.max_iter):
            # proximal-gradient step on extrapolated point y_ext
            grad = K_tik @ y_ext - r                    # ∇f(y_ext) = K̃ y - r
            v = y_ext - step * grad                      # gradient step
            x_new = self._soft_threshold(v, threshold)  # proximal step

            # convergence check
            diff = x_new - x
            if float(np.linalg.norm(diff)) < self.tol:
                x = x_new
                converged = True
                break

            # FISTA momentum
            t_new = (1.0 + sqrt(1.0 + 4.0 * t * t)) / 2.0
            momentum = (t - 1.0) / t_new

            if self.adaptive_restart:
                # O'Donoghue restart: restart when step opposes gradient direction
                # condition: (x_new - x_prev)ᵀ (x_new - v) > 0
                if float(np.dot(x_new - x_prev, x_new - v)) > 0:
                    t_new = 1.0
                    momentum = 0.0

            x_prev = x.copy()
            x = x_new
            t = t_new
            y_ext = x + momentum * (x - x_prev)

        self.n_iter_: int = k + 1
        self.alpha_ = x
        self.converged_ = converged
        self.lambda_used_ = lam
        self.step_used_ = step

        self.bias_: float = (
            float(np.mean(y - K @ self.alpha_)) if self.estimate_bias else 0.0
        )

        logger.info(
            "FISTANesterovLSSVM — n_sv=%d/%d (%.1f%% sparse), %d iters, "
            "λ=%.4f, L=%.4f, bias=%.4f, converged=%s",
            self.n_support_, n, 100.0 * self.sparsity_ratio_,
            self.n_iter_, lam, eig_max, self.bias_, converged,
        )

    def decision_function(self, X: NDArray) -> NDArray:
        check_is_fitted(self, ["alpha_", "bias_", "X_train_"])
        X = check_array(X)
        K = self.kernel_matrix(X, self.X_train_)
        return K @ self.alpha_ + self.bias_

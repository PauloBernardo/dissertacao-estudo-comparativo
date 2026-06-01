"""Sparse LSSVM in Primal via Nesterov-Accelerated ADMM with Elastic Net.

Implements Fast LSSVM-ADMM from:
    Marinho et al. "Sparse Least Square SVM in Primal via Nesterov
    Accelerated Alternating Directions Method of Multipliers", IWANN 2023.

Extended with Elastic Net regularisation (L1 + L2) for unconditional
Nesterov stability: the L2 term injects strong convexity, satisfying the
Goldstein condition without requiring manual restart tuning.

Mathematical formulation
------------------------
Primal LSSVM (Eq. 8):
    min_{w,ε}  ½ wᵀw + 1/(2τ) Σ εᵢ²   s.t. yᵢ = wᵀφ(xᵢ) + εᵢ

After representer theorem (Eq. 13–16), reduce to Elastic Net:
    min_α  ½‖Aα - b‖₂² + λ₁/2‖α‖₁ + λ₂/2‖α‖₂²
where:
    K̃  = K + σ_tik·I = LLᵀ   (Cholesky; L lower-triangular)
    A  = Lᵀ                    (upper-triangular, n×n)
    b  = (τI + K̃)⁻¹ Lᵀ y     (n-vector)

Setting λ₂=0 recovers the original LASSO formulation.

Fast ADMM with FISTA momentum (Algorithm 2+3):
    α^{k+1} = (K̃ + ρI)⁻¹ (r + ρ(ẑ^k - û^k))   where r = Aᵀb
    z^{k+1} = S_{λ₁/(2ρ)}(α^{k+1} + û^k) / (1 + λ₂/ρ)   Elastic Net prox
    u^{k+1} = û^k + α^{k+1} - z^{k+1}
    t^{k+1} = (1 + √(1 + 4t_k²)) / 2              FISTA momentum
    ẑ^{k+1} = z^{k+1} + ((t_k-1)/t_{k+1})(z^{k+1} - z^k)
    û^{k+1} = u^{k+1} + ((t_k-1)/t_{k+1})(u^{k+1} - u^k)

When λ₂=0 the z-step reduces to S_{λ₁/(2ρ)}, matching the original paper.

Adaptive restart (Fast_ADMM_restart variant):
    c_k = (1/η)(‖u^{k+1} - û^k‖² + η²‖z^{k+1} - ẑ^k‖²)
    If c_k < η·c_{k-1}: keep momentum; else: reset t=1, ẑ=z, û=u.

Prediction:
    f(x) = αᵀ K(X_train, x)    (no bias in primal; bias estimated optionally)
"""

from __future__ import annotations

import logging
from math import log, sqrt

import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import BaseLSSVM

logger = logging.getLogger(__name__)


class ADMMNesterovLSSVM(BaseLSSVM):
    """Sparse LSSVM in Primal via Nesterov-Accelerated ADMM (Marinho et al.).

    Extended with Elastic Net regularisation for unconditional Nesterov stability.

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth (γ = 1/(2σ²)).
    tau : float
        LSSVM regularisation parameter (bias/variance trade-off).
    lambda_ : float or None
        L1 regularisation — controls sparsity. If None, auto-set to
        √(2 log n) · ‖b‖∞  (scale-adaptive, avoids over-thresholding).
    lambda_ratio : float or None
        Multiplier for the auto-set λ. Optuna should tune this (e.g. 1e-4 to 1.0)
        instead of `lambda_` to preserve the scale-adaptive property.
    lambda2_ : float
        L2 regularisation for Elastic Net. Injects strong convexity for
        unconditional Nesterov stability (Goldstein condition). Default 0.0
        reduces to the original LASSO formulation.
    rho : float or None
        ADMM augmented-Lagrangian penalty. If None, auto-set to
        1/max_eigenvalue(AᵀA), matching the reference implementation.
    tol : float
        Convergence tolerance on primal and dual residuals.
    max_iter : int
        Maximum ADMM iterations.
    use_nesterov : bool
        Apply FISTA momentum (Algorithm 2). If False, runs plain ADMM.
    adaptive_restart : bool
        Lyapunov-based restart (Fast_ADMM_restart): reset momentum when
        the convergence indicator stops decreasing.
    restart_eta : float
        Restart threshold η in [0, 1]. Momentum reset when c_k ≥ η·c_{k-1}.
    tikhonov_reg : float
        Tikhonov regularisation added to K for Cholesky stability:
        K̃ = K + tikhonov_reg·I.
    estimate_bias : bool
        Whether to estimate an intercept as b = mean(y - Kα). The pure
        primal formulation has no bias; enabling this helps on imbalanced data.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        lambda_: float | None = None,
        lambda_ratio: float | None = None,
        lambda2_: float = 0.0,
        rho: float | None = None,
        tol: float = 1e-6,
        max_iter: int = 5000,
        use_nesterov: bool = True,
        adaptive_restart: bool = True,
        restart_eta: float = 0.999,
        tikhonov_reg: float = 0.01,
        estimate_bias: bool = True,
    ) -> None:
        super().__init__(sigma=sigma, tau=tau, tol=tol, max_iter=max_iter)
        self.lambda_ = lambda_
        self.lambda_ratio = lambda_ratio
        self.lambda2_ = lambda2_
        self.rho = rho
        self.use_nesterov = use_nesterov
        self.adaptive_restart = adaptive_restart
        self.restart_eta = restart_eta
        self.tikhonov_reg = tikhonov_reg
        self.estimate_bias = estimate_bias

    # ── Soft-thresholding ─────────────────────────────────────────────────────

    @staticmethod
    def _soft_threshold(v: NDArray, threshold: float) -> NDArray:
        """S_κ(v) = sign(v) · max(|v| - κ, 0)."""
        return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)

    # ── Core solver ───────────────────────────────────────────────────────────

    def _solve(self, X: NDArray, y: NDArray) -> None:
        n = len(y)

        # ── Step 1: Kernel matrix + Cholesky ──────────────────────────────────
        K = self.kernel_matrix(X)
        K_tik = K + self.tikhonov_reg * np.eye(n)

        try:
            L = np.linalg.cholesky(K_tik)
        except np.linalg.LinAlgError:
            logger.warning("Cholesky failed — adding extra regularisation 1e-3.")
            L = np.linalg.cholesky(K_tik + 1e-3 * np.eye(n))

        # A = Lᵀ  (n×n upper triangular)
        A = L.T

        # u = (τI + K̃)⁻¹ Lᵀ y
        Pt_y = A @ y
        b_vec = np.linalg.solve(self.tau * np.eye(n) + K_tik, Pt_y)

        # r = A u = L b_vec  (cached RHS contribution)
        r = L @ b_vec

        # ── Step 2: Auto-select ρ and λ if not provided ───────────────────────
        # ρ from max eigenvalue of AᵀA = K̃  (reference: rho = 1/max_eig)
        AtA = A.T @ A   # = K̃  (symmetric)
        eig_max = float(np.linalg.eigvalsh(AtA).max())
        rho = self.rho if self.rho is not None else (1.0 / eig_max if eig_max > 0 else 1.0)

        # λ auto-set: √(2 log n) scaled by ‖b‖∞ so the threshold is invariant
        # to the alpha scale (O(1/n) in kernel LSSVM, not O(1) assumed by CS).
        if self.lambda_ is not None:
            lam = self.lambda_
        else:
            b_scale = float(np.abs(b_vec).max())
            b_scale = b_scale if b_scale > 0 else 1.0
            lam = sqrt(2.0 * log(n)) * b_scale
            if self.lambda_ratio is not None:
                lam *= self.lambda_ratio

        # Soft-threshold: reference uses λ/2 convention → threshold = λ/(2ρ)
        threshold = lam / (2.0 * rho)

        # Elastic Net scaling for z-step: 1/(1 + λ₂/ρ)
        elastic_scale = 1.0 / (1.0 + self.lambda2_ / rho) if self.lambda2_ > 0.0 else 1.0

        # ── Step 3: Precompute (K̃ + ρI) factorisation ────────────────────────
        M = K_tik + rho * np.eye(n)
        L_M = np.linalg.cholesky(M)

        # ── Step 4: ADMM loop ─────────────────────────────────────────────────
        alpha = np.zeros(n)
        z = np.zeros(n)
        u = np.zeros(n)
        z_hat = np.zeros(n)
        u_hat = np.zeros(n)
        t = 1.0           # FISTA momentum variable

        # Lyapunov-based restart bookkeeping
        c_prev = np.inf

        converged = False
        for k in range(self.max_iter):
            # α-step
            rhs = r + rho * (z_hat - u_hat)
            alpha_new = scipy.linalg.cho_solve((L_M, True), rhs)

            # z-step: Elastic Net proximal — S_{λ₁/(2ρ)}(v) / (1 + λ₂/ρ)
            z_prev = z.copy()
            z_new = self._soft_threshold(alpha_new + u_hat, threshold) * elastic_scale

            # u-step
            u_prev = u.copy()
            u_new = u_hat + alpha_new - z_new

            # Convergence check
            primal_res = float(np.linalg.norm(alpha_new - z_new))
            dual_res = float(rho * np.linalg.norm(z_new - z_prev))
            if primal_res < self.tol and dual_res < self.tol:
                z = z_new
                alpha = alpha_new
                converged = True
                break

            # Nesterov / FISTA momentum
            if self.use_nesterov:
                # FISTA formula: t_{k+1} = (1 + √(1 + 4t_k²)) / 2
                t_new = (1.0 + sqrt(1.0 + 4.0 * t * t)) / 2.0
                momentum = (t - 1.0) / t_new

                if self.adaptive_restart:
                    # Lyapunov condition: c_k = ‖Δu‖²/η + η‖Δz‖²
                    eta = self.restart_eta
                    delta_u = u_new - u_hat
                    delta_z = z_new - z_hat
                    c = float(np.dot(delta_u, delta_u) / eta +
                              eta * np.dot(delta_z, delta_z))

                    if c < eta * c_prev:
                        # Good progress — keep momentum
                        z_hat = z_new + momentum * (z_new - z_prev)
                        u_hat = u_new + momentum * (u_new - u_prev)
                        c_prev = c
                    else:
                        # Restart: reset momentum
                        t_new = 1.0
                        z_hat = z_new.copy()
                        u_hat = u_new.copy()
                        c_prev = c / eta
                else:
                    z_hat = z_new + momentum * (z_new - z_prev)
                    u_hat = u_new + momentum * (u_new - u_prev)

                t = t_new
            else:
                z_hat = z_new.copy()
                u_hat = u_new.copy()

            z = z_new
            u = u_new
            alpha = alpha_new

        self.n_iter_: int = k + 1
        self.alpha_ = z       # z is always sparse (passes through soft-threshold)
        self.converged_ = converged
        self.rho_used_ = rho
        self.lambda_used_ = lam

        if not converged:
            logger.warning(
                "ADMMNesterovLSSVM did not converge in %d iters (primal_res=%.2e).",
                self.max_iter, float(np.linalg.norm(alpha - z)),
            )

        # Optional bias estimate: b = mean(y - Kα)
        self.bias_: float = float(np.mean(y - K @ self.alpha_)) if self.estimate_bias else 0.0

        logger.info(
            "ADMMNesterovLSSVM — n_sv=%d/%d (%.1f%% sparse), %d iters, "
            "λ₁=%.4f, λ₂=%.4f, ρ=%.4f, bias=%.4f, converged=%s",
            self.n_support_, n, 100.0 * self.sparsity_ratio_,
            self.n_iter_, lam, self.lambda2_, rho, self.bias_, converged,
        )

    # ── Primal decision function ──────────────────────────────────────────────

    def decision_function(self, X: NDArray) -> NDArray:
        """f(x) = αᵀ K(X_train, x) + b."""
        check_is_fitted(self, ["alpha_", "bias_", "X_train_"])
        X = check_array(X)
        K = self.kernel_matrix(X, self.X_train_)
        return K @ self.alpha_ + self.bias_

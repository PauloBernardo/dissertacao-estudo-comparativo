"""Sparse LSSVM in Primal via Nesterov-Accelerated ADMM.

Implements Algorithm 3 (Fast LSSVM-ADMM) from:
    Marinho et al. "Sparse Least Square SVM in Primal via Nesterov
    Accelerated Alternating Directions Method of Multipliers", IWANN 2023.

Mathematical formulation (following paper notation):
----------------------------------------------------
Primal LSSVM problem (Eq. 8):
    min_{w,ε}  f(w,ε) = ½ wᵀw + 1/(2τ) Σ εᵢ²
    s.t.       yᵢ = wφ(xᵢ) + εᵢ,  i = 1,...,N

After KKT conditions and representer theorem (Eq. 11):
    (τK + KᵀK) α = Ky   →   (τI + K) α = y

Tikhonov-regularised kernel matrix (Eq. 13):
    K̃ = K + σ_tik·I = PP^T   (Cholesky decomposition)

Reformulated LASSO problem (Eq. 16):
    min_α  ‖Aα - u‖₂² + λ‖α‖₁
where A = P^T,  u = (τI + PP^T)⁻¹ P^T y.

ADMM update rules (Eq. 20):
    α^{k+1} = (PP^T + ρI)⁻¹ (P(τI + PP^T)⁻¹ P^T y + ρ(ẑ^k - û^k))
    z^{k+1} = S_{λ/ρ}(α^{k+1} + û^k)
    u^{k+1} = û^k + α^{k+1} - z^{k+1}

Fast ADMM adds Nesterov momentum (Algorithm 2):
    β_{k+1}  = (1 + √(1 + β_k²)) / 2
    ẑ^{k+1}  = z^{k+1} + (β_k / β_{k+1})(z^{k+1} - z^k)
    û^{k+1}  = u^{k+1} + (β_k / β_{k+1})(u^{k+1} - u^k)

Prediction (Eq. 12):
    f(x) = Σ_{i∈B} αᵢ K(xᵢ, x)    [no bias in primal formulation]
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import BaseLSSVM

logger = logging.getLogger(__name__)


class ADMMNesterovLSSVM(BaseLSSVM):
    """Sparse LSSVM in Primal via Nesterov-Accelerated ADMM (Marinho et al.).

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth (γ = 1/(2σ²)).
    tau : float
        LSSVM regularisation parameter (trade-off bias/variance).
    lambda_ : float
        L1 (LASSO) regularisation parameter — controls sparsity.
        Larger values yield fewer support vectors.
    rho : float
        ADMM penalty parameter. Controls convergence speed.
    tol : float
        Primal and dual residual tolerance for convergence.
    max_iter : int
        Maximum ADMM iterations.
    use_nesterov : bool
        Whether to apply Nesterov momentum (Algorithm 2). If False,
        runs standard ADMM without acceleration.
    adaptive_restart : bool
        If True, resets Nesterov momentum when the primal residual
        increases significantly (stabilises oscillating trajectories).
        Mentioned as future work in the paper; enabled by default.
    tikhonov_reg : float
        Small positive constant added to K before Cholesky to ensure
        positive definiteness: K̃ = K + tikhonov_reg·I.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        lambda_: float = 1.0,
        rho: float = 1.0,
        tol: float = 1e-6,
        max_iter: int = 500,
        use_nesterov: bool = True,
        adaptive_restart: bool = True,
        tikhonov_reg: float = 1e-6,
    ) -> None:
        super().__init__(sigma=sigma, tau=tau, tol=tol, max_iter=max_iter)
        self.lambda_ = lambda_
        self.rho = rho
        self.use_nesterov = use_nesterov
        self.adaptive_restart = adaptive_restart
        self.tikhonov_reg = tikhonov_reg

    # ── Soft-thresholding operator ────────────────────────────────────────────

    @staticmethod
    def _soft_threshold(v: NDArray, threshold: float) -> NDArray:
        """S_κ(v) = sign(v) · max(|v| - κ, 0) (element-wise)."""
        return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)

    # ── Core solver ───────────────────────────────────────────────────────────

    def _solve(self, X: NDArray, y: NDArray) -> None:
        """Run Fast LSSVM-ADMM (Algorithms 2+3 from the paper).

        After convergence, populates:
        - self.alpha_  : primal Lagrange multipliers (sparse)
        - self.bias_   : estimated intercept
        - self.n_iter_ : iterations until convergence
        """
        n = len(y)

        # ── Step 1 of Algorithm 3: kernel + Cholesky factorisation ────────────
        K = self.kernel_matrix(X)  # (n, n) kernel matrix

        # K̃ = K + σ_tik·I  — Tikhonov regularisation for numerical stability
        K_tik = K + self.tikhonov_reg * np.eye(n)

        # Cholesky: K̃ = LL^T  →  P = L (lower triangular), A = P^T
        try:
            L = np.linalg.cholesky(K_tik)
        except np.linalg.LinAlgError:
            logger.warning("Cholesky failed; adding extra regularisation 1e-4.")
            L = np.linalg.cholesky(K_tik + 1e-4 * np.eye(n))

        # b_vec = (τI + K̃)⁻¹ P^T y   [vector used as RHS in ADMM]
        # Solved via the linear system (τI + K̃) b_vec = P^T y
        Pt_y = L.T @ y                                      # P^T y
        tau_eye = self.tau * np.eye(n)
        b_vec = np.linalg.solve(tau_eye + K_tik, Pt_y)     # (n,)

        # r = A^T b_vec = P b_vec   [precomputed ADMM RHS contribution]
        r = L @ b_vec                                        # (n,)

        # ── Precompute (A^T A + ρI) = (K̃ + ρI) and its Cholesky ─────────────
        # This is reused every ADMM iteration (caching the factorisation).
        M = K_tik + self.rho * np.eye(n)                   # (n, n)
        L_M = np.linalg.cholesky(M)                        # lower triangular

        # ── Step 2 of Algorithm 3: Fast ADMM (Algorithm 2) ───────────────────
        alpha = np.zeros(n)
        z = np.zeros(n)       # actual z^k
        u = np.zeros(n)       # actual u^k
        z_hat = np.zeros(n)   # ẑ^k  (extrapolated, used in α and z updates)
        u_hat = np.zeros(n)   # û^k  (extrapolated, used in α and z updates)
        beta = 1.0            # β_1 = 1
        threshold = self.lambda_ / self.rho   # λ/ρ for soft-thresholding

        prev_primal_res = np.inf
        converged = False

        for k in range(self.max_iter):
            # α^{k+1} = (K̃ + ρI)⁻¹ (r + ρ(ẑ^k - û^k))
            rhs = r + self.rho * (z_hat - u_hat)
            alpha_new = scipy.linalg.cho_solve((L_M, True), rhs)

            # z^{k+1} = S_{λ/ρ}(α^{k+1} + û^k)
            z_new = self._soft_threshold(alpha_new + u_hat, threshold)

            # u^{k+1} = û^k + α^{k+1} - z^{k+1}
            u_new = u_hat + alpha_new - z_new

            # ── Convergence check ─────────────────────────────────────────────
            primal_res = float(np.linalg.norm(alpha_new - z_new))
            dual_res = float(self.rho * np.linalg.norm(z_new - z))

            if primal_res < self.tol and dual_res < self.tol:
                z = z_new
                alpha = alpha_new
                converged = True
                break

            # ── Nesterov momentum update (Algorithm 2) ────────────────────────
            if self.use_nesterov:
                beta_new = (1.0 + np.sqrt(1.0 + beta**2)) / 2.0

                # Adaptive restart: reset momentum on oscillation
                if self.adaptive_restart and primal_res > prev_primal_res * 1.5:
                    beta_new = 1.0
                    z_hat_new = z_new.copy()
                    u_hat_new = u_new.copy()
                else:
                    # ẑ^{k+1} = z^{k+1} + (β_k/β_{k+1})(z^{k+1} - z^k)
                    # û^{k+1} = u^{k+1} + (β_k/β_{k+1})(u^{k+1} - u^k)
                    momentum = beta / beta_new
                    z_hat_new = z_new + momentum * (z_new - z)
                    u_hat_new = u_new + momentum * (u_new - u)
            else:
                beta_new = beta
                z_hat_new = z_new.copy()
                u_hat_new = u_new.copy()

            # Advance state
            z = z_new
            u = u_new
            z_hat = z_hat_new
            u_hat = u_hat_new
            beta = beta_new
            alpha = alpha_new
            prev_primal_res = primal_res

        self.n_iter_: int = k + 1
        # z is the ADMM primal variable that is always sparse (passes through
        # soft-thresholding at every step). Use z as the final solution.
        # At convergence alpha ≈ z; if not converged, z is still sparse.
        self.alpha_ = z
        self.converged_ = converged

        if not converged:
            logger.warning(
                "ADMMNesterovLSSVM did not converge in %d iterations "
                "(primal_res=%.2e).",
                self.max_iter,
                float(np.linalg.norm(alpha - z_new)),
            )

        # Bias estimation: b = mean(y - K α)
        # In the paper the bias is omitted, but we estimate it for robustness
        # on imbalanced datasets.
        self.bias_: float = float(np.mean(y - K @ alpha))

        logger.info(
            "ADMMNesterovLSSVM — n_sv=%d/%d (%.1f%% sparse), %d iters, "
            "bias=%.4f, converged=%s",
            self.n_support_,
            n,
            100.0 * self.sparsity_ratio_,
            self.n_iter_,
            self.bias_,
            converged,
        )

    # ── Primal decision function (overrides dual formula in BaseLSSVM) ────────

    def decision_function(self, X: NDArray) -> NDArray:
        """Primal prediction: f(x) = Σᵢ αᵢ K(xᵢ, x) + b  (Eq. 12).

        Note: unlike the dual LSSVM, there is no yᵢ factor here — the primal
        Lagrange multipliers αᵢ already encode the class direction.
        """
        check_is_fitted(self, ["alpha_", "bias_", "X_train_"])
        X = check_array(X)
        K = self.kernel_matrix(X, self.X_train_)
        return K @ self.alpha_ + self.bias_

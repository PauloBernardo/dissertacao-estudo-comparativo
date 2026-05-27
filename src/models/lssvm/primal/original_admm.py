"""Original ADMM-Nesterov LSSVM — faithful reproduction of Pesquisa_Tese/ADMM.py.

Wraps the functions Fast_ADMM_restart (and Fast_ADMM) from the original
Marinho et al. research code as a sklearn-compatible LSSVM classifier.

Intentionally preserved from the original:
- λ auto-set: sqrt(2 * log10(n))  [original uses log base-10, not natural]
- ρ auto-set: 1 / max|eigvals(AᵀA)| via np.linalg.eig (non-symmetric)
- Q = np.linalg.inv(AᵀA + ρI)  [full inversion, pre-computed once]
- No convergence criterion — always runs max_iter iterations
- Restart resets to z_anterior / u_anterior (previous step, not current)
- Sthresh(x, γ) = sign(x) * max(|x| - γ/2, 0)  [original γ/2 convention]
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


class OriginalADMMNesterovLSSVM(BaseLSSVM):
    """Faithful wrapper of the original Fast_ADMM_restart from Marinho et al.

    The LSSVM preprocessing (kernel → Cholesky → b_vec) is added here since
    it was not part of the original ADMM.py file.  Everything inside the
    ADMM loop is kept exactly as written in Pesquisa_Tese/ADMM.py.

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth (γ = 1/(2σ²)).
    tau : float
        LSSVM regularisation parameter (bias/variance trade-off).
    lambda_ : float or None
        L1 regularisation. If None, auto-set to sqrt(2 * log10(n))
        — exactly as in the original code.
    rho : float or None
        ADMM penalty. If None, auto-set to 1/max|eigvals(AᵀA)|
        via np.linalg.eig — exactly as in the original.
    restart_eta : float
        Lyapunov restart threshold η (called 'neta' in the original).
        Momentum is kept when c < η·c_prev; otherwise reset.
    tikhonov_reg : float
        Tikhonov regularisation for Cholesky stability: K̃ = K + reg·I.
    estimate_bias : bool
        Whether to estimate bias as mean(y - Kα) after fitting.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        lambda_: float | None = None,
        rho: float | None = None,
        restart_eta: float = 0.999,
        tikhonov_reg: float = 0.01,
        estimate_bias: bool = True,
        max_iter: int = 5000,
    ) -> None:
        super().__init__(sigma=sigma, tau=tau, max_iter=max_iter)
        self.lambda_ = lambda_
        self.rho = rho
        self.restart_eta = restart_eta
        self.tikhonov_reg = tikhonov_reg
        self.estimate_bias = estimate_bias

    # ── Soft-threshold (original convention) ─────────────────────────────────

    @staticmethod
    def _sthresh(x: NDArray, gamma: float) -> NDArray:
        """Original Sthresh: sign(x) * max(|x| - gamma/2, 0)."""
        return np.sign(x) * np.maximum(np.abs(x) - gamma / 2.0, 0.0)

    # ── Core solver ───────────────────────────────────────────────────────────

    def _solve(self, X: NDArray, y: NDArray) -> None:
        n = len(y)

        # ── LSSVM preprocessing (not in original ADMM.py) ────────────────────
        K = self.kernel_matrix(X)
        K_tik = K + self.tikhonov_reg * np.eye(n)

        try:
            L = np.linalg.cholesky(K_tik)
        except np.linalg.LinAlgError:
            logger.warning("Cholesky failed — adding extra regularisation 1e-3.")
            L = np.linalg.cholesky(K_tik + 1e-3 * np.eye(n))

        A = L.T                                              # n×n upper triangular
        Pt_y = A @ y
        b_vec = np.linalg.solve(self.tau * np.eye(n) + K_tik, Pt_y)
        r = L @ b_vec                                        # = Aᵀ b_vec

        # ── ρ and λ: original auto-selection logic ────────────────────────────
        AtA = A.T @ A   # = K_tik

        if self.rho is not None:
            rho = self.rho
        else:
            # Original: w, v = np.linalg.eig(A.T @ A)  (non-symmetric eig)
            w = np.linalg.eig(AtA)[0]
            rho = 1.0 / float(np.amax(np.absolute(w)))

        if self.lambda_ is not None:
            lam = self.lambda_
        else:
            # Original: l = sqrt(2 * log(n, 10))  — log base 10
            lam = sqrt(2.0 * log(n, 10))

        # Original threshold convention: Sthresh(v, l/rho) → gamma = l/rho
        # Combined with gamma/2 inside Sthresh → effective = l/(2*rho)
        sthresh_gamma = lam / rho   # passed as 'gamma' to Sthresh

        # ── Q = inv(AᵀA + ρI), pre-computed once (original approach) ─────────
        Q = np.linalg.inv(AtA + rho * np.eye(n))

        # ── Fast_ADMM_restart loop (verbatim from original) ───────────────────
        xhat = np.zeros(n)
        z     = np.zeros(n)
        zhat  = np.zeros(n)
        u     = np.zeros(n)
        uhat  = np.zeros(n)
        alpha = 1.0
        c     = 0.0
        neta  = self.restart_eta

        for _ in range(self.max_iter):
            # x-update
            xhat = Q @ (r + rho * (zhat - uhat))

            # z-update
            z_anterior = z.copy()
            z = self._sthresh(xhat + uhat, sthresh_gamma)

            # u-update
            u_anterior = u.copy()
            u = uhat + xhat - z

            # Lyapunov restart criterion (original: tau = neta)
            c_anterior = c
            c = ((1.0 / neta) * float(np.dot(u - uhat, u - uhat)) +
                 neta          * float(np.dot(z - zhat, z - zhat)))

            alpha_anterior = alpha
            if c < neta * c_anterior:
                # keep momentum
                alpha = (1.0 + sqrt(1.0 + 4.0 * alpha ** 2)) / 2.0
                zhat = z + (alpha_anterior - 1.0) / alpha * (z - z_anterior)
                uhat = u + (alpha_anterior - 1.0) / alpha * (u - u_anterior)
            else:
                # reset — original resets to z_anterior / u_anterior
                alpha = 1.0
                zhat  = z_anterior.copy()
                uhat  = u_anterior.copy()
                c     = c / neta

        self.alpha_ = z
        self.rho_used_    = rho
        self.lambda_used_ = lam

        self.bias_: float = (
            float(np.mean(y - K @ self.alpha_)) if self.estimate_bias else 0.0
        )

        logger.info(
            "OriginalADMMNesterovLSSVM — n_sv=%d/%d (%.1f%% sparse), "
            "λ=%.4f, ρ=%.4f, bias=%.4f",
            self.n_support_, n, 100.0 * self.sparsity_ratio_,
            lam, rho, self.bias_,
        )

    # ── Decision function ─────────────────────────────────────────────────────

    def decision_function(self, X: NDArray) -> NDArray:
        check_is_fitted(self, ["alpha_", "bias_", "X_train_"])
        X = check_array(X)
        K = self.kernel_matrix(X, self.X_train_)
        return K @ self.alpha_ + self.bias_

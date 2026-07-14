"""Sparse scalable LSSVM via Nesterov-ADMM on Nyström landmark space.

Extends Marinho et al. (IWANN 2023) to large-N datasets.  Instead of the
full N×N kernel matrix required by ADMMNesterovLSSVM, this model selects
m << N landmark points and solves a sparse regression in the resulting
m-dimensional landmark space.

Mathematical formulation
------------------------
Landmark model:
    f(x) = K(x, Z)ᵀ θ + b,   Z ∈ ℝ^{m×d} landmarks,  θ ∈ ℝ^m

Sparse objective (primal LSSVM with L1 on landmark weights):
    min_{θ,b}  (1/2τ) ‖Cθ + b·1 - y‖₂²  +  (λ/2)‖θ‖₁

where C = K(X_train, Z) ∈ ℝ^{N×m}.  As in the base ADMMNesterovLSSVM, the
pure primal L1 objective carries NO bias; an intercept is estimated post-hoc
(optionally) to help on imbalanced data:
    b = mean(y - Cθ)     (computed after θ converges, not jointly with θ)

ADMM with Nesterov momentum — identical to Marinho's Algorithm 2+3 but
operating on the m×m system instead of N×N:
    θ^{k+1} = (CᵀC/τ + ρI)⁻¹ (Cᵀy/τ + ρ(ẑᵏ - ûᵏ))   [Cholesky, m×m]
    z^{k+1} = S_{λ/(2ρ)}(θ^{k+1} + ûᵏ)                 [soft-threshold]
    u^{k+1} = ûᵏ + θ^{k+1} − z^{k+1}
    (+ Nesterov momentum on ẑ, û — same restart logic as admm_nesterov.py)

Complexity vs ADMMNesterovLSSVM
-------------------------------
    Setup memory : O(N·m)   vs  O(N²)
    CᵀC step     : O(N·m²)  vs  O(N³)     ← embarrassingly parallel
    Cholesky     : O(m³)    vs  O(N³)
    Per ADMM iter: O(m)     vs  O(N)
    Predict      : O(n_test·n_nz) where n_nz ≤ m

For N=30 000, m=3 000 (10%): ~100× memory reduction, ~1000× Cholesky speedup.

Distributed mode (n_blocks > 1)
--------------------------------
When n_blocks > 1, the CᵀC computation is split into B independent block
contributions and evaluated in parallel via joblib:

    Node i (one of B):  Cᵢ = K(Xᵢ, Z)   →   CᵢᵀCᵢ,  Cᵢᵀyᵢ
    AllReduce:          CᵀC = Σ CᵢᵀCᵢ ,   Cᵀy = Σ Cᵢᵀyᵢ

This mirrors the consensus-ADMM distributed architecture described by
Marinho.  On a real cluster each block would run on a separate machine;
here joblib spawns B processes on the same hardware for benchmarking.
Set n_jobs=-1 to use all available CPU cores.

Paper-fonte (BASE TEORICA, fora do repo — ver docs/model_references.md):
    IWANN___LSSVM_ADMM.pdf; Nyström: LSSVM/ESTADO DA ARTE/2202.11599v2.pdf (NysADMM)
"""

from __future__ import annotations

import logging
import time
from math import sqrt

import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

logger = logging.getLogger(__name__)


class ADMMNystromLSSVM(BaseEstimator, ClassifierMixin):
    """Sparse scalable LSSVM via Nesterov-ADMM on Nyström landmark space.

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth (γ = 1/(2σ²)).
    tau : float
        LSSVM regularisation (bias/variance trade-off).
    lambda_ : float
        L1 penalty on landmark weights θ.  Higher → more landmark sparsity.
    m_ratio : float
        Fraction of training samples used as landmarks (m = round(m_ratio·N)).
    rho : float or None
        ADMM augmented-Lagrangian step.  None = auto (1/max_eig(CᵀC/τ)).
    tol : float
        Convergence tolerance on primal and dual residuals.
    max_iter : int
        Maximum ADMM iterations.
    use_nesterov : bool
        Apply FISTA momentum (Algorithm 2 of Marinho).
    adaptive_restart : bool
        Lyapunov-based restart for momentum (Fast_ADMM_restart).
    restart_eta : float
        Restart threshold η ∈ (0,1).
    estimate_bias : bool
        Estimate intercept b = mean(y − Cθ) after convergence.
    landmark_method : str
        Landmark selection strategy: 'colnorm' (default), 'random', 'fps',
        'leverage', 'kmeans', 'opposite'.  'colnorm' samples proportional to
        the squared input-feature norms ‖x_i‖² (a cheap O(Nd) proxy for the
        kernel-column/leverage norms, avoiding the O(N²) kernel matrix) — the
        same criterion used by NystromLSSVMColnorm.
    n_blocks : int
        Number of data blocks for the parallel CᵀC computation.
        1 = single-machine (mode A); >1 = block-parallel (mode B).
    n_jobs : int
        joblib workers for block-parallel mode.  -1 = all CPU cores.
    random_state : int or None
        Seed for landmark selection.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        lambda_: float = 0.01,
        m_ratio: float = 0.10,
        rho: float | None = None,
        tol: float = 1e-6,
        max_iter: int = 500,
        use_nesterov: bool = True,
        adaptive_restart: bool = True,
        restart_eta: float = 0.999,
        estimate_bias: bool = True,
        landmark_method: str = "colnorm",
        n_blocks: int = 1,
        n_jobs: int = 1,
        random_state: int | None = None,
    ) -> None:
        self.sigma = sigma
        self.tau = tau
        self.lambda_ = lambda_
        self.m_ratio = m_ratio
        self.rho = rho
        self.tol = tol
        self.max_iter = max_iter
        self.use_nesterov = use_nesterov
        self.adaptive_restart = adaptive_restart
        self.restart_eta = restart_eta
        self.estimate_bias = estimate_bias
        self.landmark_method = landmark_method
        self.n_blocks = n_blocks
        self.n_jobs = n_jobs
        self.random_state = random_state

    # ── Kernel ────────────────────────────────────────────────────────────────

    def _rbf(self, X: NDArray, Y: NDArray) -> NDArray:
        gamma = 1.0 / (2.0 * self.sigma ** 2)
        Xsq = np.sum(X ** 2, axis=1, keepdims=True)
        Ysq = np.sum(Y ** 2, axis=1, keepdims=True)
        D = np.maximum(Xsq + Ysq.T - 2.0 * (X @ Y.T), 0.0)
        return np.exp(-gamma * D)

    @staticmethod
    def _soft_threshold(v: NDArray, kappa: float) -> NDArray:
        return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)

    # ── CᵀC accumulation (parallel if n_blocks > 1) ───────────────────────

    def _build_CtC_Cty(self, C: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        """Compute CᵀC and Cᵀy, optionally in parallel blocks."""
        if self.n_blocks <= 1:
            return C.T @ C, C.T @ y

        N = len(y)
        edges = np.linspace(0, N, self.n_blocks + 1, dtype=int)

        def _block(i: int) -> tuple[NDArray, NDArray]:
            s, e = int(edges[i]), int(edges[i + 1])
            Ci = C[s:e]
            return Ci.T @ Ci, Ci.T @ y[s:e]

        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_block)(i) for i in range(self.n_blocks)
            )
        except ImportError:
            results = [_block(i) for i in range(self.n_blocks)]

        CtC = sum(r[0] for r in results)
        Cty = sum(r[1] for r in results)
        return CtC, Cty

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: NDArray, y: NDArray) -> "ADMMNystromLSSVM":
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if set(self.classes_) == {0, 1}:
            y = np.where(y == 1, 1.0, -1.0)
        elif not set(self.classes_).issubset({-1, 1}):
            raise ValueError("Labels must be in {-1,+1} or {0,1}.")

        n = len(y)
        self.X_train_ = X
        self.y_train_ = y
        self.n_samples_fit_ = n
        self.n_features_in_ = X.shape[1]

        # ── Step 1: select m landmarks ────────────────────────────────────────
        m = max(2, int(round(self.m_ratio * n)))
        from src.models.landmark_selection import get_selector
        extra = {}
        # leverage needs true kernel norms; colnorm uses feature norms ||x_i||²
        # (same as NystromLSSVMColnorm) to avoid the O(N²) kernel matrix.
        if self.landmark_method == "leverage":
            extra["kernel"] = lambda Z: self._rbf(Z, Z)
        selector = get_selector(self.landmark_method, m,
                                random_state=self.random_state, **extra)
        selector.fit(X)
        lm_idx = selector.indices_
        self.landmarks_ = X[lm_idx].copy()
        self.m_ = m

        # ── Step 2: build C ∈ ℝ^{N×m} and accumulate CᵀC (parallel in mode B)
        C = self._rbf(X, self.landmarks_)

        t0 = time.perf_counter()
        CtC, Cty = self._build_CtC_Cty(C, y)
        self.ctc_wall_time_ = time.perf_counter() - t0  # for speedup benchmarks

        CtC_tau = CtC / self.tau
        Cty_tau = Cty / self.tau

        # ── Step 3: auto ρ from max eigenvalue of CᵀC/τ ──────────────────────
        eig_max = float(np.linalg.eigvalsh(CtC_tau).max())
        rho = self.rho if self.rho is not None else (
            1.0 / eig_max if eig_max > 0 else 1.0
        )

        # ── Step 4: prefactor (CᵀC/τ + ρI)  [m×m Cholesky] ──────────────────
        M = CtC_tau + rho * np.eye(m)
        L_M = np.linalg.cholesky(M + 1e-10 * np.eye(m))

        threshold = self.lambda_ / (2.0 * rho)  # Marinho λ/2 convention

        # ── Step 5: ADMM-Nesterov loop ────────────────────────────────────────
        theta = np.zeros(m)
        z     = np.zeros(m)
        u     = np.zeros(m)
        z_hat = np.zeros(m)
        u_hat = np.zeros(m)
        t_mom = 1.0
        c_prev = np.inf
        converged = False

        for k in range(self.max_iter):
            # θ-update: (CᵀC/τ + ρI)θ = Cᵀy/τ + ρ(ẑ - û)
            rhs = Cty_tau + rho * (z_hat - u_hat)
            theta_new = scipy.linalg.cho_solve((L_M, True), rhs)

            # z-update: soft-threshold
            z_prev = z.copy()
            z_new  = self._soft_threshold(theta_new + u_hat, threshold)

            # u-update
            u_prev = u.copy()
            u_new  = u_hat + theta_new - z_new

            # Convergence check
            primal_res = float(np.linalg.norm(theta_new - z_new))
            dual_res   = float(rho * np.linalg.norm(z_new - z_prev))
            if primal_res < self.tol and dual_res < self.tol:
                z = z_new
                theta = theta_new
                converged = True
                break

            # Nesterov / FISTA momentum (Algorithm 2+3 of Marinho)
            if self.use_nesterov:
                t_new  = (1.0 + sqrt(1.0 + 4.0 * t_mom * t_mom)) / 2.0
                momentum = (t_mom - 1.0) / t_new

                if self.adaptive_restart:
                    eta = self.restart_eta
                    c = (float(np.dot(u_new - u_hat, u_new - u_hat)) / eta
                         + eta * float(np.dot(z_new - z_hat, z_new - z_hat)))
                    if c < eta * c_prev:
                        z_hat = z_new + momentum * (z_new - z_prev)
                        u_hat = u_new + momentum * (u_new - u_prev)
                        c_prev = c
                    else:
                        t_new  = 1.0
                        z_hat  = z_new.copy()
                        u_hat  = u_new.copy()
                        c_prev = c / eta
                else:
                    z_hat = z_new + momentum * (z_new - z_prev)
                    u_hat = u_new + momentum * (u_new - u_prev)
                t_mom = t_new
            else:
                z_hat = z_new.copy()
                u_hat = u_new.copy()

            z = z_new
            u = u_new
            theta = theta_new

        self.theta_     = z          # sparse landmark weight vector
        self.n_iter_    = k + 1
        self.converged_ = converged
        self.rho_used_  = rho

        if not converged:
            logger.warning(
                "ADMMNystromLSSVM did not converge in %d iters (primal_res=%.2e).",
                self.max_iter, float(np.linalg.norm(theta - z)),
            )

        # ── Step 6: bias and sparse cache ─────────────────────────────────────
        self.bias_ = float(np.mean(y - C @ self.theta_)) if self.estimate_bias else 0.0

        nz = np.abs(self.theta_) > 1e-8
        self._theta_nz_     = self.theta_[nz]
        self._landmarks_nz_ = self.landmarks_[nz]

        logger.info(
            "ADMMNystromLSSVM — N=%d  m=%d  n_nz=%d (landmark sparsity %.1f%%)  "
            "%d iters  λ=%.4f  ρ=%.4f  ctc_time=%.2fs  blocks=%d  converged=%s",
            n, m, int(nz.sum()), 100.0 * (1.0 - nz.sum() / m),
            self.n_iter_, self.lambda_, rho,
            self.ctc_wall_time_, self.n_blocks, converged,
        )
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def decision_function(self, X: NDArray) -> NDArray:
        check_is_fitted(self, ["theta_", "landmarks_"])
        X = check_array(X)
        if len(self._theta_nz_) == 0:
            return np.full(len(X), self.bias_)
        return self._rbf(X, self._landmarks_nz_) @ self._theta_nz_ + self.bias_

    def predict(self, X: NDArray) -> NDArray:
        scores = self.decision_function(X)
        raw = np.sign(scores)
        raw[raw == 0] = 1.0
        if set(self.classes_) == {0, 1}:
            return np.where(raw == 1.0, 1, 0)
        return raw.astype(int)

    def predict_proba(self, X: NDArray) -> NDArray:
        scores = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - p1, p1])

    # ── Sparsity interface (compatible with lssvm_sparsity in runner) ─────────

    @property
    def alpha_(self) -> NDArray:
        """Alias for theta_ — allows lssvm_sparsity to work unchanged."""
        return self.theta_

    @property
    def n_support_(self) -> int:
        """Number of non-zero landmark weights (active landmarks)."""
        check_is_fitted(self, ["theta_"])
        return int(np.sum(np.abs(self.theta_) > 1e-6))

    @property
    def sparsity_ratio_(self) -> float:
        """Fraction of m landmarks pruned to zero by L1."""
        return 1.0 - self.n_support_ / self.m_

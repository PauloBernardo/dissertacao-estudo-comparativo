"""Sparse scalable LSSVM in Primal via FISTA + Nyström approximation.

Same Nyström substitution as ADMMNystromLSSVM but uses FISTA (pure proximal
gradient) instead of ADMM — no ρ parameter, no linear system solve.

Mathematical formulation
------------------------
Primal Nyström LSSVM (bias profiled out via centering):
    Ĉ = C - 1·mean(C)ᵀ,   ŷ = y - mean(y)

    min_{θ ∈ ℝᵐ}  f(θ) + λ‖θ‖₁
    f(θ) = ½‖Ĉθ - ŷ‖² / τ

Gradient and Lipschitz (all m×m — no N×N matrix):
    ∇f(θ) = M·θ − r      M = ĈᵀĈ/τ ∈ ℝ^{m×m},  r = Ĉᵀŷ/τ ∈ ℝᵐ
    L      = max_eig(M)   ← O(m²N + m³)

FISTA iteration (Beck & Teboulle 2009 + O'Donoghue restart):
    grad   = M·y_ext − r       ← O(m²) per iter
    v      = y_ext − (1/L)·grad
    θ_new  = S_{λ/L}(v)        soft-threshold
    (+ Nesterov momentum + gradient restart)

Bias recovery:
    b = mean(y) − mean(C)ᵀ·θ

Prediction (O(n_test·n_nz) where n_nz = ‖θ‖₀ ≤ m):
    f(x) = K(x, Z_nz)·θ_nz + b

Comparison with ADMMNystromLSSVM
---------------------------------
    Complexity : identical  O(N·m²) setup, O(m²) per iter
    Parameters : one fewer — no ρ to tune (FISTA uses L directly)
    Convergence: both O(1/k²) with Nesterov; ADMM may converge in fewer iters
                 due to implicit second-order information via the Cholesky solve
"""

from __future__ import annotations

import logging
from math import sqrt

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

logger = logging.getLogger(__name__)


class FISTANystromLSSVM(BaseEstimator, ClassifierMixin):
    """Sparse scalable LSSVM via FISTA + Nyström (primal, landmark sparsity).

    Parameters
    ----------
    sigma : float
        RBF bandwidth (γ = 1/(2σ²)).
    tau : float
        LSSVM regularisation (scales the least-squares term).
    lambda_ : float
        L1 penalty on landmark weights θ. Controls landmark sparsity.
    m_ratio : float
        Fraction of training samples used as landmarks (m = round(m_ratio·N)).
    landmark_method : str
        Landmark selection: 'colnorm' (feature norms, default) or 'random'.
    tol : float
        Convergence tolerance on ‖θ_k − θ_{k-1}‖.
    max_iter : int
        Maximum FISTA iterations.
    adaptive_restart : bool
        O'Donoghue & Candès gradient-based restart.
    estimate_bias : bool
        Recover intercept b = mean(y) − mean(C)ᵀ·θ after convergence.
    random_state : int or None
        Seed for landmark selection.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        lambda_: float = 0.01,
        m_ratio: float = 0.30,
        landmark_method: str = "colnorm",
        tol: float = 1e-6,
        max_iter: int = 5000,
        adaptive_restart: bool = True,
        estimate_bias: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.sigma = sigma
        self.tau = tau
        self.lambda_ = lambda_
        self.m_ratio = m_ratio
        self.landmark_method = landmark_method
        self.tol = tol
        self.max_iter = max_iter
        self.adaptive_restart = adaptive_restart
        self.estimate_bias = estimate_bias
        self.random_state = random_state

    def _rbf(self, X: NDArray, Y: NDArray) -> NDArray:
        gamma = 1.0 / (2.0 * self.sigma ** 2)
        Xsq = np.sum(X ** 2, axis=1, keepdims=True)
        Ysq = np.sum(Y ** 2, axis=1, keepdims=True)
        D = np.maximum(Xsq + Ysq.T - 2.0 * (X @ Y.T), 0.0)
        return np.exp(-gamma * D)

    @staticmethod
    def _soft_threshold(v: NDArray, kappa: float) -> NDArray:
        return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)

    def fit(self, X: NDArray, y: NDArray) -> "FISTANystromLSSVM":
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        if set(self.classes_) == {0, 1}:
            y = np.where(y == 1, 1.0, -1.0)
        elif not set(self.classes_).issubset({-1, 1}):
            raise ValueError("Labels must be in {-1,+1} or {0,1}.")

        n = len(y)
        self.n_samples_fit_ = n
        self.n_features_in_ = X.shape[1]

        # ── Step 1: select m landmarks (feature-norm colnorm, no N×N kernel) ─
        m = max(2, int(round(self.m_ratio * n)))
        from src.models.landmark_selection import get_selector
        extra: dict = {}
        if self.landmark_method == "leverage":
            extra["kernel"] = lambda Z: self._rbf(Z, Z)
        selector = get_selector(self.landmark_method, m,
                                random_state=self.random_state, **extra)
        selector.fit(X)
        self.landmarks_ = X[selector.indices_].copy()
        self.m_ = m

        # ── Step 2: build C ∈ ℝ^{N×m}, center for bias profiling ─────────────
        C = self._rbf(X, self.landmarks_)   # N×m
        c_mean = C.mean(axis=0)             # m
        y_mean = float(y.mean())
        C_hat  = C - c_mean                 # Ĉ = C − 1·c̄ᵀ
        y_hat  = y - y_mean                 # ŷ = y − ȳ

        # ── Step 3: precompute M = ĈᵀĈ/τ and r = Ĉᵀŷ/τ ─────────────────────
        M = (C_hat.T @ C_hat) / self.tau    # m×m — one-time O(N·m²)
        r = (C_hat.T @ y_hat) / self.tau    # m

        # ── Step 4: Lipschitz constant L = max_eig(M) ────────────────────────
        L = float(np.linalg.eigvalsh(M).max())
        if L <= 0:
            L = 1.0
        step     = 1.0 / L
        threshold = self.lambda_ * step     # λ/L for soft-threshold

        # ── Step 5: FISTA loop ────────────────────────────────────────────────
        theta      = np.zeros(m)
        theta_prev = np.zeros(m)
        y_ext      = np.zeros(m)
        t          = 1.0
        converged  = False

        for k in range(self.max_iter):
            grad      = M @ y_ext - r
            v         = y_ext - step * grad
            theta_new = self._soft_threshold(v, threshold)

            if float(np.linalg.norm(theta_new - theta)) < self.tol:
                theta = theta_new
                converged = True
                break

            t_new    = (1.0 + sqrt(1.0 + 4.0 * t * t)) / 2.0
            momentum = (t - 1.0) / t_new

            if self.adaptive_restart:
                if float(np.dot(theta_new - theta_prev, theta_new - v)) > 0:
                    t_new    = 1.0
                    momentum = 0.0

            theta_prev = theta.copy()
            theta      = theta_new
            t          = t_new
            y_ext      = theta + momentum * (theta - theta_prev)

        self.theta_     = theta
        self.n_iter_    = k + 1
        self.converged_ = converged

        # ── Step 6: bias and sparse cache ─────────────────────────────────────
        self.bias_ = float(y_mean - c_mean @ theta) if self.estimate_bias else 0.0

        nz = np.abs(theta) > 1e-8
        self._theta_nz_     = theta[nz]
        self._landmarks_nz_ = self.landmarks_[nz]

        logger.info(
            "FISTANystromLSSVM — N=%d  m=%d  n_nz=%d (sparsity %.1f%%)  "
            "%d iters  λ=%.4f  L=%.4f  bias=%.4f  converged=%s",
            n, m, int(nz.sum()), 100.0 * (1.0 - nz.sum() / m),
            self.n_iter_, self.lambda_, L, self.bias_, converged,
        )
        return self

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

    # ── Sparsity interface (compatível com lssvm_sparsity no runner) ──────────

    @property
    def alpha_(self) -> NDArray:
        """Alias para theta_ — permite lssvm_sparsity funcionar sem mudança."""
        return self.theta_

    @property
    def n_support_(self) -> int:
        check_is_fitted(self, ["theta_"])
        return int(np.sum(np.abs(self.theta_) > 1e-6))

    @property
    def sparsity_ratio_(self) -> float:
        return 1.0 - self.n_support_ / self.m_

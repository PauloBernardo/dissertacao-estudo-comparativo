"""Sparse scalable LSSVM in Dual via FISTA + Nyström approximation.

Combines the dual L1 sparsity of DualFISTALSSVM with the O(N·m) scalability
of Nyström, avoiding the O(N²) kernel matrix entirely.

Mathematical formulation
------------------------
Standard dual LSSVM (Suykens 2002):
    min_α  (1/2) αᵀ Ω α  −  1ᵀα  +  λ‖α‖₁
    Ω = YKY + (1/τ)I,    Y = diag(y),  K ∈ ℝ^{N×N}

Nyström substitution K ← K̃ = CW⁻¹Cᵀ  (C = K(X,Z), W = K(Z,Z)):
    Ω̃ = YK̃Y + (1/τ)I
    Ω̃·v = YC·(W⁻¹·(Cᵀ·(Y·v))) + v/τ      ← O(N·m), no N×N matrix

Lipschitz constant (m×m problem):
    L(Ω̃) = max_eig(W⁻¹·CᵀC) + 1/τ         ← exact, O(m²N + m³)

FISTA iteration (Beck & Teboulle 2009):
    grad   = Ω̃·y_ext − 1                    ← O(N·m) per iter
    v      = y_ext − η·grad
    α_new  = S_{ηλ}(v)                       ← soft-threshold
    (+ Nesterov momentum + O'Donoghue restart)

Prediction (O(n_test·m)):
    β = W⁻¹·Cᵀ·(α⊙y)                       ← precomputed after fit (m-dim)
    f(x) = K(x, Z)·β + b                    ← landmarks Z, not N SVs
"""

from __future__ import annotations

import logging
from math import sqrt

import numpy as np
import scipy.linalg
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

logger = logging.getLogger(__name__)


class DualFISTANystromLSSVM(BaseEstimator, ClassifierMixin):
    """Sparse scalable LSSVM in Dual via FISTA + Nyström.

    Parameters
    ----------
    sigma : float
        RBF bandwidth (γ = 1/(2σ²)).
    tau : float
        LSSVM regularisation (1/τ diagonal term in Ω̃).
    lambda_ : float
        L1 penalty on dual variables α. Controls support-vector sparsity.
    m_ratio : float
        Fraction of training samples used as Nyström landmarks (m = round(m_ratio·N)).
    landmark_method : str
        Landmark selection: 'colnorm' (feature norms, default) or 'random'.
    tol : float
        Convergence tolerance on ‖α_k − α_{k-1}‖.
    max_iter : int
        Maximum FISTA iterations.
    adaptive_restart : bool
        O'Donoghue & Candès gradient restart.
    estimate_bias : bool
        Estimate intercept b = mean(y − C·β) after convergence.
    random_state : int or None
        Seed for landmark selection.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        lambda_: float = 0.1,
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

    def fit(self, X: NDArray, y: NDArray) -> "DualFISTANystromLSSVM":
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

        # ── Step 2: build C ∈ ℝ^{N×m} and W ∈ ℝ^{m×m} ─────────────────────
        C = self._rbf(X, self.landmarks_)          # N×m
        W = self._rbf(self.landmarks_, self.landmarks_)  # m×m

        W_chol = np.linalg.cholesky(W + 1e-10 * np.eye(m))
        W_factor = (W_chol, True)  # lower triangular, for cho_solve

        # ── Step 3: Lipschitz constant via m×m problem ────────────────────────
        # Ω̃ = YK̃Y + I/τ;  max_eig(YK̃Y) = max_eig(W⁻¹·CᵀC)  (Y²=I)
        CtC = C.T @ C                                        # m×m, O(N·m²)
        W_inv_CtC = scipy.linalg.cho_solve(W_factor, CtC)   # m×m
        L = float(np.linalg.eigvalsh(W_inv_CtC).max()) + 1.0 / self.tau
        eta = 1.0 / L
        threshold = eta * self.lambda_

        # Precompute YC = diag(y)·C for gradient: Ω̃·v = YC·(W⁻¹·(YCᵀ·v)) + v/τ
        YC = y[:, None] * C   # N×m

        def omega_mv(v: NDArray) -> NDArray:
            w = YC.T @ v                                  # m
            w = scipy.linalg.cho_solve(W_factor, w)      # m
            return YC @ w + v / self.tau                  # N

        # ── Step 4: FISTA loop ────────────────────────────────────────────────
        ones = np.ones(n)
        alpha      = np.zeros(n)
        alpha_prev = np.zeros(n)
        y_ext      = np.zeros(n)
        t          = 1.0
        converged  = False

        for k in range(self.max_iter):
            grad      = omega_mv(y_ext) - ones
            v         = y_ext - eta * grad
            alpha_new = self._soft_threshold(v, threshold)

            if float(np.linalg.norm(alpha_new - alpha)) < self.tol:
                alpha = alpha_new
                converged = True
                break

            t_new    = (1.0 + sqrt(1.0 + 4.0 * t * t)) / 2.0
            momentum = (t - 1.0) / t_new

            if self.adaptive_restart:
                if float(np.dot(alpha_new - alpha_prev, alpha_new - v)) > 0:
                    t_new    = 1.0
                    momentum = 0.0

            alpha_prev = alpha.copy()
            alpha      = alpha_new
            t          = t_new
            y_ext      = alpha + momentum * (alpha - alpha_prev)

        self.n_iter_    = k + 1
        self.converged_ = converged
        self.alpha_     = alpha          # N-dim, for sparsity reporting

        # ── Step 5: precompute β = W⁻¹·Cᵀ·(α⊙y) for O(n_test·m) prediction ─
        alpha_eff = alpha * y            # N-dim (α_i · y_i)
        w = C.T @ alpha_eff             # m
        self.beta_ = scipy.linalg.cho_solve(W_factor, w)  # m

        # ── Step 6: bias ──────────────────────────────────────────────────────
        self.bias_ = float(np.mean(y - C @ self.beta_)) if self.estimate_bias else 0.0

        nz = int(np.sum(np.abs(alpha) > 1e-6))
        logger.info(
            "DualFISTANystromLSSVM — N=%d  m=%d  n_sv=%d (%.1f%% sparse)  "
            "%d iters  λ=%.4f  L=%.4f  bias=%.4f  converged=%s",
            n, m, nz, 100.0 * (1.0 - nz / n),
            self.n_iter_, self.lambda_, L, self.bias_, converged,
        )
        return self

    def decision_function(self, X: NDArray) -> NDArray:
        check_is_fitted(self, ["beta_", "landmarks_"])
        X = check_array(X)
        return self._rbf(X, self.landmarks_) @ self.beta_ + self.bias_

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

    # ── Sparsity interface ─────────────────────────────────────────────────────

    @property
    def n_support_(self) -> int:
        check_is_fitted(self, ["alpha_"])
        return int(np.sum(np.abs(self.alpha_) > 1e-6))

    @property
    def sparsity_ratio_(self) -> float:
        return 1.0 - self.n_support_ / self.n_samples_fit_

    @property
    def n_support_vectors(self) -> int:
        return self.n_support_

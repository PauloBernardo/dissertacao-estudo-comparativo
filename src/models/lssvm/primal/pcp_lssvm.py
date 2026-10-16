"""Pivoted Cholesky Primal LSSVM (PCP-LSSVM).

Direct Method: computes a low-rank approximation of the kernel matrix
using Pivoted (incomplete) Cholesky, then trains a primal LSSVM on the
reduced feature space.

Reference:
    Zhou S. et al., "PCP-LSSVM: A Pivoted Cholesky Decomposition based
    Primal Formulation of the LSSVM Classifier", 2015.

Sparsity analogy with Nyström
------------------------------
PCP selects r pivot points greedily (max residual diagonal); Nyström
selects m landmarks by column-norm sampling. Both compress N → r/m and
predict in O(n_test × r). Reported sparsity = 1 − r/N, exactly as
NystromLSSVMColnorm reports 1 − m/N.

Paper-fonte (BASE TEORICA, fora do repo — ver docs/model_references.md):
    LSSVM/CLASSICOS/zhou2016.pdf
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import BaseLSSVM

logger = logging.getLogger(__name__)


class PCPLSSVm(BaseLSSVM):
    """PCP-LSSVM via incomplete Pivoted Cholesky of the kernel matrix.

    The rank-r Cholesky approximation K ≈ L_r L_r^T is used to map samples
    to a r-dimensional feature space. Then a primal LSSVM is solved
    directly in this reduced space via normal equations.

    Prediction uses only the r pivot points: f(x) = Σ_j α_j K(x, x_j) + b — O(n_test × r).
    The kernel matrix is never formed (Zhou 2016, Algorithm 1): memory O(n·r).

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth.
    tau : float
        Regularisation parameter.
    rank : int
        Target rank for the Pivoted Cholesky approximation.
    tol : float
        Residual tolerance for early stopping of Cholesky pivoting.
    max_iter : int
        Maximum Cholesky pivot steps (bounded by n).
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        rank: int = 50,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> None:
        super().__init__(sigma=sigma, tau=tau, tol=tol, max_iter=max_iter)
        self.rank = rank

    # Cache da fatoração por (dados, σ), só dentro do processo. A fatoração não depende de τ nem de y
    # (Zhou 2016, §III-C) e é gulosa: os r primeiros pivôs de uma fatoração de posto R ≥ r são os da de
    # posto r. Na busca em grade, uma fatoração de posto máximo por (fold, σ) serve a todos os postos e
    # τ — resultado idêntico ao de refatorar a cada configuração.
    _CHOL_CACHE: "dict[tuple, tuple[NDArray, NDArray]]" = {}
    _CHOL_CACHE_MAX = 80        # ≥ 5 folds × 13 σ + refits; ≈2,5 MB por entrada com n = 1600
    _CHOL_RANK_CAP = 200

    def _cached_cholesky(self, X: NDArray) -> tuple[NDArray, NDArray]:
        import hashlib
        key = (hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest(), X.shape, float(self.sigma), float(self.tol))
        hit = self._CHOL_CACHE.get(key)
        if hit is None or (hit[0].shape[1] < min(self.rank, X.shape[0]) and hit[0].shape[1] == hit[2]):
            rank = self.rank
            self.rank = max(rank, self._CHOL_RANK_CAP)
            try:
                P, B = self._pivoted_cholesky(X)
            finally:
                self.rank = rank
            if len(self._CHOL_CACHE) >= self._CHOL_CACHE_MAX:
                self._CHOL_CACHE.pop(next(iter(self._CHOL_CACHE)))
            hit = (P, B, max(rank, self._CHOL_RANK_CAP))
            self._CHOL_CACHE[key] = hit
        P, B, _ = hit
        r = min(self.rank, P.shape[1])
        return P[:, :r], B[:r]

    def _pivoted_cholesky(self, X: NDArray) -> tuple[NDArray, NDArray]:
        """Algoritmo 1 de Zhou (2016), PCP-LSSVM: Cholesky pivotado sem montar K.

        Passos do artigo: d⁰ = diag(K) (= 1 no RBF), ε⁰ = ‖d⁰‖₁; enquanto εⁱ > ε e i < r_max:
        t = argmax_{j∈Nⁱ} dⱼ; calcula só a coluna K(X, x_t); p = ν⁻¹(K_Mt − Pⁱu), u = Pⁱ_t,·,
        ν = (K_tt − uᵀu)^{1/2} = (d_t)^{1/2}; dⱼ ← dⱼ − pⱼ²; εⁱ⁺¹ = Σ_{j∈Nⁱ⁺¹} dⱼ.
        Custo O(n·r²), memória O(n·r). Linhas de P na ordem original das amostras.

        Returns (P, B): B[j] é o índice de treino do j-ésimo pivô (conjunto básico).
        """
        n = X.shape[0]
        r_max = min(self.rank, n)
        P = np.zeros((n, r_max))
        d = np.ones(n)                       # d⁰ = diag(K); K(x, x) = 1 no RBF
        free = np.ones(n, dtype=bool)        # Nⁱ: amostras fora do conjunto básico
        B: list[int] = []
        eps = d.sum()
        while eps > self.tol and len(B) < r_max:
            i = len(B)
            t = int(np.flatnonzero(free)[np.argmax(d[free])])
            k_t = self.kernel_matrix(X, X[t:t + 1])[:, 0]          # coluna K(X, x_t)
            p = (k_t - P[:, :i] @ P[t, :i]) / np.sqrt(d[t])
            P[:, i] = p
            B.append(t); free[t] = False
            d -= p ** 2
            d[~free] = 0.0
            eps = d[free].sum()
        r = len(B)
        return P[:, :r], np.asarray(B, dtype=int)

    def _solve(self, X: NDArray, y: NDArray) -> None:
        n = len(y)
        P, B = self._cached_cholesky(X)             # 2026-10-15: fiel a Zhou (2016), Alg. 1
        r = P.shape[1]

        # Eq. (11) com K_MB = P Lᵀ, L = P_B (fator de Cholesky dos pivôs, triangular inferior):
        #   α_B = H⁻¹ K_MBᵀ (y − ȳe),  H = λ K_BB + K_MBᵀ (I − eeᵀ/m) K_MB,
        #   b = (eᵀy − eᵀ K_MB α_B)/m.
        # Como H = L(λI + P̃ᵀP̃)Lᵀ, com P̃ = P centrado nas colunas, resolve-se o sistema r×r
        # w = (λI + P̃ᵀP̃)⁻¹ P̃ᵀ(y − ȳ) e faz-se α_B = L⁻ᵀ w; o viés sai exato, b = ȳ − P̄·w.
        # O λ do artigo é o τ deste código.
        P_mean, y_mean = P.mean(axis=0), float(y.mean())
        Pc = P - P_mean
        w = np.linalg.solve(self.tau * np.eye(r) + Pc.T @ Pc, Pc.T @ (y - y_mean))
        self.bias_ = float(y_mean - P_mean @ w)
        L = P[B]
        self.coef_: NDArray = np.linalg.solve(L.T, w)              # α_B

        self.pivot_indices_: NDArray = B
        self.X_pivots_: NDArray = X[B].copy()
        self.rank_: int = r
        self.n_samples_fit_ = n

        # alpha_ (resíduo por amostra) só para compatibilidade com BaseLSSVM; não entra na predição.
        self.alpha_: NDArray = (y - P @ w - self.bias_) / self.tau
        del self.X_train_, self.y_train_

        logger.info(
            "PCPLSSVm solved — rank=%d/%d  compression=%.1f%%  bias=%.4f",
            r, self.rank, 100.0 * (1.0 - r / n), self.bias_,
        )

    def decision_function(self, X: NDArray) -> NDArray:
        """f(x) = Σ_j α_j K(x, x_j) + b sobre os r pivôs — O(n_test × r)."""
        check_is_fitted(self, ["coef_", "X_pivots_", "bias_"])
        X = check_array(X)
        return self.kernel_matrix(X, self.X_pivots_) @ self.coef_ + self.bias_

    # ── Sparsity interface (analogous to NystromLSSVMColnorm) ─────────────────

    @property
    def n_support_(self) -> int:
        """Number of pivot points used (analogous to Nyström landmarks)."""
        check_is_fitted(self, ["rank_"])
        return self.rank_

    @property
    def sparsity_ratio_(self) -> float:
        """Compression ratio: fraction of training points NOT used as pivots."""
        return 1.0 - self.rank_ / self.n_samples_fit_

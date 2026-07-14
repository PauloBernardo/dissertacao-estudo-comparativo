"""Faithful FSA-LSSVM (Jiao et al., 2007) — backfitting algorithm.

Faithful reproduction of FSALS-SVM, in contrast to ``FSALSSVm`` (the
project's variant, which used a Matching-Pursuit correlation criterion and a
project-out residual instead of Jiao's normalized-residual criterion and
backfitting re-solve).

Faithful algorithm (paper Algorithm 1, backfitting):
    1. α⁰ = 0, b = 0, r⁰ = −y, Q = {1,…,l}, P = ∅.
    2. Stop if Q = ∅ or max(|r_Q|) < ε.
    3. Selection (Eq. 30): s = argmax_{i∈Q} (r_i)² / (k(xᵢ,xᵢ) + 1/(2γ)).
    4. P ← P ∪ {s}, Q ← Q − {s}.
    5. Backfitting: RE-SOLVE the LS-SVM subproblem on the whole basis P
       (all α_P jointly re-optimised), not a one-column projection:
           [[0, 1ᵀ], [1, K̃_PP]] [b; α_P] = [0; y_P],  K̃_PP = K_PP + 1/(2γ)·I.
    6. Residual (Eq. 28): r_i = Σ_{k∈P} α_k k(xᵢ,x_k) + b − yᵢ,  ∀i.
    7. Repeat.

Loss is the function-estimation LS-SVM (target y directly), so the decision
function is f(x) = Σ_{k∈P} α_k k(x, x_k) + b — α already absorbs the sign, so
``decision_function`` is overridden to NOT multiply by y (unlike the dual
BaseLSSVM convention). Here γ ↔ ``tau``.

Reference:
    Jiao L., Bo L. & Wang L., "Fast Sparse Approximation for Least Squares
    Support Vector Machine", IEEE Trans. Neural Networks 18(3), 2007.

Paper-fonte (BASE TEORICA, fora do repo — ver docs/model_references.md):
    LSSVM/CLASSICOS/tnn07a.pdf
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import BaseLSSVM

logger = logging.getLogger(__name__)


class FSALSSVmOriginal(BaseLSSVM):
    """Faithful FSALS-SVM with Jiao's normalized-residual criterion + backfitting.

    Parameters
    ----------
    sigma, tau : float
        RBF bandwidth and LS-SVM regularisation (γ ↔ tau; reg = 1/(2·tau)).
    n_ratio : float
        Maximum number of basis functions as a fraction of N.
    tol : float
        ε-insensitive stopping threshold on the maximum residual.
    max_iter : int
        Kept for BaseLSSVM interface compatibility.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        n_ratio: float = 0.25,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> None:
        super().__init__(sigma=sigma, tau=tau, tol=tol, max_iter=max_iter)
        self.n_ratio = n_ratio

    def _solve(self, X: NDArray, y: NDArray) -> None:
        n = len(y)
        K = self.kernel_matrix(X)
        reg = 1.0 / (2.0 * self.tau)          # Jiao's 1/(2γ)
        diag = np.diag(K) + reg               # K̃_ii for the selection criterion

        max_k = max(1, min(n, int(round(self.n_ratio * n))))
        remaining = np.ones(n, dtype=bool)
        P: list[int] = []
        r = -y.astype(float)                  # residual r⁰ = −y
        b = 0.0
        alpha_P = np.zeros(0)

        # Incremental inverse of the augmented matrix  A = [[0, 1ᵀ], [1, K̃_PP]]
        # (Jiao et al. 2007, Eqs. 13–17): grown by a rank-1 (bordered) update per
        # step in O(p²), instead of re-solving from scratch in O(p³). Produces the
        # SAME (b, α_P) — só evita o gargalo O(n³) por iteração que o paper aponta.
        R = np.empty((0, 0))                  # A⁻¹ para a base atual (bias + P)
        rebuild = False                       # recomputa R direto se schur singular

        for _ in range(max_k):
            # Step 3 — selection (Eq. 30): max normalized squared residual
            score = np.where(remaining, r * r / diag, -np.inf)
            s = int(np.argmax(score))
            remaining[s] = False

            # Step 5 — backfitting via bordered-inverse update.
            e = float(K[s, s] + reg)                       # novo elemento diagonal K̃_ss
            if not P:
                R = np.linalg.inv(np.array([[0.0, 1.0], [1.0, e]]))
            else:
                Pi_old = np.asarray(P)
                c = np.concatenate(([1.0], K[Pi_old, s]))  # acoplamento [bias, K̃_{P,s}]
                Rc = R @ c
                schur = e - float(c @ Rc)
                if abs(schur) < 1e-12:                     # guarda de estabilidade
                    rebuild = True
                else:
                    p1 = R.shape[0]
                    Rn = np.empty((p1 + 1, p1 + 1))
                    Rn[:p1, :p1] = R + np.outer(Rc, Rc) / schur
                    Rn[:p1, p1] = -Rc / schur
                    Rn[p1, :p1] = -Rc / schur
                    Rn[p1, p1] = 1.0 / schur
                    R = Rn
            P.append(s)
            Pi = np.asarray(P)

            if rebuild:                                    # fallback direto (raro)
                m = len(Pi)
                M = np.zeros((m + 1, m + 1))
                M[0, 1:] = 1.0; M[1:, 0] = 1.0
                M[1:, 1:] = K[np.ix_(Pi, Pi)] + reg * np.eye(m)
                R = np.linalg.inv(M)
                rebuild = False

            # [b; α_P] = R @ [0; y_P]
            sol = R @ np.concatenate(([0.0], y[Pi]))
            b = float(sol[0])
            alpha_P = sol[1:]

            # Step 6 — residual over ALL points: r_i = f(x_i) − y_i
            r = K[:, Pi] @ alpha_P + b - y

            if not remaining.any() or float(np.max(np.abs(r[remaining]))) < self.tol:
                break

        self.alpha_ = np.zeros(n)
        self.alpha_[np.array(P)] = alpha_P
        self.bias_ = b
        self.support_indices_ = np.sort(np.array(P))

        logger.info(
            "FSALSSVmOriginal solved — %d SVs / %d (%.1f%% sparse)",
            len(P), n, 100.0 * self.sparsity_ratio_,
        )

    def decision_function(self, X: NDArray) -> NDArray:
        """f(x) = Σ_{k∈P} α_k k(x, x_k) + b  (α absorbs the sign; no ·y)."""
        check_is_fitted(self, ["alpha_", "bias_", "X_train_"])
        X = check_array(X)
        return self._kernel_predict(X, self.alpha_)

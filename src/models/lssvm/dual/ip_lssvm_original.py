"""Faithful IP-LSSVM (Carvalho & Braga, 2009) — original two-step algorithm.

Faithful reproduction of the IP-LSSVM relevance criterion, in contrast to
``IPLSSVm`` (the project's variant, which selected support vectors by
column-pivoted QR of the kernel matrix — a criterion absent from the paper).

Faithful algorithm (paper Section 4.1, the 8 steps):
    1. Solve the FULL LS-SVM linear system A x = B (square) → all αᵢ and b.
       A = [[0, -yᵀ], [y, H]],  H = Ω + I/γ,  Ω_ij = yᵢ yⱼ K(xᵢ,xⱼ),
       B = [0, 1, …, 1]ᵀ.
    2–4. Rank the training points by their SIGNED Lagrange multiplier αᵢ
         (NOT |αᵢ|). The relevance criterion (paper Figs. 1–2): αᵢ ≫ 0 marks
         a support vector (on the border / opposite-class region); αᵢ ≲ 0 or
         αᵢ ≪ 0 is eliminated. Keep the fraction τ with the LARGEST αᵢ.
    5. Build the non-square matrix A₂ by removing from A the columns of the
       eliminated αᵢ (rows/labels kept). System A₂ x₂ = B is over-determined.
    6. Solve x₂ = A₂⁺ B via the pseudo-inverse (least squares).
    7. The retained points are the support vectors.
    8. α and b come from x₂.

Prediction: f(x) = Σ_{i∈SV} αᵢ yᵢ K(xᵢ, x) + b  (BaseLSSVM.decision_function).

Reference:
    Carvalho B.P.R. & Braga A.P., "IP-LSSVM: A two-step sparse classifier",
    Pattern Recognition Letters 30 (2009) 1507–1515.

Paper-fonte (BASE TEORICA, fora do repo — ver docs/model_references.md):
    LSSVM/CLASSICOS/carvalho2009.pdf
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from ..base import BaseLSSVM

logger = logging.getLogger(__name__)


class IPLSSVmOriginal(BaseLSSVM):
    """Faithful IP-LSSVM with the signed-α relevance criterion (Carvalho & Braga, 2009)."""

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        selection_ratio: float = 0.20,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> None:
        super().__init__(sigma=sigma, tau=tau, tol=tol, max_iter=max_iter)
        self.selection_ratio = selection_ratio

    def _full_system(self, K: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        """Build the square LS-SVM system A x = B (paper Eq. 3)."""
        n = len(y)
        Omega = (y[:, None] * K) * y[None, :]
        H = Omega + np.eye(n) / self.tau
        A = np.zeros((n + 1, n + 1))
        A[0, 1:] = -y
        A[1:, 0] = y
        A[1:, 1:] = H
        B = np.concatenate(([0.0], np.ones(n)))
        return A, B

    def _solve(self, X: NDArray, y: NDArray) -> None:
        n = len(y)
        K = self.kernel_matrix(X)

        # ── Step 1: solve the FULL LS-SVM system → α (for ranking) ────────────
        A, B = self._full_system(K, y)
        x_full = np.linalg.solve(A, B)
        alpha_full = x_full[1:]                       # signed Lagrange multipliers

        # ── Steps 2–4: keep the fraction τ with the LARGEST signed α ──────────
        n_keep = max(2, min(n, int(round(self.selection_ratio * n))))
        selected = np.sort(np.argsort(alpha_full)[-n_keep:])   # top-α indices

        # ── Steps 5–6: over-determined reduced system, pseudo-inverse solve ───
        # Columns of A kept: the bias column (0) plus the α-columns of `selected`.
        cols = np.concatenate(([0], selected + 1))
        A2 = A[:, cols]                               # (n+1) × (n_keep+1), non-square
        x2, *_ = np.linalg.lstsq(A2, B, rcond=None)   # x2 = A2⁺ B
        bias = float(x2[0])
        alpha_sel = x2[1:]

        # ── Steps 7–8: store solution in full-N format ───────────────────────
        self.alpha_ = np.zeros(n)
        self.alpha_[selected] = alpha_sel
        self.bias_ = bias
        self.support_indices_ = selected

        logger.info(
            "IPLSSVmOriginal solved — %d SVs / %d (%.1f%% sparse)",
            len(selected), n, 100.0 * self.sparsity_ratio_,
        )

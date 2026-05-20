"""Opposite Maps LSSVM (Neto & Barreto, 2013).

Reduction Method: uses Vector Quantization (k-means) to identify prototype
patterns near the decision boundary from OPPOSITE classes. These "opposite
map" prototypes are the most informative candidates for support vectors.

Reference:
    Neto A.R.R. & Barreto G.A., "Opposite Maps: Vector Quantization
    Algorithms for Building Reduced-Set SVM and LSSVM Classifiers",
    Neural Processing Letters, 2013.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import cg
from sklearn.cluster import KMeans

from ..base import BaseLSSVM

logger = logging.getLogger(__name__)


class OppositeMapsLSSVM(BaseLSSVM):
    """Opposite Maps LSSVM (Neto & Barreto, 2013).

    Algorithm:
        1. Separate training data into positive (y=+1) and negative (y=-1).
        2. Run k-means with `n_prototypes` centroids per class.
        3. For each positive prototype, find its nearest NEGATIVE training
           sample (and vice versa). These pairs define the "opposite maps" —
           the boundary-critical patterns.
        4. Collect all opposite-map samples as the reduced training set.
        5. Retrain LSSVM on this reduced set.

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth.
    tau : float
        Regularisation parameter.
    n_prototypes : int
        Number of k-means centroids per class. The final SV set has at
        most 2 × n_prototypes samples.
    random_state : int
        Seed for k-means.
    tol : float
        CG convergence tolerance.
    max_iter : int
        Maximum CG iterations.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        n_prototypes: int = 10,
        random_state: int = 42,
        tol: float = 1e-6,
        max_iter: int = 1000,
    ) -> None:
        super().__init__(sigma=sigma, tau=tau, tol=tol, max_iter=max_iter)
        self.n_prototypes = n_prototypes
        self.random_state = random_state

    def _solve(self, X: NDArray, y: NDArray) -> None:
        """Opposite-maps prototype selection + LSSVM retrain."""
        n = len(y)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == -1)[0]

        n_proto = min(self.n_prototypes, len(pos_idx) // 2, len(neg_idx) // 2)
        n_proto = max(1, n_proto)

        # ── k-means per class ─────────────────────────────────────────────────
        pos_proto = self._cluster(X[pos_idx], n_proto)  # (n_proto, p)
        neg_proto = self._cluster(X[neg_idx], n_proto)

        # ── Opposite map: for each positive prototype, find nearest negative ──
        selected_set = set()

        # Nearest negative sample to each positive prototype
        for proto in pos_proto:
            dists = np.sum((X[neg_idx] - proto) ** 2, axis=1)
            selected_set.add(int(neg_idx[np.argmin(dists)]))

        # Nearest positive sample to each negative prototype
        for proto in neg_proto:
            dists = np.sum((X[pos_idx] - proto) ** 2, axis=1)
            selected_set.add(int(pos_idx[np.argmin(dists)]))

        selected = np.array(sorted(selected_set))

        # ── Retrain LSSVM on prototype subset ────────────────────────────────
        X_sub = X[selected]
        y_sub = y[selected]
        n_sub = len(y_sub)

        if n_sub < 2 or len(np.unique(y_sub)) < 2:
            logger.warning(
                "OppositeMaps: degenerate subset (n=%d, classes=%s). "
                "Falling back to full training set.",
                n_sub,
                np.unique(y_sub),
            )
            selected = np.arange(n)
            X_sub, y_sub = X, y
            n_sub = n

        K_sub = self.kernel_matrix(X_sub)
        Omega = (y_sub[:, None] * K_sub) * y_sub[None, :]
        H = Omega + np.eye(n_sub) / self.tau
        ones = np.ones(n_sub)

        eta, _ = cg(H, y_sub, rtol=self.tol, maxiter=self.max_iter)
        mu, _ = cg(H, ones, rtol=self.tol, maxiter=self.max_iter)

        s = float(y_sub @ eta)
        self.bias_: float = float(ones @ eta) / s if abs(s) > 1e-12 else 0.0
        alpha_sub = mu - self.bias_ * eta

        self.alpha_: NDArray = np.zeros(n)
        self.alpha_[selected] = alpha_sub
        self.support_indices_: NDArray = selected

        logger.info(
            "OppositeMapsLSSVM solved — %d SVs / %d (%.1f%% sparse)",
            len(selected),
            n,
            100.0 * self.sparsity_ratio_,
        )

    def _cluster(self, X: NDArray, k: int) -> NDArray:
        """Run k-means and return centroids."""
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
        km.fit(X)
        return km.cluster_centers_

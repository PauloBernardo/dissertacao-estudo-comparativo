"""Faithful Opposite Maps LSSVM (Neto & Barreto, 2013) — original algorithm.

This is a faithful reproduction of the OM-LSSVM method as described in the
paper, in contrast to ``OppositeMapsLSSVM`` (the project's adapted variant
with input-space k-means, same-class anchors and a dense-fallback).

Faithful algorithm (paper Section 4, K2M variant for RBF kernels):
    STEP 1  Split training data into class +1 (D1) and class -1 (D2).
    STEP 2  Run Kernel K-means IN FEATURE SPACE on each class separately,
            using the SAME RBF kernel as the classifier (paper: "the kernel
            function used by the classifier and the VQ must be the same").
            Each cluster's representative is the data point closest to the
            centroid in feature space (a medoid) — so prototypes ARE data
            points.
    STEP 3  Prune prototypes whose cluster is empty (never selected).
    STEP 4  Opposite map: for each point of D2 find its nearest prototype in
            D1 (in feature space), and vice-versa. The selected prototypes
            are the boundary ones.
    STEP 5  Collect the selected prototypes from both classes.
    STEP 6  For K2M the prototype is already a data point, so its nearest
            same-class data point is itself.
    STEP 7  Reduced set = union of the selected class-1 and class-2 points.

No dense-fallback and no same-class anchor are used here — the reduced set is
taken as-is, exactly as the paper reports it (accuracy "equivalent to or
slightly worse than full-set, sometimes better").

Reference:
    Neto A.R.R. & Barreto G.A., "Opposite Maps: Vector Quantization
    Algorithms for Building Reduced-Set SVM and LSSVM Classifiers",
    Neural Processing Letters, 2013.

Paper-fonte (BASE TEORICA, fora do repo — ver docs/model_references.md):
    Opposite Maps + Outras propostas/CLASSICOS/NPL_Ajalmar.pdf
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import cg

from ..base import BaseLSSVM

logger = logging.getLogger(__name__)


class OppositeMapsOriginalLSSVM(BaseLSSVM):
    """Faithful OM-LSSVM with feature-space Kernel K-means (K2M variant).

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth (shared by the VQ and the classifier).
    tau : float
        Regularisation parameter of the LSSVM.
    proto_ratio : float
        Number of Kernel K-means clusters per class, expressed as a fraction
        of that class's sample count: ``k = round(proto_ratio * n_class)``.
        Tying the codebook size to N keeps the compression level comparable
        across dataset sizes (Tier 1 vs Tier 2). Typical grid: 0.2/0.3/0.5.
    random_state : int
        Seed for the Kernel K-means initialisation.
    tol : float
        CG convergence tolerance for the LSSVM solve.
    max_iter : int
        Maximum CG iterations for the LSSVM solve.
    max_kmeans_iter : int
        Maximum Kernel K-means iterations.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        proto_ratio: float = 0.3,
        random_state: int = 42,
        tol: float = 1e-6,
        max_iter: int = 1000,
        max_kmeans_iter: int = 20,
    ) -> None:
        super().__init__(sigma=sigma, tau=tau, tol=tol, max_iter=max_iter)
        self.proto_ratio = proto_ratio
        self.random_state = random_state
        self.max_kmeans_iter = max_kmeans_iter

    # ── LSSVM solve on a subset (same convention as OppositeMapsLSSVM) ──────────
    def _solve_on_subset(
        self, X_sub: NDArray, y_sub: NDArray
    ) -> tuple[NDArray, float]:
        n_sub = len(y_sub)
        K_sub = self.kernel_matrix(X_sub)
        Omega = (y_sub[:, None] * K_sub) * y_sub[None, :]
        H = Omega + np.eye(n_sub) / self.tau
        ones = np.ones(n_sub)
        eta, _ = cg(H, y_sub, rtol=self.tol, maxiter=self.max_iter)
        mu, _ = cg(H, ones, rtol=self.tol, maxiter=self.max_iter)
        s = float(y_sub @ eta)
        bias = float(ones @ eta) / s if abs(s) > 1e-12 else 0.0
        alpha_sub = mu - bias * eta
        return alpha_sub, bias

    # ── Kernel K-means in feature space ────────────────────────────────────────
    def _kernel_kmeans(self, K: NDArray, k: int, rng: np.random.Generator
                       ) -> NDArray:
        """Cluster points whose kernel matrix is ``K`` into ``k`` clusters.

        Returns the medoid indices (into ``K``) of the non-empty clusters —
        the feature-space representative of each cluster (paper Section 3).
        """
        m = K.shape[0]
        k = max(1, min(k, m))
        diag = np.diag(K)                       # K(xi, xi) — = 1 for RBF
        labels = rng.integers(0, k, size=m)

        def _d2(labels: NDArray) -> NDArray:
            """Feature-space distance of every point to every cluster centroid.

            Vectorised (BLAS): with the one-hot membership M (m×k),
            d²(i,j) = K(i,i) − 2·(K M)[i,j]/|Rj| + diag(Mᵀ K M)[j]/|Rj|².
            Empty clusters get +inf so no point is assigned to them.
            """
            M = np.zeros((m, k))
            M[np.arange(m), labels] = 1.0
            sz = M.sum(0)                        # (k,)
            KM = K @ M                           # (m×k)
            h = np.einsum("ij,ij->j", M, KM)     # diag(Mᵀ K M) = mask_jᵀ K mask_j
            with np.errstate(divide="ignore", invalid="ignore"):
                d2 = diag[:, None] - 2.0 * KM / sz[None, :] + (h / sz**2)[None, :]
            d2[:, sz == 0] = np.inf
            return d2

        for _ in range(self.max_kmeans_iter):
            new_labels = np.argmin(_d2(labels), axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

        # medoid of each non-empty cluster = point closest to centroid (min d2)
        d2 = _d2(labels)                         # STEP 3: empty clusters → inf
        medoids: list[int] = []
        for j in range(k):
            members = np.where(labels == j)[0]
            if members.size == 0:               # pruned (empty cluster)
                continue
            medoids.append(int(members[np.argmin(d2[members, j])]))
        return np.array(sorted(set(medoids)), dtype=int)

    def _solve(self, X: NDArray, y: NDArray) -> None:
        rng = np.random.default_rng(self.random_state)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == -1)[0]

        # k clusters per class = proto_ratio × (class size), capped at the class
        k_pos = max(1, min(len(pos_idx), round(self.proto_ratio * len(pos_idx))))
        k_neg = max(1, min(len(neg_idx), round(self.proto_ratio * len(neg_idx))))

        # STEP 2–3: feature-space Kernel K-means per class → medoid prototypes
        K_pos = self.kernel_matrix(X[pos_idx])
        K_neg = self.kernel_matrix(X[neg_idx])
        pos_medoids = pos_idx[self._kernel_kmeans(K_pos, k_pos, rng)]
        neg_medoids = neg_idx[self._kernel_kmeans(K_neg, k_neg, rng)]

        # STEP 4–5: opposite map. Nearest prototype in feature space ⇔ highest
        # kernel value, since d²(Φ(a),Φ(b)) = K(a,a)+K(b,b)-2K(a,b).
        selected: set[int] = set()
        if len(pos_medoids) > 0:
            # each negative point picks its nearest positive-class prototype
            K_neg_pos = self.kernel_matrix(X[neg_idx], X[pos_medoids])
            chosen = np.unique(np.argmax(K_neg_pos, axis=1))
            selected.update(int(i) for i in pos_medoids[chosen])
        if len(neg_medoids) > 0:
            # each positive point picks its nearest negative-class prototype
            K_pos_neg = self.kernel_matrix(X[pos_idx], X[neg_medoids])
            chosen = np.unique(np.argmax(K_pos_neg, axis=1))
            selected.update(int(i) for i in neg_medoids[chosen])

        sel = np.array(sorted(selected), dtype=int)
        n = len(y)

        # Numerical guard (NOT the performance fallback): a degenerate reduced
        # set cannot train an LSSVM. Rare; keeps the estimator well-defined.
        if sel.size < 2 or np.unique(y[sel]).size < 2:
            logger.warning(
                "OppositeMapsOriginal: degenerate reduced set (n=%d). Using full set.",
                sel.size,
            )
            alpha_full, bias_full = self._solve_on_subset(X, y)
            self.alpha_ = alpha_full
            self.bias_ = bias_full
            self.support_indices_ = np.arange(n)
            return

        # STEP 6–7: train LSSVM on the reduced set as-is (no fallback)
        alpha_sub, bias = self._solve_on_subset(X[sel], y[sel])
        self.alpha_ = np.zeros(n)
        self.alpha_[sel] = alpha_sub
        self.bias_ = bias
        self.support_indices_ = sel

        logger.info(
            "OppositeMapsOriginalLSSVM solved — %d SVs / %d (%.1f%% sparse)",
            len(self.support_indices_), n, 100.0 * self.sparsity_ratio_,
        )

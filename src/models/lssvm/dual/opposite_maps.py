"""Opposite Maps LSSVM (Neto & Barreto, 2013).

Reduction Method: uses Vector Quantization (k-means) to identify prototype
patterns near the decision boundary from OPPOSITE classes. These "opposite
map" prototypes are the most informative candidates for support vectors.

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
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

from ..base import BaseLSSVM

logger = logging.getLogger(__name__)


class OppositeMapsLSSVM(BaseLSSVM):
    """Opposite Maps LSSVM (Neto & Barreto, 2013).

    Algorithm:
        1. Separate training data into positive (y=+1) and negative (y=-1).
        2. Run k-means with `n_prototypes` centroids per class.
        3. For each prototype, select its nearest SAME-class sample (an
           anchor for its own class's core) and its nearest OPPOSITE-class
           sample (the boundary-critical pattern). These pairs form the
           "opposite maps".
        4. Collect all opposite-map samples as the reduced training set.
        5. Retrain LSSVM on this reduced set.
        6. Safety fallback: compare training F1 against the full LSSVM
           baseline; if the prototype model drops by more than
           ``drop_tolerance``, keep the full model instead.

    Anchor points (2026-07-07): earlier versions selected only the nearest
    OPPOSITE-class sample per prototype, never a same-class anchor. That
    concentrates the whole reduced set on boundary/overlap points — with an
    LSSVM's least-squares loss (unlike hinge loss), fitting exact +1/-1
    targets on a set of only closely-spaced, opposite-class pairs forces a
    violently oscillating decision function (observed empirically as a
    near-saturated kernel matrix and near-random/inverted test predictions
    on hard datasets like HAB). Adding a same-class anchor per prototype
    gives the least-squares fit stable "interior" targets to anchor +1/-1
    against, alongside the boundary pairs.

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth.
    tau : float
        Regularisation parameter.
    n_prototypes : int
        Number of k-means centroids per class. The final SV set has at
        most 4 × n_prototypes samples (same-class anchor + opposite-class
        pair, per prototype, per class).
    random_state : int
        Seed for k-means.
    tol : float
        CG convergence tolerance.
    max_iter : int
        Maximum CG iterations.
    drop_tolerance : float
        Maximum allowed drop in training F1-macro of the prototype model
        compared to the full LSSVM baseline. If the drop is larger, the
        full baseline is kept instead. Set to ``np.inf`` to disable the
        safety fallback. Default 0.05.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        tau: float = 1.0,
        n_prototypes: int = 10,
        random_state: int = 42,
        tol: float = 1e-6,
        max_iter: int = 1000,
        drop_tolerance: float = 0.05,
    ) -> None:
        super().__init__(sigma=sigma, tau=tau, tol=tol, max_iter=max_iter)
        self.n_prototypes = n_prototypes
        self.random_state = random_state
        self.drop_tolerance = drop_tolerance

    def _solve_on_subset(
        self, X_sub: NDArray, y_sub: NDArray
    ) -> tuple[NDArray, float]:
        """Solve LSSVM on (X_sub, y_sub); return (alpha_sub, bias)."""
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

    def _train_f1(
        self, X: NDArray, y: NDArray, X_sub: NDArray, y_sub: NDArray,
        alpha_sub: NDArray, bias: float,
    ) -> float:
        """Macro-F1 on the training set for the model defined by (X_sub, α, b).

        ``alpha_sub`` is the raw dual multiplier (same convention as
        ``BaseLSSVM.decision_function``, which computes ``alpha_ * y_train_``
        before the kernel sum) — it must be weighted by ``y_sub`` here too,
        or the score collapses to a label-agnostic (effectively degenerate)
        decision function.
        """
        K = self.kernel_matrix(X, X_sub)
        scores = K @ (alpha_sub * y_sub) + bias
        y_pred = np.sign(scores)
        y_pred[y_pred == 0] = 1
        return float(f1_score(y, y_pred, average="macro", zero_division=0))

    def _solve(self, X: NDArray, y: NDArray) -> None:
        """Opposite-maps prototype selection + LSSVM retrain with safety fallback."""
        n = len(y)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == -1)[0]

        n_proto = min(self.n_prototypes, len(pos_idx) // 2, len(neg_idx) // 2)
        n_proto = max(1, n_proto)

        # ── k-means per class ─────────────────────────────────────────────────
        pos_proto = self._cluster(X[pos_idx], n_proto)  # (n_proto, p)
        neg_proto = self._cluster(X[neg_idx], n_proto)

        # ── Opposite map: each prototype contributes a same-class anchor and
        #    an opposite-class boundary point ──────────────────────────────────
        selected_set = set()

        for proto in pos_proto:
            dists_pos = np.sum((X[pos_idx] - proto) ** 2, axis=1)
            selected_set.add(int(pos_idx[np.argmin(dists_pos)]))

            dists_neg = np.sum((X[neg_idx] - proto) ** 2, axis=1)
            selected_set.add(int(neg_idx[np.argmin(dists_neg)]))

        for proto in neg_proto:
            dists_neg = np.sum((X[neg_idx] - proto) ** 2, axis=1)
            selected_set.add(int(neg_idx[np.argmin(dists_neg)]))

            dists_pos = np.sum((X[pos_idx] - proto) ** 2, axis=1)
            selected_set.add(int(pos_idx[np.argmin(dists_pos)]))

        selected = np.array(sorted(selected_set))

        # ── Train on prototype subset ────────────────────────────────────────
        X_sub = X[selected]
        y_sub = y[selected]
        n_sub = len(y_sub)
        proto_degenerate = n_sub < 2 or len(np.unique(y_sub)) < 2

        if proto_degenerate:
            logger.warning(
                "OppositeMaps: degenerate prototype subset (n=%d, classes=%s). "
                "Using full training set.",
                n_sub, np.unique(y_sub),
            )
            self._set_full_solution(X, y)
            return

        alpha_proto, bias_proto = self._solve_on_subset(X_sub, y_sub)

        # ── Safety fallback: compare against full LSSVM baseline ──────────────
        if not np.isfinite(self.drop_tolerance):
            self._set_proto_solution(n, selected, alpha_proto, bias_proto)
            return

        f1_proto = self._train_f1(X, y, X_sub, y_sub, alpha_proto, bias_proto)
        alpha_full, bias_full = self._solve_on_subset(X, y)
        f1_full = self._train_f1(X, y, X, y, alpha_full, bias_full)

        if f1_proto < f1_full - self.drop_tolerance:
            logger.info(
                "OppositeMaps fallback: prototype F1=%.4f vs full F1=%.4f "
                "(drop %.4f > tol %.4f). Keeping full model.",
                f1_proto, f1_full, f1_full - f1_proto, self.drop_tolerance,
            )
            self.alpha_: NDArray = alpha_full
            self.bias_: float = bias_full
            self.support_indices_: NDArray = np.arange(n)
        else:
            self._set_proto_solution(n, selected, alpha_proto, bias_proto)

        logger.info(
            "OppositeMapsLSSVM solved — %d SVs / %d (%.1f%% sparse)",
            len(self.support_indices_), n, 100.0 * self.sparsity_ratio_,
        )

    def _set_proto_solution(
        self, n: int, selected: NDArray,
        alpha_sub: NDArray, bias: float,
    ) -> None:
        self.alpha_: NDArray = np.zeros(n)
        self.alpha_[selected] = alpha_sub
        self.bias_: float = bias
        self.support_indices_: NDArray = selected

    def _set_full_solution(self, X: NDArray, y: NDArray) -> None:
        alpha_full, bias_full = self._solve_on_subset(X, y)
        self.alpha_: NDArray = alpha_full
        self.bias_: float = bias_full
        self.support_indices_: NDArray = np.arange(len(y))

    def _cluster(self, X: NDArray, k: int) -> NDArray:
        """Run k-means and return centroids."""
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
        km.fit(X)
        return km.cluster_centers_

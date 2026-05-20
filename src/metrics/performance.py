"""Classification performance metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)


def compute_performance(
    y_true: NDArray,
    y_pred: NDArray,
    y_proba: NDArray | None = None,
) -> dict[str, float]:
    """Compute standard binary classification metrics.

    Parameters
    ----------
    y_true : ground-truth labels
    y_pred : predicted class labels
    y_proba : (N, 2) probability array; if None, AUC/AP are omitted

    Returns
    -------
    dict with keys: accuracy, f1_macro, f1_binary, mcc, [auc_roc, avg_precision]
    """
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_binary": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    if y_proba is not None:
        p_pos = y_proba[:, 1]
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, p_pos))
        except ValueError:
            metrics["auc_roc"] = float("nan")
        try:
            metrics["avg_precision"] = float(average_precision_score(y_true, p_pos))
        except ValueError:
            metrics["avg_precision"] = float("nan")

    return metrics

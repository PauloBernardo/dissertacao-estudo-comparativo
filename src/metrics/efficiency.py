"""Training and inference efficiency metrics."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

import numpy as np
from numpy.typing import NDArray


@contextmanager
def timer():
    """Context manager that yields elapsed wall-clock seconds."""
    state = {"elapsed": 0.0}
    t0 = time.perf_counter()
    try:
        yield state
    finally:
        state["elapsed"] = time.perf_counter() - t0


def measure_fit_time(model, X_train: NDArray, y_train: NDArray) -> tuple[Any, float]:
    """Fit a model and return (fitted_model, train_time_seconds)."""
    with timer() as t:
        model.fit(X_train, y_train)
    return model, t["elapsed"]


def measure_predict_time(model, X_test: NDArray) -> tuple[NDArray, float]:
    """Run predict and return (predictions, predict_time_seconds)."""
    with timer() as t:
        preds = model.predict(X_test)
    return preds, t["elapsed"]


def efficiency_metrics(
    train_time: float,
    predict_time: float,
    n_train: int,
    n_test: int,
) -> dict[str, float]:
    """Summarise timing results.

    Returns
    -------
    dict with keys: train_time_s, predict_time_s,
                    train_time_per_sample_ms, predict_time_per_sample_ms
    """
    return {
        "train_time_s": train_time,
        "predict_time_s": predict_time,
        "train_time_per_sample_ms": 1000.0 * train_time / max(n_train, 1),
        "predict_time_per_sample_ms": 1000.0 * predict_time / max(n_test, 1),
    }

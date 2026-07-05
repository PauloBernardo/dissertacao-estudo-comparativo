#!/usr/bin/env python3
"""Quick test: does smaller embedding_dim help on small Tier 1 datasets?

Runs FTTransformer_softmax with embedding_dim in [16, 32, 64] on a subset
of Tier 1 datasets, 10 seeds each. Prints mean F1 per (dim, dataset).

Usage
-----
    python scripts/test_embedding_dim.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import DatasetLoader
from src.experiments.reproducibility import set_global_seed
from src.experiments.runner import _build_model

DATASETS     = ["BCW", "HAB", "TWC", "TWS"]
EMB_DIMS     = [16, 32, 64]
SEEDS        = list(range(10))
NUM_BLOCKS   = 2
NUM_HEADS    = 2


def run_one(dataset: str, embedding_dim: int, seed: int) -> float:
    set_global_seed(seed)
    X, y, _ = DatasetLoader.load(dataset)

    # binary labels
    classes = np.unique(y)
    mapping = {c: i for i, c in enumerate(classes)}
    y = np.array([mapping[c] for c in y])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )

    params = {
        "attention_type": "softmax",
        "embedding_dim":  embedding_dim,
        "num_blocks":     NUM_BLOCKS,
        "num_heads":      NUM_HEADS,
        "dropout":        0.1,
        "lr":             1e-3,
        "batch_size":     256,
        "max_epochs":     15,
        "patience":       3,
    }

    estimator, _ = _build_model("FTTransformer", params, label_format="binary")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    return float(f1_score(y_te, y_pred, average="macro", zero_division=0))


def main() -> None:
    results: dict[tuple, list] = {}

    total = len(DATASETS) * len(EMB_DIMS) * len(SEEDS)
    done  = 0
    t0    = time.perf_counter()

    for ds in DATASETS:
        for dim in EMB_DIMS:
            scores = []
            for seed in SEEDS:
                f1 = run_one(ds, dim, seed)
                scores.append(f1)
                done += 1
                elapsed = time.perf_counter() - t0
                eta = elapsed / done * (total - done)
                print(f"[{done}/{total}] {ds} dim={dim} seed={seed} → F1={f1:.3f}  (ETA {eta:.0f}s)")
            results[(ds, dim)] = scores

    print("\n── Resultados ────────────────────────────────")
    print(f"{'Dataset':<8}", end="")
    for dim in EMB_DIMS:
        print(f"  dim={dim}", end="")
    print()
    for ds in DATASETS:
        print(f"{ds:<8}", end="")
        for dim in EMB_DIMS:
            mu = np.mean(results[(ds, dim)])
            print(f"  {mu:.3f} ", end="")
        # best dim
        best = max(EMB_DIMS, key=lambda d: np.mean(results[(ds, d)]))
        print(f"  ← best: {best}")


if __name__ == "__main__":
    main()

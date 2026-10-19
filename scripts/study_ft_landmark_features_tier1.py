#!/usr/bin/env python3
"""FT-LF nas bases reais do Tier 1 com d ≥ 6 (BCW 30, GCR 24, AUS 14, PID 8, VCP 6, AI4I 6).

Protocolo do Tier 1 (hold-out 70/30 estratificado com random_state = semente; refit 200 épocas sem
paciência), arquitetura FIXA (3 blocos, 4 cabeças, d = 64, lr 1e-3, lote cheio como no Tier 1).
m/d = 1,0 é o FT softmax denso (referência). Seletores: random, attn_colnorm, sparsemax.

    .venv/bin/python scripts/study_ft_landmark_features_tier1.py
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import run_tier2_gridcv as g  # noqa: E402
from src.data.preprocessing import make_splits, _convert_labels  # noqa: E402
from src.models.transformers.ft_landmark_features import FTLandmarkFeatures  # noqa: E402

ARCH = {"num_blocks": 3, "num_heads": 4, "embedding_dim": 64, "dropout": 0.1, "lr": 1e-3,
        "batch_size": 4096, "max_epochs": 200, "patience": 200}   # lote cheio + refit 200 (Tier 1)
DATASETS = ["BCW", "GCR", "AUS", "PID", "VCP", "AI4I"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--methods", nargs="+", default=["random", "attn_colnorm", "sparsemax"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--output", default=str(ROOT / "results/study_ft_landmark_features_tier1.json"))
    a = p.parse_args()
    out = Path(a.output)
    recs = json.loads(out.read_text()) if out.exists() else []
    done = {(r["dataset"], r["m_ratio"], r["method"], r["seed"]) for r in recs}
    for ds in a.datasets:
        X, y, _ = g.DatasetLoader.load(ds)
        for seed in a.seeds:
            X_tr, X_te, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
            ytr, yte = _convert_labels(ytr, "binary"), _convert_labels(yte, "binary")
            for r in a.ratios:
                for meth in (["dense"] if r == 1.0 else a.methods):
                    if (ds, r, meth, seed) in done:
                        continue
                    g.set_global_seed(seed)
                    pipe = Pipeline([("scaler", StandardScaler()), ("clf", FTLandmarkFeatures(
                        m_ratio=r, landmark_method="random" if meth == "dense" else meth,
                        random_state=seed, **ARCH))])
                    t0 = time.perf_counter(); pipe.fit(X_tr, ytr); fit_s = time.perf_counter() - t0
                    clf = pipe.named_steps["clf"]
                    f1 = float(f1_score(yte, pipe.predict(X_te), average="macro"))
                    rec = {"dataset": ds, "n_features": X.shape[1], "m_ratio": r, "m": clf.n_landmarks_,
                           "method": meth, "seed": seed, "test_f1_macro": f1,
                           "sparsity_ratio": clf.sparsity_ratio_, "landmarks": clf.landmark_features_.tolist(),
                           "fit_time_s": fit_s}
                    recs.append(rec); out.write_text(json.dumps(recs, indent=1))
                    print(f"{ds:5s} s{seed} m/d={r:.2f} m={clf.n_landmarks_:2d}/{X.shape[1]:2d} {meth:12s} "
                          f"F1 {f1:.4f}  {fit_s:5.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

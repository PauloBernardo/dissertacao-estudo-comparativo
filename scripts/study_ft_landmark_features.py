#!/usr/bin/env python3
"""FT-LF (landmark features, ver src/models/transformers/ft_landmark_features.py) na curva de ruído.

Mesmo dado de study_noise_features_sweep.py (TWS + k features N(0,1); as 2 primeiras são as úteis) e a
mesma arquitetura fixa dos FT (3 blocos, 4 cabeças, lote 128, 200 épocas, paciência 16).
Varre m/d e o seletor (random, attn_colnorm); registra F1, esparsidade 1 − m/d e quantas das 2 features
úteis caíram entre os landmarks.

    .venv/bin/python scripts/study_ft_landmark_features.py
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
import study_noise_features_sweep as S  # noqa: E402
from src.data.preprocessing import _convert_labels  # noqa: E402
from src.models.transformers.ft_landmark_features import FTLandmarkFeatures  # noqa: E402

ARCH = {"num_blocks": 3, "num_heads": 4, "batch_size": 128, "embedding_dim": 64, "dropout": 0.1,
        "lr": 1e-3, "max_epochs": 200, "patience": 16}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--noise", nargs="+", type=int, default=[0, 3, 10, 50])
    p.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    p.add_argument("--methods", nargs="+", default=["random", "attn_colnorm"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--output", default=str(ROOT / "results/study_ft_landmark_features.json"))
    a = p.parse_args()
    out = Path(a.output)
    recs = json.loads(out.read_text()) if out.exists() else []
    done = {(r["n_noise"], r["m_ratio"], r["method"], r["seed"]) for r in recs}
    for k in a.noise:
        for seed in a.seeds:
            X_tr, X_te, ytr, yte = S.data(k, seed, 2500)
            ytr, yte = _convert_labels(ytr, "binary"), _convert_labels(yte, "binary")
            for r in a.ratios:
                for meth in a.methods:
                    if (k, r, meth, seed) in done or (k == 0 and r < 0.5):   # d = 2: m/d = 0,25 ≡ 0,5
                        continue
                    g.set_global_seed(seed)
                    pipe = Pipeline([("scaler", StandardScaler()), ("clf", FTLandmarkFeatures(
                        m_ratio=r, landmark_method=meth, random_state=seed, **ARCH))])
                    t0 = time.perf_counter(); pipe.fit(X_tr, ytr); fit_s = time.perf_counter() - t0
                    clf = pipe.named_steps["clf"]
                    f1 = float(f1_score(yte, pipe.predict(X_te), average="macro"))
                    lm = clf.landmark_features_.tolist()
                    rec = {"n_noise": k, "n_features": 2 + k, "m_ratio": r, "m": clf.n_landmarks_,
                           "method": meth, "seed": seed, "test_f1_macro": f1,
                           "sparsity_ratio": clf.sparsity_ratio_, "landmarks": lm,
                           "n_useful_in_landmarks": int(sum(j < 2 for j in lm)), "fit_time_s": fit_s}
                    recs.append(rec); out.write_text(json.dumps(recs, indent=1))
                    print(f"k={k:3d} s{seed} m/d={r:.2f} m={clf.n_landmarks_:3d} {meth:12s} F1 {f1:.4f}  "
                          f"esparsidade {clf.sparsity_ratio_:.2f}  úteis {rec['n_useful_in_landmarks']}/2  {fit_s:5.0f}s",
                          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

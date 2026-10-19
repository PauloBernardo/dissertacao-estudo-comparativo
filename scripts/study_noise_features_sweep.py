#!/usr/bin/env python3
"""Ablação B ampliada (piloto): a atenção esparsa vira seleção de features quando há muito ruído?

TWS (espiral, 2 features informativas) + k features gaussianas de ruído (N(0,1)), k ∈ {0,3,10,50,200}.
N = 2500 (70/30 → 1750 de treino). Hipótese declarada antes de rodar: a vantagem de Top-k/entmax/sparsemax
sobre a softmax cresce com k; XGBoost (seleção por árvore) é a referência.

Piloto: os FT usam arquitetura FIXA (3 blocos, 4 cabeças, lote 128, lr 1e-3, 200 épocas, paciência 16); XGBoost e
Nyström-LSSVM fazem GridSearchCV 3-fold nas grades do Tier 2. Retoma pelo arquivo de saída.

    .venv/bin/python scripts/study_noise_features_sweep.py
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_tier2_gridcv as g  # noqa: E402
from src.data.synthetic import make_tws  # noqa: E402
from src.data.preprocessing import make_splits, _convert_labels  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

FT = ["FTTransformer_softmax", "FTTransformer_topk", "FTTransformer_entmax", "FTTransformer_sparsemax"]
CPU = ["XGBoost", "NystromLSSVMRandom"]
FT_ARCH = {"num_blocks": 3, "num_heads": 4, "batch_size": 128}   # 512 estoura a MX350 com d = 202
# FT-CUR e SAINT: d_model 32 do estudo, 2 camadas; FT-CUR em per_batch (m/N = 10%) + predição streaming
ARCH = {v: FT_ARCH for v in FT}
ARCH["FTTransformer_sparsemax"] = {**FT_ARCH, "batch_size": 64}   # sparsemax: OOM com 128 em d = 202
ARCH["FTTransformerCURRandom"] = {"n_layers": 2, "n_heads": 4, "m_ratio": 0.1, "batch_size": 128,
                                  "predict_mode": "streaming"}
ARCH["SAINTColnorm"] = {"n_layers": 2, "n_heads": 4, "batch_size": 128}
FT = FT + ["FTTransformerCURRandom", "SAINTColnorm"]

# Avaliação em pedaços de 256 (o padrão 4096 estoura a MX350 com 203 tokens); exato: o FT não mistura linhas.
import src.models.transformers.ft_transformer as _ftm  # noqa: E402
_orig_chunked = _ftm.FTTransformer._forward_chunked
_ftm.FTTransformer._forward_chunked = lambda self, X_t, chunk=256: _orig_chunked(self, X_t, chunk)


def data(k: int, seed: int, n: int):
    X, y = make_tws(n_samples=n, random_state=seed)
    if k:
        X = np.hstack([X, np.random.default_rng(seed + 1000).normal(0, 1, (n, k))])
    return make_splits(X, y, test_size=0.30, seed=seed)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--noise", nargs="+", type=int, default=[0, 3, 10, 50, 200])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--variants", nargs="+", default=CPU + FT)
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--output", default=str(ROOT / "results/study_noise_features_sweep.json"))
    a = p.parse_args()
    g.BUDGET_OVERRIDE.update({"epochs": 200, "patience": 16})
    out = Path(a.output)
    recs = json.loads(out.read_text()) if out.exists() else []
    done = {(r["variant"], r["n_noise"], r["seed"]) for r in recs}
    for k in a.noise:
        for seed in a.seeds:
            X_tr, X_te, ytr, yte = data(k, seed, a.n)
            for v in a.variants:
                if (v, k, seed) in done:
                    continue
                fmt = g._label_format(v)
                g.set_global_seed(seed)
                pipe, grid = g._build_pipeline(v, seed)
                t0 = time.perf_counter()
                if v in FT:
                    arch = ARCH[v]
                    pipe.set_params(**{f"clf__{p_}": x for p_, x in arch.items()})
                    pipe.fit(X_tr, _convert_labels(ytr, fmt)); best = arch
                else:
                    cv = GridSearchCV(pipe, grid, scoring="f1_macro", n_jobs=-1,
                                      cv=StratifiedKFold(3, shuffle=True, random_state=seed))
                    cv.fit(X_tr, _convert_labels(ytr, fmt)); pipe = cv.best_estimator_
                    best = {p_.removeprefix("clf__"): x for p_, x in cv.best_params_.items()}
                fit_s = time.perf_counter() - t0
                yp, yt = pipe.predict(X_te), _convert_labels(yte, fmt)
                if fmt == "signed":
                    yp, yt = (yp + 1) // 2, (yt + 1) // 2
                f1 = float(f1_score(yt, yp, average="macro"))
                recs.append({"variant": v, "n_noise": k, "n_features": 2 + k, "seed": seed, "n_train": len(X_tr),
                             "test_f1_macro": f1, "fit_time_s": fit_s, "params": best})
                out.write_text(json.dumps(recs, indent=1))
                print(f"k={k:3d} s{seed} {v:24s} F1 {f1:.4f}  {fit_s:6.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

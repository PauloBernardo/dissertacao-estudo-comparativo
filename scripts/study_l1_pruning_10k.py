#!/usr/bin/env python3
"""Poda por l1 vs menos landmarks, com N_treino = 10 000 (2026-10-06).

Pergunta. No Tier 1 (N_treino = 280), com a mesma esparsidade, o Nyström-SVM com menos landmarks
aleatórios iguala ou supera o ADMM-/FISTA-Nyström (30% de landmarks + poda l1): o l1 não escolhe
pontos melhor do que o acaso. Mas com N pequeno há pouca redundância a podar. Aqui o teste é
repetido em N_treino = 10 000.

Desenho (curvas F1 x esparsidade; sem seleção, logo sem validação cruzada):
  - ADULT e HIGGS50K, 5 sementes; subamostragem e split 70/30 idênticos ao Tier 2.
  - sigma, tau, gamma fixos na configuração MODAL do Tier 2 de cada modelo e dataset.
  - ADMM-Nyström e FISTA-Nyström: m_ratio = 0,1 e lambda varrido (inclusive acima de 1: o
    FISTA-Nyström no HIGGS50K escolhe lambda = 1, o topo da grade publicada). O ADMM roda com
    max_iter = 500 (protocolo publicado) e 5000 (para ver se a truncagem muda a conclusão).
  - Nyström-SVM: m_ratio de 0,01 a 0,1 (esparsidade 0,90 a 0,99), sem l1.
  - Cada ponto: ajuste no treino, F1-macro no teste, esparsidade, iterações, convergência.

Uso: python scripts/study_l1_pruning_10k.py [--seeds 0 1 2 3 4] [--n-jobs 3]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data.loaders import DatasetLoader  # noqa: E402
from src.data.preprocessing import _convert_labels  # noqa: E402
from src.experiments.runner import _build_model  # noqa: E402
from src.tuning.grids import GRIDS  # noqa: E402
from run_tier2_gridcv import _subsample  # noqa: E402

N_TRAIN = 10_000
LAMBDAS = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.01, 0.001]
M_NYSTROM = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
M_L1 = 0.10
# configuração modal do Tier 2 (results/tier2_gridcv.json, 30 sementes)
MODAL = {
    ("NystromLSSVMRandom", "ADULT"):    {"sigma": 5.0, "gamma": 50.0},
    ("NystromLSSVMRandom", "HIGGS50K"): {"sigma": 5.0, "gamma": 1.0},
    ("ADMMNystromRandom", "ADULT"):     {"sigma": 2.0, "tau": 50.0},
    ("ADMMNystromRandom", "HIGGS50K"):  {"sigma": 2.0, "tau": 5.0},
    ("FISTANystromRandom", "ADULT"):    {"sigma": 2.0, "tau": 0.05},
    ("FISTANystromRandom", "HIGGS50K"): {"sigma": 2.0, "tau": 0.5},
}
FMT = {"NystromLSSVMRandom": "signed", "ADMMNystromRandom": "binary", "FISTANystromRandom": "binary"}


def _points(ds: str) -> list[tuple[str, dict]]:
    pts = [("NystromLSSVMRandom", {**MODAL[("NystromLSSVMRandom", ds)], "m_ratio": m}) for m in M_NYSTROM]
    for lam in LAMBDAS:
        for it in (500, 5000):
            pts.append(("ADMMNystromRandom", {**MODAL[("ADMMNystromRandom", ds)], "m_ratio": M_L1,
                                              "lambda_": lam, "max_iter": it}))
        pts.append(("FISTANystromRandom", {**MODAL[("FISTANystromRandom", ds)], "m_ratio": M_L1, "lambda_": lam}))
    return pts


def _fit_point(variant, params, Xtr, ytr, Xte, yte, seed):
    logging.disable(logging.WARNING)
    fmt = FMT[variant]
    est, _ = _build_model(GRIDS[variant]["model_name"], {**GRIDS[variant]["fixed"], **params}, label_format=fmt)
    if "random_state" in est.get_params():
        est.set_params(random_state=seed)
    yt_tr, yt_te = _convert_labels(ytr, fmt), _convert_labels(yte, fmt)
    t0 = time.perf_counter()
    est.fit(Xtr, yt_tr)
    dt = time.perf_counter() - t0
    yp = est.predict(Xte)
    if fmt == "signed":
        yt_te, yp = (yt_te + 1) // 2, (yp + 1) // 2
    return dict(variant=variant, params=params, test_f1=float(f1_score(yt_te, yp, average="macro", zero_division=0)),
                sparsity=float(est.sparsity_ratio_), n_iter=getattr(est, "n_iter_", None),
                converged=getattr(est, "converged_", None), fit_time_s=dt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ADULT", "HIGGS50K"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--n-jobs", type=int, default=3)
    ap.add_argument("--output", type=Path, default=Path("results/study_l1_pruning_10k.json"))
    a = ap.parse_args()
    out = json.loads(a.output.read_text()) if a.output.exists() else []
    done = {(r["dataset"], r["seed"]) for r in out}
    for ds in a.datasets:
        X, y, _ = DatasetLoader.load(ds)
        for seed in a.seeds:
            if (ds, seed) in done:
                continue
            Xs, ys = _subsample(X, y, round(N_TRAIN / 0.70), seed)
            Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.30, stratify=ys, random_state=seed)
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            t0 = time.time()
            res = Parallel(n_jobs=a.n_jobs)(delayed(_fit_point)(v, p, Xtr, ytr, Xte, yte, seed) for v, p in _points(ds))
            out.append(dict(dataset=ds, seed=seed, n_train=len(ytr), points=res))
            a.output.write_text(json.dumps(out))
            print(f"{ds:9s} seed{seed}  n_train={len(ytr)}  {len(res)} pontos  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

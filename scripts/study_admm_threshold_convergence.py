#!/usr/bin/env python3
"""Estudo de convergência do ADMM sob as duas parametrizações do limiar.

Pergunta
--------
Os experimentos da dissertação usaram o limiar ``S_{λ/(2ρ)}``; o paper-fonte
(Marinho et al.) usa ``S_{λ/ρ}``. As duas são o MESMO operador sob λ' = λ/2.
Este estudo mede (i) que a equivalência de fato se verifica e (ii) qual é o
custo de convergência de trocar a convenção sem reindexar a grade de λ.

Como se implementa sem tocar no modelo
--------------------------------------
O código atual aplica ``threshold = lambda_ / rho``.  Logo:

    convenção ANTIGA com λ nominal  ==  código atual com ``lambda_ = λ/2``
    convenção NOVA   com λ nominal  ==  código atual com ``lambda_ = λ``

Basta, portanto, variar ``lambda_``.

Note-se que a equivalência entre as convenções é uma IDENTIDADE ALGÉBRICA do
operador de soft-threshold, não uma hipótese empírica: não há o que testar
nesse ponto. O que este estudo mede é o efeito de trocar a convenção SEM
reindexar a grade de λ — isto é, comparar antiga(λ) contra nova(λ) para o
mesmo λ nominal, caso em que o limiar efetivo DOBRA. Foi esse o cenário
produzido pela rodada harmonizada.

Mede-se também, como subproduto, a TAXA DE NÃO-CONVERGÊNCIA em 500 iterações
no ponto de operação real (σ, τ modais do GridSearchCV), que se verifica ser
elevada em AMBAS as convenções — portanto uma propriedade do ADMM neste
regime, e não um efeito da harmonização.

Saídas:
    results/admm_threshold_convergence.json
    dissertacao-latex/tables/app_admm_convergence.tex
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import DatasetLoader                    # noqa: E402
from src.data.preprocessing import _convert_labels, make_splits  # noqa: E402
from src.models.lssvm.primal.admm_nesterov import ADMMNesterovLSSVM  # noqa: E402
from src.models.lssvm.primal.admm_nystrom import ADMMNystromLSSVM    # noqa: E402
from sklearn.metrics import f1_score                          # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit     # noqa: E402

THESIS = ROOT.parent / "dissertacao-latex"

DATASETS = ["ADULT", "BANK", "CREDIT"]
LAMBDAS = [1.0, 0.1, 0.01]        # grade nominal da dissertação
# (σ, τ) modais do GridSearchCV do Tier 2, por dataset — evita medir a
# convergência num ponto de operação artificial em que o modelo colapsa.
PARAMS = {"ADULT": (2.0, 5.0), "BANK": (2.0, 0.5), "CREDIT": (2.0, 5.0),
          "HIGGS50K": (2.0, 0.5), "SHOPPERS": (2.0, 5.0), "TELCO": (2.0, 5.0)}
N_TOTAL_CAP = 2857                # mesmo protocolo do Tier 2 (N_train=2000)
MAX_ITER = 500                    # mesmo teto usado nos experimentos


def prepare(dataset: str, seed: int):
    X, y, _ = DatasetLoader.load(dataset)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(N_TOTAL_CAP, len(y)),
                                 random_state=seed)
    idx, _ = next(sss.split(X, y))
    X, y = X[idx], y[idx]
    Xtr, Xte, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
    sc = StandardScaler().fit(Xtr)
    return (sc.transform(Xtr), sc.transform(Xte),
            _convert_labels(ytr, "signed"), _convert_labels(yte, "signed"))


def run_one(model_cls, Xtr, ytr, Xte, yte, lam: float, sigma: float, tau: float) -> dict:
    kw = dict(sigma=sigma, tau=tau, lambda_=lam, max_iter=MAX_ITER)
    if model_cls is ADMMNystromLSSVM:
        kw.update(m_ratio=0.30, landmark_method="colnorm", rho=None)
    est = model_cls(**kw)
    t0 = time.perf_counter()
    est.fit(Xtr, ytr)
    dt = time.perf_counter() - t0
    return {
        "n_iter": int(getattr(est, "n_iter_", -1)),
        "converged": bool(getattr(est, "converged_", False)),
        "sparsity": float(getattr(est, "sparsity_ratio_", float("nan"))),
        "f1_macro": float(f1_score(yte, est.predict(Xte), average="macro")),
        "fit_time_s": dt,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    a = p.parse_args()
    logging.getLogger().setLevel(logging.ERROR)   # silencia avisos por run

    recs = []
    for ds in a.datasets:
        for seed in a.seeds:
            Xtr, Xte, ytr, yte = prepare(ds, seed)
            sigma, tau = PARAMS[ds]
            for name, cls in [("ADMM-Nesterov", ADMMNesterovLSSVM),
                              ("ADMM-Nystrom", ADMMNystromLSSVM)]:
                for lam in LAMBDAS:
                    # convenção ANTIGA (λ/2ρ) com λ nominal  ==  lambda_ = λ/2
                    old = run_one(cls, Xtr, ytr, Xte, yte, lam / 2.0, sigma, tau)
                    # convenção NOVA (λ/ρ) com o MESMO λ nominal
                    new = run_one(cls, Xtr, ytr, Xte, yte, lam, sigma, tau)
                    recs.append({"dataset": ds, "seed": seed, "model": name,
                                 "lambda_nominal": lam,
                                 "old": old, "new": new})
                    print(f"{ds:9s} s{seed} {name:14s} λ={lam:<5} "
                          f"antiga: {old['n_iter']:4d} it {'ok ' if old['converged'] else 'NAO'} "
                          f"esp={old['sparsity']:.3f} F1={old['f1_macro']:.3f} | "
                          f"nova: {new['n_iter']:4d} it {'ok ' if new['converged'] else 'NAO'} "
                          f"esp={new['sparsity']:.3f} F1={new['f1_macro']:.3f}", flush=True)
    out = ROOT / "results" / "admm_threshold_convergence.json"
    out.write_text(json.dumps(recs, indent=1), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Quanto do desempenho do ADMM-Nyström é limitado pelo teto de iterações?

Contexto
--------
O objetivo do ADMM-Nyström é (1/2τ)‖Cθ − y‖² + λ‖θ‖₁, cujo minimizador
depende APENAS do produto λτ. Já o passo do ADMM é ρ ∝ τ (auto-selecionado
como 1/max_eig(CᵀC/τ)), de modo que o limiar por iteração é λ/ρ ∝ λ/τ.

Consequência: duas configurações com o MESMO λτ definem o mesmo problema,
mas convergem a velocidades muito diferentes. Sob um teto fixo de iterações,
o iterando devolvido depende de λ e τ separadamente — e o GridSearchCV passa
a selecionar, em parte, velocidade de convergência em vez de ajuste.

Este script mede o tamanho desse efeito no ponto de operação publicado:
para cada dataset do Tier 2, roda a configuração modal do GridSearchCV com
o teto original (500) e com um teto folgado, e compara.

Saída: results/admm_truncation_effect.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loaders import DatasetLoader                       # noqa: E402
from src.data.preprocessing import _convert_labels, make_splits  # noqa: E402
from src.models.lssvm.primal.admm_nystrom import ADMMNystromLSSVM  # noqa: E402

# (sigma, tau, lambda_) modais do GridSearchCV do Tier 2 (N_treino=2000).
MODAL = {
    "ADULT":    (2.0, 50.0, 0.01),
    "BANK":     (2.0,  0.5, 0.01),
    "CREDIT":   (2.0,  5.0, 0.10),
    "HIGGS50K": (2.0,  0.5, 1.00),
    "SHOPPERS": (2.0,  5.0, 0.01),
    "TELCO":    (2.0,  5.0, 0.10),
}
N_TOTAL_CAP = 2857
CAPS = [500, 20000]        # teto publicado vs teto folgado


def prepare(dataset: str, seed: int):
    X, y, _ = DatasetLoader.load(dataset)
    n = min(N_TOTAL_CAP, len(y))
    idx, _ = next(StratifiedShuffleSplit(1, train_size=n, random_state=seed).split(X, y))
    X, y = X[idx], y[idx]
    Xtr, Xte, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
    sc = StandardScaler().fit(Xtr)
    return (sc.transform(Xtr), sc.transform(Xte),
            _convert_labels(ytr, "signed"), _convert_labels(yte, "signed"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(30)))
    p.add_argument("--datasets", nargs="+", default=list(MODAL))
    p.add_argument("--output", type=Path,
                   default=ROOT / "results" / "admm_truncation_effect.json")
    a = p.parse_args()
    logging.getLogger().setLevel(logging.ERROR)

    recs = list(json.loads(a.output.read_text())) if a.output.exists() else []
    done = {(r["dataset"], r["seed"], r["max_iter"]) for r in recs}

    for ds in a.datasets:
        sigma, tau, lam = MODAL[ds]
        for seed in a.seeds:
            Xtr, Xte, ytr, yte = prepare(ds, seed)
            for cap in CAPS:
                if (ds, seed, cap) in done:
                    continue
                m = ADMMNystromLSSVM(sigma=sigma, tau=tau, lambda_=lam,
                                     m_ratio=0.30, landmark_method="colnorm",
                                     rho=None, max_iter=cap)
                t0 = time.perf_counter()
                m.fit(Xtr, ytr)
                dt = time.perf_counter() - t0
                recs.append({
                    "dataset": ds, "seed": seed, "max_iter": cap,
                    "sigma": sigma, "tau": tau, "lambda_": lam,
                    "lambda_times_tau": lam * tau, "lambda_over_tau": lam / tau,
                    "n_iter": int(m.n_iter_), "converged": bool(m.converged_),
                    "sparsity": float(m.sparsity_ratio_),
                    "f1_macro": float(f1_score(yte, m.predict(Xte), average="macro")),
                    "fit_time_s": dt,
                })
                a.output.write_text(json.dumps(recs, indent=1), encoding="utf-8")
            r5, r2 = recs[-2], recs[-1]
            print(f"{ds:9s} s{seed:<2d} λτ={lam*tau:<6} λ/τ={lam/tau:<8.4f} | "
                  f"500: {r5['n_iter']:5d}it {'ok ' if r5['converged'] else 'NAO'} "
                  f"F1={r5['f1_macro']:.4f} esp={r5['sparsity']:.3f} | "
                  f"20k: {r2['n_iter']:5d}it {'ok ' if r2['converged'] else 'NAO'} "
                  f"F1={r2['f1_macro']:.4f} esp={r2['sparsity']:.3f} | "
                  f"ΔF1={r2['f1_macro']-r5['f1_macro']:+.4f}", flush=True)
    print(f"\n-> {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Ablação D completa do ADMM-Nyström, com e sem Elastic Net.

Hipótese sob teste (H1)
-----------------------
A queda do ADMM-Nyström ao transferir hiperparâmetros de N=2000 para N=5000 é,
em parte dos datasets, efeito do TRUNCAMENTO: com o teto de 500 iterações a
solução ainda está sendo construída quando o laço para. Se for isso, o termo
ℓ2 do Elastic Net — que acelera a aproximação do ótimo sem alterar o suporte —
deve recuperar o desempenho SEM re-tunar e SEM aumentar o teto.

Protocolo: idêntico ao da Ablação D publicada (N_treino=5000, hiperparâmetros
fixos herdados de N=2000, teto de 500 iterações, 30 sementes, 6 datasets),
variando λ₂ ∈ {0.01, 0.1, 1.0}. O braço λ₂=0 NÃO é re-executado: ele é a
própria Ablação D publicada (results/tier2_fixedparams_n5000_lssvm.json), usada
aqui como linha de base em PUB_5K.

Saída: results/ablationD_elasticnet.json  (retomável)
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

# (σ, τ, λ) fixos da Ablação D + F1 publicado em N=2000 e em N=5000 (referência)
CFG = {
    "ADULT":    (2.0, 50.0, 0.01),
    "BANK":     (2.0,  0.5, 0.01),
    "CREDIT":   (2.0,  5.0, 0.10),
    "HIGGS50K": (2.0,  0.5, 1.00),
    "SHOPPERS": (2.0,  5.0, 0.01),
    "TELCO":    (2.0,  5.0, 0.10),
}
N_TARGET = 5000          # N_treino alvo (TELCO fica em ~4922 pela dimensão total)

# Referências publicadas (ADMM-Nyström): Ablação D com λ₂=0 e alvo em N=2000.
PUB_5K = {"ADULT": 0.7584, "BANK": 0.5598, "CREDIT": 0.4670,
          "HIGGS50K": 0.4810, "SHOPPERS": 0.7490, "TELCO": 0.6798}
PUB_2K = {"ADULT": 0.747, "BANK": 0.667, "CREDIT": 0.646,
          "HIGGS50K": 0.594, "SHOPPERS": 0.733, "TELCO": 0.697}
MAX_ITER = 500           # teto da Ablação D
L2S = (0.01, 0.1, 1.0)   # braços de λ₂ (o braço λ₂=0 é a Ablação D já publicada)


def prepare(dataset: str, seed: int):
    X, y, _ = DatasetLoader.load(dataset)
    cap = int(round(N_TARGET / 0.7))
    # Se o dataset é menor que o alvo, usa-se o conjunto inteiro (caso do TELCO,
    # com 7032 amostras → 4922 de treino).  O StratifiedShuffleSplit exige
    # train_size estritamente menor que n_samples, daí o desvio.
    if cap < len(y):
        idx, _ = next(StratifiedShuffleSplit(1, train_size=cap,
                                             random_state=seed).split(X, y))
        X, y = X[idx], y[idx]
    Xtr, Xte, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
    sc = StandardScaler().fit(Xtr)
    return (sc.transform(Xtr), sc.transform(Xte),
            _convert_labels(ytr, "signed"), _convert_labels(yte, "signed"))


def summarise(recs: list[dict]) -> None:
    """Resumo por dataset: F1 de cada braço de λ₂ contra as referências publicadas."""
    w = 78 + 12 * len(L2S)
    print("\n" + "=" * w)
    head = (f"{'dataset':10s} {'n':>3s} | {'N=2k':>7s} {'N=5k λ2=0':>10s} |")
    for l2 in L2S:
        head += f" {'λ2=' + str(l2):>10s}"
    head += f" | {'melhor Δ':>9s} {'recupera?':>10s}"
    print(head); print("-" * w)
    for ds in CFG:
        arms = {l2: [x["f1_macro"] for x in recs
                     if x["dataset"] == ds and x["lambda2"] == l2] for l2 in L2S}
        if not all(arms.values()):
            continue
        n = min(len(v) for v in arms.values())
        base, alvo = PUB_5K[ds], PUB_2K[ds]
        row = f"{ds:10s} {n:>3d} | {alvo:>7.4f} {base:>10.4f} |"
        best = -9.0
        for l2 in L2S:
            m = float(np.mean(arms[l2])); best = max(best, m)
            row += f" {m:>10.4f}"
        # "recupera" = melhor braço alcança o nível de N=2000 (tolerância 0,01)
        rec = "SIM" if best >= alvo - 0.01 else ("parcial" if best >= base + 0.02 else "não")
        row += f" | {best - base:>+9.4f} {rec:>10s}"
        print(row)
    print("=" * w, flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(30)))
    p.add_argument("--datasets", nargs="+", default=list(CFG))
    p.add_argument("--output", type=Path,
                   default=ROOT / "results" / "ablationD_elasticnet.json")
    a = p.parse_args()
    logging.getLogger().setLevel(logging.ERROR)

    recs = json.loads(a.output.read_text()) if a.output.exists() else []
    done = {(r["dataset"], r["seed"], r["lambda2"]) for r in recs}
    total = len(a.datasets) * len(a.seeds) * len(L2S)
    t_start = time.perf_counter()

    for ds in a.datasets:
        sigma, tau, lam = CFG[ds]
        for seed in a.seeds:
            if all((ds, seed, l2) in done for l2 in L2S):
                continue
            Xtr, Xte, ytr, yte = prepare(ds, seed)
            out = {}
            for l2 in L2S:
                if (ds, seed, l2) in done:
                    prev = next(r for r in recs if (r["dataset"], r["seed"], r["lambda2"]) == (ds, seed, l2))
                    out[l2] = prev
                    continue
                m = ADMMNystromLSSVM(sigma=sigma, tau=tau, lambda_=lam, lambda2_=l2,
                                     m_ratio=0.30, landmark_method="colnorm",
                                     rho=None, max_iter=MAX_ITER, random_state=seed)
                t0 = time.perf_counter()
                m.fit(Xtr, ytr)
                rec = {"dataset": ds, "seed": seed, "lambda2": l2,
                       "sigma": sigma, "tau": tau, "lambda_": lam,
                       "n_train": int(len(ytr)), "n_iter": int(m.n_iter_),
                       "converged": bool(m.converged_),
                       "sparsity": float(m.sparsity_ratio_),
                       "f1_macro": float(f1_score(yte, m.predict(Xte), average="macro")),
                       "fit_time_s": time.perf_counter() - t0}
                recs.append(rec); out[l2] = rec
                a.output.write_text(json.dumps(recs, indent=1), encoding="utf-8")
            n_done = len(recs)
            parts = " | ".join(f"λ2={l2}: esp={out[l2]['sparsity']:.3f} F1={out[l2]['f1_macro']:.4f}"
                               for l2 in L2S)
            print(f"[{n_done:4d}/{total}] {ds:9s} s{seed:<2d} {parts}", flush=True)
        summarise(recs)
    summarise(recs)
    print(f"\n-> {a.output}   ({(time.perf_counter()-t_start)/60:.0f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

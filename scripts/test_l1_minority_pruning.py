#!/usr/bin/env python3
"""A penalidade ℓ1 poda preferencialmente a classe minoritária?

Testa a hipótese levantada como explicação do "colapso preditivo" das formulações
primais ℓ1 DENSAS em datasets desbalanceados (Tier 2).

MECANISMO PROPOSTO: sob desbalanceamento, o soft-threshold zeraria de forma
desproporcional os coeficientes ligados à classe minoritária (que contribuem menos
para o termo de mínimos quadrados), esvaziando sua representação no modelo.

DESENHO: para cada modelo e cada λ, mede-se a fração da classe minoritária ENTRE OS
COEFICIENTES SOBREVIVENTES (α ≠ 0) e compara-se com a proporção original no treino.
    - Se a fração cai muito abaixo da original -> ℓ1 poda a minoria (hipótese OK).
    - Se acompanha a original -> a poda é neutra quanto à classe (hipótese refutada).

CONTROLE: BANK (1:7,5, desbalanceado) vs HIGGS50K (1:1,1, ~balanceado). O efeito
deve aparecer no primeiro e não no segundo, se for causado pelo desbalanceamento.

COMPARAÇÃO: ADMM-Nesterov (DENSO, colapsa no Tier 2) vs ADMM-Nyström (NÃO colapsa),
para entender o que os separa.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels, make_splits
from src.models.lssvm.primal.admm_nesterov import ADMMNesterovLSSVM
from src.models.lssvm.primal.admm_nystrom import ADMMNystromLSSVM

# (sigma, tau) por dataset: escolhidos por varredura no LSSVM denso SEM ℓ1, de modo
# que o modelo de fato FUNCIONE antes de o ℓ1 entrar. Com sigma=0,5 o kernel degenera
# e o classificador já prediz tudo majoritário — o teste não mediria o efeito do ℓ1.
DATASETS = [
    ("BANK",     "desbalanceado 1:7,5", 2.0, 50.0),   # F1 sem ℓ1 = 0,683
    ("HIGGS50K", "balanceado 1:1,1",   10.0,  5.0),   # F1 sem ℓ1 = 0,633
]
LAMBDAS = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
SEED, NTRAIN = 0, 2000


def prep(ds):
    X, y, _ = DatasetLoader.load(ds)
    cap = round(NTRAIN / 0.70)
    if len(X) > cap:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=cap, random_state=SEED)
        idx, _ = next(sss.split(X, y)); X, y = X[idx], y[idx]
    Xtr, Xte, ytr, yte = make_splits(X, y, test_size=0.30, seed=SEED)
    ytr = _convert_labels(ytr, "signed"); yte = _convert_labels(yte, "signed")
    sc = StandardScaler(); return sc.fit_transform(Xtr), sc.transform(Xte), ytr, yte


def main():
    for ds, tag, SIGMA, TAU in DATASETS:
        Xtr, Xte, ytr, yte = prep(ds)
        # classe minoritária no treino
        vals, cnts = np.unique(ytr, return_counts=True)
        minor_cls = vals[np.argmin(cnts)]
        p_orig = float((ytr == minor_cls).mean())
        print(f"\n{'='*104}\n{ds}  ({tag})   n_train={len(ytr)}   "
              f"minoritária no treino = {100*p_orig:.1f}%   [σ={SIGMA}, τ={TAU}]\n{'='*104}")
        print(f"{'modelo':16s} {'λ':>6s} {'F1-macro':>9s} {'esparso':>8s} {'sobrev.':>8s} "
              f"{'%min entre sobrev.':>20s} {'%min PREDITA':>13s}")

        for name, cls, kw in [
            # ADMMNesterovLSSVM (denso) não expõe random_state — é determinístico.
            ("ADMM-Nesterov", ADMMNesterovLSSVM, dict(max_iter=500)),
            ("ADMM-Nyström",  ADMMNystromLSSVM,  dict(max_iter=500, m_ratio=0.30,
                                                      landmark_method="colnorm",
                                                      random_state=SEED)),
        ]:
            for lam in LAMBDAS:
                m = cls(sigma=SIGMA, tau=TAU, lambda_=lam, **kw)
                m.fit(Xtr, ytr)
                pred = m.predict(Xte)
                f1 = f1_score(yte, pred, average="macro", zero_division=0)
                # fração da minoritária que o modelo AINDA prediz: 0% = colapso preditivo
                p_pred = float((pred == minor_cls).mean())

                # coeficientes e as CLASSES às quais estão ligados
                if name.startswith("ADMM-Nes"):
                    coef = m.alpha_                    # α_i <-> amostra i
                    y_of_coef = ytr
                else:
                    coef = m.theta_                    # θ_j <-> landmark j
                    y_of_coef = ytr[m._landmark_idx_] if hasattr(m, "_landmark_idx_") \
                        else ytr[m.landmark_indices_] if hasattr(m, "landmark_indices_") else None
                    if y_of_coef is None:              # recupera pelos landmarks
                        lm = m.landmarks_ if hasattr(m, "landmarks_") else m._landmarks_
                        idx = [int(np.argmin(np.linalg.norm(Xtr - z, axis=1))) for z in lm]
                        y_of_coef = ytr[idx]

                nz = np.abs(coef) > 1e-8
                n_surv = int(nz.sum())
                spars = 1.0 - n_surv / len(ytr)        # esparsidade AMOSTRAL
                p_surv = float((y_of_coef[nz] == minor_cls).mean()) if n_surv else float("nan")
                delta = p_surv - p_orig
                flag = ""
                if n_surv and delta < -0.03:
                    flag = "  <-- MINORIA PODADA"
                if p_pred == 0.0:
                    flag += "  <-- COLAPSO PREDITIVO"
                print(f"{name:16s} {lam:6g} {f1:9.4f} {spars:8.3f} {n_surv:8d} "
                      f"{100*p_surv:8.1f}% (Δ={100*delta:+5.1f}pp) {100*p_pred:12.1f}%{flag}")


if __name__ == "__main__":
    main()

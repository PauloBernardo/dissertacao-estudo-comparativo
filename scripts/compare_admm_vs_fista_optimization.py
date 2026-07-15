#!/usr/bin/env python3
"""ADMM-Nyström vs FISTA-Nyström: são o mesmo problema de otimização?

Responde à crítica clássica: "se dois algoritmos resolvem o mesmo problema convexo,
deveriam chegar ao mesmo ótimo; se diferem, um convergiu prematuramente".

HIPÓTESE A TESTAR: eles NÃO resolvem o mesmo problema. Diferem no tratamento do
intercepto:

  ADMM  (fiel a Marinho et al., que omite o bias):
      min_θ  (1/2τ)‖Cθ − y‖²  +  λ‖θ‖₁          [dados CRUS; b post-hoc]

  FISTA (bias "profiled out" por centralização):
      min_θ  (1/2τ)‖Ĉθ − ŷ‖²  +  λ‖θ‖₁          [Ĉ = C − mean(C), ŷ = y − mean(y)]

Se a hipótese estiver certa, cada um será ÓTIMO NO SEU PRÓPRIO objetivo e pior no
do outro — provando que ambos convergem corretamente, mas para problemas distintos.
Se, ao contrário, um deles for pior nos DOIS objetivos, aí sim houve convergência
prematura.

Também reporta as métricas de otimização pedidas: nº de iterações, tempo total e
tempo por iteração.

Uso:
    python scripts/compare_admm_vs_fista_optimization.py [--dataset ADULT] [--seed 0]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels, make_splits
from src.models.lssvm.primal.admm_nystrom import ADMMNystromLSSVM
from src.models.lssvm.primal.fista_nystrom import FISTANystromLSSVM


def objectives(C: np.ndarray, y: np.ndarray, theta: np.ndarray,
               tau: float, lam: float) -> tuple[float, float]:
    """(objetivo CRU — o do ADMM, objetivo CENTRADO — o do FISTA)."""
    raw = 0.5 / tau * np.sum((C @ theta - y) ** 2) + lam * np.abs(theta).sum()
    Cc = C - C.mean(axis=0)
    yc = y - y.mean()
    cen = 0.5 / tau * np.sum((Cc @ theta - yc) ** 2) + lam * np.abs(theta).sum()
    return float(raw), float(cen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ADULT")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--lam", type=float, default=0.1)
    a = ap.parse_args()

    X, y, _ = DatasetLoader.load(a.dataset)
    if len(X) > 2857:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=2857, random_state=a.seed)
        idx, _ = next(sss.split(X, y))
        X, y = X[idx], y[idx]
    Xtr, _, ytr, _ = make_splits(X, y, test_size=0.30, seed=a.seed)
    ytr = _convert_labels(ytr, "signed").astype(float)
    Xtr = StandardScaler().fit_transform(Xtr)

    kw = dict(sigma=a.sigma, tau=a.tau, lambda_=a.lam,
              m_ratio=0.30, landmark_method="colnorm", random_state=a.seed)

    fits = {}
    for name, cls, extra in [("ADMM", ADMMNystromLSSVM, dict(max_iter=500)),
                             ("FISTA", FISTANystromLSSVM, dict(max_iter=5000))]:
        m = cls(**kw, **extra)
        t0 = time.perf_counter()
        m.fit(Xtr, ytr)
        dt = time.perf_counter() - t0
        fits[name] = (m, dt)

    # C comum: os dois usam colnorm com o mesmo random_state -> mesmos landmarks?
    Cs = {}
    for name, (m, _) in fits.items():
        lm = getattr(m, "_landmarks_", None)
        if lm is None:
            lm = getattr(m, "landmarks_", None)
        Cs[name] = m._rbf(Xtr, lm) if hasattr(m, "_rbf") else None

    same_lm = (Cs["ADMM"] is not None and Cs["FISTA"] is not None
               and np.allclose(Cs["ADMM"], Cs["FISTA"]))
    print(f"dataset={a.dataset} seed={a.seed} | sigma={a.sigma} tau={a.tau} lambda={a.lam}")
    print(f"n_train={len(Xtr)}  m={fits['ADMM'][0].theta_.size}  "
          f"landmarks idênticos entre os dois: {same_lm}")
    if not same_lm:
        print("  (usando o C do ADMM para avaliar ambos os objetivos)")
    C = Cs["ADMM"]

    print("\n" + "=" * 78)
    print(f"{'':7s} {'obj CRU (ADMM)':>16s} {'obj CENTRADO (FISTA)':>22s} "
          f"{'n_iter':>7s} {'tempo':>8s} {'ms/iter':>9s} {'esparso':>8s}")
    print("=" * 78)
    objs = {}
    for name in ("ADMM", "FISTA"):
        m, dt = fits[name]
        th = m.theta_
        raw, cen = objectives(C, ytr, th, a.tau, a.lam)
        objs[name] = (raw, cen)
        ni = getattr(m, "n_iter_", 0)
        sp = 100.0 * (np.abs(th) <= 1e-8).mean()
        print(f"{name:7s} {raw:16.4f} {cen:22.4f} {ni:7d} {dt:7.2f}s "
              f"{1000*dt/max(ni,1):8.2f} {sp:7.1f}%")
    print("=" * 78)

    dtheta = np.linalg.norm(fits["ADMM"][0].theta_ - fits["FISTA"][0].theta_)
    print(f"\n‖θ_ADMM − θ_FISTA‖₂ = {dtheta:.4f}")

    print("\nVEREDITO:")
    admm_wins_raw = objs["ADMM"][0] < objs["FISTA"][0]
    fista_wins_cen = objs["FISTA"][1] < objs["ADMM"][1]
    if admm_wins_raw and fista_wins_cen:
        print("  ✅ Cada um é ÓTIMO NO SEU PRÓPRIO objetivo e pior no do outro.")
        print("     -> Ambos convergem corretamente; resolvem PROBLEMAS DIFERENTES")
        print("        (o ADMM omite o intercepto; o FISTA o absorve por centralização).")
        print("     -> A diferença de F1 NÃO é convergência prematura, é de FORMULAÇÃO.")
    elif not admm_wins_raw:
        print("  ⚠️  O ADMM NÃO minimiza melhor o próprio objetivo -> possível convergência prematura.")
    elif not fista_wins_cen:
        print("  ⚠️  O FISTA NÃO minimiza melhor o próprio objetivo -> possível convergência prematura.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

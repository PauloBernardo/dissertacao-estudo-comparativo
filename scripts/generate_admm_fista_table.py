#!/usr/bin/env python3
"""Tabela ADMM-Nyström vs FISTA-Nyström no MESMO objetivo (formulação de 2026-10-14).

Os dois solvers minimizam (1/2τ)‖Ĉθ−ŷ‖² + ½θᵀWθ + λ‖θ‖₁ (Ĉ, ŷ centrados, bias perfilado),
com λ = lambda_rel·‖Ĉᵀŷ/τ‖∞. Roda, nos mesmos (σ, τ, lambda_rel) e landmarks:
  ADMM ρ fixo   (ρ = 1/λ_max(H), regra da referência) com 500 e 20 000 iterações
  ADMM ρ adapt. (balanceamento de resíduos, Boyd 2011 §3.4.1) com 500 iterações (o da grade)
  FISTA         com 5 000 iterações (o da grade) e 50 000 (referência do ótimo)
e avalia todas as soluções no mesmo objetivo. Config: moda do Tier 2 (σ=5, τ=0,01, f=0,001).
Saída: dissertacao-latex/tables/admm_fista_opt.tex
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels, make_splits
from src.models.lssvm.primal.admm_nystrom import ADMMNystromLSSVM
from src.models.lssvm.primal.fista_nystrom import FISTANystromLSSVM

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT.parent / "dissertacao-latex" / "tables" / "admm_fista_opt.tex"

CONFIGS = [("ADULT", 0), ("CREDIT", 0)]
SIGMA, TAU, LREL = 5.0, 0.01, 0.001
SOLVERS = [("ADMM, $\\rho$ fixo", ADMMNystromLSSVM, dict(rho=None, max_iter=500)),
           ("ADMM, $\\rho$ fixo", ADMMNystromLSSVM, dict(rho=None, max_iter=20000)),
           ("ADMM, $\\rho$ adaptativo", ADMMNystromLSSVM, dict(rho="adaptive", max_iter=500)),
           ("FISTA", FISTANystromLSSVM, dict(max_iter=5000)),
           ("FISTA", FISTANystromLSSVM, dict(max_iter=50000))]
REPS = 3  # repetições para o tempo (mediana)


def load(ds, seed):
    X, y, _ = DatasetLoader.load(ds)
    if len(X) > 2857:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=2857, random_state=seed)
        idx, _ = next(sss.split(X, y)); X, y = X[idx], y[idx]
    Xtr, Xte, ytr, yte = make_splits(X, y, test_size=0.30, seed=seed)
    sc = StandardScaler().fit(Xtr)
    return (sc.transform(Xtr), sc.transform(Xte),
            _convert_labels(ytr, "signed").astype(float), _convert_labels(yte, "signed").astype(float))


def objective(m, X, y, th):
    """Objetivo comum, com o mesmo λ efetivo nos dois solvers."""
    C = m._rbf(X, m.landmarks_); Cc, yc = C - C.mean(0), y - y.mean()
    W = m._rbf(m.landmarks_, m.landmarks_)
    lam = LREL * float(np.abs(Cc.T @ yc / TAU).max())
    return 0.5 / TAU * float(np.sum((Cc @ th - yc) ** 2)) + 0.5 * float(th @ W @ th) + lam * float(np.abs(th).sum())


def run(ds, seed):
    Xtr, Xte, ytr, yte = load(ds, seed)
    out = []
    for name, cls, extra in SOLVERS:
        ts = []
        for _ in range(REPS if extra["max_iter"] <= 5000 else 1):
            m = cls(sigma=SIGMA, tau=TAU, lambda_rel=LREL, kernel_reg=True, m_ratio=0.30,
                    landmark_method="random", random_state=seed, **extra)
            t = time.perf_counter(); m.fit(Xtr, ytr); ts.append(time.perf_counter() - t)
        out.append(dict(name=name, m=m, obj=objective(m, Xtr, ytr, m.theta_), it=int(m.n_iter_),
                        conv=bool(getattr(m, "converged_", True)), t=float(np.median(ts)),
                        sp=float(np.mean(m.theta_ == 0)),
                        f1=f1_score(yte, m.predict(Xte), average="macro")))
    lm = [o["m"].landmarks_ for o in out]
    assert all(np.array_equal(lm[0], l) for l in lm), "landmarks diferentes"
    best = min(o["obj"] for o in out)
    for o in out:
        o["gap"] = (o["obj"] - best) / abs(best)
    return out


def num(x, d=2):
    return f"{x:.{d}f}".replace(".", "{,}")


def sci(x):
    if x < 1e-9:
        return r"$<10^{-9}$"
    e = int(np.floor(np.log10(x))); b = x / 10 ** e
    return rf"${num(b, 1)}\times10^{{{e}}}$"


def main():
    L = [r"\begin{table}[H]", r"  \centering",
         r"  \caption[ADMM-Nyström \emph{vs.}\ FISTA-Nyström no mesmo objetivo]{ADMM-Nyström \emph{vs.}\ FISTA-Nyström "
         r"no \textbf{mesmo} objetivo (Equação~\ref{eq:admm_vs_fista}), com os mesmos $(\sigma,\tau,\lambda)$ e os mesmos "
         r"\textit{landmarks} ($\sigma=5$, $\tau=0{,}01$, $\lambda = 0{,}001\,\lambda_{\max}$, a configuração mais escolhida no Tier~2; "
         r"$N=2\,000$, $m=600$). \emph{Gap}: excesso relativo do objetivo sobre o melhor valor obtido. "
         r"Com o $\rho$ fixo da referência, o ADMM não converge nem em $20\,000$ iterações; com o $\rho$ adaptativo, "
         r"chega ao ótimo. O FISTA, com o teto da grade, para perto do ótimo, mas com menos coeficientes zerados; com $50\,000$ iterações, chega ao mesmo ponto. Tempo: mediana de 3 execuções.}",
         r"  \label{tab:admm_fista_opt}", r"  \resizebox{\textwidth}{!}{%",
         r"  \begin{tabular}{llrrrrrr}", r"    \toprule",
         r"    \emph{Dataset} & Solver & Iterações & \emph{Gap} & Esparsidade de $\boldsymbol{\theta}$ & $F_1$ (teste) & Tempo (s) & ms/iter \\",
         r"    \midrule"]
    for j, (ds, seed) in enumerate(CONFIGS):
        res = run(ds, seed)
        for i, o in enumerate(res):
            first = (r"\multirow{5}{*}{" + ds + "}") if i == 0 else ""
            it = f"{o['it']:,}".replace(",", r"\,") + ("" if o["conv"] else r"$^{\dagger}$")
            L.append(f"    {first} & {o['name']} & {it} & {sci(o['gap'])} & {num(100 * o['sp'], 1)}\\% & "
                     f"{num(o['f1'], 3)} & {num(o['t'], 3)} & {num(1000 * o['t'] / max(o['it'], 1))} \\\\")
            print(f"{ds:7s} {o['name']:28s} it={o['it']:6d} conv={o['conv']} gap={o['gap']:.2e} "
                  f"sp={o['sp']:.3f} f1={o['f1']:.4f} t={o['t']:.3f}s")
        if j < len(CONFIGS) - 1:
            L.append(r"    \midrule")
    L += [r"    \bottomrule", r"  \end{tabular}}",
          r"  \par\smallskip{\footnotesize $^{\dagger}$Atingiu o teto de iterações sem satisfazer a tolerância de $10^{-6}$ nos resíduos.}",
          r"\end{table}", ""]
    TAB.write_text("\n".join(L))
    print("escrito", TAB)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Gera a tabela LaTeX da comparação de otimização ADMM-Nyström vs FISTA-Nyström.

Roda os dois solvers nos MESMOS (sigma, tau, lambda) e nos mesmos landmarks, e
avalia AMBAS as soluções nos DOIS objetivos (cru e centrado). Emite
app_admm_fista_opt.tex.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.data.loaders import DatasetLoader
from src.data.preprocessing import _convert_labels, make_splits
from src.models.lssvm.primal.admm_nystrom import ADMMNystromLSSVM
from src.models.lssvm.primal.fista_nystrom import FISTANystromLSSVM

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT.parent / "dissertacao-latex" / "tables" / "admm_fista_opt.tex"

CONFIGS = [("ADULT", 0, 0.5, 5.0, 0.1), ("CREDIT", 0, 0.5, 5.0, 0.1)]
REPS = 5  # repetições para o tempo (mediana)


def objectives(C, y, th, tau, lam):
    raw = 0.5 / tau * np.sum((C @ th - y) ** 2) + lam * np.abs(th).sum()
    Cc, yc = C - C.mean(axis=0), y - y.mean()
    cen = 0.5 / tau * np.sum((Cc @ th - yc) ** 2) + lam * np.abs(th).sum()
    return float(raw), float(cen)


def run(ds, seed, sigma, tau, lam):
    X, y, _ = DatasetLoader.load(ds)
    if len(X) > 2857:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=2857, random_state=seed)
        idx, _ = next(sss.split(X, y)); X, y = X[idx], y[idx]
    Xtr, _, ytr, _ = make_splits(X, y, test_size=0.30, seed=seed)
    ytr = _convert_labels(ytr, "signed").astype(float)
    Xtr = StandardScaler().fit_transform(Xtr)
    kw = dict(sigma=sigma, tau=tau, lambda_=lam, m_ratio=0.30,
              landmark_method="colnorm", random_state=seed)
    out = {}
    for name, cls, extra in [("ADMM", ADMMNystromLSSVM, dict(max_iter=500)),
                             ("FISTA", FISTANystromLSSVM, dict(max_iter=5000))]:
        ts = []
        for _ in range(REPS):
            m = cls(**kw, **extra)
            t = time.perf_counter(); m.fit(Xtr, ytr); ts.append(time.perf_counter() - t)
        dt = float(np.median(ts))
        C = m._rbf(Xtr, m._landmarks_ if hasattr(m, "_landmarks_") else m.landmarks_)
        raw, cen = objectives(C, ytr, m.theta_, tau, lam)
        out[name] = dict(raw=raw, cen=cen, it=int(m.n_iter_), t=dt,
                         ms=1000 * dt / max(m.n_iter_, 1))
    return out


def num(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def main():
    lines = [
        r"\begin{table}[ht]", r"  \centering",
        r"  \caption[ADMM-Nyström \emph{vs.}\ FISTA-Nyström sob os mesmos hiperparâmetros]{ADMM-Nyström \emph{vs.}\ FISTA-Nyström: os dois solvers avaliados "
        r"nos \textbf{mesmos} $(\sigma,\tau,\lambda)$ e sobre os \textbf{mesmos} "
        r"\textit{landmarks}. Cada solução é avaliada nos \emph{dois} objetivos --- o "
        r"\textbf{cru} (sem intercepto, o do ADMM) e o \textbf{centrado} (intercepto "
        r"absorvido, o do FISTA). Cada algoritmo minimiza melhor o \emph{seu próprio} "
        r"objetivo (em negrito) e pior o do outro: ambos convergem corretamente, para "
        r"ótimos de \emph{problemas distintos}. Tempo: mediana de 5 execuções.}",
        r"  \label{tab:admm_fista_opt}",
        r"  \begin{tabular}{llrrrrr}", r"    \toprule",
        r"    \emph{Dataset} & Solver & Obj.\ cru & Obj.\ centrado & Iterações & "
        r"Tempo (s) & ms/iter \\", r"    \midrule",
    ]
    for ds, seed, sg, tau, lam in CONFIGS:
        r = run(ds, seed, sg, tau, lam)
        for i, s in enumerate(("ADMM", "FISTA")):
            v = r[s]
            raw = rf"\textbf{{{num(v['raw'])}}}" if s == "ADMM" else num(v["raw"])
            cen = rf"\textbf{{{num(v['cen'])}}}" if s == "FISTA" else num(v["cen"])
            first = (r"\multirow{2}{*}{" + ds + "}") if i == 0 else ""
            lines.append(f"    {first} & \\texttt{{{s}}} & {raw} & {cen} & {v['it']} & "
                         f"{num(v['t'], 3)} & {num(v['ms'])} \\\\")
        lines.append(r"    \midrule" if ds != CONFIGS[-1][0] else "")
    lines = [l for l in lines if l != ""]
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    TAB.write_text("\n".join(lines))
    print("escrito", TAB.name)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

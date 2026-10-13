#!/usr/bin/env python3
"""Protótipo: Nyström-LSSVM com l1 E o regularizador do LSSVM (2026-10-06).

O ADMM-/FISTA-Nyström minimizam (1/2tau)||C theta - y||^2 + lambda ||theta||_1: o termo l2 do
LSSVM (norma do kernel, 1/2 theta' W theta, W = K(Z, Z)) foi SUBSTITUÍDO pelo l1, e por isso só
o produto lambda*tau importa. Aqui ele é restaurado:

    min_theta  (1/2tau) ||Ĉ theta - ŷ||^2 + 1/2 theta' W theta + lambda ||theta||_1

(Ĉ, ŷ centrados -> bias b = ȳ - c̄' theta, como no FISTA-Nyström). Com lambda = 0 é exatamente o
Nyström-SVM (LSSVM reduzido) -- serve de checagem de sanidade.

Pergunta: na MESMA esparsidade, podar landmarks com l1 + norma do kernel bate usar menos
landmarks aleatórios? Para cada (dataset, semente):
  - sigma e gamma (tau = 1/gamma) do Nyström-SVM com m = 0,3 escolhido por CV
    (results/study_l1_grid_tier1.json, mesma semente);
  - landmarks aleatórios ANINHADOS (prefixos de uma permutação fixa por semente);
  - "menos landmarks": lambda = 0, m em {0,30; 0,20; 0,15; 0,10; 0,05};
  - "poda l1 + RKHS": m = 0,30, lambda = f * lambda_max, f em F_PATH (caminho com warm start);
  - F1-macro de TESTE e esparsidade = 1 - nnz(theta)/N (denominador da tese).
Solver: FISTA com restart de O'Donoghue & Candès e parada relativa no objetivo.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data.loaders import DatasetLoader  # noqa: E402
from src.data.preprocessing import _convert_labels  # noqa: E402

DATASETS = ["BCW", "PID", "HAB", "VCP", "GCR", "AUS", "AI4I", "TWS", "TWM", "TWC"]
M_FEW = [0.30, 0.20, 0.15, 0.10, 0.05]
F_PATH = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
M_L1 = 0.30


def rbf(A, B, sigma):
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-np.maximum(d, 0) / (2 * sigma ** 2))


def fista(H, g, lam, theta0, max_iter=50_000, rtol=1e-10):
    """min 1/2 theta'H theta - g'theta + lam ||theta||_1 (H psd)."""
    L = float(np.linalg.eigvalsh(H).max())
    step, thr = 1.0 / L, lam / L
    obj = lambda t: 0.5 * t @ H @ t - g @ t + lam * np.abs(t).sum()
    x = theta0.copy(); y = x.copy(); t = 1.0; f_old = obj(x)
    for k in range(max_iter):
        v = y - step * (H @ y - g)
        xn = np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        mom = (t - 1) / tn
        if np.dot(y - xn, xn - x) > 0:      # O'Donoghue & Candès (2015), eq. (12)
            tn, mom = 1.0, 0.0
        y = xn + mom * (xn - x); x = xn; t = tn
        if k % 50 == 49:
            f = obj(x)
            if abs(f_old - f) <= rtol * max(1.0, abs(f)):
                return x, k + 1, True
            f_old = f
    return x, max_iter, False


def admm(H, g, lam, theta0, max_iter=20_000, rtol=1e-8):
    """Mesmo problema do fista(), por ADMM com fatoração única de (H + rho I) -- robusto a mau
    condicionamento. rho = média geométrica dos extremos do espectro de H; parada relativa nos
    resíduos primal e dual (Boyd et al., 2011, sec. 3.3)."""
    import scipy.linalg as sl
    ev = np.linalg.eigvalsh(H); rho = float(np.sqrt(max(ev.max(), 1e-12) * max(ev.min(), 1e-12 * ev.max())))
    F = sl.cho_factor(H + rho * np.eye(len(g)))
    z = theta0.copy(); u = np.zeros_like(z)
    for k in range(max_iter):
        x = sl.cho_solve(F, g + rho * (z - u))
        z_old = z
        v = x + u; z = np.sign(v) * np.maximum(np.abs(v) - lam / rho, 0.0)
        u = u + x - z
        r = np.linalg.norm(x - z); s_ = rho * np.linalg.norm(z - z_old)
        if r <= rtol * max(np.linalg.norm(x), np.linalg.norm(z), 1e-12) and s_ <= rtol * max(rho * np.linalg.norm(u), 1e-12):
            return z, k + 1, True
    return z, max_iter, False


SOLVER = {"fista": fista, "admm": admm}


def run_unit(ds, seed, sigma, gamma, n_train=None, m_few=None, m_l1=None, solver="fista"):
    logging.disable(logging.WARNING)
    X, yraw, _ = DatasetLoader.load(ds)
    y = _convert_labels(yraw, "signed").astype(float)
    if n_train is not None:   # mesma subamostragem do Tier 2
        from run_tier2_gridcv import _subsample
        X, y = _subsample(X, y, round(n_train / 0.70), seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    N = len(ytr); tau = 1.0 / gamma
    perm = np.random.default_rng(seed).permutation(N)
    out = []

    def fit_eval(m_ratio, lam_fracs):
        m = max(2, int(round(m_ratio * N)))
        Z = Xtr[perm[:m]]
        C = rbf(Xtr, Z, sigma); W = rbf(Z, Z, sigma) + 1e-8 * np.eye(m)
        cm, ym = C.mean(0), ytr.mean()
        Ch, yh = C - cm, ytr - ym
        H = Ch.T @ Ch / tau + W; g = Ch.T @ yh / tau
        lam_max = float(np.abs(g).max())
        theta = np.linalg.solve(H, g)        # lambda = 0: solução exata
        res = []
        for f in lam_fracs:
            if f > 0:
                theta, it, conv = SOLVER[solver](H, g, f * lam_max, theta)
            else:
                it, conv = 0, True
            b = ym - cm @ theta
            yp = np.sign(rbf(Xte, Z, sigma) @ theta + b); yp[yp == 0] = 1
            nnz = int((np.abs(theta) > 1e-8).sum())
            res.append(dict(m_ratio=m_ratio, lam_frac=f, test_f1=float(f1_score((yte + 1) // 2, (yp + 1) // 2,
                            average="macro", zero_division=0)), sparsity=1 - nnz / N, nnz=nnz, iters=it, converged=conv))
        return res

    for m in (m_few or M_FEW):
        out += [dict(arm="menos_landmarks", **r) for r in fit_eval(m, [0.0])]
    out += [dict(arm="l1_rkhs", **r) for r in fit_eval(m_l1 or M_L1, F_PATH)]
    return dict(dataset=ds, seed=seed, sigma=sigma, gamma=gamma, n_train=N, points=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--output", type=Path, default=Path("results/study_l1_rkhs_nystrom_tier1.json"))
    ap.add_argument("--solver", choices=["fista", "admm"], default="fista")
    ap.add_argument("--datasets", nargs="+", default=None, help="só com --n10k")
    ap.add_argument("--n10k", action="store_true",
                    help="ADULT e HIGGS50K com N_treino = 10 000; l1 parte de m = 0,1; sigma/gamma modais do Tier 2")
    a = ap.parse_args()
    if a.n10k:
        modal = {"ADULT": (5.0, 50.0), "HIGGS50K": (5.0, 1.0)}   # Nyström-SVM, results/tier2_gridcv.json
        units = [(ds, sd, *modal[ds]) for ds in (a.datasets or modal) for sd in a.seeds]
        res = Parallel(n_jobs=a.n_jobs)(delayed(run_unit)(*u, n_train=10_000,
                     m_few=[0.10, 0.07, 0.05, 0.03, 0.02, 0.01], m_l1=0.10, solver=a.solver) for u in units)
        a.output.write_text(json.dumps(res)); print(f"{len(res)} unidades -> {a.output}"); return
    grid = json.load(open("results/study_l1_grid_tier1.json"))
    units = []
    for r in grid:
        if r["variant"] != "NystromLSSVMRandom" or r["seed"] not in a.seeds:
            continue
        c = max((c for c in r["grid"] if abs(c["params"]["m_ratio"] - 0.3) < 1e-9), key=lambda c: c["cv_f1"])
        units.append((r["dataset"], r["seed"], c["params"]["sigma"], c["params"]["gamma"]))
    res = Parallel(n_jobs=a.n_jobs)(delayed(run_unit)(*u) for u in units)
    a.output.write_text(json.dumps(res))
    print(f"{len(res)} unidades -> {a.output}")


if __name__ == "__main__":
    main()

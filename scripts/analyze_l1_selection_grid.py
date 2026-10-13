#!/usr/bin/env python3
"""Analisa results/study_l1_grid_tier1.json (ver scripts/study_l1_selection_grid.py).

1. Censura da grade em lambda: com lambda estendido, com que frequência o melhor F1 de CV cai
   abaixo da grade publicada, e quanto F1 isso acrescenta.
2. Combinações lentas x ruins: fração do tempo de ajuste gasta em combinações "ruins"
   (F1 de CV mais de 0,02 abaixo do melhor da mesma semente) e correlação de Spearman
   tempo x F1 dentro de cada (dataset, semente).
3. Esparsidade equiparada: melhor F1 de CV entre as combinações com esparsidade >= s, por
   modelo, e em quantos (dataset, semente) o modelo alcança esse nível.
Tudo dentro da grade publicada, salvo onde indicado. F1 é o de validação cruzada (o mesmo que a
seleção usa); nada do conjunto de teste entra aqui.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "results/study_l1_grid_tier1.json")
BAD = 0.02
LEVELS = [0.5, 0.7, 0.8, 0.9]
LABEL = {"DualFISTA": "DualFISTA", "FISTANesterov": "FISTA-Nesterov", "ADMMNesterovLSSVM": "ADMM-Nesterov",
         "ADMMNystromRandom": "ADMM-Nyström", "FISTANystromRandom": "FISTA-Nyström",
         "PCPLSSVm": "LSSVM-PCP", "NystromLSSVMRandom": "Nyström-SVM"}

recs = json.loads(PATH.read_text())
by = defaultdict(list)
for r in recs:
    by[r["variant"]].append(r)
order = [v for v in LABEL if v in by]

print("\n== 1. Censura em lambda (melhor da grade ESTENDIDA vs melhor da publicada) ==")
for v in order:
    if "lambda_" not in by[v][0]["grid"][0]["params"]:
        continue
    pub_min = min(c["params"]["lambda_"] for c in by[v][0]["grid"] if c["in_published_grid"])
    below, gains = 0, []
    for r in by[v]:
        best_ext = max(r["grid"], key=lambda c: c["cv_f1"])
        best_pub = max((c for c in r["grid"] if c["in_published_grid"]), key=lambda c: c["cv_f1"])
        below += best_ext["params"]["lambda_"] < pub_min
        gains.append(best_ext["cv_f1"] - best_pub["cv_f1"])
    print(f"  {LABEL[v]:15s} melhor abaixo da grade em {below}/{len(by[v])}  "
          f"ganho de F1 CV: médio {np.mean(gains):+.4f}, máximo {np.max(gains):+.4f}")

print(f"\n== 2. Tempo gasto em combinações ruins (F1 CV < melhor − {BAD}), grade publicada ==")
for v in order:
    frac, rhos, nbad = [], [], []
    for r in by[v]:
        g = [c for c in r["grid"] if c["in_published_grid"]]
        best = max(c["cv_f1"] for c in g)
        t = np.array([c["fit_time_s"] for c in g]); f = np.array([c["cv_f1"] for c in g])
        bad = f < best - BAD
        frac.append(t[bad].sum() / t.sum()); nbad.append(bad.mean())
        if np.ptp(t) > 0 and np.ptp(f) > 0:
            rhos.append(spearmanr(t, f).statistic)
    print(f"  {LABEL[v]:15s} combinações ruins: {np.mean(nbad):.0%}  consomem {np.mean(frac):.0%} do tempo  "
          f"| Spearman tempo×F1 mediano {np.median(rhos):+.2f}")

print("\n== 3. Melhor F1 de CV com esparsidade >= s (grade publicada; [alcança s em k/n]) ==")
print("  " + f"{'modelo':15s}" + "".join(f"{'s≥'+str(s):>20s}" for s in LEVELS) + f"{'sem restrição':>18s}")
for v in order:
    cells = []
    for s in LEVELS:
        vals = []
        for r in by[v]:
            ok = [c["cv_f1"] for c in r["grid"] if c["in_published_grid"] and c["sparsity"] >= s - 1e-9]
            if ok:
                vals.append(max(ok))
        cells.append(f"{np.mean(vals):.4f} [{len(vals)}/{len(by[v])}]" if vals else f"— [0/{len(by[v])}]")
    free = np.mean([max(c["cv_f1"] for c in r["grid"] if c["in_published_grid"]) for r in by[v]])
    print("  " + f"{LABEL[v]:15s}" + "".join(f"{c:>20s}" for c in cells) + f"{free:>18.4f}")
print("\n  Média sobre os (dataset, semente) em que o nível é alcançável; comparar só colunas com k/n completo.")

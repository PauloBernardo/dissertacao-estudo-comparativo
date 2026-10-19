#!/usr/bin/env python3
"""Intervalos de confiança das diferenças pareadas de F1-macro (Tiers 1 e 2).

Para cada comparação A − B, a diferença é pareada por (base, semente) — mesmos splits.
O IC de 95% é por bootstrap em dois níveis (10 000 réplicas): reamostram-se as bases
com reposição e, dentro de cada base, as sementes. Assim a incerteza entre bases, que
domina, não é subestimada tratando as 300 (ou 180) células como independentes.

    .venv/bin/python scripts/paired_ci.py [--out ../dissertacao-latex/tables/paired_ci.tex]
"""
from __future__ import annotations

import argparse, collections, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NAMES = {"NystromLSSVMRandom": "Nyström-SVM", "StandardLSSVM": "LSSVM-Std", "ADMMNystromRandom": "ADMM-Nyström",
         "FISTANystromRandom": "FISTA-Nyström", "FTTransformerCURRandom": "FT-CUR", "SAINTColnorm": "SAINT",
         "FTTransformer_softmax": "FT-Softmax", "FTTransformer_topk": "FT-TopK",
         "FTTransformer_sparsemax": "FT-Sparsemax", "XGBoost": "XGBoost"}
COMPARISONS = [("NystromLSSVMRandom", "StandardLSSVM"), ("ADMMNystromRandom", "NystromLSSVMRandom"),
               ("FISTANystromRandom", "NystromLSSVMRandom"), ("FTTransformerCURRandom", "SAINTColnorm"),
               ("FTTransformer_sparsemax", "FTTransformer_softmax"), ("FTTransformer_topk", "FTTransformer_softmax"),
               ("XGBoost", "NystromLSSVMRandom")]
TIERS = {"Cenário~1": ["results/tier1_gridcv.json"],
         "Cenário~2": ["results/tier2_gridcv.json", "results/tier2_transformers.json"]}


def load(files):
    F = collections.defaultdict(dict)
    for f in files:
        for r in json.load(open(ROOT / f)):
            if r.get("status", "ok") == "ok":
                F[r["variant"]][(r["dataset"], r["seed"])] = r["test_f1_macro"]
    return F


def ci(F, a, b, rng, B=10_000):
    keys = sorted(set(F[a]) & set(F[b]))
    by_ds = collections.defaultdict(list)
    for d, s in keys:
        by_ds[d].append(F[a][(d, s)] - F[b][(d, s)])
    ds = sorted(by_ds); D = [np.array(by_ds[d]) for d in ds]
    point = float(np.mean([x.mean() for x in D]))
    boot = np.empty(B)
    for i in range(B):
        pick = rng.integers(0, len(D), len(D))
        boot[i] = np.mean([D[j][rng.integers(0, len(D[j]), len(D[j]))].mean() for j in pick])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi), len(ds), len(keys)


def fmt(x):
    return ("+" if x >= 0 else "-") + f"{abs(x):.3f}".replace(".", "{,}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT.parent / "dissertacao-latex" / "tables" / "paired_ci.tex"))
    a = p.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for tier, files in TIERS.items():
        F = load(files)
        for x, y in COMPARISONS:
            if x in F and y in F:
                pt, lo, hi, nd, nk = ci(F, x, y, rng)
                rows.append((tier, f"{NAMES[x]} $-$ {NAMES[y]}", pt, lo, hi, nd, nk))
                print(f"{tier:7s} {NAMES[x]:>14s} - {NAMES[y]:<12s} {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]  ({nd} bases, {nk} pares)")
    body, last = [], None
    for tier, lab, pt, lo, hi, nd, nk in rows:
        if last and tier != last:
            body.append(r"    \midrule")
        last = tier
        body.append(f"    {tier} & {lab} & {fmt(pt)} & [{fmt(lo)}; {fmt(hi)}] \\\\")
    Path(a.out).write_text("\n".join([
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption[Diferenças pareadas de F1-macro com IC de 95\%]{Diferenças pareadas de F1-macro (A $-$ B), pareadas "
        r"por base e semente, com intervalo de confiança de 95\% por \textit{bootstrap} em dois níveis (bases e, dentro "
        r"delas, sementes; $10\,000$ réplicas).}",
        r"  \label{tab:paired_ci}",
        r"  \small",
        r"  \begin{tabular}{llrc}",
        r"    \toprule",
        r"    & Comparação (A $-$ B) & $\Delta$ médio & IC de 95\% \\",
        r"    \midrule",
        *body,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \legend{Fonte: \texttt{results/tier1\_gridcv.json}, \texttt{results/tier2\_gridcv.json} e "
        r"\texttt{results/tier2\_transformers.json}; gerada por \texttt{scripts/paired\_ci.py}.}",
        r"\end{table}", ""]))
    print("ok:", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

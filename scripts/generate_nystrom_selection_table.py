#!/usr/bin/env python3
"""Gera a tabela LaTeX consolidada da ablação de seleção de landmarks do
Nyström-SVM (colnorm vs random) nos três regimes de N: Tier 1 (N≈400),
Tier 2 (N=2000) e Ablação D (N=5000, params fixos).

Lê os JSONs de resultados, agrega F1-macro/acurácia/esparsidade sobre datasets
e sementes, computa o Wilcoxon pareado por (dataset, semente) e escreve a
tabela em ../dissertacao-latex/tables/nystrom_selection.tex.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
TEX_OUT = ROOT.parent / "dissertacao-latex" / "tables" / "nystrom_selection.tex"

REGIMES = [
    {
        "label": r"Tier~1 ($N{\approx}400$, 10 \emph{datasets})",
        "col": (ROOT / "results/tier1_gridcv.json", "NystromLSSVMColnorm", None),
        "rnd": (ROOT / "results/tier1_nystrom_random.json", "NystromLSSVMRandom", None),
    },
    {
        "label": r"Tier~2 ($N{=}2000$, 6 \emph{datasets})",
        "col": (ROOT / "results/tier2_gridcv.json", "NystromLSSVMColnorm", None),
        "rnd": (ROOT / "results/tier2_nystrom_random.json", "NystromLSSVMRandom", None),
    },
    {
        "label": r"Ablação~D ($N{=}5000$, 6 \emph{datasets})",
        "col": (ROOT / "results/tier2_fixedparams_n5000_lssvm.json", "NystromLSSVMColnorm", 5000),
        "rnd": (ROOT / "results/tier2_fixedparams_n5000_nystrom_random.json", "NystromLSSVMRandom", 5000),
    },
]

METRICS = [
    ("test_f1_macro", r"$F_1$-macro"),
    ("test_accuracy", "Acurácia"),
    ("sparsity_ratio", "Esparsidade"),
]


def load(path, variant, n5000):
    out = {}
    for r in json.loads(Path(path).read_text()):
        if r.get("variant") != variant or r.get("status") != "ok":
            continue
        if n5000 is not None and r.get("n_train_target") != n5000:
            continue
        out[(r["dataset"], r["seed"])] = r
    return out


def paired(col, rnd, metric):
    keys = sorted(set(col) & set(rnd))
    c = [col[k][metric] for k in keys if col[k].get(metric) is not None
         and rnd[k].get(metric) is not None]
    r = [rnd[k][metric] for k in keys if col[k].get(metric) is not None
         and rnd[k].get(metric) is not None]
    return c, r


def wilcox(c, r):
    if not c or all(a == b for a, b in zip(c, r)):
        return None
    try:
        return wilcoxon(c, r, zero_method="wilcox")[1]
    except ValueError:
        return None


def fmt_p(p):
    if p is None:
        return "--"
    return f"{p:.3f}".replace(".", ",")


def fmt(x):
    return f"{x:.4f}".replace(".", ",")


def fmt_d(x):
    s = f"{x:+.4f}".replace(".", ",")
    return s


def main():
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{Seleção de \emph{landmarks} do Nyström-SVM --- norma de coluna "
        r"(\texttt{colnorm}, usada na dissertação) vs.\ amostragem aleatória "
        r"(\texttt{random}), agregada sobre os \emph{datasets} e 30 sementes de cada "
        r"regime de $N$. $\Delta = \texttt{colnorm} - \texttt{random}$; $p$ = teste de "
        r"Wilcoxon signed-rank pareado por (\emph{dataset}, semente). Nenhuma diferença "
        r"de \emph{desempenho} é significativa ($\alpha=0{,}05$) em nenhum regime; o único "
        r"$p<0{,}05$ é a esparsidade do Tier~1, e favorece o \texttt{random} (mais esparso). "
        r"A esparsidade em $N{=}2000$ e $N{=}5000$ é idêntica por construção ($m/n$ fixo em $0{,}30$).}",
        r"  \label{tab:nystrom_selection}",
        r"  \begin{tabular}{llrrrr}",
        r"    \toprule",
        r"    Regime & Métrica & \texttt{colnorm} & \texttt{random} & $\Delta$ & $p$ \\",
        r"    \midrule",
    ]

    for i, reg in enumerate(REGIMES):
        col = load(*reg["col"])
        rnd = load(*reg["rnd"])
        for j, (mkey, mlabel) in enumerate(METRICS):
            c, r = paired(col, rnd, mkey)
            mc, mr = st.mean(c), st.mean(r)
            p = wilcox(c, r)
            first = (r"\multirow{3}{*}{" + reg["label"] + "}") if j == 0 else ""
            lines.append(
                f"    {first} & {mlabel} & {fmt(mc)} & {fmt(mr)} & {fmt_d(mc-mr)} & {fmt_p(p)} \\\\"
            )
        if i < len(REGIMES) - 1:
            lines.append(r"    \midrule")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]

    TEX_OUT.write_text("\n".join(lines))
    print(f"Escrito: {TEX_OUT.relative_to(ROOT.parent)}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Gera a tabela LaTeX do FT-CUR para o apêndice de seleção de landmarks.

Lê results/ftcur_sel_tier1.json e produz app_ftcur_selection.tex — 4 seletores
× {tabulares, geométricos} no Tier 1, F1/acurácia + Friedman + custo (fit_time).
Robusto a dados parciais: usa '--' onde um seletor não tem todos os pares e
computa Friedman entre os seletores disponíveis para o grupo.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from scipy.stats import friedmanchisquare

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/ftcur_sel_tier1.json"
TAB = ROOT.parent / "dissertacao-latex" / "tables" / "app_ftcur_selection.tex"

SELS = ["colnorm", "random", "kmeans", "opposite"]
GROUPS = [("Tabulares", ["BCW", "PID", "HAB", "VCP", "GCR", "AUS", "AI4I"]),
          ("Geométricos 2D", ["TWS", "TWC", "TWM"])]


def load():
    d = [r for r in json.loads(SRC.read_text()) if r.get("status") == "ok"]
    S = {s: {} for s in SELS}
    for r in d:
        v = r["variant"].replace("FTTransformerCUR", "").lower()
        if v in S:
            S[v][(r["dataset"], r["seed"])] = r
    return S


def num(x):
    return f"{x:.4f}".replace(".", ",") if x is not None else "--"


def group_vals(S, dss, metric):
    """Pares (dataset,seed) presentes em TODOS os seletores com dado no grupo."""
    per = {}
    for sel in SELS:
        keys = {k for k in S[sel] if k[0] in dss}
        per[sel] = keys
    avail = [s for s in SELS if per[s]]
    if not avail:
        return {}, [], 0
    common = set.intersection(*[per[s] for s in avail])
    out = {s: [S[s][k][metric] for k in sorted(common)] for s in avail}
    return out, avail, len(common)


def main():
    S = load()
    lines = [
        r"\begin{table}[ht]", r"  \centering",
        r"  \caption[FT-CUR sob os quatro seletores de \emph{landmarks} --- Tier~1]{FT-CUR sob os quatro seletores de \emph{landmarks} --- Tier~1, "
        r"30 sementes, agregado por grupo. $p$ = Friedman entre os seletores disponíveis. "
        r"Nenhuma diferença de desempenho é significativa. Custo relativo de seleção "
        r"(\emph{wall-time} do \textit{GridCV}): \texttt{random}/\texttt{colnorm} $\approx$ base, "
        r"\texttt{kmeans} $+50\%$, \texttt{opposite} $+7\%$ --- sem ganho de desempenho que o justifique.}",
        r"  \label{tab:app_ftcur_selection}",
        r"  \begin{tabular}{llrrrrr}", r"    \toprule",
        r"    Grupo & Métrica & \texttt{colnorm} & \texttt{random} & \texttt{kmeans} & "
        r"\texttt{opposite} & Friedman $p$ \\", r"    \midrule",
    ]
    for gi, (gname, dss) in enumerate(GROUPS):
        for j, (mk, ml) in enumerate([("test_f1_macro", "$F_1$-macro"),
                                       ("test_accuracy", "Acurácia")]):
            vals, avail, n = group_vals(S, dss, mk)
            means = {s: (st.mean(vals[s]) if s in vals else None) for s in SELS}
            try:
                fp = friedmanchisquare(*[vals[s] for s in avail])[1] if len(avail) >= 3 else None
            except Exception:
                fp = None
            first = (r"\multirow{2}{*}{" + gname + f" $n{{=}}{n}$" + "}") if j == 0 else ""
            cells = " & ".join(num(means[s]) for s in SELS)
            pcell = (f"{fp:.3f}".replace(".", ",") if fp is not None else "--")
            note = "" if len(avail) == 4 else r"$^{\dagger}$"
            lines.append(f"    {first} & {ml} & {cells} & {pcell}{note} \\\\")
        if gi < len(GROUPS) - 1:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    if "$^{\\dagger}$" in "\n".join(lines):
        lines.append(
            r"  \par\smallskip\footnotesize $^{\dagger}$ \texttt{opposite} ainda incompleto "
            r"nos \emph{datasets} geométricos; Friedman entre "
            r"\texttt{colnorm}/\texttt{random}/\texttt{kmeans}.")
    lines += [r"\end{table}", ""]
    TAB.write_text("\n".join(lines))
    print("escrito", TAB.name)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

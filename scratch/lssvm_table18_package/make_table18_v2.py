#!/usr/bin/env python3
"""Tabela 18, versão 2026-10-15: por N, F1-macro de teste, pontos guardados para a predição,
tempo e pico de RAM de treino e tempo de predição. A coluna de RAM de predição saiu: o
aumento de RSS na predição fica abaixo da resolução da medida (marcava 0,0); o custo de
predição aparece pelos pontos guardados (o kernel de teste é n_teste × pontos).

Uso:
    python make_table18_v2.py --input table18_v2.json --output tabela18.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {
    "StandardLSSVM":             "LSSVM padrão",
    "ADMMNesterovLSSVM":         "ADMM-Nesterov",
    "NystromLSSVMColnorm":       "Nyström-SVM",
    "ADMMNystromLSSVM":          "ADMM-Nyström",
    "FISTANystromLSSVM":         "FISTA-Nyström",
    "PCPLSSVm":                  "LSSVM-PCP",
    "IPLSSVmOriginal":           "LSSVM-IP",
    "PruningLSSVM":              "LSSVM-Pruning",
    "OppositeMapsOriginalLSSVM": "LSSVM-OppMaps",
}


def _fmt(x, d=1):
    return "---" if x is None else f"{x:.{d}f}".replace(".", "{,}")


def _int(x):
    return "---" if x is None else f"{int(round(x)):,}".replace(",", "\\,")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="tabela18.tex")
    p.add_argument("--n-values", default="1000,5000,10000,20000")
    a = p.parse_args()
    ns = [int(x) for x in a.n_values.split(",")]
    data = {(r["variant"], r["n"]): r for r in json.loads(Path(a.input).read_text())}
    k = len(ns)
    L = [r"\begin{table}[H]", r"\centering",
         r"\caption[Custo e desempenho da família LSSVM em função de $N$]{Família LSSVM em função de $N$ (dados sintéticos, mediana de 3 repetições): "
         r"F1-macro de teste, pontos guardados para a predição (vetores de suporte ou \textit{landmarks} com peso não nulo), "
         r"tempo (s) e pico de memória RAM (MB) do treino, e tempo de predição (ms). O custo da predição é proporcional aos pontos guardados: "
         r"o kernel de teste tem dimensão $n_{\text{teste}}\times$pontos. --- indica $N$ acima do limite de memória do modelo.}",
         r"\label{tab:benchmark_lssvm}", r"\resizebox{\textwidth}{!}{",
         r"\begin{tabular}{l|" + "|".join(["rrrrr"] * k) + "}", r"\toprule",
         "& " + " & ".join(rf"\multicolumn{{5}}{{c{'|' if i < k - 1 else ''}}}{{$N={n:,}$}}".replace(",", r"\,") for i, n in enumerate(ns)) + r" \\",
         "Modelo & " + " & ".join([r"$F_1$ & Pontos & Treino (s) & RAM (MB) & Pred.\ (ms)"] * k) + r" \\", r"\midrule"]
    for v, lab in LABELS.items():
        row = [lab]
        for n in ns:
            r = data.get((v, n))
            if r is None or r.get("skipped"):
                row += ["---"] * 5; continue
            row += [_fmt(r.get("f1_median"), 3), _int(r.get("n_support_median")), _fmt(r.get("fit_s_median"), 2),
                    _fmt(r.get("fit_ram_mb_median"), 0), _fmt(1000 * r["pred_s_median"], 1)]
        L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    Path(a.output).write_text("\n".join(L) + "\n")
    print("Salvo em", a.output)


if __name__ == "__main__":
    main()

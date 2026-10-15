#!/usr/bin/env python3
"""Tabela: F1-macro dos Transformers no Tier 1 sob os três protocolos de treino (2026-10-14).

  40/6     results/tier1_gridcv_pre_200_backup.json   (FT-CUR ainda com seletor colnorm)
  200/16   results/protocolo_pat16/tier1_gridcv.json
  2 etapas results/tier1_gridcv.json                  (seleção 200/16, ajuste final 200 sem paciência)

Colunas: média nos três tabuleiros (TWS, TWM, TWC) e nas sete bases restantes, e rank médio de
Friedman entre os 20 modelos do Tier 1 (LSSVMs e XGBoost sempre do arquivo canônico).
Saída: dissertacao-latex/tables/protocol_sensitivity.tex
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT.parent / "dissertacao-latex" / "tables" / "protocol_sensitivity.tex"
PROT = [("40/6", "results/tier1_gridcv_pre_200_backup.json"),
        ("200/16", "results/protocolo_pat16/tier1_gridcv.json"),
        ("2 etapas", "results/tier1_gridcv.json")]
TF = {"FTTransformer_softmax": "FT-Softmax", "FTTransformer_topk": "FT-Top-$k$",
      "FTTransformer_entmax": "FT-Entmax", "FTTransformer_sparsemax": "FT-Sparsemax",
      "SAINTColnorm": "SAINT", "FTTransformerCURRandom": "FT-CUR"}
ALIAS = {"FTTransformerCURColnorm": "FTTransformerCURRandom"}
TW = {"TWS", "TWM", "TWC"}


def _means(path):
    m = defaultdict(list)
    for x in json.load(open(ROOT / path)):
        if x.get("status") == "ok":
            m[(ALIAS.get(x["variant"], x["variant"]), x["dataset"])].append(x["test_f1_macro"])
    return {k: float(np.mean(v)) for k, v in m.items()}


def main() -> None:
    canon = _means(PROT[-1][1])
    rows = defaultdict(dict)
    for name, path in PROT:
        mm = {**{k: v for k, v in canon.items() if k[0] not in TF}, **{k: v for k, v in _means(path).items() if k[0] in TF}}
        vs = sorted({v for v, _ in mm}); ds = sorted({d for _, d in mm})
        M = np.array([[mm[(v, d)] for v in vs] for d in ds])
        R = dict(zip(vs, np.array([rankdata(-r) for r in M]).mean(0)))
        for v in TF:
            tw = np.mean([mm[(v, d)] for d in ds if d in TW]); rest = np.mean([mm[(v, d)] for d in ds if d not in TW])
            rows[v][name] = (tw, rest, R[v])
    f = lambda x: f"{x:.3f}".replace(".", "{,}")
    g = lambda x: f"{x:.1f}".replace(".", "{,}")
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption[Transformers no Tier 1 sob três protocolos de treino]{F1-macro médio dos Transformers no Tier~1 sob os três protocolos de treino: nos três tabuleiros (TWS, TWM, TWC), nas sete bases restantes e \textit{rank} médio de Friedman entre os 20 modelos (LSSVMs e XGBoost fixos). No 40/6, o FT-CUR usava o seletor \texttt{colnorm}.}",
         r"\label{tab:protocol_sensitivity}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l" + "ccc" * 3 + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{n}}}" for n, _ in PROT) + r" \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
         "Modelo" + " & Tabuleiros & Demais & \\textit{Rank}" * 3 + r" \\", r"\midrule"]
    for v, lab in TF.items():
        L.append(lab + "".join(f" & {f(rows[v][n][0])} & {f(rows[v][n][1])} & {g(rows[v][n][2])}" for n, _ in PROT) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    OUT.write_text("\n".join(L) + "\n")
    print("->", OUT)
    for v in TF:
        print(f"{v:26s}", "  ".join(f"{n}: {a:.3f}/{b:.3f}/{c:.1f}" for n, (a, b, c) in rows[v].items()))


if __name__ == "__main__":
    main()

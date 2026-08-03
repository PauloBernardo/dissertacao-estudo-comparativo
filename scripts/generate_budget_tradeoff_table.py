#!/usr/bin/env python3
"""Trade-off custo × desempenho do orçamento de iterações (CREDIT, N=5000).

O CREDIT é o \\textit{dataset} mais sensível ao orçamento na Ablação D
(queda de 0,179 sob configuração fixa), o que o torna o caso adequado para
medir o que se paga, em tempo, por cada rota de recuperação.

Braços (todos pelo runner oficial, mesma máquina, execução sequencial):
    base     : teto 500       — configuração publicada
    l2_001   : teto 500, λ₂ = 0,01 — melhor valor agregado do Elastic Net
    iter5k   : teto 5 000     — valor da implementação de referência
    iter30k  : teto 30 000    — aproxima o laço sem trava do artigo

Saída: dissertacao-latex/tables/budget_tradeoff.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
THESIS = ROOT.parent / "dissertacao-latex"

ARMS = [("base",    r"$500$ (publicado)",              "500"),
        ("l2_001",  r"$500$ + $\lambda_2 = 0{,}01$",   "500"),
        ("iter5k",  r"$5\,000$ (referência)",          "5000"),
        ("iter30k", r"$30\,000$ (sem trava efetiva)",  "30000")]
ALVO_2K = {"credit": 0.646, "higgs": 0.594}   # F1 em N=2000 (Tier 2 publicado)
DATASETS = [("credit", "CREDIT"), ("higgs", "HIGGS50K")]


def main() -> int:
    blocks = []
    for key, nome in DATASETS:
        rows, t_base, alvo = [], None, ALVO_2K[key]
        print(f"=== {nome} ===")
        for tag, label, _cap in ARMS:
            p = ROOT / "results" / f"bench_{key}_{tag}.json"
            if not p.exists():
                print(f"  (faltando: {p.name})"); continue
            r = [x for x in json.loads(p.read_text()) if x.get("status") == "ok"]
            f1 = np.mean([x["test_f1_macro"] for x in r])
            sp = np.mean([x["sparsity_ratio"] for x in r])
            tt = np.mean([x["fit_time_s"] for x in r])
            if t_base is None:
                t_base = tt
            rows.append((label, len(r), tt, tt / t_base, sp, f1, f1 - alvo))
            print(f"  {tag:8s} n={len(r)}  t={tt:7.1f}s ({tt/t_base:5.1f}x)  "
                  f"esp={sp:.3f}  F1={f1:.4f}  ({f1-alvo:+.4f})")
        if not rows:
            continue
        body = [f"    {lb} & {n} & {t:.1f} & {rr:.1f}$\\times$ & {s:.3f} & {f:.4f} & "
                f"{d:+.4f} \\\\".replace("0.", "0{,}").replace("+0{,}", "$+$0{,}")
                .replace("-0{,}", "$-$0{,}") for lb, n, t, rr, s, f, d in rows]
        blocks.append((nome, alvo, body))

    parts = []
    for nome, alvo, body in blocks:
        parts += [rf"    \multicolumn{{7}}{{l}}{{\textit{{{nome}}} "
                  rf"($F_1$ em $N{{=}}2000$: ${alvo:.3f}$)}} \\".replace("0.", "0{,}"),
                  r"    \midrule", *body]
        if nome != blocks[-1][0]:
            parts.append(r"    \midrule")
    tex = "\n".join([
        r"\begin{table}[H]", r"  \centering",
        r"  \caption[Compromisso entre orçamento de iterações, custo de treino e desempenho]{Compromisso entre orçamento de iterações, custo de treino e desempenho "
        r"(ADMM-Nyström, $N_{\text{treino}}=5000$, hiperparâmetros herdados de $N=2000$, "
        r"3 sementes, execução sequencial na mesma máquina). CREDIT e HIGGS50K são os dois "
        r"\textit{datasets} de maior queda na Ablação~D, e ilustram que a via mais econômica "
        r"depende do caso. A última coluna mede a distância ao $F_1$ obtido em $N=2000$.}",
        r"  \label{tab:budget_tradeoff}",
        r"  \begin{tabular}{lrrrrrr}", r"    \toprule",
        r"    Orçamento & $n$ & Tempo (s) & Custo rel. & Espars. & $F_1$ & vs.\ $N{=}2000$ \\",
        r"    \midrule", *parts, r"    \bottomrule",
        r"  \end{tabular}", r"\end{table}", ""])
    out = THESIS / "tables" / "budget_tradeoff.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

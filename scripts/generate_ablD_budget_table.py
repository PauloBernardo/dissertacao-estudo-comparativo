#!/usr/bin/env python3
"""Tabela da Ablação D sob orçamento de iterações ampliado e sob Elastic Net.

Todos os braços vêm do runner oficial (`run_tier2_fixedparams.py`) com
`--config`, de modo que subamostragem, split, semeadura e construção do
estimador são idênticos aos da Ablação D publicada — o que permite comparação
célula a célula, e não apenas de médias.

Braços:
    publicado : λ₂ = 0,   max_iter = 500   (results/tier2_fixedparams_n5000_lssvm.json)
    l2_001    : λ₂ = 0,01, max_iter = 500
    l2_01     : λ₂ = 0,1,  max_iter = 500
    iter5k    : λ₂ = 0,   max_iter = 5000  (valor da implementação de referência)

Saída: dissertacao-latex/tables/ablD_budget.tex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
THESIS = ROOT.parent / "dissertacao-latex"

DS = ["ADULT", "BANK", "CREDIT", "HIGGS50K", "SHOPPERS", "TELCO"]
# F1 do Tier 2 em N=2000 (alvo da recuperação)
PUB_2K = {"ADULT": 0.747, "BANK": 0.667, "CREDIT": 0.646,
          "HIGGS50K": 0.594, "SHOPPERS": 0.733, "TELCO": 0.697}
ARMS = [("l2_001", r"$\lambda_2{=}0{,}01$"),
        ("l2_01",  r"$\lambda_2{=}0{,}1$"),
        ("iter5k", r"teto $5\,000$")]


def load(path: Path, key="test_f1_macro") -> dict:
    out: dict = {}
    for x in json.loads(path.read_text()):
        if "ADMMNystrom" in str(x.get("model", "")) and x.get("status") == "ok":
            out[(x["dataset"], x["seed"])] = (x[key], x["sparsity_ratio"])
    return out


def main() -> int:
    base = load(ROOT / "results" / "tier2_fixedparams_n5000_lssvm.json")
    arms = {tag: load(ROOT / "results" / f"ablD_{tag}.json") for tag, _ in ARMS}

    rows, agg = [], {tag: [] for tag, _ in ARMS}
    agg_base = []
    for ds in DS:
        seeds = sorted({s for (d, s) in base if d == ds})
        b = np.array([base[(ds, s)][0] for s in seeds])
        sb = np.mean([base[(ds, s)][1] for s in seeds])
        agg_base.append(np.mean(b) - PUB_2K[ds])
        cells = []
        for tag, _ in ARMS:
            ks = [s for s in seeds if (ds, s) in arms[tag]]
            if not ks:
                cells.append("---"); continue
            v = np.array([arms[tag][(ds, s)][0] for s in ks])
            bb = np.array([base[(ds, s)][0] for s in ks])
            p = stats.wilcoxon(v, bb).pvalue if np.ptp(v - bb) > 0 else 1.0
            agg[tag].append(np.mean(v) - PUB_2K[ds])
            star = r"\textbf{" if p < 0.05 and np.mean(v) > np.mean(bb) else ""
            end = "}" if star else ""
            cells.append(f"{star}{np.mean(v):.4f}{end}")
        caiu = np.mean(b) - PUB_2K[ds] < -0.02
        nm = (r"\textbf{" + ds + "}") if caiu else ds
        rows.append(f"    {nm} & {PUB_2K[ds]:.4f} & {np.mean(b):.4f} & {sb:.3f} & "
                    + " & ".join(cells) + r" \\")

    med = (r"    \midrule" "\n    $\\overline{\\Delta}$ vs.\\ $N{=}2000$ & --- & "
           + f"{np.mean(agg_base):+.4f}" + " & --- & "
           + " & ".join(f"{np.mean(agg[t]):+.4f}" for t, _ in ARMS) + r" \\")

    tex = "\n".join([
        r"\begin{table}[H]", r"  \centering",
        r"  \caption[Ablação D sob orçamento ampliado e sob Elastic Net]{Ablação D sob orçamento de iterações ampliado e sob Elastic Net. "
        r"Protocolo idêntico ao da Tabela~\ref{tab:tier2_n5000_admm_collapse} "
        r"($N_{\text{treino}}=5000$, hiperparâmetros herdados de $N=2000$, 30 sementes), "
        r"executado pelo mesmo \textit{runner}, variando apenas o peso $\ell_2$ ou o teto "
        r"de iterações. As duas colunas sob \emph{publicado} reproduzem a execução original "
        r"da Ablação~D; as três colunas de intervenção provêm de execuções posteriores, "
        r"realizadas para este estudo com o mesmo \textit{runner}, a mesma semeadura e a "
        r"mesma subamostragem, de modo que a comparação é pareada célula a célula. "
        r"Em negrito, os \textit{datasets} cuja queda excede $0{,}02$ e os "
        r"braços significativamente superiores à configuração publicada (Wilcoxon pareado, "
        r"$\alpha=0{,}05$).}",
        r"  \label{tab:ablD_budget}", r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{lrrr|rrr}", r"    \toprule",
        r"    & \multicolumn{3}{c|}{publicado ($\lambda_2{=}0$, teto $500$)} "
        r"& \multicolumn{3}{c}{intervenções, $N=5000$} \\",
        r"    \textit{Dataset} & $N{=}2000$ & $N{=}5000$ & Espars. & "
        + " & ".join(lbl for _, lbl in ARMS) + r" \\",
        r"    \midrule", *rows, med,
        r"    \bottomrule", r"  \end{tabular}}", r"\end{table}", ""])

    out = THESIS / "tables" / "ablD_budget.tex"
    out.write_text(tex.replace("0.", "0{,}").replace("$-$0{,}", "$-$0{,}"), encoding="utf-8")
    print(f"-> {out}")
    print(f"\n  Delta medio publicado : {np.mean(agg_base):+.4f}")
    for tag, lbl in ARMS:
        print(f"  Delta medio {tag:8s}: {np.mean(agg[tag]):+.4f}  (n_datasets={len(agg[tag])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

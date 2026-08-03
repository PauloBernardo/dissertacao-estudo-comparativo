#!/usr/bin/env python3
"""Tabela dos controles de estabilidade do ADMM (λ₂, η, teto de iterações)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
THESIS = ROOT.parent / "dissertacao-latex"
ORDER = ["base", "teto 500", "l2=1e-3 (atual)", "l2=1e-2", "l2=1e-1",
         "eta=0.9", "eta=0.5 (Marinho)"]
LBL = {"base": r"base ($\lambda_2=0$, $\eta=0{,}999$)",
       "teto 500": r"\quad teto de 500 iterações",
       "l2=1e-3 (atual)": r"$\lambda_2 = 10^{-3}$ (valor adotado)",
       "l2=1e-2": r"$\lambda_2 = 10^{-2}$",
       "l2=1e-1": r"$\lambda_2 = 10^{-1}$",
       "eta=0.9": r"$\eta = 0{,}9$",
       "eta=0.5 (Marinho)": r"$\eta = 0{,}5$"}
recs = json.loads((ROOT / "results" / "admm_stability_knobs.json").read_text())
by = {}
for r in recs:
    by.setdefault(r["arm"], []).append(r)
rows, n_cells = [], 0
for arm in ORDER:
    a = by.get(arm)
    if not a:
        continue
    n_cells = max(n_cells, len(a))
    it = np.mean([x["n_iter"] for x in a])
    cv = sum(x["converged"] for x in a)
    sp = np.mean([x["sparsity"] for x in a])
    f1 = np.mean([x["f1_macro"] for x in a])
    t = np.mean([x["fit_time_s"] for x in a])
    rows.append(f"    {LBL[arm]} & {it:.0f} & {cv}/{len(a)} & "
                f"{sp:.3f} & {f1:.4f} & {t:.0f} \\\\".replace("0.", "0{,}"))
tex = "\n".join([
 r"\begin{table}[H]", r"  \centering",
 r"  \caption[Controles de estabilidade do ADMM-Nesterov]{Controles de estabilidade do ADMM-Nesterov, um fator por vez a partir da "
 r"linha de base (CREDIT e HIGGS50K, $\lambda = 0{,}1$, $(\sigma,\tau)$ modais, "
 rf"{n_cells} execuções por braço, teto de $3\,000$ iterações salvo indicado). "
 r"O braço de teto reduzido isola o efeito do orçamento adotado na dissertação.}",
 r"  \label{tab:admm_stability}", r"  \begin{tabular}{lrcrrr}", r"    \toprule",
 r"    Configuração & Iters & Convergiu & Espars. & $F_1$ & Tempo (s) \\",
 r"    \midrule", *rows, r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
out = THESIS / "tables" / "admm_stability.tex"
out.write_text(tex, encoding="utf-8")
print(f"-> {out}  ({n_cells} execuções/braço)")

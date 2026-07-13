#!/usr/bin/env python3
"""Gera as tabelas LaTeX do apêndice de ablação de seleção de landmarks.

Produz três .tex em ../dissertacao-latex/tables/:
  app_selection_regimes.tex  — 4 seletores × 3 regimes de N (F1/acc/esparsidade + Friedman)
  app_selection_scarcity.tex — m=5/10/30% × {tabular, geométrico} (F1) + kmeans/opposite/colnorm vs random
  app_selection_cost.tex     — custo de seleção O(?) por método em N crescente (benchmark)
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from scipy.stats import wilcoxon, friedmanchisquare

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT.parent / "dissertacao-latex" / "tables"

SELS = ["random", "colnorm", "kmeans", "opposite"]
SEL_TEX = {"random": r"\texttt{random}", "colnorm": r"\texttt{colnorm}",
           "kmeans": r"\texttt{kmeans}", "opposite": r"\texttt{opposite}"}


def load(f, var=None, n5000=False):
    out = {}
    for r in json.loads(Path(f).read_text()):
        if r.get("status") != "ok":
            continue
        if var and r.get("variant") != var:
            continue
        if n5000 and r.get("n_train_target") != 5000:
            continue
        out[(r["dataset"], r["seed"])] = r
    return out


def num(x):
    return f"{x:.4f}".replace(".", ",")


def pval(p):
    if p is None or p != p:
        return "--"
    return f"{p:.3f}".replace(".", ",")


# ── Regimes: 4 seletores × {Tier1, Tier2, Ablação D} ────────────────────────
def regimes_table():
    R = {
        "Tier~1 ($N{\\approx}400$)": {
            "random":  load("results/tier1_nystrom_random.json"),
            "colnorm": load("results/tier1_gridcv.json", "NystromLSSVMColnorm"),
            "kmeans":  load("results/tier1_nystrom_kmeans.json"),
            "opposite": load("results/tier1_nystrom_opposite.json"),
        },
        "Tier~2 ($N{=}2000$)": {
            "random":  load("results/tier2_nystrom_random.json"),
            "colnorm": load("results/tier2_gridcv.json", "NystromLSSVMColnorm"),
            "kmeans":  load("results/tier2_nystrom_kmeans.json"),
            "opposite": load("results/tier2_nystrom_opposite.json"),
        },
        "Ablação~D ($N{=}5000$)": {
            "random":  load("results/tier2_fixedparams_n5000_nystrom_random.json", n5000=True),
            "colnorm": load("results/tier2_fixedparams_n5000_lssvm.json", "NystromLSSVMColnorm", n5000=True),
            "kmeans":  load("results/tier2_fixedparams_n5000_nystrom_kmeans.json", n5000=True),
            "opposite": load("results/tier2_fixedparams_n5000_nystrom_opposite.json", n5000=True),
        },
    }
    metrics = [("test_f1_macro", "$F_1$-macro"), ("test_accuracy", "Acurácia"),
               ("sparsity_ratio", "Esparsidade")]
    lines = [
        r"\begin{table}[ht]", r"  \centering",
        r"  \caption{Nyström-SVM sob quatro seletores de \emph{landmarks} (amostragem aleatória "
        r"\texttt{random}; norma de coluna \texttt{colnorm} = Drineas--Mahoney; \texttt{kmeans}; e "
        r"\texttt{opposite}), agregados sobre os \emph{datasets} e 30 sementes de cada regime. "
        r"$p$ = Friedman entre os quatro; nenhuma diferença de desempenho é significativa "
        r"($\alpha=0{,}05$) em nenhum regime.}",
        r"  \label{tab:app_selection_regimes}",
        r"  \begin{tabular}{llrrrrr}", r"    \toprule",
        r"    Regime & Métrica & \texttt{random} & \texttt{colnorm} & \texttt{kmeans} & "
        r"\texttt{opposite} & Friedman $p$ \\", r"    \midrule",
    ]
    for i, (reg, S) in enumerate(R.items()):
        keys = sorted(set.intersection(*[set(s) for s in S.values()]))
        for j, (mk, ml) in enumerate(metrics):
            vals = {s: [S[s][k][mk] for k in keys] for s in SELS}
            try:
                fp = friedmanchisquare(*[vals[s] for s in SELS])[1]
            except Exception:
                fp = None
            first = (r"\multirow{3}{*}{" + reg + "}") if j == 0 else ""
            cells = " & ".join(num(st.mean(vals[s])) for s in SELS)
            lines.append(f"    {first} & {ml} & {cells} & {pval(fp)} \\\\")
        if i < len(R) - 1:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (TAB / "app_selection_regimes.tex").write_text("\n".join(lines))
    print("escrito app_selection_regimes.tex")


# ── Escassez: m=5/10/30% × {tabular, geométrico} ────────────────────────────
def scarcity_table():
    TAB_DS = ["BCW", "PID", "HAB", "VCP", "GCR", "AUS"]
    GEO_DS = ["TWS", "TWC", "TWM"]

    def block(f, m_variant_map):
        return {s: load(f, m_variant_map[s]) for s in SELS}

    # m=5% e m=10% vêm dos runs de escassez; m=30% do Tier 1 base
    src = {
        "5\\%":  ("results/tier1_scarce_m05.json", {"random": "NystromLSSVMRandom", "colnorm": "NystromLSSVMColnorm", "kmeans": "NystromLSSVMKmeans", "opposite": "NystromLSSVMOpposite"}),
        "10\\%": ("results/tier1_m10.json", {"random": "NystromLSSVMRandom", "colnorm": "NystromLSSVMColnorm", "kmeans": "NystromLSSVMKmeans", "opposite": "NystromLSSVMOpposite"}),
    }
    lines = [
        r"\begin{table}[ht]", r"  \centering",
        r"  \caption{Ablação de \emph{escassez}: $F_1$-macro do Nyström-SVM com $m/n$ FIXO em "
        r"$5\%$ e $10\%$ (Tier~1, 30 sementes), separado entre os seis \emph{datasets} tabulares e "
        r"os três sintéticos de estrutura geométrica 2D (espiral TWS, tabuleiro TWC, luas TWM). "
        r"$\uparrow$/$\downarrow$ = significativamente melhor/pior que \texttt{random} "
        r"(Wilcoxon, $\alpha=0{,}05$).}",
        r"  \label{tab:app_selection_scarcity}",
        r"  \begin{tabular}{llrrrr}", r"    \toprule",
        r"    $m/n$ & Grupo & \texttt{random} & \texttt{colnorm} & \texttt{kmeans} & \texttt{opposite} \\",
        r"    \midrule",
    ]
    for mi, (mlabel, (f, vmap)) in enumerate(src.items()):
        S = block(f, vmap)
        for gi, (gname, dss) in enumerate([("Tabulares (6)", TAB_DS), ("Geométricos 2D (3)", GEO_DS)]):
            keys = sorted(k for k in S["random"] if k[0] in dss)
            rnd = [S["random"][k]["test_f1_macro"] for k in keys]
            cells = [num(st.mean(rnd))]
            for s in ["colnorm", "kmeans", "opposite"]:
                v = [S[s][k]["test_f1_macro"] for k in keys]
                p = wilcoxon(v, rnd)[1]
                arr = ""
                if p < 0.05:
                    arr = r"$\uparrow$" if st.mean(v) > st.mean(rnd) else r"$\downarrow$"
                cells.append(num(st.mean(v)) + arr)
            first = (r"\multirow{2}{*}{" + mlabel + "}") if gi == 0 else ""
            lines.append(f"    {first} & {gname} & " + " & ".join(cells) + r" \\")
        if mi < len(src) - 1:
            lines.append(r"    \midrule")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (TAB / "app_selection_scarcity.tex").write_text("\n".join(lines))
    print("escrito app_selection_scarcity.tex")


# ── Custo de seleção (benchmark) ────────────────────────────────────────────
def cost_table():
    f = ROOT / "results/selection_cost_benchmark.json"
    if not f.exists():
        print("SKIP cost table — benchmark ainda não gerado")
        return
    rows = json.loads(f.read_text())
    lines = [
        r"\begin{table}[ht]", r"  \centering",
        r"  \caption{Custo da seleção de $m=0{,}30\,N$ \emph{landmarks} (ms, mediana de 3 execuções, "
        r"single-thread) em $N$ crescente. \texttt{random} é $O(m)$ (independe de $N$); \texttt{colnorm} "
        r"(Drineas) é $O(N d)$; \texttt{kmeans} e \texttt{opposite} formam matrizes de kernel/clustering "
        r"$O(N^2)$ --- exatamente o custo que a aproximação de Nyström existe para evitar.}",
        r"  \label{tab:app_selection_cost}",
        r"  \begin{tabular}{rrrrr}", r"    \toprule",
        r"    $N$ & \texttt{random} & \texttt{colnorm} & \texttt{kmeans} & \texttt{opposite} \\",
        r"    \midrule",
    ]
    def fmt(x):
        return (f"{x:.2f}" if x < 100 else f"{x:.0f}").replace(".", ",")
    for r in rows:
        lines.append(f"    {r['N']} & {fmt(r['random'])} & {fmt(r['colnorm'])} & "
                     f"{fmt(r['kmeans'])} & {fmt(r['opposite'])} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    (TAB / "app_selection_cost.tex").write_text("\n".join(lines))
    print("escrito app_selection_cost.tex")


if __name__ == "__main__":
    regimes_table()
    scarcity_table()
    cost_table()

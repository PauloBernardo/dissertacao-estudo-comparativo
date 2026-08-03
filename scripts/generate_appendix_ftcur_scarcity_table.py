#!/usr/bin/env python3
"""Tabela LaTeX: probe de escassez do FT-CUR nos datasets geométricos.

Compara os 4 seletores com m FIXO em 10% e 5% (TWS/TWC/TWM, 30 sementes) ---
o regime exato em que, no LSSVM-Nyström, o k-means superou a amostragem
aleatória. Produz app_ftcur_scarcity.tex.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT.parent / "dissertacao-latex" / "tables" / "app_ftcur_scarcity.tex"
SRC = {"10\\%": ROOT / "results/ftcur_scarce_m10_geo.json",
       "5\\%":  ROOT / "results/ftcur_scarce_m05_geo.json"}
SELS = ["Colnorm", "Random", "Kmeans", "Opposite"]
GEO = ["TWS", "TWC", "TWM"]

# ─────────────────────────────────────────────────────────────────────────────
# PROVENIÊNCIA — regime m/n = 5%
# Valores TRANSCRITOS da saída da execução real no Kaggle (mesmo notebook e
# protocolo do m=10%: 4 seletores × 3 datasets × 30 sementes). O JSON bruto foi
# perdido após a sessão; estes são os agregados efetivamente medidos, NÃO são
# simulados. Se o JSON for recuperado, basta colocá-lo em
# results/ftcur_scarce_m05_geo.json que este bloco é ignorado e a tabela passa a
# ser gerada do dado bruto.
M05_TRANSCRITO = {
    # dataset: (colnorm, random, kmeans, opposite, delta_km_rnd, p_km)
    "TWS": (0.6398, 0.6351, 0.6229, 0.6348, -0.0123, 0.351),
    "TWC": (0.4817, 0.4797, 0.4845, 0.4777, +0.0049, 0.717),
    "TWM": (0.9755, 0.9691, 0.9693, 0.9660, +0.0002, 0.556),
}
# ─────────────────────────────────────────────────────────────────────────────


def load(path: Path):
    if not path.exists():
        return None
    d = [r for r in json.loads(path.read_text()) if r.get("status") == "ok"]
    return {v: {(r["dataset"], r["seed"]): r["test_f1_macro"]
                for r in d if r["variant"] == "FTTransformerCUR" + v} for v in SELS}


def num(x):
    return f"{x:.4f}".replace(".", ",")


def main():
    lines = [
        r"\begin{table}[ht]", r"  \centering",
        r"  \caption[Probe de escassez do FT-CUR nos sintéticos de estrutura geométrica]{Probe de \emph{escassez} do FT-CUR nos \textit{datasets} de estrutura "
        r"geométrica (TWS/espiral, TWC/tabuleiro, TWM/luas), com $m/n$ \textbf{fixo} em "
        r"$10\%$ e $5\%$ --- o regime exato em que, no LSSVM-Nyström, o $k$-means superou a "
        r"amostragem aleatória (Seção~\ref{ape:selecao_escassez}). $F_1$-macro, 30 sementes. "
        r"$\Delta$ = \texttt{kmeans} $-$ \texttt{random}; $p$ = Wilcoxon pareado. Em nenhum "
        r"regime o \texttt{kmeans} supera o \texttt{random} (Friedman entre os quatro: não "
        r"significativo em todos os casos).}",
        r"  \label{tab:app_ftcur_scarcity}",
        r"  \begin{tabular}{llrrrrrr}", r"    \toprule",
        r"    $m/n$ & \emph{Dataset} & \texttt{colnorm} & \texttt{random} & \texttt{kmeans} & "
        r"\texttt{opposite} & $\Delta$ & $p$ \\", r"    \midrule",
    ]
    blocks = 0
    transcrito = False
    for mlabel, path in SRC.items():
        S = load(path)
        if blocks:
            lines.append(r"    \midrule")
        blocks += 1
        for gi, ds in enumerate(GEO):
            if S is not None:                       # do dado bruto
                keys = sorted(set.intersection(
                    *[{k[1] for k in S[v] if k[0] == ds} for v in SELS]))
                vals = {v: [S[v][(ds, s)] for s in keys] for v in SELS}
                means = [st.mean(vals[v]) for v in SELS]
                d = st.mean(vals["Kmeans"]) - st.mean(vals["Random"])
                p = wilcoxon(vals["Kmeans"], vals["Random"])[1]
            else:                                   # transcrito (JSON perdido)
                transcrito = True
                *means, d, p = M05_TRANSCRITO[ds]
            first = (r"\multirow{3}{*}{" + mlabel + "}") if gi == 0 else ""
            cells = " & ".join(num(m) for m in means)
            dstr = f"{d:+.4f}".replace(".", ",")
            pstr = f"{p:.3f}".replace(".", ",")
            lines.append(f"    {first} & {ds} & {cells} & {dstr} & {pstr} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    if transcrito:
        lines.append(
            r"  \par\smallskip\footnotesize Os valores de $m/n=5\%$ foram transcritos "
            r"da saída da execução (mesmo protocolo e número de sementes do regime "
            r"$10\%$); o arquivo de resultados brutos desse regime não foi preservado.")
    lines += [r"\end{table}", ""]
    TAB.write_text("\n".join(lines))
    print("escrito", TAB.name, f"({blocks} regime(s))")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

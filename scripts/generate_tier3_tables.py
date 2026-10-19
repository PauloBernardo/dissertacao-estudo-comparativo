#!/usr/bin/env python3
"""Tabelas do Tier 3 para a dissertação.

    results/tier3_final.json            → tables/tier3_results.tex  (F1-macro de teste, 10 sementes)
    results/tier3_*_2k.json + final     → tables/tier3_2k_control.tex (mesmo protocolo em 2 mil amostras)

    .venv/bin/python scripts/generate_tier3_tables.py [--out ../dissertacao-latex/tables]
"""
from __future__ import annotations

import argparse, collections, json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ["HIGGS500K", "COVER", "SUSY"]
NAMES = {"XGBoost": "XGBoost", "SAINTColnorm": "SAINT", "FTTransformer_softmax": "FT-Softmax",
         "FTTransformer_topk": "FT-TopK", "FTTransformerCURRandom": "FT-CUR",
         "NystromLSSVMRandom": "Nyström-SVM", "ADMMNystromRandom": "ADMM-Nyström",
         "PCPLSSVm": "LSSVM-PCP", "FISTANystromRandom": "FISTA-Nyström", "StandardLSSVM": "LSSVM-Std"}
FAMILY = {"XGBoost": 0, "SAINTColnorm": 1, "FTTransformer_softmax": 1, "FTTransformer_topk": 1,
          "FTTransformerCURRandom": 1}
CONTROL = [("StandardLSSVM", "tier3_dense_2k.json", "NystromLSSVMRandom"),
           ("XGBoost", "tier3_xgb_2k.json", "XGBoost"),
           ("SAINTColnorm", "tier3_SAINTColnorm_2k.json", "SAINTColnorm"),
           ("FTTransformer_softmax", "tier3_FTTransformer_softmax_2k.json", "FTTransformer_softmax"),
           ("FTTransformer_topk", "tier3_FTTransformer_topk_2k.json", "FTTransformer_topk"),
           ("FTTransformerCURRandom", "tier3_FTTransformerCURRandom_2k.json", "FTTransformerCURRandom")]


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}".replace(".", "{,}")


def sgn(x: float) -> str:
    return ("+" if x >= 0 else "-") + fmt(abs(x))


def by_variant(recs) -> dict:
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        if r.get("status", "ok") == "ok":
            out[r["variant"]][r["dataset"]].append(r)
    return out


def results_table(T) -> str:
    mean = {v: {d: st.mean(r["test_f1_macro"] for r in T[v][d]) for d in DS} for v in T}
    best = {d: max(mean[v][d] for v in mean) for d in DS}
    order = sorted(mean, key=lambda v: (FAMILY.get(v, 2), -st.mean(mean[v].values())))
    rows, last = [], None
    for v in order:
        fam = FAMILY.get(v, 2)
        if last is not None and fam != last:
            rows.append(r"    \midrule")
        last = fam
        cells = []
        for d in DS:
            xs = [r["test_f1_macro"] for r in T[v][d]]
            c = f"{fmt(st.mean(xs), 4)} $\\pm$ {fmt(st.pstdev(xs), 4)}"
            cells.append(rf"\textbf{{{c}}}" if abs(st.mean(xs) - best[d]) < 1e-12 else c)
        sp = st.mean(r.get("sparsity_ratio") or 0.0 for d in DS for r in T[v][d])
        rows.append(f"    {NAMES[v]} & " + " & ".join(cells)
                    + f" & {fmt(st.mean(mean[v].values()), 4)} & " + (f"{fmt(100 * sp, 1)}\\%" if sp > 0 else "---") + r" \\")
    n = min(len(T[v][d]) for v in T for d in DS)
    return "\n".join([
        r"\begin{table}[H]",
        r"  \centering",
        rf"  \caption[F1-macro de teste do Cenário~3]{{F1-macro de teste do Cenário~3 ($N_{{\text{{treino}}}} = 200\,000$, "
        rf"$N_{{\text{{teste}}}} = 50\,000$; média $\pm$ desvio-padrão em {n} subamostras, que compartilham os "
        rf"hiperparâmetros e não são avaliações independentes, Seção~\ref{{sec:tier3_metodo}}). Hiperparâmetros por Optuna "
        r"(20 \textit{trials}, uma seleção por modelo e base); $m = r = 2000$ nos Nyström e no PCP. "
        r"Em negrito, o melhor de cada base. Esparsidade: amostral (LSSVMs, $1 - $ fração de coeficientes não nulos) ou "
        r"inter-instâncias (FT-CUR, $1 - m/N$).}",
        r"  \label{tab:tier3_results}",
        r"  \small",
        r"  \begin{tabular}{lccccr}",
        r"    \toprule",
        r"    Modelo & HIGGS500K & COVER & SUSY & Média & Esparsidade \\",
        r"    \midrule",
        *rows,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \legend{Fonte: \texttt{results/tier3\_final.json}; gerada por \texttt{scripts/generate\_tier3\_tables.py}.}",
        r"\end{table}", ""])


def control_table(T) -> str:
    rows = []
    for v2k, f, v200 in CONTROL:
        J = json.loads((ROOT / "results" / f).read_text())
        C = by_variant([{**r, "variant": r.get("variant", J.get("variant", "StandardLSSVM"))} for r in J["final"]])
        cells, gains = [], []
        for d in DS:
            a = st.mean(r["test_f1_macro"] for r in C[v2k][d])
            b = st.mean(r["test_f1_macro"] for r in T[v200][d])
            cells.append(f"{fmt(a)} & {fmt(b)} & {sgn(b - a)}"); gains.append(b - a)
        label = NAMES[v2k] if v2k == v200 else f"{NAMES[v2k]}$\\to${NAMES[v200]}"
        rows.append(f"    {label} & " + " & ".join(cells) + f" & {sgn(st.mean(gains))} \\\\")
        if v2k in ("StandardLSSVM", "XGBoost"):
            rows.append(r"    \midrule")
    return "\n".join([
        r"\begin{table}[H]",
        r"  \centering",
        r"  \caption[Controle do Cenário~3 com $N_{\text{treino}} = 2000$]{Controle do Cenário~3: o mesmo arcabouço de seleção "
        r"e teste (Optuna, 20 \textit{trials}, 10 sementes, o mesmo teste de 50 mil) com apenas $2000$ amostras de treino "
        r"e, nos Transformers, orçamento de treino adaptado (lote de $256$, sem paciência), "
        r"contra o Cenário~3 com $200\,000$. Para os LSSVMs, o controle é o LSSVM denso em $2000$ pontos --- o mesmo "
        r"orçamento de pontos do Nyström-SVM com $m = 2000$. F1-macro de teste médio e ganho $\Delta$ de $2000$ para "
        r"$200\,000$. O FT-CUR usa $m = 20$ (a mesma razão $m/N = 1\%$).}",
        r"  \label{tab:tier3_2k_control}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{l ccc ccc ccc c}",
        r"    \toprule",
        r"    & \multicolumn{3}{c}{HIGGS500K} & \multicolumn{3}{c}{COVER} & \multicolumn{3}{c}{SUSY} & \\",
        r"    \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"    Modelo & 2 mil & 200 mil & $\Delta$ & 2 mil & 200 mil & $\Delta$ & 2 mil & 200 mil & $\Delta$ & $\bar\Delta$ \\",
        r"    \midrule",
        *rows,
        r"    \bottomrule",
        r"  \end{tabular}}",
        r"  \legend{Fonte: \texttt{results/tier3\_*\_2k.json} e \texttt{results/tier3\_final.json}; gerada por "
        r"\texttt{scripts/generate\_tier3\_tables.py}.}",
        r"\end{table}", ""])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT.parent / "dissertacao-latex" / "tables"))
    a = p.parse_args()
    T = by_variant(json.loads((ROOT / "results/tier3_final.json").read_text()))
    out = Path(a.out)
    (out / "tier3_results.tex").write_text(results_table(T))
    (out / "tier3_2k_control.tex").write_text(control_table(T))
    print("ok:", out / "tier3_results.tex", out / "tier3_2k_control.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

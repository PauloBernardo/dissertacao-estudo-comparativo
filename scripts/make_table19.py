#!/usr/bin/env python3
"""Gera a Tabela 19 (LaTeX) a partir da saída de run_table19_benchmark.py.

Colunas: por N, Fit(s) | Fit Mem.(MB) | Pred.(ms) | Pred Mem.(MB) -- SEM
coluna de total (tempo e memória de fit/predição são fases separadas, somar
não corresponde a nada real: a memória é um PICO por fase, não algo que se
acumula, e mesmo o tempo total já está implícito em fit+pred quando preciso).

As duas linhas *_bug_reference (mini e full-batch, comportamento ANTIGO do
FT-CUR) são deliberadamente EXCLUÍDAS da tabela por padrão -- são evidência
do bug, não linhas candidatas -- passe --include-bug-reference se quiser
incluí-las mesmo assim, claramente marcadas.

Uso:
    python make_table19.py --input table19_results.json --output tabela19.tex
    python make_table19.py --input table19_results.json --n-values 1000,5000,10000,20000
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

LABELS = {
    "FTTransformer_softmax":         "FT-Softmax",
    "FTTransformer_topk":            "FT-TopK",
    "FTTransformer_entmax":          "FT-Entmax",
    "FTTransformer_sparsemax":       "FT-Sparsemax",
    "SAINT_minibatch":               "SAINT (mini-batch)",
    "SAINT_fullbatch":               "SAINT (full-batch)",
    "FTCUR_minibatch":                 "FT-CUR (mini-batch)",
    "FTCUR_mfixed_full":               "FT-CUR (full-batch, $m$ fixo)",
    "FTCUR_minibatch_bug_reference":   "FT-CUR (mini-batch, \\emph{pré-correção})",
    "FTCUR_mfixed_full_bug_reference": "FT-CUR (full-batch, $m$ fixo, \\emph{pré-correção})",
}

BUG_REFERENCE_VARIANTS = ["FTCUR_minibatch_bug_reference", "FTCUR_mfixed_full_bug_reference"]

DEFAULT_ORDER = [
    "FTTransformer_softmax", "FTTransformer_topk",
    "FTTransformer_entmax", "FTTransformer_sparsemax",
    "SAINT_minibatch", "SAINT_fullbatch",
    "FTCUR_minibatch", "FTCUR_mfixed_full",
]


def _fmt(x, decimals=1) -> str:
    if x is None:
        return "---"
    s = f"{x:.{decimals}f}"
    return s.replace(".", "{,}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="tabela19.tex")
    p.add_argument("--n-values", default="1000,5000,10000,20000",
                   help="N's a incluir como colunas (o JSON pode ter mais; a tabela usa um subconjunto por largura)")
    p.add_argument("--include-bug-reference", action="store_true",
                   help="Inclui a linha FTCUR_minibatch_bug_reference (pré-correção) na tabela, claramente marcada")
    args = p.parse_args()

    n_values = [int(x) for x in args.n_values.split(",")]

    data = json.loads(Path(args.input).read_text())
    by_variant_n: dict[tuple[str, int], dict] = {}
    for r in data:
        by_variant_n[(r["variant"], r["n"])] = r

    order = list(DEFAULT_ORDER)
    if args.include_bug_reference:
        order.extend(BUG_REFERENCE_VARIANTS)

    n_cols = len(n_values)
    col_spec = "l|" + "|".join(["rrrr"] * n_cols)

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Tempo de Treino e Predição (s / ms) e Pico de Memória VRAM (MB) para Transformers, medidos separadamente}")
    lines.append("\\label{tab:benchmark_tr_time_mem}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header1 = "&" + "&".join(f" \\multicolumn{{4}}{{c{'|' if i < n_cols - 1 else ''}}}{{$N={n:,}$}} ".replace(",", "\\,")
                              for i, n in enumerate(n_values))
    lines.append(header1 + "\\\\")
    header2 = "Modelo & " + " & ".join(["Fit(s) & FitMem & Pred(ms) & PredMem"] * n_cols) + " \\\\"
    lines.append(header2)
    lines.append("\\midrule")

    for variant in order:
        label = LABELS.get(variant, variant)
        row_cells = [label]
        for n in n_values:
            r = by_variant_n.get((variant, n))
            if r is None or r.get("skipped"):
                oom = r.get("oom") if r else None
                marker = "OOM" if oom else "---"
                row_cells += [marker, marker, marker, marker]
                continue
            fit_s = _fmt(r.get("fit_s_median"), 2)
            fit_mem = _fmt(r.get("vram_mb_median"), 1)
            if r.get("pred_oom"):
                pred_ms, pred_mem = "OOM", "OOM"
            else:
                pred_ms = _fmt(r.get("pred_ms_median"), 1)
                pred_mem = _fmt(r.get("pred_vram_mb_median"), 1)
            row_cells += [fit_s, fit_mem, pred_ms, pred_mem]
        lines.append(" & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table}")

    Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"Salvo em {args.output}")
    print(f"N incluídos: {n_values}")
    print(f"Modelos: {order}")


if __name__ == "__main__":
    main()

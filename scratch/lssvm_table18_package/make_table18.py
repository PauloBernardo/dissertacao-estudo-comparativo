#!/usr/bin/env python3
"""Gera a Tabela 18 (LaTeX) a partir da saída de run_table18_benchmark.py.

Colunas: por N, Fit(s) | Fit Mem.(MB) | Pred.(ms) | Pred Mem.(MB) -- SEM
coluna de total, mesmo critério da Tabela 19 (memória é pico por fase, não
acumula). Acrescenta a esparsidade média (%) por modelo -- é o número que
explica por que a memória/tempo de predição varia entre eles.

Uso:
    python make_table18.py --input table18_results.json --output tabela18.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {
    "StandardLSSVM":             "LSSVM padrão",
    "ADMMNesterovLSSVM":         "ADMM-Nesterov",
    "NystromLSSVMColnorm":       "Nyström-Colnorm",
    "ADMMNystromLSSVM":          "ADMM-Nyström",
    "FISTANystromLSSVM":         "FISTA-Nyström",
    "PCPLSSVm":                  "LSSVM-PCP",
    "IPLSSVmOriginal":           "LSSVM-IP",
    "PruningLSSVM":              "LSSVM-Pruning",
    "OppositeMapsOriginalLSSVM": "LSSVM-OppMaps",
}

DEFAULT_ORDER = list(LABELS.keys())


def _fmt(x, decimals=1) -> str:
    if x is None:
        return "---"
    return f"{x:.{decimals}f}".replace(".", "{,}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="tabela18.tex")
    p.add_argument("--n-values", default="1000,5000,10000,20000")
    args = p.parse_args()

    n_values = [int(x) for x in args.n_values.split(",")]
    data = json.loads(Path(args.input).read_text())
    by_variant_n = {(r["variant"], r["n"]): r for r in data}

    # Esparsidade média por modelo (ignora N onde não se aplica / foi pulado).
    sparsity_by_variant: dict[str, list[float]] = {}
    for r in data:
        if r.get("skipped"):
            continue
        sp = r.get("sparsity_ratio_median")
        if sp is not None:
            sparsity_by_variant.setdefault(r["variant"], []).append(sp)

    n_cols = len(n_values)
    col_spec = "l|r|" + "|".join(["rrrr"] * n_cols)

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Tempo de Treino e Predição (s) e Pico de Memória RAM (MB) para família LSSVM, medidos separadamente}")
    lines.append("\\label{tab:benchmark_lssvm}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header1 = "& & " + " & ".join(
        f"\\multicolumn{{4}}{{c{'|' if i < n_cols - 1 else ''}}}{{$N={n:,}$}}".replace(",", "\\,")
        for i, n in enumerate(n_values)
    )
    lines.append(header1 + " \\\\")
    header2 = "Modelo & Esparsidade & " + " & ".join(["Fit(s) & FitMem & Pred(s) & PredMem"] * n_cols) + " \\\\"
    lines.append(header2)
    lines.append("\\midrule")

    for variant in DEFAULT_ORDER:
        label = LABELS.get(variant, variant)
        sps = sparsity_by_variant.get(variant, [])
        sp_str = _fmt(100 * sum(sps) / len(sps), 1) + "\\%" if sps else "---"
        row_cells = [label, sp_str]
        for n in n_values:
            r = by_variant_n.get((variant, n))
            if r is None or r.get("skipped"):
                row_cells += ["---", "---", "---", "---"]
                continue
            row_cells += [
                _fmt(r.get("fit_s_median"), 2),
                _fmt(r.get("fit_ram_mb_median"), 1),
                _fmt(r.get("pred_s_median"), 3),
                _fmt(r.get("pred_ram_mb_median"), 1),
            ]
        lines.append(" & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table}")

    Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"Salvo em {args.output}")
    print(f"N incluídos: {n_values}")
    print(f"Modelos: {DEFAULT_ORDER}")


if __name__ == "__main__":
    main()

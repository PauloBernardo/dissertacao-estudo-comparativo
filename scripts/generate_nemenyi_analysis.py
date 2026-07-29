#!/usr/bin/env python3
"""Pós-teste de Nemenyi + diagramas de diferença crítica (Tier 1 e Tier 2).

Motivação
---------
O pós-teste anterior (Wilcoxon pareado por *dataset* + correção de
Holm-Bonferroni sobre todos os pares) é degenerado neste desenho: com
``n`` datasets, o menor p-valor bilateral que o Wilcoxon signed-rank
consegue produzir é ``2 / 2**n`` — um piso de resolução do teste, não uma
propriedade dos dados. Multiplicado pelo número de comparações da família
(190 no Tier 1, 120 no Tier 2), esse piso ultrapassa alfa para *qualquer*
conjunto de dados:

    Tier 1: 2/2**10 = 0,00195  ×190 = 0,371   (nada < 0,05 é atingível)
    Tier 2: 2/2**6  = 0,03125  ×120 = 3,75    (truncado em 1,000)

O teste de Nemenyi escapa disso por operar sobre os *ranks médios*, cuja
distribuição amostral é contínua (amplitude studentizada), e é o
procedimento recomendado por Demšar (2006) para muitos modelos e poucos
datasets.

Saídas
------
    dissertacao-latex/tables/tier{1,2}_nemenyi.tex
    dissertacao-latex/Figuras/fig_cd_tier{1,2}.pdf
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.plots import cd_diagram          # noqa: E402
from src.metrics.statistical import nemenyi_cd     # noqa: E402

THESIS = ROOT.parent / "dissertacao-latex"

# Piso de resolução do Wilcoxon signed-rank exato, bilateral, com n pares.
def wilcoxon_floor(n_datasets: int) -> float:
    return 2.0 / (2.0 ** n_datasets)


def read_ranks(path: Path) -> list[tuple[str, float]]:
    """Lê a tabela de ranks médios de Friedman já publicada."""
    out: list[tuple[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(\d+)\s*&\s*(.+?)\s*&\s*([\d.]+)\s*\\\\", line)
        if m:
            out.append((m.group(2).strip(), float(m.group(3))))
    if not out:
        raise ValueError(f"nenhum rank lido de {path}")
    return out


def nemenyi_table(tier: int, rows: list[tuple[str, float]], n_datasets: int,
                  cd: float) -> str:
    """Tabela: para cada modelo, de quantos/quais ele difere significativamente."""
    names = [r[0] for r in rows]
    ranks = np.array([r[1] for r in rows])
    best_name, best_rank = rows[0]

    body = []
    for name, rk in rows:
        diff = rk - best_rank
        n_beaten = int(np.sum(ranks - rk > cd))     # quantos ele supera signif.
        n_lost = int(np.sum(rk - ranks > cd))       # quantos o superam signif.
        vs_best = "---" if name == best_name else (
            r"\textbf{sim}" if diff > cd else "não")
        body.append(
            f"    {name} & {rk:.3f} & {diff:.3f} & {vs_best} & {n_beaten} & {n_lost} \\\\"
        )

    n_pairs = len(rows) * (len(rows) - 1) // 2
    n_sig = sum(1 for i in range(len(rows)) for j in range(i + 1, len(rows))
                if abs(ranks[i] - ranks[j]) > cd)

    return "\n".join([
        r"\begin{table}[H]",
        r"  \centering",
        rf"  \caption{{Pós-teste de Nemenyi --- Tier {tier} "
        rf"($k={len(rows)}$ modelos, $n={n_datasets}$ \textit{{datasets}}, "
        rf"$\mathrm{{CD}}={cd:.3f}$ a $\alpha=0{{,}}05$). "
        rf"``Difere do 1º'' indica diferença significativa em relação ao "
        rf"{best_name}; as duas últimas colunas contam de quantos modelos "
        rf"cada um difere significativamente para melhor e para pior. "
        rf"Ao todo, {n_sig} dos {n_pairs} pares são significativos.}}",
        rf"  \label{{tab:tier{tier}_nemenyi}}",
        r"  \begin{tabular}{lrrccc}",
        r"    \toprule",
        r"    Modelo & \textit{Rank} & $\Delta$ p/ 1º & Difere do 1º & Supera & É superado \\",
        r"    \midrule",
        *body,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ])


def process(tier: int, ranks_file: Path, n_datasets: int, dry: bool) -> None:
    rows = read_ranks(ranks_file)
    k = len(rows)
    cd = nemenyi_cd(k, n_datasets, alpha=0.05)
    floor = wilcoxon_floor(n_datasets)
    n_pairs = k * (k - 1) // 2

    print(f"\n===== TIER {tier} =====")
    print(f"  k={k} modelos, n={n_datasets} datasets")
    print(f"  CD (Nemenyi, alpha=0,05) = {cd:.3f}")
    print(f"  piso do Wilcoxon: 2/2^{n_datasets} = {floor:.5f}; "
          f"x{n_pairs} (Holm) = {min(floor * n_pairs, 1.0):.3f} "
          f"-> {'DEGENERADO' if floor * n_pairs > 0.05 else 'ok'}")

    ranks = np.array([r[1] for r in rows])
    n_sig = sum(1 for i in range(k) for j in range(i + 1, k)
                if abs(ranks[i] - ranks[j]) > cd)
    print(f"  pares significativos pelo Nemenyi: {n_sig}/{n_pairs}")

    tbl = nemenyi_table(tier, rows, n_datasets, cd)
    fig = cd_diagram(ranks, [r[0] for r in rows], cd,
                     title=f"Diagrama de Diferença Crítica — F1-macro (Tier {tier})")

    if dry:
        print("  [dry-run] nada gravado")
        return

    out_tbl = THESIS / "tables" / f"tier{tier}_nemenyi.tex"
    out_fig = THESIS / "Figuras" / f"fig_cd_tier{tier}.pdf"
    out_tbl.write_text(tbl, encoding="utf-8")
    fig.savefig(out_fig, bbox_inches="tight")
    print(f"  -> {out_tbl.relative_to(THESIS.parent)}")
    print(f"  -> {out_fig.relative_to(THESIS.parent)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    for tier, n_datasets in [(1, 10), (2, 6)]:
        process(tier, THESIS / "tables" / f"tier{tier}_ranks.tex",
                n_datasets, a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

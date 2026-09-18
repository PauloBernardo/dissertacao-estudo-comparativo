#!/usr/bin/env python3
"""Substitui, num JSON canônico de resultados, os registros de determinadas
variantes pelos registros de um JSON de re-execução.

Uso típico (pós-auditoria 2026-09-18, SAINT fiel + entmax corrigido):

    python scripts/merge_rerun_results.py \
        --target results/tier1_gridcv.json \
        --source results/rerun_saint_entmax_tier1.json \
        --variants SAINTColnorm FTTransformer_entmax

O alvo recebe um backup ``<nome>_pre_<tag>_backup.json`` antes de ser
sobrescrito. Só registros com ``status == "ok"`` do source entram; os
registros do alvo das mesmas variantes são REMOVIDOS integralmente (não se
faz união por (dataset, seed)), para que nenhum resultado da versão antiga
sobreviva. ``--check`` apenas relata o que seria feito.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--tag", default="rerun", help="sufixo do backup")
    ap.add_argument("--check", action="store_true", help="não escreve nada")
    args = ap.parse_args()

    target = json.loads(args.target.read_text())
    source = json.loads(args.source.read_text())
    if not isinstance(target, list) or not isinstance(source, list):
        raise SystemExit("ambos os arquivos devem ser listas de registros")

    variants = set(args.variants)
    new = [r for r in source if r.get("variant") in variants and r.get("status", "ok") == "ok"]
    missing = variants - {r["variant"] for r in new}
    if missing:
        raise SystemExit(f"source não contém registros ok para: {sorted(missing)}")

    kept = [r for r in target if r.get("variant") not in variants]
    removed = len(target) - len(kept)

    print(f"alvo: {args.target}  ({len(target)} registros)")
    print(f"  removidos (variantes antigas): {removed}")
    print(f"  adicionados (re-execução):     {len(new)}")
    for v, c in sorted(Counter(r["variant"] for r in new).items()):
        old = sum(1 for r in target if r.get("variant") == v)
        flag = "" if old == c else f"   <<< ATENÇÃO: alvo tinha {old}"
        print(f"    {v:<28} {c:>5}{flag}")

    if args.check:
        print("(--check: nada escrito)")
        return

    backup = args.target.with_name(f"{args.target.stem}_pre_{args.tag}_backup.json")
    shutil.copy(args.target, backup)
    args.target.write_text(json.dumps(kept + new, indent=2))
    print(f"backup: {backup}\nescrito: {args.target} ({len(kept) + len(new)} registros)")


if __name__ == "__main__":
    main()

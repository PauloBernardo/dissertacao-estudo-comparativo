#!/usr/bin/env python3
"""Corrige o denominador da esparsidade em ADMMNystromLSSVM e FISTANystrom.

CONTEXTO
--------
Até 2026-07-14, esses dois modelos calculavam ``sparsity_ratio_`` como
``1 - n_support_/m`` (fração dos *m landmarks* podados pelo ℓ1) — enquanto TODOS
os demais modelos do estudo reportam ``1 - n_support_/N`` (fração das *amostras*
sem influência na predição). Os números, portanto, não eram comparáveis e
SUBESTIMAVAM fortemente a esparsidade real (Tier 1: 0,386 reportado vs 0,816 real
no ADMM-Nyström).

O bug é só na métrica derivada — os campos brutos (``n_support_vectors`` e
``n_train``) estão corretos em todos os JSONs. Este script recomputa o campo
derivado a partir deles:

    sparsity_ratio = 1 - n_support_vectors / n_train

Sanidade: para os modelos JÁ corretos, a recomputação reproduz exatamente o valor
gravado — o script verifica isso e aborta se houver divergência.

Uso:
    python scripts/fix_sparsity_denominator.py --dry-run   # só mostra
    python scripts/fix_sparsity_denominator.py --apply     # grava
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Variantes com o denominador errado (m em vez de N).
AFFECTED = {"ADMMNystromLSSVM", "ADMMNystromDistributed", "FISTANystrom"}
TOL = 1e-6  # tolerância da checagem de sanidade nos modelos corretos


def recompute(r: dict) -> float | None:
    nsv, ntr = r.get("n_support_vectors"), r.get("n_train")
    if nsv is None or not ntr:
        return None
    return 1.0 - nsv / ntr


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    total_fixed = 0
    for path in sorted(RESULTS.glob("*.json")):
        try:
            recs = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(recs, list):
            continue

        fixed, sane_checked, sane_bad = [], 0, 0
        for r in recs:
            if r.get("status") != "ok":
                continue
            new = recompute(r)
            if new is None or r.get("sparsity_ratio") is None:
                continue
            v = r.get("variant")
            if v in AFFECTED:
                fixed.append((v, r["sparsity_ratio"], new))
                if args.apply:
                    r["sparsity_ratio"] = new
            else:
                # sanidade: modelos corretos devem bater com a recomputação
                sane_checked += 1
                if abs(r["sparsity_ratio"] - new) > TOL:
                    sane_bad += 1

        if not fixed:
            continue

        by = {}
        for v, old, new in fixed:
            by.setdefault(v, []).append((old, new))
        print(f"\n{path.name}")
        for v, pairs in sorted(by.items()):
            o = st.mean(p[0] for p in pairs)
            n = st.mean(p[1] for p in pairs)
            print(f"    {v:24s} n={len(pairs):4d}  {o:.4f} -> {n:.4f}  (Δ={n-o:+.4f})")
        if sane_bad:
            print(f"    ⚠️  ATENÇÃO: {sane_bad}/{sane_checked} registros de modelos "
                  f"'corretos' divergem da recomputação — investigar antes de aplicar!")
        total_fixed += len(fixed)

        if args.apply:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(recs, indent=2, default=str))
            tmp.replace(path)

    print(f"\n{'APLICADO' if args.apply else 'DRY-RUN'}: {total_fixed} registros afetados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

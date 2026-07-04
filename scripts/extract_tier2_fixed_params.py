#!/usr/bin/env python3
"""Extrai hiperparâmetros FIXOS para a ablação Tier 2 em N maior (5000).

Para cada (modelo, dataset), toma a MODA dos ``best_params`` escolhidos pelo
GridSearchCV do Tier 2 em N=2000 (arquivos de resultados existentes), sobre as
30 sementes. O resultado é a configuração vencedora mais frequente em N=2000,
que será mantida FIXA ao rodar em N=5000 (ablação: isola o efeito de N sem
re-tunar).

Fontes:
    results/tier2_gridcv.json          — LSSVMs + XGBoost (9 variantes)
    results/tier2_transformers*.json    — 6 variantes de Transformer

Saída:
    config/tier2_fixed_params.json      — {variant: {dataset: {param: valor}}}

Uso:
    python scripts/extract_tier2_fixed_params.py
"""

from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "config" / "tier2_fixed_params.json"

DATASETS = ["ADULT", "BANK", "CREDIT", "HIGGS50K", "SHOPPERS", "TELCO"]

# Arquivos-fonte. Transformers estão espalhados em vários dumps; todos são lidos
# e deduplicados por (variant, dataset, seed).
LSSVM_FILE = RESULTS / "tier2_gridcv.json"
TRANSFORMER_GLOBS = ["tier2_transformers*.json"]

N_TRAIN_SOURCE = 2000  # o GridCV foi feito em N=2000


def _load(path: Path) -> list[dict]:
    try:
        d = json.loads(path.read_text())
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _records() -> list[dict]:
    recs = list(_load(LSSVM_FILE))
    seen_files = {LSSVM_FILE.name}
    for pattern in TRANSFORMER_GLOBS:
        for f in sorted(RESULTS.glob(pattern)):
            if f.name in seen_files:
                continue
            seen_files.add(f.name)
            recs.extend(_load(f))
    return recs


def main() -> None:
    recs = _records()

    # dedup por (variant, dataset, seed) — mantém a 1ª ocorrência
    dedup: dict[tuple, dict] = {}
    for r in recs:
        variant = r.get("variant") or r.get("model")
        bp = r.get("best_params")
        nt = r.get("n_train_target", r.get("n_train"))
        if bp is None or variant is None:
            continue
        if nt is not None and nt != N_TRAIN_SOURCE:
            continue
        key = (variant, r.get("dataset"), r.get("seed"))
        dedup.setdefault(key, r)

    # moda dos best_params por (variant, dataset)
    by = collections.defaultdict(list)
    for (variant, dataset, _seed), r in dedup.items():
        by[(variant, dataset)].append(tuple(sorted(r["best_params"].items())))

    config: dict[str, dict] = collections.defaultdict(dict)
    report_rows = []
    for (variant, dataset), vals in sorted(by.items()):
        mode, cnt = collections.Counter(vals).most_common(1)[0]
        config[variant][dataset] = dict(mode)
        report_rows.append((variant, dataset, dict(mode), cnt, len(vals)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(config, indent=2, sort_keys=True))

    # relatório legível
    print(f"Variantes: {len(config)} | destino: {OUT.relative_to(ROOT)}\n")
    last = None
    for variant, dataset, params, cnt, tot in report_rows:
        if variant != last:
            print(f"=== {variant} ===")
            last = variant
        print(f"  {dataset:<10} {params}   [moda {cnt}/{tot} seeds]")
    missing = [(v, ds) for v in config for ds in DATASETS if ds not in config[v]]
    if missing:
        print("\nAVISO — sem dados para:", missing)


if __name__ == "__main__":
    main()

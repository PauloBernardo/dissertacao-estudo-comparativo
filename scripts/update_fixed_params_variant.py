#!/usr/bin/env python3
"""Atualiza em config/tier2_fixed_params.json SÓ as entradas de uma variante, com a moda dos best_params
de um JSON de Tier 2 (N=2000). Mesma regra de extract_tier2_fixed_params.py (moda por dataset, 1ª
ocorrência por semente), sem reescrever as outras variantes. Faz backup do config antes.

Uso: python scripts/update_fixed_params_variant.py --source results/tier2_pcp_fiel.json --variant PCPLSSVm
"""
import argparse, collections, json, shutil
from pathlib import Path

CFG = Path(__file__).resolve().parent.parent / "config" / "tier2_fixed_params.json"
ap = argparse.ArgumentParser()
ap.add_argument("--source", type=Path, required=True)
ap.add_argument("--variant", required=True)
ap.add_argument("--tag", default="pre_update")
a = ap.parse_args()

seen, by = set(), collections.defaultdict(list)
for r in json.loads(a.source.read_text()):
    if r.get("variant") != a.variant or r.get("status") != "ok" or not r.get("best_params"):
        continue
    if (r["dataset"], r["seed"]) in seen:
        continue
    seen.add((r["dataset"], r["seed"]))
    by[r["dataset"]].append(tuple(sorted(r["best_params"].items())))
cfg = json.loads(CFG.read_text())
shutil.copy(CFG, CFG.with_name(f"tier2_fixed_params_{a.tag}_backup.json"))
for ds, vals in sorted(by.items()):
    mode, cnt = collections.Counter(vals).most_common(1)[0]
    print(f"{ds:9s} {dict(mode)}  ({cnt}/{len(vals)})   antes: {cfg.get(a.variant, {}).get(ds)}")
    cfg.setdefault(a.variant, {})[ds] = dict(mode)
CFG.write_text(json.dumps(cfg, indent=2) + "\n")

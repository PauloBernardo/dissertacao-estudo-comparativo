#!/usr/bin/env python3
"""Tier 3: a semente 0 (teste limpo, nunca visto pela seleção) contra a média das sementes 1–9
(cujo teste compartilha ~40% das linhas com o treino da semente 0 usado na seleção).
Se o vazamento favorecesse as sementes 1–9, a semente 0 ficaria sistematicamente abaixo.

    .venv/bin/python scripts/tier3_seed0_check.py
"""
import collections, json, statistics as st
from pathlib import Path

R = [r for r in json.load(open(Path(__file__).resolve().parent.parent / "results/tier3_final.json"))
     if r.get("status", "ok") == "ok"]
D = collections.defaultdict(dict)
for r in R:
    D[(r["variant"], r["dataset"])][r["seed"]] = r["test_f1_macro"]
zs = []
for (v, d), S in sorted(D.items()):
    o = [S[s] for s in range(1, 10) if s in S]
    if 0 in S and len(o) >= 5:
        z = (S[0] - st.mean(o)) / st.stdev(o); zs.append(z)
        print(f"{v:24s} {d:10s} s0 {S[0]:.4f}  s1-9 {st.mean(o):.4f} ± {st.stdev(o):.4f}  z {z:+.2f}")
print(f"pares {len(zs)}; s0 acima em {sum(z > 0 for z in zs)}; z médio {st.mean(zs):+.2f}")
